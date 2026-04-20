# ====================== src/rag/vectorstore.py ======================
"""
VectorStoreManager - Gerenciador do Banco de Dados Vetorial.

Este módulo gerencia o ChromaDB para armazenamento e recuperação
de documentos vetoriais, permitindo busca semântica eficiente.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import os

from ..config import Config


class VectorStoreManager:
    """
    Gerenciador do banco de dados vetorial ChromaDB.
    
    Responsável por:
    - Inicializar conexão com ChromaDB
    - Gerenciar embeddings com sentence-transformers
    - Adicionar documentos ao banco
    - Realizar buscas semânticas
    """
    
    def __init__(self, collection_name: str = "grok_knowledge"):
        """
        Inicializa o gerenciador do vector store.
        
        Args:
            collection_name: Nome da coleção no ChromaDB
        """
        self.collection_name = collection_name
        
        # Inicializar cliente ChromaDB com persistência
        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_PERSIST_DIR,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Inicializar modelo de embedding local
        print(f"[VectorStore] Carregando modelo de embedding: {Config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
        # Obter ou criar coleção
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Similaridade por cosseno
        )
        
        print(f"[VectorStore] Coleção '{collection_name}' pronta.")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Gera embedding vetorial para um texto.
        
        Args:
            text: Texto para gerar embedding
            
        Returns:
            Lista de floats representando o embedding
        """
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def add_document(
        self, 
        doc_id: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Adiciona um documento ao banco vetorial.
        
        Args:
            doc_id: Identificador único do documento
            content: Conteúdo textual do documento
            metadata: Metadados opcionais (autor, data, tags, etc.)
        """
        # Gerar embedding
        embedding = self._generate_embedding(content)
        
        # Metadados padrão
        if metadata is None:
            metadata = {}
        metadata["content"] = content
        
        # Adicionar à coleção
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[content]
        )
        
        print(f"[VectorStore] Documento '{doc_id}' adicionado.")
    
    def add_documents(
        self, 
        documents: List[Dict[str, Any]], 
        batch_size: int = 100
    ):
        """
        Adiciona múltiplos documentos em lote.
        
        Args:
            documents: Lista de dicionários com 'id', 'content' e 'metadata'
            batch_size: Tamanho do lote para processamento
        """
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            ids = []
            embeddings = []
            metadatas = []
            contents = []
            
            for doc in batch:
                ids.append(doc["id"])
                contents.append(doc["content"])
                embeddings.append(self._generate_embedding(doc["content"]))
                metadatas.append(doc.get("metadata", {}))
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=contents
            )
            
            print(f"[VectorStore] Lote de {len(batch)} documentos adicionado.")
    
    def search(
        self, 
        query: str, 
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca documentos relevantes no banco vetorial.
        
        Args:
            query: Texto da consulta
            n_results: Número máximo de resultados
            filter_metadata: Filtro opcional por metadados
            
        Returns:
            Lista de documentos encontrados com scores de relevância
        """
        # Gerar embedding da query
        query_embedding = self._generate_embedding(query)
        
        # Realizar busca
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        # Formatrar resultados
        formatted_results = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                result = {
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    "relevance_score": 1.0 - results["distances"][0][i] if results["distances"] else 1.0
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def get_document_count(self) -> int:
        """
        Retorna o número total de documentos no banco.
        
        Returns:
            Número de documentos
        """
        return self.collection.count()
    
    def delete_document(self, doc_id: str):
        """
        Remove um documento do banco.
        
        Args:
            doc_id: ID do documento a remover
        """
        self.collection.delete(ids=[doc_id])
        print(f"[VectorStore] Documento '{doc_id}' removido.")
    
    def reset_collection(self):
        """
        Reseta completamente a coleção (remove todos os documentos).
        """
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[VectorStore] Coleção '{self.collection_name}' resetada.")
    
    def __call__(self, state: dict) -> dict:
        """
        Método callable para integração com LangGraph.
        
        Args:
            state: Estado atual contendo 'query' e possivelmente 'query_analysis'
            
        Returns:
            Estado atualizado com 'retrieved_documents'
        """
        query = state.get("query", "")
        query_analysis = state.get("query_analysis", {})
        
        # Usar query esclarecida se disponível
        search_query = query_analysis.get("clarified_query", query)
        
        # Determinar número de resultados baseado na complexidade
        complexity = query_analysis.get("complexity", "media")
        n_results = {
            "baixa": 3,
            "media": 5,
            "alta": 7
        }.get(complexity, 5)
        
        # Realizar busca
        documents = self.search(search_query, n_results=n_results)
        
        # Formatar contexto para os próximos agentes
        context = "\n\n".join([
            f"[Documento {i+1}] (Relevância: {doc['relevance_score']:.2f})\n{doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        return {
            **state,
            "retrieved_documents": documents,
            "context": context if documents else "Nenhum documento relevante encontrado no banco de dados."
        }
