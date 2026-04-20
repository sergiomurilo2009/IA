# ====================== src/agents/retriever.py ======================
"""
RetrieverAgent - Agente de Recuperação (RAG).

Este agente é responsável por buscar informações relevantes no banco
de dados vetorial usando técnicas de Retrieval-Augmented Generation (RAG).
"""

from typing import Optional, List
from langchain_ollama import ChatOllama

from ..config import Config
from ..rag.vectorstore import VectorStoreManager


class RetrieverAgent:
    """
    Agente especializado em recuperação de informações do banco vetorial.
    
    Este agente usa o VectorStoreManager para buscar documentos relevantes
    que possam ajudar a responder a query do usuário.
    """
    
    def __init__(self, vectorstore: Optional[VectorStoreManager] = None):
        """
        Inicializa o RetrieverAgent.
        
        Args:
            vectorstore: Instância do VectorStoreManager (cria uma nova se não fornecida)
        """
        if vectorstore is None:
            self.vectorstore = VectorStoreManager()
        else:
            self.vectorstore = vectorstore
        
        # LLM para processamento adicional se necessário
        self.llm = ChatOllama(
            model=Config.LLM_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.3
        )
    
    def retrieve(self, query: str, n_results: int = 5) -> List[dict]:
        """
        Busca documentos relevantes no banco vetorial.
        
        Args:
            query: Texto da consulta
            n_results: Número máximo de resultados
            
        Returns:
            Lista de documentos encontrados
        """
        return self.vectorstore.search(query, n_results=n_results)
    
    def get_context_string(self, documents: List[dict]) -> str:
        """
        Formata documentos recuperados como string de contexto.
        
        Args:
            documents: Lista de documentos recuperados
            
        Returns:
            String formatada com o conteúdo dos documentos
        """
        if not documents:
            return "Nenhum documento relevante encontrado no banco de dados."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            relevance = doc.get("relevance_score", 0)
            content = doc.get("content", "")
            title = doc.get("metadata", {}).get("title", f"Documento {i}")
            
            context_parts.append(
                f"[Fonte {i}] {title}\n"
                f"Relevância: {relevance:.2f}\n"
                f"Conteúdo: {content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def __call__(self, state: dict) -> dict:
        """
        Método callable para integração com LangGraph.
        
        Este método busca documentos relevantes baseados na query
        e análise prévias, adicionando o contexto ao estado.
        
        Args:
            state: Estado atual contendo 'query' e 'query_analysis'
            
        Returns:
            Estado atualizado com 'retrieved_documents' e 'context'
        """
        query = state.get("query", "")
        query_analysis = state.get("query_analysis", {})
        
        # Usar query esclarecida se disponível
        search_query = query_analysis.get("clarified_query", query)
        
        # Determinar número de resultados baseado na complexidade
        complexity = query_analysis.get("complexity", "media")
        n_results_map = {"baixa": 3, "media": 5, "alta": 7}
        n_results = n_results_map.get(complexity, 5)
        
        print(f"[Retriever] Buscando por: '{search_query}' ({n_results} resultados)")
        
        # Realizar busca
        documents = self.retrieve(search_query, n_results=n_results)
        
        print(f"[Retriever] {len(documentos)} documentos encontrados.")
        
        # Formatar contexto
        context = self.get_context_string(documents)
        
        # Atualizar estado
        return {
            **state,
            "retrieved_documents": documents,
            "context": context,
            "has_context": len(documents) > 0
        }
