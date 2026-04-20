# ====================== src/rag/document_loader.py ======================
"""
DocumentLoader - Carregador de Documentos.

Este módulo é responsável por carregar documentos de diversas fontes
e prepará-los para armazenamento no banco vetorial.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib


class DocumentLoader:
    """
    Carregador de documentos para o sistema RAG.
    
    Suporta carregamento de:
    - Arquivos de texto (.txt)
    - Arquivos JSON
    - Strings diretas
    - Diretórios inteiros
    """
    
    def __init__(self):
        """Inicializa o DocumentLoader."""
        self.supported_extensions = [".txt", ".json", ".md"]
    
    def _generate_id(self, content: str) -> str:
        """
        Gera um ID único baseado no conteúdo.
        
        Args:
            content: Conteúdo do documento
            
        Returns:
            Hash MD5 do conteúdo (primeiros 16 caracteres)
        """
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def load_text_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Carrega um arquivo de texto simples.
        
        Args:
            file_path: Caminho para o arquivo .txt
            metadata: Metadados opcionais
            
        Returns:
            Dicionário com id, content e metadata
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        
        if path.suffix not in self.supported_extensions:
            raise ValueError(f"Extensão não suportada: {path.suffix}")
        
        # Ler conteúdo
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Metadados padrão
        if metadata is None:
            metadata = {}
        
        metadata["source"] = str(path)
        metadata["type"] = "text_file"
        metadata["filename"] = path.name
        
        return {
            "id": self._generate_id(content),
            "content": content,
            "metadata": metadata
        }
    
    def load_json_file(self, file_path: str, content_key: str = "content") -> List[Dict[str, Any]]:
        """
        Carrega documentos de um arquivo JSON.
        
        Espera-se que o JSON seja uma lista de objetos ou um objeto único.
        
        Args:
            file_path: Caminho para o arquivo .json
            content_key: Chave que contém o conteúdo textual
            
        Returns:
            Lista de documentos
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        documents = []
        
        # Se for lista, processar cada item
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    content = item.get(content_key, str(item))
                    doc_metadata = {k: v for k, v in item.items() if k != content_key}
                    doc_metadata["source"] = str(path)
                    doc_metadata["type"] = "json_file"
                    
                    documents.append({
                        "id": self._generate_id(content),
                        "content": content,
                        "metadata": doc_metadata
                    })
                else:
                    # Item não é dicionário, usar como string
                    content = str(item)
                    documents.append({
                        "id": self._generate_id(content),
                        "content": content,
                        "metadata": {"source": str(path), "type": "json_file"}
                    })
        elif isinstance(data, dict):
            # Objeto único
            content = data.get(content_key, str(data))
            doc_metadata = {k: v for k, v in data.items() if k != content_key}
            doc_metadata["source"] = str(path)
            doc_metadata["type"] = "json_file"
            
            documents.append({
                "id": self._generate_id(content),
                "content": content,
                "metadata": doc_metadata
            })
        
        return documents
    
    def load_string(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Carrega uma string direta como documento.
        
        Args:
            content: Conteúdo textual
            metadata: Metadados opcionais
            doc_id: ID opcional (gera automático se não fornecido)
            
        Returns:
            Dicionário com id, content e metadata
        """
        if metadata is None:
            metadata = {}
        
        metadata["type"] = "string"
        
        return {
            "id": doc_id or self._generate_id(content),
            "content": content,
            "metadata": metadata
        }
    
    def load_directory(
        self, 
        dir_path: str, 
        recursive: bool = True,
        metadata_template: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Carrega todos os documentos de um diretório.
        
        Args:
            dir_path: Caminho do diretório
            recursive: Se deve buscar em subdiretórios
            metadata_template: Metadados base para todos os documentos
            
        Returns:
            Lista de documentos encontrados
        """
        path = Path(dir_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {dir_path}")
        
        if not path.is_dir():
            raise NotADirectoryError(f"Não é um diretório: {dir_path}")
        
        documents = []
        
        # Padrão de busca
        pattern = "**/*" if recursive else "*"
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix in self.supported_extensions:
                try:
                    metadata = metadata_template.copy() if metadata_template else {}
                    metadata["directory"] = str(path)
                    
                    if file_path.suffix == ".json":
                        docs = self.load_json_file(str(file_path))
                        for doc in docs:
                            doc["metadata"].update(metadata)
                        documents.extend(docs)
                    else:
                        doc = self.load_text_file(str(file_path), metadata)
                        documents.append(doc)
                        
                except Exception as e:
                    print(f"[DocumentLoader] Erro ao carregar {file_path}: {e}")
        
        print(f"[DocumentLoader] {len(documents)} documentos carregados de {dir_path}")
        return documents
    
    def create_sample_documents(self) -> List[Dict[str, Any]]:
        """
        Cria documentos de exemplo sobre IA e tecnologia.
        Útil para testar o sistema sem dados externos.
        
        Returns:
            Lista de documentos de exemplo
        """
        samples = [
            {
                "title": "O que é Inteligência Artificial?",
                "content": """Inteligência Artificial (IA) é um campo da ciência da computação que visa criar sistemas capazes de realizar tarefas que normalmente requerem inteligência humana. 
                
Isso inclui aprendizado, raciocínio, percepção, compreensão de linguagem natural e resolução de problemas. 

A IA pode ser classificada em:
- IA Estreita (ANI): Especializada em uma tarefa específica
- IA Geral (AGI): Capacidade intelectual humana completa (ainda teórica)
- Superinteligência:超越 capacidade humana (conceito futurista)

Aplicações comuns incluem: assistentes virtuais, carros autônomos, diagnósticos médicos, tradução automática e muito mais."""
            },
            {
                "title": "Machine Learning - Aprendizado de Máquina",
                "content": """Machine Learning (ML) é um subcampo da Inteligência Artificial que permite aos sistemas aprenderem e melhorarem automaticamente através da experiência.

Tipos principais:
1. Aprendizado Supervisionado: Modelo treinado com dados rotulados
2. Aprendizado Não-Supervisionado: Modelo encontra padrões em dados não-rotulados
3. Aprendizado por Reforço: Agente aprende através de recompensas e punições

Algoritmos populares incluem: Redes Neurais, Decision Trees, SVM, K-Means, entre outros."""
            },
            {
                "title": "Redes Neurais e Deep Learning",
                "content": """Redes Neurais Artificiais são sistemas computacionais inspirados no cérebro humano.

Deep Learning usa redes neurais com múltiplas camadas para aprender representações complexas de dados.

Arquiteturas comuns:
- CNN (Convolutional Neural Networks): Para processamento de imagens
- RNN (Recurrent Neural Networks): Para sequências temporais
- Transformers: Arquitetura moderna para NLP (ex: GPT, BERT)

Aplicações: Reconhecimento facial, tradução, geração de texto, diagnóstico médico."""
            },
            {
                "title": "Processamento de Linguagem Natural (NLP)",
                "content": """NLP é o campo da IA que foca na interação entre computadores e linguagem humana.

Tarefas principais:
- Análise de sentimentos
- Tradução automática
- Resumo de texto
- Resposta a perguntas
- Geração de texto

Modelos modernos como GPT, BERT e LLaMA revolucionaram o campo, permitindo conversas naturais e compreensão contextual profunda."""
            },
            {
                "title": "Ética em Inteligência Artificial",
                "content": """A ética em IA é crucial para garantir que sistemas sejam desenvolvidos e usados responsavelmente.

Princípios importantes:
- Transparência: Sistemas devem ser explicáveis
- Justiça: Evitar viés e discriminação
- Privacidade: Proteger dados dos usuários
- Responsabilidade: Definir quem responde pelos resultados
- Segurança: Prevenir uso malicioso

Desafios atuais incluem: viés algorítmico, deepfakes, impacto no emprego, e autonomia de sistemas."""
            }
        ]
        
        documents = []
        for sample in samples:
            doc = {
                "id": self._generate_id(sample["content"]),
                "content": sample["content"],
                "metadata": {
                    "title": sample["title"],
                    "type": "sample",
                    "language": "pt-BR"
                }
            }
            documents.append(doc)
        
        print(f"[DocumentLoader] {len(documents)} documentos de exemplo criados.")
        return documents
