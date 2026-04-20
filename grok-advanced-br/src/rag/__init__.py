# ====================== src/rag/__init__.py ======================
"""
Pacote RAG (Retrieval-Augmented Generation) do sistema Grok-Advanced-BR.

Contém módulos para:
- VectorStore: Gerenciamento do banco de dados vetorial ChromaDB
- DocumentLoader: Carregamento e processamento de documentos
"""

from .vectorstore import VectorStoreManager
from .document_loader import DocumentLoader

__all__ = ["VectorStoreManager", "DocumentLoader"]
