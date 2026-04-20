# ====================== src/agents/__init__.py ======================
"""
Pacote de agentes do sistema Grok-Advanced-BR.

Contém todos os agentes especializados que compõem o sistema multi-agente:
- QueryAnalyzerAgent: Analisa a query do usuário
- RetrieverAgent: Busca informações no banco vetorial (RAG)
- ThinkerAgent: Realiza raciocínio profundo
- CriticAgent: Revisa e corrige respostas
- FinalAnswerAgent: Gera a resposta final
"""

from .query_analyzer import QueryAnalyzerAgent
from .retriever import RetrieverAgent
from .thinker import ThinkerAgent
from .critic import CriticAgent
from .final_answer import FinalAnswerAgent

__all__ = [
    "QueryAnalyzerAgent",
    "RetrieverAgent",
    "ThinkerAgent",
    "CriticAgent",
    "FinalAnswerAgent",
]
