# ====================== src/agents/final_answer.py ======================
"""
FinalAnswerAgent - Agente de Resposta Final.

Este agente é responsável por formatar e entregar a resposta final ao usuário,
após todo o processo de análise, recuperação, raciocínio e revisão.
"""

from typing import Optional, Dict, Any
from langchain_ollama import ChatOllama

from ..config import Config


class FinalAnswerAgent:
    """
    Agente especializado em formatar e entregar a resposta final.
    
    Este agente compila todas as informações processadas pelos agentes
    anteriores e entrega uma resposta clara, formatada e completa ao usuário.
    """
    
    def __init__(self, llm: Optional[ChatOllama] = None):
        """
        Inicializa o FinalAnswerAgent.
        
        Args:
            llm: Instância do ChatOllama (cria uma nova se não fornecida)
        """
        if llm is None:
            self.llm = ChatOllama(
                model=Config.LLM_MODEL,
                base_url=Config.OLLAMA_BASE_URL,
                temperature=0.3  # Temperatura baixa para formatação consistente
            )
        else:
            self.llm = llm
    
    def format_response(
        self,
        query: str,
        final_answer: str,
        query_analysis: Dict[str, Any],
        has_context: bool,
        total_iterations: int,
        thought_process: Optional[str] = None,
        include_sources: bool = False,
        retrieved_documents: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Formata a resposta final com metadados opcionais.
        
        Args:
            query: Query original do usuário
            final_answer: Resposta final processada
            query_analysis: Análise da query
            has_context: Se houve contexto do banco de dados
            total_iterations: Número de iterações de revisão
            thought_process: Processo de pensamento (opcional)
            include_sources: Se deve incluir fontes
            retrieved_documents: Documentos recuperados (se include_sources=True)
            
        Returns:
            Dicionário com resposta formatada e metadados
        """
        response_data = {
            "query": query,
            "answer": final_answer,
            "metadata": {
                "query_type": query_analysis.get("query_type", "desconhecido"),
                "complexity": query_analysis.get("complexity", "media"),
                "used_context": has_context,
                "review_iterations": total_iterations,
                "model": Config.LLM_MODEL
            }
        }
        
        # Adicionar fontes se solicitado
        if include_sources and retrieved_documents:
            sources = []
            for i, doc in enumerate(retrieved_documents, 1):
                source_info = {
                    "id": i,
                    "title": doc.get("metadata", {}).get("title", f"Fonte {i}"),
                    "relevance": doc.get("relevance_score", 0),
                    "content_preview": doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
                }
                sources.append(source_info)
            
            response_data["sources"] = sources
        
        # Adicionar pensamento se disponível (para debug/verbose)
        if thought_process:
            response_data["thought_process"] = thought_process
        
        return response_data
    
    def create_verbose_response(self, state: dict) -> str:
        """
        Cria uma resposta detalhada com todo o processo.
        
        Args:
            state: Estado completo do grafo
            
        Returns:
            String formatada com resposta completa e detalhes do processo
        """
        query = state.get("query", "")
        final_answer = state.get("final_answer", "")
        query_analysis = state.get("query_analysis", {})
        thought_process = state.get("thought_process", "")
        retrieved_documents = state.get("retrieved_documents", [])
        total_iterations = state.get("total_iterations", 1)
        
        # Construir resposta verbose
        parts = []
        
        # Cabeçalho
        parts.append("=" * 60)
        parts.append("GROK-ADVANCED-BR - RESPOSTA COMPLETA")
        parts.append("=" * 60)
        parts.append("")
        
        # Query do usuário
        parts.append(f"📝 PERGUNTA: {query}")
        parts.append("")
        
        # Análise da query
        parts.append("🔍 ANÁLISE DA QUERY:")
        parts.append(f"   Tipo: {query_analysis.get('query_type', 'N/A')}")
        parts.append(f"   Complexidade: {query_analysis.get('complexity', 'N/A')}")
        parts.append(f"   Palavras-chave: {', '.join(query_analysis.get('keywords', [])) or 'N/A'}")
        parts.append("")
        
        # Contexto usado
        has_context = len(retrieved_documents) > 0
        parts.append(f"📚 CONTEXTO DO BANCO DE DADOS: {'Sim' if has_context else 'Não'}")
        if has_context:
            parts.append(f"   Documentos encontrados: {len(retrieved_documents)}")
            for i, doc in enumerate(retrieved_documents[:3], 1):  # Mostrar até 3
                title = doc.get("metadata", {}).get("title", f"Documento {i}")
                relevance = doc.get("relevance_score", 0)
                parts.append(f"   [{i}] {title} (Relevância: {relevance:.2f})")
        parts.append("")
        
        # Processo de pensamento (resumido)
        if thought_process:
            parts.append("🧠 PROCESSO DE PENSAMENTO (resumo):")
            # Mostrar primeiras 500 caracteres do pensamento
            thought_preview = thought_process[:500]
            if len(thought_process) > 500:
                thought_preview += "..."
            parts.append(f"   {thought_preview}")
            parts.append("")
        
        # Iterações de revisão
        parts.append(f"✅ REVISÕES REALIZADAS: {total_iterations}")
        parts.append("")
        
        # Resposta final
        parts.append("=" * 60)
        parts.append("RESPOSTA FINAL:")
        parts.append("=" * 60)
        parts.append(final_answer)
        parts.append("")
        parts.append("=" * 60)
        
        return "\n".join(parts)
    
    def create_simple_response(self, state: dict) -> str:
        """
        Cria apenas a resposta final simples (sem metadados).
        
        Args:
            state: Estado completo do grafo
            
        Returns:
            Apenas a resposta final
        """
        return state.get("final_answer", "Desculpe, não foi possível gerar uma resposta.")
    
    def __call__(self, state: dict) -> dict:
        """
        Método callable para integração com LangGraph.
        
        Args:
            state: Estado atual contendo todas as informações processadas
            
        Returns:
            Estado atualizado com 'formatted_response' e 'simple_answer'
        """
        query = state.get("query", "")
        final_answer = state.get("final_answer", "")
        query_analysis = state.get("query_analysis", {})
        thought_process = state.get("thought_process", "")
        retrieved_documents = state.get("retrieved_documents", [])
        total_iterations = state.get("total_iterations", 1)
        has_context = len(retrieved_documents) > 0
        
        print("[FinalAnswer] Formatando resposta final...")
        
        # Criar resposta formatada com metadados
        formatted_response = self.format_response(
            query=query,
            final_answer=final_answer,
            query_analysis=query_analysis,
            has_context=has_context,
            total_iterations=total_iterations,
            thought_process=thought_process,
            include_sources=True,
            retrieved_documents=retrieved_documents
        )
        
        # Criar resposta simples
        simple_answer = self.create_simple_response(state)
        
        # Criar resposta verbose para logging
        verbose_response = self.create_verbose_response(state)
        
        print("[FinalAnswer] Resposta finalizada.")
        
        return {
            **state,
            "formatted_response": formatted_response,
            "simple_answer": simple_answer,
            "verbose_response": verbose_response,
            "complete": True
        }
