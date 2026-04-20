# ====================== src/agents/thinker.py ======================
"""
ThinkerAgent - Agente de Raciocínio Profundo.

Este agente implementa Chain of Thought (CoT) e Tree of Thoughts (ToT)
para realizar raciocínio profundo sobre a query do usuário e o contexto
recuperado do banco de dados.

Segue rigorosamente o System Prompt interno obrigatório.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional, Dict, Any

from ..config import Config


class ThinkerAgent:
    """
    Agente especializado em raciocínio profundo usando CoT e ToT.
    
    Este agente segue o System Prompt interno obrigatório, realizando:
    1. Entendimento da pergunta
    2. Análise do contexto do banco de dados
    3. Identificação de informações faltantes
    4. Pensamento em múltiplos caminhos (Tree of Thoughts)
    5. Criação de rascunho
    6. Auto-crítica rigorosa
    7. Versão final corrigida
    """
    
    def __init__(self, llm: Optional[ChatOllama] = None):
        """
        Inicializa o ThinkerAgent.
        
        Args:
            llm: Instância do ChatOllama (cria uma nova se não fornecida)
        """
        if llm is None:
            self.llm = ChatOllama(
                model=Config.LLM_MODEL,
                base_url=Config.OLLAMA_BASE_URL,
                temperature=Config.TEMPERATURE
            )
        else:
            self.llm = llm
        
        # System Prompt Interno OBRIGATÓRIO
        self.system_prompt = Config.SYSTEM_PROMPT
        
        # Prompt para o processo de pensamento
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """
QUERY DO USUÁRIO: {query}

CONTEXTO DO BANCO DE DADOS:
{context}

ANÁLISE DA QUERY:
{query_analysis}

---

Agora siga rigorosamente o system prompt: faça seu raciocínio completo dentro de <pensamento> </pensamento> 
e depois responda apenas com <resposta_final> sua resposta aqui </resposta_final>
""")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def think(self, query: str, context: str, query_analysis: Dict[str, Any]) -> str:
        """
        Realiza o processo de raciocínio profundo.
        
        Args:
            query: Query original do usuário
            context: Contexto recuperado do banco de dados
            query_analysis: Análise da query feita pelo QueryAnalyzer
            
        Returns:
            Resposta completa com pensamento e resposta final
        """
        # Formatar análise da query
        analysis_str = "\n".join([f"- {k}: {v}" for k, v in query_analysis.items()])
        
        # Executar cadeia de pensamento
        response = self.chain.invoke({
            "query": query,
            "context": context,
            "query_analysis": analysis_str
        })
        
        return response
    
    def extract_thought(self, response: str) -> str:
        """
        Extrai a seção de pensamento da resposta.
        
        Args:
            response: Resposta completa do modelo
            
        Returns:
            Conteúdo dentro das tags <pensamento>
        """
        start_tag = "<pensamento>"
        end_tag = "</pensamento>"
        
        start_idx = response.find(start_tag)
        end_idx = response.find(end_tag)
        
        if start_idx != -1 and end_idx != -1:
            return response[start_idx + len(start_tag):end_idx].strip()
        
        return "Pensamento não encontrado na resposta."
    
    def extract_final_answer(self, response: str) -> str:
        """
        Extrai a resposta final da resposta completa.
        
        Args:
            response: Resposta completa do modelo
            
        Returns:
            Conteúdo dentro das tags <resposta_final>
        """
        start_tag = "<resposta_final>"
        end_tag = "</resposta_final>"
        
        start_idx = response.find(start_tag)
        end_idx = response.find(end_tag)
        
        if start_idx != -1 and end_idx != -1:
            return response[start_idx + len(start_tag):end_idx].strip()
        
        # Se não encontrar tags, retornar tudo após possível pensamento
        if "</pensamento>" in response:
            return response.split("</pensamento>")[-1].strip()
        
        return response.strip()
    
    def __call__(self, state: dict) -> dict:
        """
        Método callable para integração com LangGraph.
        
        Args:
            state: Estado atual contendo 'query', 'context' e 'query_analysis'
            
        Returns:
            Estado atualizado com 'thought_process', 'draft_answer' e 'final_answer'
        """
        query = state.get("query", "")
        context = state.get("context", "Nenhum contexto disponível.")
        query_analysis = state.get("query_analysis", {})
        
        print("[Thinker] Iniciando processo de raciocínio profundo...")
        
        # Realizar pensamento
        response = self.think(query, context, query_analysis)
        
        # Extrair componentes
        thought = self.extract_thought(response)
        draft_answer = self.extract_final_answer(response)
        
        print("[Thinker] Raciocínio completado.")
        
        # Atualizar estado
        return {
            **state,
            "thought_process": thought,
            "draft_answer": draft_answer,
            "full_response": response,
            "iteration_count": state.get("iteration_count", 0) + 1
        }
