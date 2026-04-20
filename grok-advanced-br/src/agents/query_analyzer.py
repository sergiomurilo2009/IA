# ====================== src/agents/query_analyzer.py ======================
"""
QueryAnalyzerAgent - Agente de Análise de Query.

Este agente é responsável por analisar e classificar a pergunta do usuário,
extraindo informações importantes como:
- Tipo de pergunta (factual, conceitual, procedimental, etc.)
- Palavras-chave principais
- Contexto necessário
- Complexidade da query
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Optional
import json

from ..config import Config


class QueryAnalysis(BaseModel):
    """
    Estrutura para análise da query.
    
    Atributos:
        query_type: Tipo de pergunta (factual, conceitual, procedimental, opinativa)
        keywords: Lista de palavras-chave extraídas
        context_needed: Se precisa de contexto adicional do banco de dados
        complexity: Nível de complexidade (baixa, media, alta)
        clarified_query: Query reformulada para melhor entendimento
    """
    query_type: str = Field(description="Tipo de pergunta: factual, conceitual, procedimental ou opinativa")
    keywords: list[str] = Field(description="Lista de palavras-chave principais")
    context_needed: bool = Field(description="Se precisa buscar contexto no banco de dados")
    complexity: str = Field(description="Nível de complexidade: baixa, media ou alta")
    clarified_query: str = Field(description="Query reformulada e esclarecida")


class QueryAnalyzerAgent:
    """
    Agente especializado em analisar queries de usuários.
    
    Este agente usa um modelo LLM local via Ollama para entender
    a intenção e estrutura da pergunta do usuário.
    """
    
    def __init__(self, llm: Optional[ChatOllama] = None):
        """
        Inicializa o QueryAnalyzerAgent.
        
        Args:
            llm: Instância do ChatOllama (cria uma nova se não fornecida)
        """
        if llm is None:
            self.llm = ChatOllama(
                model=Config.LLM_MODEL,
                base_url=Config.OLLAMA_BASE_URL,
                temperature=0.3  # Temperatura baixa para análise mais consistente
            )
        else:
            self.llm = llm
        
        # Prompt para análise da query
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um analista de queries especializado.
Sua tarefa é analisar a pergunta do usuário e extrair informações importantes.

Responda APENAS com um JSON válido neste formato:
{{
    "query_type": "tipo da pergunta",
    "keywords": ["palavra1", "palavra2"],
    "context_needed": true/false,
    "complexity": "baixa/media/alta",
    "clarified_query": "query reformulada"
}}

Tipos de pergunta:
- factual: Pergunta sobre fatos específicos
- conceitual: Pergunta sobre conceitos ou definições
- procedimental: Pergunta sobre como fazer algo
- opinativa: Pergunta que pede opinião ou avaliação"""),
            ("human", "Analise esta query: {query}")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def analyze(self, query: str) -> dict:
        """
        Analisa uma query do usuário.
        
        Args:
            query: Texto da pergunta do usuário
            
        Returns:
            Dicionário com a análise da query
        """
        try:
            # Executar a cadeia de análise
            response = self.chain.invoke({"query": query})
            
            # Tentar parsear como JSON
            # Remover possíveis marcações de código markdown
            response_clean = response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            response_clean = response_clean.strip()
            
            analysis = json.loads(response_clean)
            
            # Validar campos obrigatórios
            result = {
                "query_type": analysis.get("query_type", "desconhecido"),
                "keywords": analysis.get("keywords", []),
                "context_needed": analysis.get("context_needed", True),
                "complexity": analysis.get("complexity", "media"),
                "clarified_query": analysis.get("clarified_query", query),
                "original_query": query
            }
            
            return result
            
        except Exception as e:
            # Em caso de erro, retornar análise básica
            print(f"[QueryAnalyzer] Erro na análise: {e}")
            return {
                "query_type": "desconhecido",
                "keywords": [],
                "context_needed": True,
                "complexity": "media",
                "clarified_query": query,
                "original_query": query,
                "error": str(e)
            }
    
    def __call__(self, state: dict) -> dict:
        """
        Método callable para integração com LangGraph.
        
        Args:
            state: Estado atual do grafo contendo 'query'
            
        Returns:
            Estado atualizado com 'query_analysis'
        """
        query = state.get("query", "")
        analysis = self.analyze(query)
        
        # Atualizar o estado com a análise
        return {
            **state,
            "query_analysis": analysis,
            "needs_retrieval": analysis.get("context_needed", True)
        }
