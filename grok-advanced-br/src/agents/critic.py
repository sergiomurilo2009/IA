# ====================== src/agents/critic.py ======================
"""
CriticAgent - Agente Revisor com Auto-Correção.

Este agente realiza revisão crítica das respostas geradas pelo ThinkerAgent,
identificando problemas e sugerindo correções em um loop de até 3 iterações.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional, Dict, Any, Tuple

from ..config import Config


class CriticAgent:
    """
    Agente especializado em revisão e auto-correção de respostas.
    
    Este agente analisa criticamente a resposta gerada, identificando:
    - Erros factuais
    - Inconsistências lógicas
    - Informações faltantes
    - Problemas de clareza
    - Violações do system prompt
    
    Realiza até 3 iterações de correção conforme configurado.
    """
    
    def __init__(self, llm: Optional[ChatOllama] = None):
        """
        Inicializa o CriticAgent.
        
        Args:
            llm: Instância do ChatOllama (cria uma nova se não fornecida)
        """
        if llm is None:
            self.llm = ChatOllama(
                model=Config.LLM_MODEL,
                base_url=Config.OLLAMA_BASE_URL,
                temperature=0.5  # Temperatura moderada para análise equilibrada
            )
        else:
            self.llm = llm
        
        # Prompt para crítica e revisão
        self.critic_prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um revisor crítico rigoroso e construtivo.
Sua tarefa é analisar respostas e identificar problemas para melhoria.

Avalie os seguintes aspectos:
1. PRECISÃO: A resposta está factualmente correta?
2. COMPLETUDE: Todas as partes da pergunta foram respondidas?
3. CLAREZA: A resposta é clara e fácil de entender?
4. CONSISTÊNCIA: Não há contradições internas?
5. FORMATO: Segue o formato solicitado (<resposta_final>)?

Se encontrar problemas, explique O QUE precisa ser corrigido.
Se a resposta estiver boa, diga "APROVADO"."""),
            ("human", """
QUERY ORIGINAL: {query}

RESPOSTA ATUAL:
{draft_answer}

CONTEXTO DISPONÍVEL:
{context}

---

Analise criticamente esta resposta. Liste os problemas encontrados (se houver)
e explique o que precisa ser corrigido. Se estiver boa, escreva apenas "APROVADO".
""")
        ])
        
        # Prompt para reescrita baseada na crítica
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", Config.SYSTEM_PROMPT),
            ("human", """
QUERY DO USUÁRIO: {query}

CONTEXTO DO BANCO DE DADOS:
{context}

RESPOSTA ANTERIOR:
{draft_answer}

CRÍTICA RECEBIDA:
{criticism}

---

Com base na crítica acima, reescreva a resposta corrigindo todos os problemas identificados.
Siga o system prompt: raciocínio em <pensamento> e resposta em <resposta_final>.
""")
        ])
        
        self.critic_chain = self.critic_prompt | self.llm | StrOutputParser()
        self.rewrite_chain = self.rewrite_prompt | self.llm | StrOutputParser()
    
    def criticize(self, query: str, draft_answer: str, context: str) -> str:
        """
        Realiza crítica construtiva da resposta.
        
        Args:
            query: Query original do usuário
            draft_answer: Resposta atual para revisão
            context: Contexto do banco de dados
            
        Returns:
            Texto da crítica com problemas identificados
        """
        criticism = self.critic_chain.invoke({
            "query": query,
            "draft_answer": draft_answer,
            "context": context
        })
        
        return criticism
    
    def rewrite(self, query: str, draft_answer: str, context: str, criticism: str) -> str:
        """
        Reescreve a resposta baseada na crítica.
        
        Args:
            query: Query original do usuário
            draft_answer: Resposta anterior
            context: Contexto do banco de dados
            criticism: Crítica recebida
            
        Returns:
            Nova versão da resposta
        """
        rewritten = self.rewrite_chain.invoke({
            "query": query,
            "draft_answer": draft_answer,
            "context": context,
            "criticism": criticism
        })
        
        return rewritten
    
    def is_approved(self, criticism: str) -> bool:
        """
        Verifica se a resposta foi aprovada pela crítica.
        
        Args:
            criticism: Texto da crítica
            
        Returns:
            True se aprovado, False caso contrário
        """
        return "APROVADO" in criticism.upper()
    
    def review_and_correct(
        self, 
        query: str, 
        draft_answer: str, 
        context: str,
        max_iterations: int = None
    ) -> Tuple[str, str, int]:
        """
        Realiza loop de revisão e correção.
        
        Args:
            query: Query original do usuário
            draft_answer: Resposta inicial para revisão
            context: Contexto do banco de dados
            max_iterations: Número máximo de iterações (usa Config.MAX_ITERATIONS se None)
            
        Returns:
            Tupla com (resposta_final, histórico_de_críticas, número_de_iterações)
        """
        if max_iterations is None:
            max_iterations = Config.MAX_ITERATIONS
        
        current_answer = draft_answer
        criticism_history = []
        iterations = 0
        
        print(f"[Critic] Iniciando revisão (máximo de {max_iterations} iterações)...")
        
        for i in range(max_iterations):
            iterations = i + 1
            
            # Realizar crítica
            print(f"[Critic] Iteração {iterations}: Analisando resposta...")
            criticism = self.criticize(query, current_answer, context)
            criticism_history.append({
                "iteration": iterations,
                "criticism": criticism
            })
            
            # Verificar aprovação
            if self.is_approved(criticism):
                print(f"[Critic] Resposta APROVADA na iteração {iterations}.")
                break
            
            print(f"[Critic] Problemas encontrados. Reescrevendo...")
            
            # Reescrever resposta
            current_answer = self.rewrite(query, current_answer, context, criticism)
        
        # Extrair resposta final
        final_answer = self._extract_final_answer(current_answer)
        
        print(f"[Critic] Revisão completada em {iterations} iteração(ões).")
        
        return final_answer, criticism_history, iterations
    
    def _extract_final_answer(self, response: str) -> str:
        """Extrai a resposta final das tags."""
        start_tag = "<resposta_final>"
        end_tag = "</resposta_final>"
        
        start_idx = response.find(start_tag)
        end_idx = response.find(end_tag)
        
        if start_idx != -1 and end_idx != -1:
            return response[start_idx + len(start_tag):end_idx].strip()
        
        if "</pensamento>" in response:
            return response.split("</pensamento>")[-1].strip()
        
        return response.strip()
    
    def __call__(self, state: dict) -> dict:
        """
        Método callable para integração com LangGraph.
        
        Args:
            state: Estado atual contendo 'query', 'draft_answer', 'context'
            
        Returns:
            Estado atualizado com 'final_answer', 'criticism_history', 'approved'
        """
        query = state.get("query", "")
        draft_answer = state.get("draft_answer", "")
        context = state.get("context", "")
        
        # Realizar revisão e correção
        final_answer, criticism_history, iterations = self.review_and_correct(
            query=query,
            draft_answer=draft_answer,
            context=context,
            max_iterations=Config.MAX_ITERATIONS
        )
        
        # Verificar se foi aprovado na primeira tentativa
        approved = len(criticism_history) == 1 and self.is_approved(criticism_history[0]["criticism"])
        
        # Atualizar estado
        return {
            **state,
            "final_answer": final_answer,
            "criticism_history": criticism_history,
            "approved": approved,
            "total_iterations": iterations,
            "answer_refined": not approved
        }
