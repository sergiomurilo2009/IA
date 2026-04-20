# ====================== src/graph.py ======================
"""
Graph - Definição do Grafo LangGraph.

Este módulo define o grafo de fluxo de trabalho multi-agente usando LangGraph,
orquestrando todos os agentes em uma sequência lógica de processamento.
"""

from typing import TypedDict, Annotated, List, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from .config import Config
from .agents.query_analyzer import QueryAnalyzerAgent
from .agents.retriever import RetrieverAgent
from .agents.thinker import ThinkerAgent
from .agents.critic import CriticAgent
from .agents.final_answer import FinalAnswerAgent
from .rag.vectorstore import VectorStoreManager


class GraphState(TypedDict):
    """
    Estrutura do estado que flui através do grafo.
    
    Atributos:
        query: Pergunta original do usuário
        query_analysis: Análise da query (tipo, keywords, etc.)
        needs_retrieval: Se precisa buscar no banco de dados
        retrieved_documents: Documentos recuperados do RAG
        context: Contexto formatado dos documentos
        thought_process: Processo de pensamento do ThinkerAgent
        draft_answer: Resposta inicial antes da revisão
        final_answer: Resposta após revisão do CriticAgent
        formatted_response: Resposta formatada com metadados
        simple_answer: Apenas a resposta final
        verbose_response: Resposta detalhada para debug
        complete: Se o processamento foi completado
        iteration_count: Contador de iterações
        total_iterations: Total de iterações de revisão
        approved: Se a resposta foi aprovada sem revisões
        answer_refined: Se a resposta passou por refinamento
        error: Mensagem de erro se ocorrer falha
    """
    query: str
    query_analysis: dict
    needs_retrieval: bool
    retrieved_documents: List[dict]
    context: str
    thought_process: str
    draft_answer: str
    final_answer: str
    formatted_response: dict
    simple_answer: str
    verbose_response: str
    complete: bool
    iteration_count: int
    total_iterations: int
    approved: bool
    answer_refined: bool
    error: str
    has_context: bool


def create_graph(
    vectorstore: VectorStoreManager = None,
    verbose: bool = True
) -> StateGraph:
    """
    Cria e configura o grafo LangGraph com todos os agentes.
    
    Fluxo do grafo:
    1. QueryAnalyzerAgent → Analisa a query
    2. RetrieverAgent → Busca contexto no banco (se necessário)
    3. ThinkerAgent → Realiza raciocínio profundo
    4. CriticAgent → Revisa e corrige a resposta
    5. FinalAnswerAgent → Formata e entrega resposta final
    
    Args:
        vectorstore: Instância opcional do VectorStoreManager
        verbose: Se deve imprimir logs detalhados
        
    Returns:
        Grafo LangGraph compilado pronto para uso
    """
    if verbose:
        print("=" * 60)
        print("GROK-ADVANCED-BR - INICIALIZANDO SISTEMA MULTI-AGENTE")
        print("=" * 60)
        Config.print_config()
    
    # Inicializar agentes
    if verbose:
        print("\n[Graph] Inicializando agentes...")
    
    query_analyzer = QueryAnalyzerAgent()
    retriever = RetrieverAgent(vectorstore=vectorstore)
    thinker = ThinkerAgent()
    critic = CriticAgent()
    final_answer = FinalAnswerAgent()
    
    if verbose:
        print("[Graph] Todos os agentes inicializados.\n")
    
    # Criar o grafo
    workflow = StateGraph(GraphState)
    
    # Adicionar nós (nodes)
    workflow.add_node("query_analyzer", query_analyzer)
    workflow.add_node("retriever", retriever)
    workflow.add_node("thinker", thinker)
    workflow.add_node("critic", critic)
    workflow.add_node("final_answer", final_answer)
    
    # Definir arestas (edges) - fluxo sequencial
    workflow.set_entry_point("query_analyzer")
    
    # Query Analyzer → Retriever
    workflow.add_edge("query_analyzer", "retriever")
    
    # Retriever → Thinker
    workflow.add_edge("retriever", "thinker")
    
    # Thinker → Critic
    workflow.add_edge("thinker", "critic")
    
    # Critic → Final Answer
    workflow.add_edge("critic", "final_answer")
    
    # Final Answer → END
    workflow.add_edge("final_answer", END)
    
    # Compilar o grafo
    graph = workflow.compile()
    
    if verbose:
        print("[Graph] Grafo compilado com sucesso!")
        print("=" * 60)
        print("FLUXO DO GRAFO:")
        print("  query_analyzer → retriever → thinker → critic → final_answer → END")
        print("=" * 60)
    
    return graph


def initialize_system_with_samples() -> tuple:
    """
    Inicializa o sistema com documentos de exemplo.
    
    Returns:
        Tupla com (grafo, vectorstore)
    """
    from .rag.document_loader import DocumentLoader
    
    # Criar vector store
    vectorstore = VectorStoreManager(collection_name="grok_knowledge")
    
    # Verificar se já tem documentos
    if vectorstore.get_document_count() == 0:
        print("\n[Graph] Banco vazio. Carregando documentos de exemplo...")
        
        # Carregar documentos de exemplo
        loader = DocumentLoader()
        sample_docs = loader.create_sample_documents()
        
        # Adicionar ao vector store
        vectorstore.add_documents(sample_docs)
        
        print(f"[Graph] {len(sample_docs)} documentos de exemplo carregados.")
    else:
        count = vectorstore.get_document_count()
        print(f"\n[Graph] Banco já contém {count} documentos.")
    
    # Criar grafo
    graph = create_graph(vectorstore=vectorstore, verbose=True)
    
    return graph, vectorstore


def process_query(
    graph: StateGraph, 
    query: str, 
    verbose: bool = True
) -> dict:
    """
    Processa uma query através do grafo multi-agente.
    
    Args:
        graph: Grafo LangGraph compilado
        query: Pergunta do usuário
        verbose: Se deve imprimir logs
        
    Returns:
        Estado final com todas as respostas e metadados
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"PROCESSANDO QUERY: {query}")
        print(f"{'=' * 60}\n")
    
    # Estado inicial
    initial_state = {
        "query": query,
        "query_analysis": {},
        "needs_retrieval": True,
        "retrieved_documents": [],
        "context": "",
        "thought_process": "",
        "draft_answer": "",
        "final_answer": "",
        "formatted_response": {},
        "simple_answer": "",
        "verbose_response": "",
        "complete": False,
        "iteration_count": 0,
        "total_iterations": 0,
        "approved": False,
        "answer_refined": False,
        "error": "",
        "has_context": False
    }
    
    # Executar o grafo
    try:
        result = graph.invoke(initial_state)
        
        if verbose:
            print(f"\n{'=' * 60}")
            print("PROCESSAMENTO COMPLETADO!")
            print(f"{'=' * 60}")
            print(result.get("verbose_response", ""))
        
        return result
        
    except Exception as e:
        error_msg = f"Erro ao processar query: {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        
        return {
            **initial_state,
            "error": error_msg,
            "simple_answer": f"Desculpe, ocorreu um erro: {str(e)}"
        }


# Função convenience para uso direto
def ask(query: str, verbose: bool = True) -> str:
    """
    Função simples para fazer uma pergunta e receber resposta.
    
    Args:
        query: Pergunta do usuário
        verbose: Se deve imprimir logs
        
    Returns:
        Resposta final como string
    """
    # Inicializar sistema (com documentos de exemplo se necessário)
    graph, vectorstore = initialize_system_with_samples()
    
    # Processar query
    result = process_query(graph, query, verbose=verbose)
    
    # Retornar resposta simples
    return result.get("simple_answer", "Desculpe, não foi possível gerar uma resposta.")
