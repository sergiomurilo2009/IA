# ====================== main.py ======================
"""
Grok-Advanced-BR - Sistema Multi-Agente com LangGraph

Este é o ponto de entrada principal do sistema.
Fornece uma interface Gradio para interação com o usuário.

Uso:
    python main.py

O servidor Gradio será iniciado em http://localhost:7860
"""

import gradio as gr
from typing import Dict, Any
import sys
from pathlib import Path

# Adicionar src ao path para imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.graph import initialize_system_with_samples, process_query
from src.rag.vectorstore import VectorStoreManager
from src.rag.document_loader import DocumentLoader


# Variáveis globais para o sistema
_graph = None
_vectorstore = None
_initialized = False


def initialize_system():
    """
    Inicializa o sistema uma única vez.
    """
    global _graph, _vectorstore, _initialized
    
    if not _initialized:
        print("\n" + "=" * 60)
        print("INICIALIZANDO GROK-ADVANCED-BR...")
        print("=" * 60)
        
        # Validar configuração
        Config.validate()
        
        # Inicializar sistema com documentos de exemplo
        _graph, _vectorstore = initialize_system_with_samples()
        
        _initialized = True
        
        print("\n" + "=" * 60)
        print("SISTEMA PRONTO! 🚀")
        print("=" * 60 + "\n")
    
    return _graph, _vectorstore


def chat_with_grok(
    message: str, 
    history: list,
    verbose_mode: bool = False
) -> str:
    """
    Processa uma mensagem do usuário e retorna a resposta.
    
    Args:
        message: Mensagem do usuário
        history: Histórico da conversa (não usado diretamente, mas necessário para Gradio)
        verbose_mode: Se deve mostrar detalhes do processo
        
    Returns:
        Resposta do sistema
    """
    global _graph, _vectorstore, _initialized
    
    # Validar entrada
    if not message or not message.strip():
        return "Por favor, faça uma pergunta válida."
    
    try:
        # Inicializar sistema se necessário
        if not _initialized:
            initialize_system()
        
        # Processar query
        result = process_query(
            graph=_graph,
            query=message.strip(),
            verbose=verbose_mode
        )
        
        # Verificar erros
        if result.get("error"):
            return f"⚠️ Erro: {result['error']}"
        
        # Retornar resposta
        answer = result.get("simple_answer", "Desculpe, não foi possível gerar uma resposta.")
        
        # Adicionar metadados se verbose
        if verbose_mode:
            metadata = result.get("formatted_response", {}).get("metadata", {})
            sources = result.get("formatted_response", {}).get("sources", [])
            
            extra_info = "\n\n---\n\n**Detalhes do Processamento:**\n"
            extra_info += f"- Tipo de pergunta: {metadata.get('query_type', 'N/A')}\n"
            extra_info += f"- Complexidade: {metadata.get('complexity', 'N/A')}\n"
            extra_info += f"- Usou banco de dados: {'Sim' if metadata.get('used_context') else 'Não'}\n"
            extra_info += f"- Revisões: {metadata.get('review_iterations', 0)}\n"
            
            if sources:
                extra_info += "\n**Fontes Consultadas:**\n"
                for source in sources[:3]:
                    extra_info += f"- {source['title']} (Relevância: {source['relevance']:.2f})\n"
            
            answer += extra_info
        
        return answer
        
    except Exception as e:
        error_msg = f"⚠️ Ocorreu um erro inesperado: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return error_msg


def add_document_to_knowledge(text: str, title: str) -> str:
    """
    Adiciona um documento ao banco de conhecimento.
    
    Args:
        text: Conteúdo do documento
        title: Título do documento
        
    Returns:
        Mensagem de confirmação
    """
    global _vectorstore
    
    try:
        # Inicializar se necessário
        if not _initialized:
            initialize_system()
        
        # Validar entrada
        if not text or not text.strip():
            return "❌ O conteúdo não pode estar vazio."
        
        # Criar documento
        loader = DocumentLoader()
        doc = loader.load_string(
            content=text.strip(),
            metadata={"title": title.strip() or "Documento sem título"},
            doc_id=None
        )
        
        # Adicionar ao vector store
        _vectorstore.add_document(
            doc_id=doc["id"],
            content=doc["content"],
            metadata=doc["metadata"]
        )
        
        count = _vectorstore.get_document_count()
        return f"✅ Documento adicionado com sucesso!\n\nTotal de documentos no banco: {count}"
        
    except Exception as e:
        return f"❌ Erro ao adicionar documento: {str(e)}"


def get_knowledge_stats() -> str:
    """
    Retorna estatísticas do banco de conhecimento.
    
    Returns:
        String com estatísticas
    """
    global _vectorstore
    
    try:
        if not _initialized:
            initialize_system()
        
        count = _vectorstore.get_document_count()
        
        stats = f"📊 **Estatísticas do Banco de Conhecimento**\n\n"
        stats += f"Total de documentos: {count}\n"
        stats += f"Diretório: {Config.CHROMA_PERSIST_DIR}\n"
        stats += f"Modelo de embedding: {Config.EMBEDDING_MODEL}\n"
        
        return stats
        
    except Exception as e:
        return f"❌ Erro ao obter estatísticas: {str(e)}"


def reset_knowledge() -> str:
    """
    Reseta completamente o banco de conhecimento.
    
    Returns:
        Mensagem de confirmação
    """
    global _vectorstore
    
    try:
        if not _initialized:
            initialize_system()
        
        confirm = "Tem certeza? Isso apagará todos os documentos."
        return f"⚠️ {confirm}\n\n(Digite 'CONFIRMAR' na caixa de texto para resetar)"
        
    except Exception as e:
        return f"❌ Erro: {str(e)}"


def create_gradio_interface():
    """
    Cria e configura a interface Gradio.
    
    Returns:
        Aplicação Gradio configurada
    """
    # Tema personalizado
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )
    
    with gr.Blocks(theme=theme, title="Grok-Advanced-BR") as demo:
        gr.Markdown("""
        # 🤖 Grok-Advanced-BR
        
        **Sistema Multi-Agente com IA Local**
        
        Este sistema usa LangGraph com múltiplos agentes especializados:
        - 🔍 QueryAnalyzerAgent: Analisa sua pergunta
        - 📚 RetrieverAgent: Busca no banco de dados (RAG)
        - 🧠 ThinkerAgent: Raciocínio profundo (CoT + ToT)
        - ✅ CriticAgent: Revisão e auto-correção
        - 📝 FinalAnswerAgent: Formata a resposta final
        
        Tudo roda 100% localmente usando Ollama!
        """)
        
        with gr.Tabs():
            # Tab de Chat
            with gr.TabItem("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Conversa",
                            height=500,
                            show_copy_button=True
                        )
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="Sua pergunta",
                                placeholder="Digite sua pergunta aqui...",
                                scale=4
                            )
                            send_btn = gr.Button("Enviar", variant="primary", scale=1)
                        
                        clear_btn = gr.Button("🗑️ Limpar Conversa")
                    
                    with gr.Column(scale=1):
                        verbose_checkbox = gr.Checkbox(
                            label="Modo Detalhado",
                            value=False,
                            info="Mostra detalhes do processamento"
                        )
                        
                        gr.Markdown("""
                        ### Dicas:
                        - Faça perguntas sobre IA e tecnologia
                        - O sistema busca no banco de dados
                        - Quanto mais específica a pergunta, melhor!
                        """)
                
                # Funções do chat
                def respond(message, chat_history, verbose):
                    if not message:
                        return "", chat_history
                    
                    # Obter resposta
                    response = chat_with_grok(message, chat_history, verbose_mode=verbose)
                    
                    # Atualizar histórico
                    chat_history.append((message, response))
                    
                    return "", chat_history
                
                msg_input.submit(
                    respond,
                    inputs=[msg_input, chatbot, verbose_checkbox],
                    outputs=[msg_input, chatbot]
                )
                
                send_btn.click(
                    respond,
                    inputs=[msg_input, chatbot, verbose_checkbox],
                    outputs=[msg_input, chatbot]
                )
                
                clear_btn.click(lambda: [], None, chatbot, queue=False)
            
            # Tab de Gerenciamento de Conhecimento
            with gr.TabItem("📚 Base de Conhecimento"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Adicionar Novo Documento")
                        
                        doc_title = gr.Textbox(
                            label="Título",
                            placeholder="Ex: Introdução ao Machine Learning"
                        )
                        
                        doc_content = gr.Textbox(
                            label="Conteúdo",
                            placeholder="Cole o conteúdo do documento aqui...",
                            lines=10
                        )
                        
                        add_doc_btn = gr.Button("📥 Adicionar ao Banco", variant="primary")
                        doc_output = gr.Textbox(label="Resultado", lines=3)
                    
                    with gr.Column():
                        gr.Markdown("### Estatísticas")
                        
                        stats_btn = gr.Button("📊 Ver Estatísticas")
                        stats_output = gr.Textbox(label="Estatísticas", lines=6)
                        
                        gr.Markdown("---")
                        gr.Markdown("### Resetar Banco")
                        gr.Markdown("*Atenção: Isso apaga todos os documentos!*")
                        reset_btn = gr.Button("🗑️ Resetar Banco", variant="stop")
                        reset_output = gr.Textbox(label="Confirmação", lines=2)
                
                # Funções de conhecimento
                add_doc_btn.click(
                    add_document_to_knowledge,
                    inputs=[doc_content, doc_title],
                    outputs=[doc_output]
                )
                
                stats_btn.click(
                    get_knowledge_stats,
                    inputs=[],
                    outputs=[stats_output]
                )
                
                reset_btn.click(
                    reset_knowledge,
                    inputs=[],
                    outputs=[reset_output]
                )
            
            # Tab de Configurações
            with gr.TabItem("⚙️ Configurações"):
                gr.Markdown("### Configurações do Sistema")
                
                config_text = gr.Textbox(
                    label="Configurações Atuais",
                    value=f"""
OLLAMA_BASE_URL: {Config.OLLAMA_BASE_URL}
LLM_MODEL: {Config.LLM_MODEL}
EMBEDDING_MODEL: {Config.EMBEDDING_MODEL}
CHROMA_PERSIST_DIR: {Config.CHROMA_PERSIST_DIR}
MAX_ITERATIONS: {Config.MAX_ITERATIONS}
TEMPERATURE: {Config.TEMPERATURE}
                    """.strip(),
                    lines=8,
                    interactive=False
                )
                
                gr.Markdown("""
                ### Para alterar as configurações:
                
                1. Edite o arquivo `.env` na raiz do projeto
                2. Reinicie o servidor
                3. As novas configurações serão carregadas automaticamente
                """)
        
        # Rodapé
        gr.Markdown("""
        ---
        **Grok-Advanced-BR** v1.0.0 | Sistema Multi-Agente com LangGraph e Ollama
        """)
    
    return demo


def main():
    """
    Função principal que inicia a aplicação.
    """
    print("\n" + "=" * 60)
    print("🚀 INICIANDO GROK-ADVANCED-BR")
    print("=" * 60)
    
    # Validar configuração
    try:
        Config.validate()
        print("✅ Configuração validada com sucesso!")
    except Exception as e:
        print(f"⚠️ Aviso: {e}")
    
    # Criar interface
    demo = create_gradio_interface()
    
    print("\n" + "=" * 60)
    print("🌐 Servidor Gradio iniciando...")
    print("Acesse: http://localhost:7860")
    print("=" * 60 + "\n")
    
    # Iniciar servidor
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
