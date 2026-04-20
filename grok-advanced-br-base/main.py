"""
main.py - Arquivo principal do Grok-Advanced-BR-Base

Este é o ponto de entrada da aplicação. Oferece:
1. Interface Gradio (web) para interação
2. Modo terminal para uso sem dependências gráficas
3. Documentos iniciais de exemplo sobre IA e programação

Autor: Grok-Advanced-BR-Base
Licença: MIT
"""

import sys
import argparse
from typing import Optional

# Importa os módulos do projeto
from rag_system import RAGSystem
from reasoner import Reasoner


# ==================== DOCUMENTOS INICIAIS ====================
# Estes documentos formam a base de conhecimento inicial do sistema

DOCUMENTOS_INICIAIS = [
    # Python e Programação
    "Python é uma linguagem de programação de alto nível, interpretada, criada por Guido van Rossum em 1991 na Holanda.",
    "Python é conhecido por sua sintaxe clara e legível, sendo ideal para iniciantes em programação.",
    "Variáveis em Python não precisam de declaração explícita de tipo, pois é uma linguagem dinamicamente tipada.",
    "Listas em Python são coleções ordenadas e mutáveis de elementos, definidas com colchetes [].",
    "Dicionários em Python são estruturas de dados que armazenam pares chave-valor, definidos com chaves {}.",
    
    # Inteligência Artificial
    "Inteligência Artificial (IA) é a simulação de processos de inteligência humana por sistemas computacionais.",
    "Machine Learning (Aprendizado de Máquina) é um subcampo da IA que permite sistemas aprenderem com dados sem serem explicitamente programados.",
    "Deep Learning (Aprendizado Profundo) usa redes neurais artificiais com múltiplas camadas para aprender padrões complexos.",
    "Redes neurais artificiais são algoritmos inspirados na estrutura e funcionamento do cérebro humano.",
    "Processamento de Linguagem Natural (PLN) é a área da IA que permite computadores entenderem, interpretarem e gerarem linguagem humana.",
    
    # Conceitos Gerais de Tecnologia
    "Algoritmo é uma sequência finita de instruções bem definidas para resolver um problema ou executar uma tarefa.",
    "Banco de dados é um conjunto organizado de dados armazenados eletronicamente, acessíveis por sistemas computacionais.",
    "API (Interface de Programação de Aplicações) é um conjunto de definições que permite comunicação entre diferentes softwares.",
    "Cloud Computing (Computação em Nuvem) é a entrega de serviços computacionais através da internet.",
    "Open Source (Código Aberto) refere-se a software cujo código fonte está disponível para modificação e distribuição."
]


class GrokAdvancedBR:
    """
    Classe principal que integra todos os componentes do sistema.
    
    Funcionalidades:
    - Gerencia o banco de dados de documentos (RAG)
    - Processa perguntas usando o reasoner
    - Fornece interface para interação
    """
    
    def __init__(self):
        """Inicializa o sistema Grok-Advanced-BR."""
        print("=" * 60)
        print("  Grok-Advanced-BR-Base")
        print("  IA leve e local - Sem APIs externas")
        print("=" * 60)
        print()
        
        # Inicializa sistema RAG (carrega automaticamente da pasta data/)
        print("[SISTEMA] Inicializando banco de dados...")
        self.rag = RAGSystem(data_folder="data")
        
        # Adiciona documentos iniciais se a base estiver vazia
        if self.rag.get_total_documentos() == 0:
            print("[SISTEMA] Carregando documentos iniciais de exemplo...")
            self.rag.adicionar_documento(DOCUMENTOS_INICIAIS[0], fonte="exemplo")
            for doc in DOCUMENTOS_INICIAIS[1:]:
                self.rag.adicionar_documento(doc, fonte="exemplo")
        else:
            print(f"[SISTEMA] {self.rag.get_total_documentos()} documento(s) encontrado(s) na pasta data/")
        
        # Inicializa reasoner
        print("[SISTEMA] Inicializando motor de raciocínio...")
        self.reasoner = Reasoner(self.rag)
        
        print("[SISTEMA] Sistema pronto!")
        print()
    
    def perguntar(self, pergunta: str, mostrar_pensamento: bool = True) -> str:
        """
        Processa uma pergunta e retorna a resposta.
        
        Args:
            pergunta: Texto da pergunta do usuário
            mostrar_pensamento: Se True, exibe o processo de pensamento
        
        Returns:
            Resposta gerada pelo sistema
        """
        if not pergunta or not pergunta.strip():
            return "Por favor, faça uma pergunta válida."
        
        # Processa a pergunta com o reasoner
        resultado = self.reasoner.pensar(pergunta)
        
        # Monta a resposta
        if mostrar_pensamento:
            resposta_completa = "<pensamento>\n"
            resposta_completa += self.reasoner.get_pensamentos_formatados()
            resposta_completa += "\n</pensamento>\n\n"
            resposta_completa += f"<resposta_final>\n{resultado['resposta_final']}\n</resposta_final>"
        else:
            resposta_completa = resultado['resposta_final']
        
        return resposta_completa
    
    def adicionar_documento(self, texto: str) -> bool:
        """
        Adiciona um novo documento ao banco de dados.
        
        Args:
            texto: Conteúdo do documento a ser adicionado
        
        Returns:
            True se sucesso, False caso contrário
        """
        if not texto or len(texto.strip()) < 10:
            return False
        
        self.rag.adicionar_documento(texto, fonte="usuario")
        return True
    
    def listar_documentos(self) -> int:
        """Retorna o número total de documentos no banco de dados."""
        return self.rag.get_total_documentos()


def criar_interface_gradio(grok: GrokAdvancedBR):
    """
    Cria interface web usando Gradio.
    
    Args:
        grok: Instância do sistema GrokAdvancedBR
    """
    try:
        import gradio as gr
    except ImportError:
        print("\n[ERRO] Gradio não instalado. Execute: pip install gradio")
        print("Ou use o modo terminal: python main.py --terminal")
        return
    
    def responder(pergunta, mostrar_pensamento):
        """Função callback para o Gradio."""
        return grok.perguntar(pergunta, mostrar_pensamento)
    
    def adicionar_doc(texto):
        """Função para adicionar documentos via interface."""
        if grok.adicionar_documento(texto):
            total = grok.listar_documentos()
            return f"✓ Documento adicionado! Total: {total} documentos"
        else:
            return "✗ Documento muito curto ou vazio"
    
    # Cria interface Gradio
    with gr.Blocks(title="Grok-Advanced-BR-Base", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 Grok-Advanced-BR-Base
        
        IA inteligente e leve, rodando 100% localmente sem APIs externas.
        
        **Como usar:**
        1. Digite sua pergunta na caixa abaixo
        2. Marque "Mostrar pensamento" para ver o processo de raciocínio
        3. Clique em "Enviar Pergunta"
        
        **Documentos disponíveis:** Consulte o README.md para adicionar mais conhecimento.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                pergunta_input = gr.Textbox(
                    label="Sua Pergunta",
                    placeholder="Ex: O que é Python? Ou: Explique Inteligência Artificial",
                    lines=3
                )
                mostrar_pensamento = gr.Checkbox(
                    label="Mostrar processo de pensamento (Chain of Thought)",
                    value=True
                )
                btn_perguntar = gr.Button("🚀 Enviar Pergunta", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("""
                ### 📚 Base de Conhecimento
                
                Este sistema usa RAG (Retrieval-Augmented Generation)
                para buscar informações em documentos cadastrados.
                
                **Tópicos disponíveis:**
                - Python e Programação
                - Inteligência Artificial
                - Machine Learning
                - Conceitos de Tecnologia
                """)
        
        resposta_output = gr.Textbox(
            label="Resposta",
            lines=10,
            show_copy_button=True
        )
        
        # Área para adicionar novos documentos
        gr.Markdown("---")
        gr.Markdown("### 📝 Adicionar Novo Documento")
        
        with gr.Row():
            doc_input = gr.Textbox(
                label="Novo Documento",
                placeholder="Digite um fato ou informação para adicionar ao banco de conhecimento...",
                lines=2
            )
            btn_adicionar = gr.Button("➕ Adicionar Documento")
        
        status_doc = gr.Textbox(label="Status", interactive=False)
        
        # Configura eventos
        btn_perguntar.click(
            fn=responder,
            inputs=[pergunta_input, mostrar_pensamento],
            outputs=resposta_output
        )
        
        btn_adicionar.click(
            fn=adicionar_doc,
            inputs=doc_input,
            outputs=status_doc
        )
        
        # Permite Enter para enviar
        pergunta_input.submit(
            fn=responder,
            inputs=[pergunta_input, mostrar_pensamento],
            outputs=resposta_output
        )
    
    # Lança a interface
    print("\n" + "=" * 60)
    print("Iniciando interface web...")
    print("Acesse no navegador: http://localhost:7860")
    print("Pressione Ctrl+C para parar")
    print("=" * 60 + "\n")
    
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


def modo_terminal(grok: GrokAdvancedBR):
    """
    Interface de linha de comando simples.
    
    Args:
        grok: Instância do sistema GrokAdvancedBR
    """
    print("\n" + "=" * 60)
    print("  MODO TERMINAL")
    print("  Digite suas perguntas (ou 'sair' para encerrar)")
    print("=" * 60 + "\n")
    
    print(f"📚 Documentos na base: {grok.listar_documentos()}")
    print()
    
    while True:
        try:
            # Pede pergunta ao usuário
            pergunta = input("🤔 Pergunta: ").strip()
            
            # Verifica comandos especiais
            if pergunta.lower() in ['sair', 'exit', 'quit', 'q']:
                print("\n👋 Encerrando sistema. Até logo!")
                break
            
            if pergunta.lower() in ['ajuda', 'help', '?']:
                print("\n--- COMANDOS DISPONÍVEIS ---")
                print("  Digite sua pergunta normalmente")
                print("  'sair' - Encerra o programa")
                print("  'ajuda' - Mostra esta mensagem")
                print("  'add <texto>' - Adiciona documento")
                print("  'docs' - Mostra número de documentos")
                print("  'p <pergunta>' - Pergunta sem mostrar pensamento")
                print()
                continue
            
            if pergunta.lower() == 'docs':
                print(f"\n📚 Documentos na base: {grok.listar_documentos()}\n")
                continue
            
            # Verifica se é comando para adicionar documento
            if pergunta.lower().startswith('add '):
                texto = pergunta[4:].strip()
                if grok.adicionar_documento(texto):
                    print(f"✓ Documento adicionado! Total: {grok.listar_documentos()}\n")
                else:
                    print("✗ Documento muito curto (mínimo 10 caracteres)\n")
                continue
            
            # Verifica se é pergunta silenciosa (sem pensamento)
            mostrar_pensamento = True
            if pergunta.lower().startswith('p '):
                pergunta = pergunta[2:].strip()
                mostrar_pensamento = False
            
            # Processa pergunta
            if pergunta:
                print("\n🧠 Pensando...\n")
                resposta = grok.perguntar(pergunta, mostrar_pensamento)
                print(resposta)
                print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrompido pelo usuário. Até logo!")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}\n")


def main():
    """Função principal do programa."""
    # Configura parser de argumentos
    parser = argparse.ArgumentParser(
        description="Grok-Advanced-BR-Base - IA leve e local"
    )
    parser.add_argument(
        '--terminal',
        action='store_true',
        help='Rodar em modo terminal (sem interface gráfica)'
    )
    parser.add_argument(
        '--sem-pensamento',
        action='store_true',
        help='Não mostrar processo de pensamento nas respostas'
    )
    
    args = parser.parse_args()
    
    # Cria instância do sistema
    grok = GrokAdvancedBR()
    
    # Escolhe modo de execução
    if args.terminal:
        modo_terminal(grok)
    else:
        # Tenta iniciar interface Gradio
        try:
            criar_interface_gradio(grok)
        except Exception as e:
            print(f"\n[AVISO] Não foi possível iniciar interface gráfica: {e}")
            print("Iniciando modo terminal automaticamente...\n")
            modo_terminal(grok)


if __name__ == "__main__":
    main()
