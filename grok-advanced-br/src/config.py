# ====================== src/config.py ======================
"""
Configurações globais do sistema Grok-Advanced-BR.

Este módulo carrega as variáveis de ambiente e define configurações
padrão para todo o sistema.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Carregar variáveis de ambiente do arquivo .env
# Procura o arquivo .env no diretório raiz do projeto
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)


class Config:
    """
    Classe de configuração com todas as constantes do sistema.
    
    Atributos:
        OLLAMA_BASE_URL: URL do servidor Ollama local
        LLM_MODEL: Nome do modelo de linguagem a ser usado
        EMBEDDING_MODEL: Modelo de embedding para o banco vetorial
        CHROMA_PERSIST_DIR: Diretório para persistência do ChromaDB
        MAX_ITERATIONS: Número máximo de iterações do CriticAgent
        TEMPERATURE: Temperatura do modelo LLM
        SYSTEM_PROMPT: Prompt interno obrigatório para raciocínio
    """
    
    # Configurações do Ollama
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
    
    # Configurações de Embedding
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL", 
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Configurações do ChromaDB
    CHROMA_PERSIST_DIR = os.getenv(
        "CHROMA_PERSIST_DIR", 
        str(project_root / "chroma_db")
    )
    
    # Configurações do sistema
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    
    # System Prompt Interno (OBRIGATÓRIO)
    SYSTEM_PROMPT = """Você é Grok-Advanced-BR, uma IA brasileira muito inteligente e honesta.
Para cada pergunta do usuário, sempre faça seu raciocínio dentro de <pensamento> </pensamento> seguindo esta ordem:
1. Entender a pergunta
2. Analisar o contexto do banco de dados
3. Identificar se falta informação
4. Pensar em pelo menos 2 caminhos diferentes (Tree of Thoughts)
5. Criar rascunho da resposta
6. Auto-crítica rigorosa
7. Versão final corrigida
Depois responda apenas com <resposta_final> sua resposta aqui </resposta_final>"""

    @classmethod
    def validate(cls):
        """
        Valida se todas as configurações necessárias estão presentes.
        
        Raises:
            ValueError: Se alguma configuração crítica estiver faltando
        """
        if not cls.OLLAMA_BASE_URL:
            raise ValueError("OLLAMA_BASE_URL não configurada")
        
        if not cls.LLM_MODEL:
            raise ValueError("LLM_MODEL não configurado")
        
        # Verificar se o diretório do ChromaDB existe, criar se necessário
        chroma_path = Path(cls.CHROMA_PERSIST_DIR)
        chroma_path.mkdir(parents=True, exist_ok=True)
        
        return True
    
    @classmethod
    def print_config(cls):
        """Imprime a configuração atual para debug."""
        print("=" * 50)
        print("CONFIGURAÇÃO DO GROK-ADVANCED-BR")
        print("=" * 50)
        print(f"Ollama URL: {cls.OLLAMA_BASE_URL}")
        print(f"Modelo LLM: {cls.LLM_MODEL}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"ChromaDB Dir: {cls.CHROMA_PERSIST_DIR}")
        print(f"Max Iterações: {cls.MAX_ITERATIONS}")
        print(f"Temperatura: {cls.TEMPERATURE}")
        print("=" * 50)


# Instância global de configuração
config = Config()
