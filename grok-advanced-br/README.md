# ====================== README.md ======================
# Grok-Advanced-BR

Uma IA inteligente multi-agente que roda 100% localmente usando Ollama, LangGraph e ChromaDB.

## 📋 Requisitos

- Python 3.10 ou superior
- Ollama instalado (https://ollama.ai)
- Modelo LLM local (recomendado: llama3.2 ou mistral)

## 🚀 Instalação Passo a Passo

### 1. Instalar o Ollama

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows/Mac:** Baixe em https://ollama.ai

### 2. Baixar um modelo local

```bash
# Recomendado: llama3.2 (leve e eficiente)
ollama pull llama3.2

# Ou alternativa: mistral
ollama pull mistral
```

### 3. Configurar o projeto

```bash
cd grok-advanced-br

# Criar ambiente virtual (opcional mas recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente

```bash
# Copiar arquivo de exemplo
cp .env.example .env

# Editar o arquivo .env com suas configurações
nano .env  # ou use seu editor preferido
```

### 5. Rodar o projeto

```bash
# Iniciar a interface Gradio
python main.py
```

Acesse http://localhost:7860 no seu navegador.

## 📁 Estrutura do Projeto

```
grok-advanced-br/
├── main.py                 # Ponto de entrada principal
├── .env.example            # Exemplo de configuração
├── requirements.txt        # Dependências
├── README.md              # Este arquivo
└── src/
    ├── __init__.py
    ├── graph.py           # Definição do grafo LangGraph
    ├── config.py          # Configurações globais
    ├── agents/
    │   ├── __init__.py
    │   ├── query_analyzer.py   # Analisa a query do usuário
    │   ├── retriever.py        # Busca no banco vetorial (RAG)
    │   ├── thinker.py          # Raciocínio Chain of Thought + Tree of Thoughts
    │   ├── critic.py           # Revisora com auto-correção
    │   └── final_answer.py     # Gera resposta final
    └── rag/
        ├── __init__.py
        ├── vectorstore.py      # Gerencia ChromaDB
        └── document_loader.py  # Carrega documentos
```

## 🤖 Agentes

1. **QueryAnalyzerAgent**: Analisa e classifica a pergunta do usuário
2. **RetrieverAgent**: Busca informações relevantes no banco vetorial (RAG)
3. **ThinkerAgent**: Realiza raciocínio profundo com Chain of Thought e Tree of Thoughts
4. **CriticAgent**: Revisa e corrige a resposta (máximo 3 iterações)
5. **FinalAnswerAgent**: Formata e entrega a resposta final

## ⚙️ Configuração (.env)

```env
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./chroma_db
MAX_ITERATIONS=3
```

## 🧪 Testando sem Interface Gráfica

```bash
python -c "from src.graph import create_graph; graph = create_graph(); print(graph.invoke({'query': 'O que é IA?'}))"
```

## 📝 Licença

MIT License
