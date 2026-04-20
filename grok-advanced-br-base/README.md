# Grok-Advanced-BR-Base

IA simples e leve, 100% local, sem APIs externas, sem LLMs pesados.

## 🎯 Objetivo
Rodar em notebooks fracos usando apenas Python puro + técnicas leves de RAG.

## 📋 Requisitos
- Python 3.8+
- Notebook com pelo menos 4GB RAM (sem GPU necessária)

## 🚀 Instalação

```bash
# 1. Clone ou acesse a pasta do projeto
cd grok-advanced-br-base

# 2. Instale as dependências mínimas
pip install -r requirements.txt

# 3. Execute o projeto
python main.py
```

## 📁 Estrutura do Projeto

```
grok-advanced-br-base/
├── main.py           # Arquivo principal com interface Gradio
├── rag_system.py     # Sistema RAG (banco de dados + busca)
├── reasoner.py       # Simula Chain of Thought com regras
├── requirements.txt  # Dependências mínimas
├── README.md         # Este arquivo
├── data/             # 📚 PASTA DE DOCUMENTOS (adicione seus arquivos aqui!)
│   ├── *.txt         # Arquivos de texto simples
│   ├── *.md          # Arquivos Markdown
│   ├── *.json        # Arquivos JSON
│   ├── *.csv         # Planilhas CSV
│   └── *.zip         # Arquivos ZIP com documentos dentro
└── teste_sistema.py  # Script de teste automatizado
```

## 🔧 Como Funciona

### 1. RAG System (rag_system.py)
- Usa **TF-IDF** para extração de palavras-chave (super leve!)
- Usa **similaridade de cosseno** para encontrar documentos relevantes
- Fallback para sentence-transformers com modelo pequeno se disponível
- Armazena documentos em memória (pode persistir em JSON se necessário)

### 2. Reasoner (reasoner.py)
- Simula **Chain of Thought** usando regras em código Python
- Passos do raciocínio:
  1. Entender a pergunta (extrair palavras-chave)
  2. Buscar contexto no banco de dados
  3. Analisar se há informação suficiente
  4. Gerar resposta baseada em regras
  5. Auto-verificação simples

### 3. Interface (main.py)
- Interface Gradio simples e leve
- Ou modo terminal se preferir

## 📝 Adicionando Documentos

### Método 1: Pasta data/ (RECOMENDADO) ⭐

Basta colocar seus arquivos na pasta `data/`! O sistema lê automaticamente:

**Formatos suportados:**
- `.txt` - Arquivos de texto simples (cada parágrafo é um documento)
- `.md` - Arquivos Markdown
- `.json` - Listas ou objetos com campos de texto
- `.csv` - Planilhas (cada linha vira um documento)
- `.zip` - Arquivos compactados com documentos dentro

**Exemplo:**
```bash
# Crie arquivos na pasta data/
echo "Python é uma linguagem de programação." > data/python.txt
echo "IA simula inteligência humana." > data/ia.txt

# Ou adicione um ZIP com vários documentos
cp meus_documentos.zip data/
```

O sistema carrega tudo automaticamente ao iniciar!

### Método 2: Interface Web

Na interface Gradio, use a seção "Adicionar Novo Documento" para adicionar textos manualmente.

### Método 3: Modo Terminal

No terminal, use o comando `add`:
```
🤔 Pergunta: add Python é uma linguagem criada em 1991.
✓ Documento adicionado! Total: 15
```

### Método 4: Código (para desenvolvedores)

Edite o arquivo `main.py` e adicione documentos na lista `DOCUMENTOS_INICIAIS`:

```python
DOCUMENTOS_INICIAIS = [
    "Python é uma linguagem de programação criada por Guido van Rossum em 1991.",
    "IA significa Inteligência Artificial, que simula inteligência humana em máquinas.",
    # Adicione seus documentos aqui...
]
```

## ⚡ Modo Terminal (sem Gradio)

Se quiser rodar apenas no terminal, execute:

```bash
python main.py --terminal
```

## 🎛️ Personalização

### Usar apenas TF-IDF (mais leve ainda)
No arquivo `rag_system.py`, defina:
```python
USAR_SENTENCE_TRANSFORMERS = False
```

### Ajustar número de documentos retornados
No arquivo `rag_system.py`, altere:
```python
TOP_K = 3  # Número de documentos relevantes
```

## 📊 Vantagens desta Versão

✅ Roda em qualquer notebook (até 4GB RAM)  
✅ Sem necessidade de GPU  
✅ Sem APIs externas ou internet  
✅ Código 100% em português e comentado  
✅ Fácil de entender e modificar  
✅ RAG funcional com busca semântica  

## ⚠️ Limitações

- Não é um LLM completo (respostas baseadas em documentos)
- Raciocínio simulado via regras (não é rede neural)
- Melhor para perguntas sobre documentos cadastrados

## 📞 Uso Básico

1. Inicie o programa
2. Digite sua pergunta
3. O sistema busca documentos relevantes
4. Aplica regras de raciocínio
5. Retorna a resposta

Exemplo:
```
Pergunta: O que é Python?
Resposta: Baseado nos documentos encontrados: Python é uma linguagem 
de programação criada por Guido van Rossum em 1991.
```

---

**Criado para ser simples, leve e funcional!** 🚀
