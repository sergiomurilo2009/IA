# 🤖 IA Conversacional do Zero - Projeto Completo

Projeto completo para criar uma IA conversacional em Python/PyTorch, implementando arquitetura Transformer Decoder-only **do zero**, sem usar modelos pré-treinados.

## 📁 Estrutura do Projeto

```
projeto-ia/
├── config.py                 # Configurações (edite este arquivo!)
├── validate_dataset_windows.py  # Valida seu dataset JSON
├── tokenizer.py              # Treina o tokenizer BPE
├── model.py                  # Arquitetura Transformer
├── train.py                  # Script de treinamento
├── generate.py               # Geração de texto/chat
├── requirements.txt          # Dependências
└── README.md                 # Este arquivo
```

## 🚀 Instalação

### 1. Instale as dependências

```bash
pip install torch numpy
```

**Opcional (para GPU NVIDIA):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. Configure o caminho do dataset

Edite o arquivo `config.py` e ajuste o caminho:

```python
DATASET_PATH = r"C:\Users\SEU_USUARIO\Downloads\chat_pt.jsonl"
```

Substitua `SEU_USUARIO` pelo seu nome de usuário do Windows.

## 📋 Passo a Passo

### Passo 1: Validar o Dataset

```bash
python validate_dataset_windows.py
```

Este script vai:
- ✅ Encontrar automaticamente seu arquivo `chat_pt.jsonl`
- 📊 Analisar tamanho e estrutura
- 💡 Sugerir colunas para extração de texto
- 📝 Gerar configuração automática

### Passo 2: Treinar o Tokenizer

```bash
python tokenizer.py
```

Este script vai:
- 🔤 Carregar seu dataset
- 🧠 Treinar um tokenizer BPE (Byte Pair Encoding)
- 💾 Salvar `tokenizer.pkl` e `vocab.json`

**Tempo estimado:** 5-30 minutos (dependendo do tamanho do dataset)

### Passo 3: Treinar o Modelo

```bash
python train.py
```

Este script vai:
- 🧠 Criar modelo Transformer (6 camadas, 8 heads, ~50M parâmetros)
- 📚 Treinar no seu dataset
- 💾 Salvar checkpoints a cada 500 steps
- 📊 Logar métricas de perda e learning rate

**Tempo estimado:**
- CPU: 50-100 horas (não recomendado)
- GPU RTX 3060: 10-20 horas
- GPU RTX 4070+: 5-10 horas

### Passo 4: Conversar com a IA

```bash
python generate.py
```

Este script vai:
- 🤖 Carregar o modelo treinado
- 💬 Iniciar chat interativo
- 🎮 Permitir ajustar temperatura e máximo de tokens

**Comandos no chat:**
- `/quit` - Sair
- `/clear` - Limpar histórico
- `/temp 0.8` - Ajustar criatividade
- `/max 200` - Ajustar tamanho da resposta

## ⚙️ Configurações Recomendadas

### Para GPU com 6GB VRAM (RTX 3060, etc.)

```python
# config.py
BATCH_SIZE = 8
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
MAX_SEQ_LENGTH = 256
```

### Para GPU com 8GB+ VRAM (RTX 4070, etc.)

```python
# config.py
BATCH_SIZE = 16
D_MODEL = 512
NUM_LAYERS = 8
NUM_HEADS = 8
MAX_SEQ_LENGTH = 512
```

### Para CPU (apenas teste)

```python
# config.py
BATCH_SIZE = 4
D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 4
MAX_SEQ_LENGTH = 128
NUM_EPOCHS = 1
```

## 📊 Entendendo a Arquitetura

### Transformer Decoder-only

```
Input → Embedding → Positional Encoding → [Transformer Block]×N → Output
                                              ↓
                                    Multi-Head Attention
                                    Feed-Forward Network
                                    Layer Normalization
```

### Componentes Principais

| Componente | Função | Explicação Matemática |
|------------|--------|----------------------|
| **Embedding** | Converte tokens em vetores | `E[token] ∈ ℝ^d_model` |
| **Positional Encoding** | Adiciona informação de posição | `PE(pos,2i) = sin(pos/10000^(2i/d))` |
| **Multi-Head Attention** | Permite atender diferentes posições | `Attention(Q,K,V) = softmax(QK^T/√d_k)V` |
| **Feed-Forward** | Adiciona não-linearidade | `FFN(x) = ReLU(xW₁+b₁)W₂+b₂` |
| **Layer Norm** | Estabiliza treinamento | `Norm(x) = γ·(x-μ)/σ + β` |

## 🔧 Troubleshooting

### Erro: "Dataset não encontrado"

✅ Solução: Edite `config.py` com o caminho correto:
```python
DATASET_PATH = r"D:\MeusArquivos\chat_pt.jsonl"
```

### Erro: "CUDA out of memory"

✅ Solução: Reduza batch_size e max_seq_length:
```python
BATCH_SIZE = 4
MAX_SEQ_LENGTH = 256
```

### Erro: "ModuleNotFoundError: No module named 'torch'"

✅ Solução: Instale PyTorch:
```bash
pip install torch
```

### Treinamento muito lento

✅ Soluções:
1. Use GPU (instale versão CUDA do PyTorch)
2. Reduza número de épocas para teste
3. Diminua tamanho do dataset inicialmente

## 📈 Dicas de Treinamento

### Para melhores resultados:

1. **Dataset maior**: Mínimo 100MB, ideal 1GB+
2. **Mais épocas**: 10-50 épocas (com early stopping)
3. **Ajuste learning rate**: Tente 1e-4, 5e-5, 1e-5
4. **Monitore loss**: Deve diminuir consistentemente

### Sinais de bom treinamento:

- ✅ Loss inicial: 8-10
- ✅ Após 1 época: 4-6
- ✅ Após 5 épocas: 2-4
- ✅ Final: 1-3 (quanto menor, melhor)

## 🎯 Próximos Passos

Depois de treinar:

1. **Fine-tuning**: Treine mais em dados específicos
2. **Instruction Tuning**: Adicione exemplos de instruções
3. **Deploy**: Integre com API web ou aplicativo
4. **Experimente**: Mude hiperparâmetros e veja o que acontece!

## 📚 Recursos Adicionais

- [Paper Original do Transformer](https://arxiv.org/abs/1706.03762)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

## ⚠️ Aviso Importante

Este projeto é **educacional**. Modelos treinados do zero em hardware doméstico serão limitados comparados a LLMs grandes (GPT, Llama, etc.). O objetivo é aprender como funciona, não competir com modelos de produção.

---

**Boa sorte na criação da sua IA! 🚀**

Se tiver dúvidas, revise os logs de erro e ajuste as configurações conforme necessário.
