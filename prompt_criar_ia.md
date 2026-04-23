# 📝 Prompt para Criar sua Própria IA Conversacional

## Contexto
Atue como um Engenheiro de Machine Learning sênior especializado em Processamento de Linguagem Natural (NLP). Meu objetivo é construir uma IA conversacional do zero, sem utilizar modelos pré-treinados (como Llama, GPT ou Mistral) e sem depender de APIs externas.

---

## Requisitos do Projeto

### 1. Arquitetura
- Desenvolva o código em **Python** utilizando a biblioteca **PyTorch**
- A arquitetura deve ser baseada em **Transformers (Decoder-only)**, focada em geração de texto e diálogo
- Considere uma escala que possa rodar em **hardware doméstico** (ajuste o número de camadas e cabeças de atenção para ser eficiente)

### 2. Componentes do Código

#### a) Classe do Modelo
Implemente uma classe completa com:
- **Multi-Head Attention**: Mecanismo de atenção com múltiplas cabeças
- **Feed-Forward Networks**: Camadas densas intermediárias
- **Positional Encoding**: Codificação posicional para capturar ordem sequencial
- **Layer Normalization**: Para estabilização do treinamento
- **Masked Attention**: Para garantir que cada token só veja tokens anteriores

#### b) Tokenizer
Crie um script para o **Tokenizer** implementando:
- **Byte Pair Encoding (BPE)** do zero
- Funcionalidades: treinamento do vocabulário, tokenização e detokenização
- Salvamento e carregamento do vocabulário treinado

#### c) Loop de Treinamento
Desenvolva um script completo com:
- **Loss Function**: CrossEntropy
- **Optimizer**: AdamW
- **Learning Rate Scheduler**: Com warmup e decay
- **Suporte a Checkpoints**: Salvamento e retomada do treinamento
- **Gradient Clipping**: Para evitar exploding gradients
- **Mixed Precision Training**: Opcional, para eficiência de memória

### 3. Funcionalidade de Raciocínio
- Estruture o modelo para suportar **Instruction Fine-Tuning**
- Permita que o modelo simule um processo de pensamento lógico
- Inclua exemplos de formatação de prompts para raciocínio passo-a-passo

### 4. Independência
- O código deve ser **autossuficiente**
- Permita carregar datasets próprios em formatos:
  - `.txt` (texto bruto)
  - `.jsonl` (JSON Lines com pares de conversa)
- Não dependa de APIs externas ou modelos pré-treinados

### 5. Explicações
Para **cada bloco de código**, inclua comentários explicando:
- A função matemática por trás da implementação
- A lógica de decisão arquiteturais
- Por que determinada abordagem foi escolhida

---

## Entregáveis Esperados

Forneça **TRÊS scripts completos e funcionais**:

### Script 1: `tokenizer.py`
- Implementação completa do BPE
- Treinamento do vocabulário a partir de dataset
- Funções de tokenização/detokenização
- Exemplo de uso no final do script

### Script 2: `model.py`
- Classe `TransformerDecoder` completa
- Todas as camadas necessárias
- Forward pass com masked attention
- Função de geração de texto (inference)
- Configurações para hardware doméstico (ex: 6 camadas, 8 heads, 512 dimensões)

### Script 3: `train.py`
- Carregamento e pré-processamento de dataset
- DataLoader eficiente
- Loop de treinamento completo
- Validação e métricas
- Salvamento de checkpoints
- Logging de progresso
- Suporte a argumentos via linha de comando

### Script 4 (Bônus): `generate.py`
- Script para gerar texto com o modelo treinado
- Suporte a diferentes estratégias de sampling (greedy, top-k, nucleus)
- Interface interativa para conversação

---

## Especificações Técnicas para Hardware Doméstico

Para garantir execução em GPUs consumer (ex: RTX 3060/4070 com 12-16GB VRAM):

| Parâmetro | Valor Sugerido |
|-----------|---------------|
| Número de Camadas (n_layers) | 6-8 |
| Número de Heads (n_heads) | 8 |
| Dimensão do Modelo (d_model) | 512-768 |
| Dimensão FFN (d_ff) | 2048-3072 |
| Tamanho do Vocabulário | 8.000-16.000 |
| Context Window | 1024-2048 tokens |
| Batch Size | 16-64 (dependendo da GPU) |

---

## Formato de Dataset Esperado

### Para `.txt`:
```
Texto bruto contendo conversas, livros, ou qualquer conteúdo linguístico.
Cada linha pode ser um exemplo independente.
```

### Para `.jsonl`:
```json
{"prompt": "Qual é a capital da França?", "completion": "A capital da França é Paris."}
{"prompt": "Explique o conceito de gravidade.", "completion": "A gravidade é uma força fundamental..."}
```

---

## Instruções Adicionais

1. **Comece pelas explicações conceituais** antes de mostrar o código
2. **Inclua fórmulas matemáticas** quando relevante (ex: fórmula da atenção)
3. **Adicione tratamento de erros** robusto em todos os scripts
4. **Documente todas as funções** com docstrings claras
5. **Inclua um arquivo `requirements.txt`** com as dependências
6. **Forneça um exemplo de execução** passo-a-passo no final

---

## Exemplo de Estrutura de Resposta Esperada

```markdown
## 1. Visão Geral da Arquitetura

[Explicação conceitual do Transformer Decoder-only]

### Fórmula da Atenção Multi-Head:
Attention(Q,K,V) = softmax(QK^T / √d_k)V

## 2. Implementação do Tokenizer (tokenizer.py)

[Código completo com comentários detalhados]

## 3. Implementação do Modelo (model.py)

[Código completo com comentários detalhados]

## 4. Script de Treinamento (train.py)

[Código completo com comentários detalhados]

## 5. Como Executar

### Passo 1: Preparar o dataset
### Passo 2: Treinar o tokenizer
### Passo 3: Treinar o modelo
### Passo 4: Gerar texto

## 6. Próximos Passos e Melhorias

[Sugestões de otimização e escalabilidade]
```

---

## Nota Importante sobre Dataset

⚠️ **Atenção**: Quando a IA entregar o código, você precisará de um **arquivo imenso de texto** (como conversas de chat, livros, artigos, etc.) para alimentá-la. 

**Sem dados, ela será apenas um código vazio.**

### Fontes Sugeridas de Dataset:
- [Common Crawl](https://commoncrawl.org/)
- [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/)
- [Wikipedia Dumps](https://dumps.wikimedia.org/)
- [Project Gutenberg](https://www.gutenberg.org/) (livros em domínio público)
- Conversas próprias exportadas de chats
- Datasets do [Hugging Face](https://huggingface.co/datasets) (apenas os dados brutos, não modelos)

### Quantidade Mínima Recomendada:
- **Mínimo**: 100 MB de texto limpo
- **Ideal**: 1 GB+ para resultados coerentes
- **Ótimo**: 10 GB+ para qualidade superior

---

## Checklist de Validação

Antes de considerar o projeto completo, verifique:

- [ ] O tokenizer treina e salva vocabulário corretamente
- [ ] O modelo inicializa sem erros
- [ ] O forward pass funciona com tensors de exemplo
- [ ] O loop de treinamento executa pelo menos 1 epoch
- [ ] Checkpoints são salvos e podem ser carregados
- [ ] A geração de texto produz output legível
- [ ] Todo o código roda em uma GPU consumer (ou CPU se necessário)
- [ ] As explicações matemáticas estão presentes em cada componente

---

**Boa sorte na construção da sua IA! 🚀**
