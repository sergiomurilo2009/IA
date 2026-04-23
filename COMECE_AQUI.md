# 🚀 GUIA RÁPIDO - Comece Agora!

## Seu dataset está no Windows em: `C:\Users\SEU_USUARIO\Downloads\chat_pt.jsonl`

### ⚡ 3 Passos para Começar:

---

## **PASSO 1: Configurar o Caminho**

Abra o arquivo **`config.py`** e edite a linha do `DATASET_PATH`:

```python
# Troque SEU_USUARIO pelo seu nome de usuário do Windows
DATASET_PATH = r"C:\Users\SEU_USUARIO\Downloads\chat_pt.jsonl"
```

**Como descobrir seu nome de usuário:**
1. Abra o Prompt de Comando (cmd)
2. Digite: `echo %USERNAME%`
3. Copie o nome que aparecer

**Exemplo:** Se seu usuário é `Joao`, fica:
```python
DATASET_PATH = r"C:\Users\Joao\Downloads\chat_pt.jsonl"
```

---

## **PASSO 2: Validar o Dataset**

No terminal, na pasta do projeto:

```bash
python validate_dataset_windows.py
```

✅ Se tudo estiver certo, você verá:
- Tamanho do arquivo (~1GB)
- Número de linhas
- Colunas encontradas
- Configuração gerada automaticamente

---

## **PASSO 3: Treinar!**

### 3.1 Instalar dependências (só na primeira vez):
```bash
pip install torch numpy
```

### 3.2 Criar o Tokenizer:
```bash
python tokenizer.py
```
⏱️ Tempo: 5-30 minutos

### 3.3 Treinar o Modelo:
```bash
python train.py
```
⏱️ Tempo: 
- GPU: 5-20 horas
- CPU: 50+ horas (não recomendado)

### 3.4 Conversar com a IA:
```bash
python generate.py
```
🎉 Pronto! Sua IA está funcionando!

---

## 🔧 Problemas Comuns

### ❌ "Dataset não encontrado"
**Solução:** Verifique se o caminho em `config.py` está correto. Use o caminho completo entre `r"..."`.

### ❌ "CUDA out of memory"
**Solução:** Edite `config.py` e reduza:
```python
BATCH_SIZE = 4  # era 16
MAX_SEQ_LENGTH = 256  # era 512
```

### ❌ "ModuleNotFoundError: No module named 'torch'"
**Solução:** 
```bash
pip install torch
```

---

## 📊 O Que Esperar

### Durante o Treinamento:
```
Epoch 1/10 | Step 50 | Loss: 7.2341 | LR: 1.00e-04
Epoch 1/10 | Step 100 | Loss: 6.8923 | LR: 1.00e-04
...
```

**Loss diminuindo = ✅ Treinando bem!**

### Após o Treinamento:
```
👤 Você: Olá, como vai?
🤖 IA: Olá! Estou bem, obrigado por perguntar...
```

---

## 💡 Dicas Importantes

1. **Seu dataset de 1GB+ é excelente!** Vai dar bons resultados.

2. **Não pule o tokenizer!** Ele precisa ser criado antes do treino.

3. **Salve os checkpoints!** O script salva automaticamente a cada 500 steps.

4. **Teste após 1-2 épocas** para ver se está funcionando antes de completar tudo.

5. **Use GPU se possível!** É 10-50x mais rápido que CPU.

---

## 🎯 Resumo dos Arquivos

| Arquivo | O que faz | Quando usar |
|---------|-----------|-------------|
| `config.py` | Configurações gerais | **Edite primeiro!** |
| `validate_dataset_windows.py` | Valida dataset | Passo 1 |
| `tokenizer.py` | Cria vocabulário | Passo 2 |
| `train.py` | Treina modelo | Passo 3 |
| `generate.py` | Chat com IA | Depois de treinar |
| `model.py` | Arquitetura | Não precisa mexer |
| `README.md` | Documentação completa | Consulte se tiver dúvidas |

---

## 🆘 Precisa de Ajuda?

1. Leia o `README.md` para detalhes completos
2. Verifique os logs de erro
3. Ajuste `config.py` conforme seu hardware

**Boa sorte! 🚀**
