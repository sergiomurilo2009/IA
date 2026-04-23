# README - Banco de Dados para Treinamento

## Pasta "banco de dados"

Esta pasta é destinada a armazenar arquivos de dataset para treinamento do sistema Grok-Advanced-BR.

## Como usar

1. **Coloque seu arquivo `chat_pt.jsonl` nesta pasta**
   - O sistema carregará automaticamente este arquivo ao iniciar
   - Cada linha do arquivo deve ser um objeto JSON válido

2. **Formato do arquivo JSONL**
   
   O arquivo `chat_pt.jsonl` deve seguir o formato JSON Lines, onde cada linha é um objeto JSON independente. Exemplo:

   ```jsonl
   {"content": "O que é inteligência artificial?", "response": "IA é um campo da ciência da computação..."}
   {"content": "Como funciona machine learning?", "response": "Machine Learning permite que sistemas aprendam..."}
   {"prompt": "Explique redes neurais", "answer": "Redes neurais são inspiradas no cérebro humano..."}
   ```

3. **Campos suportados**
   
   O sistema busca automaticamente por campos como:
   - `content`
   - `text`
   - `message`
   - `prompt`
   - `response`
   - `answer`
   
   Se nenhum desses campos for encontrado, o sistema concatenará todos os valores string do objeto.

4. **Carregamento automático**
   
   Ao iniciar o sistema com `python grok-advanced-br/main.py`, o arquivo será carregado automaticamente se existir na pasta.

5. **Carregamento manual (opcional)**
   
   Você também pode carregar manualmente executando:
   ```bash
   python load_chat_data.py
   ```

## Estrutura da pasta

```
/workspace/
├── banco de dados/
│   └── chat_pt.jsonl    ← Coloque seu arquivo aqui
├── grok-advanced-br/
│   └── main.py
└── load_chat_data.py
```

## Dicas

- Mantenha o arquivo `chat_pt.jsonl` codificado em UTF-8
- Cada linha deve ser um JSON válido e independente
- Linhas vazias serão ignoradas
- O sistema mostrará o progresso do carregamento no terminal
