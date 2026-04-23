===============================================================================
LUNA - IA CONVERSACIONAL FEMININA COM PENSAMENTO PROFUNDO
===============================================================================

📁 ESTRUTURA DO PROJETO:
------------------------
/workspace/
├── rodar_ia.py           ← Script principal para RODAR A IA
├── dados/                ← PASTA PARA SEUS ARQUIVOS DE TREINO (NÃO APAGAR!)
│   └── exemplo_treinamento.jsonl
└── memoria_ia.json       ← Gerado automaticamente (memória das conversas)


🚀 COMO USAR (WINDOWS OU LINUX):
---------------------------------

1. COLOQUE SEUS ARQUIVOS DE TREINO NA PASTA "dados":
   - Copie seu arquivo chat_pt.jsonl (ou qualquer outro) para dentro de:
     /workspace/dados/
   
   Formatos aceitos:
   • .jsonl  (recomendado) - uma linha por conversa em JSON
   • .json   - lista de objetos ou dicionário
   • .txt    - formato "Usuário: ... \n IA: ..."

2. RODE A IA:
   Abra o terminal/cmd e execute:
   
   cd /workspace
   python rodar_ia.py

3. CONVERSE!
   - Digite suas mensagens
   - Use "sair" ou "tchau" para encerrar
   - Use "status" para ver estatísticas
   - Use "limpar" para resetar memória recente


📝 FORMATOS DE DADOS SUPORTADOS:
---------------------------------

FORMATO JSONL (Recomendado):
Cada linha é um objeto JSON independente:

{"content": "Pergunta", "response": "Resposta"}
{"prompt": "Outra pergunta", "answer": "Outra resposta"}
{"input": "Mais uma", "output": "Resposta aqui"}

Campos reconhecidos para PERGUNTA:
  content, prompt, question, input, user, texto, mensagem, pergunta, query, text

Campos reconhecidos para RESPOSTA:
  response, answer, output, assistant, resposta, reply, bot, ia, luna, ai


FORMATO JSON:
Pode ser uma lista ou dicionário:

[
  {"content": "Pergunta 1", "response": "Resposta 1"},
  {"content": "Pergunta 2", "response": "Resposta 2"}
]

OU

{
  "conversas": [
    {"content": "Pergunta", "response": "Resposta"}
  ]
}


FORMATO TXT:
Formato conversacional simples:

Usuário: Olá, tudo bem?
IA: Oi! Tudo ótimo e você?

User: Qual seu nome?
Assistant: Me chamo Luna!

Padrões reconhecidos:
  Usuário:, User:, Humano:, Pergunta:, Input:, Cliente:, Tu:, Você:
  IA:, Assistant:, Resposta:, Bot:, Output:, Atendente:, Eu:, Luna:


✨ RECURSOS DA LUNA:
---------------------
• Personalidade feminina doce e empática
• Pensamento profundo antes de responder
• Análise de intenção e emoção
• Memória persistente entre sessões
• Suporte a múltiplos formatos de treino
• Super robusta - não quebra com dados inválidos
• Carregamento automático ao iniciar
• Emojis sutis nas respostas
• Expressões carinhosas ("querido", "amor", "vida")


🎯 COMANDOS DURANTE A CONVERSA:
--------------------------------
• sair / tchau / quit / exit / adeus → Encerra o programa
• status → Mostra estatísticas da memória
• limpar → Limpa memória recente da conversa


💡 DICAS:
---------
1. Quanto mais dados de treino, melhor a IA fica
2. Os dados são carregados automaticamente ao iniciar
3. A memória é salva automaticamente a cada 5 mensagens
4. Você pode adicionar novos arquivos na pasta "dados" a qualquer momento
5. A IA busca automaticamente campos variados nos seus arquivos


⚠️ IMPORTANTE:
--------------
• NUNCA apague a pasta "dados" - é lá que ficam seus arquivos de treino
• O arquivo "memoria_ia.json" é gerado automaticamente - não precisa criar
• Se houver erro em algum arquivo, a IA ignora e continua com os outros


===============================================================================
DIVIRTA-SE CONVERSANDO COM LUNA! 🌙💕
===============================================================================
