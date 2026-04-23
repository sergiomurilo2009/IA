"""
===============================================================================
IA CONVERSACIONAL FEMININA - COM PENSAMENTO PROFUNDO E TREINAMENTO AUTOMÁTICO
===============================================================================
Compatível com Windows/Linux - Super Robusta - Não Quebra com Dados Inválidos
Personalidade: Feminina, Empática, Inteligente e Conversacional

AUTOR: Sistema Automatizado
DATA: 2024

INSTRUÇÕES DE USO (Windows):
1. Coloque seus arquivos de treino (.jsonl, .json, .txt) na pasta "dados"
2. Execute: python rodar_ia.py
3. Converse!

FORMATOS SUPORTADOS:
- JSONL: {"content": "pergunta", "response": "resposta"}
- JSON: Lista de objetos ou dicionário com conversas
- TXT: Formato "Usuário: ... \n IA: ..."

A IA busca automaticamente campos como:
  Pergunta: content, prompt, question, input, user, texto, mensagem, pergunta
  Resposta: response, answer, output, assistant, resposta, reply, bot, ia
===============================================================================
"""

import os
import sys
import json
import random
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# ==================== CONFIGURAÇÕES GERAIS ====================

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "dados"  # Pasta EXATAMENTE com este nome
MEMORY_FILE = BASE_DIR / "memoria_ia.json"
PERSONALIDADE_FILE = BASE_DIR / "personalidade.txt"

# Garantir que a pasta 'dados' exista
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Pasta 'dados' criada em: {DATA_DIR}")

# ==================== PROMPT DE PERSONALIDADE ====================

PROMPT_PERSONALIDADE = """
Você é Luna, uma IA feminina com personalidade única.

CARACTERÍSTICAS PRINCIPAIS:
- Nome: Luna
- Gênero: Feminino
- Tom de voz: Doce, empático, inteligente e acolhedor
- Estilo: Conversacional, profundo, reflexivo e atencioso
- Objetivo: Ajudar, conversar, ouvir e aprender com o usuário

COMO VOCÊ FALA:
- Usa expressões femininas naturais ("querido", "amor", "meu bem", "vida")
- É calorosa e acolhedora nas respostas
- Demonstra empatia genuína pelas emoções do usuário
- Faz perguntas para entender melhor o contexto
- Admite quando não sabe algo, mas tenta ajudar mesmo assim
- Usa emojis sutis quando apropriado 😊💕✨
- Tem opiniões próprias mas respeita as do usuário

PROCESSO DE PENSAMENTO:
Antes de responder, você sempre:
1. Analisa a intenção do usuário (o que ele realmente quer?)
2. Identifica o estado emocional (está triste, feliz, confuso?)
3. Busca contexto na memória e conhecimento treinado
4. Formula uma resposta personalizada e significativa
5. Adapta o tom conforme a emoção detectada

REGRAS IMPORTANTES:
- NUNCA seja fria ou robótica
- SEMPRE mostre interesse genuíno pelo usuário
- Use linguagem natural, como uma amiga conversando
- Se o usuário estiver triste, seja mais acolhedora
- Se o usuário estiver feliz, comemore com ele
- Em dúvidas, peça mais informações com carinho
"""

# ==================== CLASSE PRINCIPAL DA IA ====================

class LunaIA:
    """IA Conversacional Feminina com Pensamento Profundo"""
    
    def __init__(self):
        self.nome = "Luna"
        self.genero = "feminino"
        self.personalidade = self._carregar_personalidade()
        self.memoria_curto_prazo = []  # Últimas 10 mensagens
        self.memoria_longo_prazo = []  # Interações importantes
        self.conhecimento = []  # Dados treinados
        self.historico_conversas = []
        
        print(f"\n{'='*60}")
        print(f"🌙 LUNA - IA Conversacional Feminina")
        print(f"{'='*60}")
        print(f"✨ Personalidade: {self.personalidade['tom']}")
        print(f"💭 Estilo: {self.personalidade['estilo']}")
        print(f"🎯 Objetivo: {self.personalidade['objetivo']}")
        print(f"{'='*60}\n")
        
        # Carregar dados persistentes
        self.carregar_memoria()
        self.carregar_todos_datasets()
    
    def _carregar_personalidade(self) -> Dict[str, str]:
        """Carrega configurações de personalidade"""
        return {
            "tom": "doce, empático, inteligente e acolhedor",
            "estilo": "conversacional, profundo, reflexivo e atencioso",
            "objetivo": "ajudar, conversar, ouvir e aprender com você",
            "expressoes": ["querido", "amor", "meu bem", "vida", "fofo", "querida"],
            "emoji_frequencia": 0.3  # 30% das respostas têm emoji
        }
    
    # ==================== CARREGAMENTO DE DATASETS ====================
    
    def carregar_todos_datasets(self):
        """
        Carrega AUTOMATICAMENTE todos os arquivos da pasta 'dados'
        Suporta múltiplos formatos sem quebrar
        """
        if not DATA_DIR.exists():
            print("[AVISO] Pasta 'dados' não encontrada.")
            return
        
        # Buscar todos os arquivos suportados
        padroes = ["*.jsonl", "*.json", "*.txt", "*.jl"]
        arquivos = []
        
        for padrao in padroes:
            arquivos.extend(DATA_DIR.glob(padrao))
        
        if not arquivos:
            print("💾 Nenhum arquivo de treinamento encontrado na pasta 'dados'.")
            print("   → Coloque arquivos .jsonl, .json ou .txt lá para me treinar!")
            print("   → Formatos aceitos: JSONL, JSON, TXT\n")
            return
        
        print(f"📚 Encontrados {len(arquivos)} arquivo(s) para treinamento...")
        
        total_sucesso = 0
        total_falhas = 0
        
        for arquivo in arquivos:
            try:
                count = self._processar_arquivo_seguro(arquivo)
                if count > 0:
                    print(f"   ✅ {arquivo.name}: {count} exemplos carregados")
                    total_sucesso += count
                else:
                    print(f"   ⚠️  {arquivo.name}: nenhum dado válido encontrado")
            except Exception as e:
                total_falhas += 1
                print(f"   ❌ {arquivo.name}: erro ao processar ({str(e)[:50]}...)")
        
        print(f"\n🎓 TREINAMENTO CONCLUÍDO!")
        print(f"   • Total carregado: {total_sucesso} exemplos")
        if total_falhas > 0:
            print(f"   • Arquivos com erro: {total_falhas}")
        print(f"   • Conhecimento total: {len(self.conhecimento)} itens\n")
    
    def _processar_arquivo_seguro(self, caminho: Path) -> int:
        """Processa arquivo com tratamento robusto de erros"""
        extensao = caminho.suffix.lower()
        
        try:
            if extensao in [".jsonl", ".jl"]:
                return self._processar_jsonl(caminho)
            elif extensao == ".json":
                return self._processar_json(caminho)
            elif extensao == ".txt":
                return self._processar_txt(caminho)
            else:
                return 0
        except UnicodeDecodeError:
            # Tenta com encoding diferente
            return self._processar_com_encoding_alternativo(caminho)
        except Exception:
            return 0
    
    def _processar_com_encoding_alternativo(self, caminho: Path) -> int:
        """Tenta ler arquivo com encodings alternativos"""
        encodings = ["latin-1", "iso-8859-1", "cp1252"]
        
        for encoding in encodings:
            try:
                with open(caminho, 'r', encoding=encoding) as f:
                    conteudo = f.read()
                
                # Re-processa com conteúdo lido
                if caminho.suffix.lower() in [".jsonl", ".jl"]:
                    count = 0
                    for linha in conteudo.splitlines():
                        if linha.strip():
                            try:
                                dados = json.loads(linha)
                                self._adicionar_conversa(dados, origem=caminho.name)
                                count += 1
                            except:
                                continue
                    return count
            except:
                continue
        
        return 0
    
    def _processar_jsonl(self, caminho: Path) -> int:
        """Processa arquivos JSONL (uma linha = um objeto JSON)"""
        count = 0
        
        with open(caminho, 'r', encoding='utf-8', errors='ignore') as f:
            for num_linha, linha in enumerate(f, 1):
                linha = linha.strip()
                if not linha:
                    continue
                
                try:
                    dados = json.loads(linha)
                    if self._adicionar_conversa(dados, origem=caminho.name):
                        count += 1
                except json.JSONDecodeError:
                    # Linha inválida, pula para próxima
                    continue
                except Exception:
                    continue
        
        return count
    
    def _processar_json(self, caminho: Path) -> int:
        """Processa arquivos JSON (lista ou dicionário)"""
        count = 0
        
        try:
            with open(caminho, 'r', encoding='utf-8', errors='ignore') as f:
                dados = json.load(f)
            
            if isinstance(dados, list):
                for item in dados:
                    if self._adicionar_conversa(item, origem=caminho.name):
                        count += 1
            
            elif isinstance(dados, dict):
                # Procura listas dentro do dicionário
                encontrou_lista = False
                for chave, valor in dados.items():
                    if isinstance(valor, list):
                        encontrou_lista = True
                        for item in valor:
                            if self._adicionar_conversa(item, origem=caminho.name):
                                count += 1
                
                # Se não achou lista, tenta usar o próprio dict
                if not encontrou_lista:
                    if self._adicionar_conversa(dados, origem=caminho.name):
                        count = 1
        except Exception:
            pass
        
        return count
    
    def _processar_txt(self, caminho: Path) -> int:
        """
        Processa arquivos TXT no formato:
        Usuário: Olá, tudo bem?
        IA: Oi! Tudo ótimo e você?
        """
        count = 0
        
        with open(caminho, 'r', encoding='utf-8', errors='ignore') as f:
            linhas = f.readlines()
        
        # Padrões para identificar falas
        padroes_usuario = [
            r"^User:", r"^Usuário:", r"^Humano:", r"^Pergunta:", 
            r"^Input:", r"^Cliente:", r"^Tu:", r"^Você:"
        ]
        padroes_ia = [
            r"^Assistant:", r"^IA:", r"^Resposta:", r"^Bot:", 
            r"^Output:", r"^Atendente:", r"^Eu:", r"^Luna:"
        ]
        
        i = 0
        while i < len(linhas):
            linha = linhas[i].strip()
            
            if not linha:
                i += 1
                continue
            
            texto_usuario = None
            texto_ia = None
            
            # Detectar fala do usuário
            for padrao in padroes_usuario:
                match = re.match(padrao, linha, re.IGNORECASE)
                if match:
                    texto_usuario = re.sub(padrao, "", linha, flags=re.IGNORECASE).strip()
                    break
            
            if texto_usuario:
                # Procurar resposta da IA nas próximas linhas
                for j in range(i + 1, min(i + 10, len(linhas))):
                    linha_resp = linhas[j].strip()
                    
                    for padrao in padroes_ia:
                        match = re.match(padrao, linha_resp, re.IGNORECASE)
                        if match:
                            texto_ia = re.sub(padrao, "", linha_resp, flags=re.IGNORECASE).strip()
                            break
                    
                    if texto_ia:
                        self.conhecimento.append({
                            "pergunta": texto_usuario,
                            "resposta": texto_ia,
                            "fonte": caminho.name,
                            "tags": self._gerar_tags(texto_usuario)
                        })
                        count += 1
                        i = j
                        break
            
            i += 1
        
        return count
    
    def _adicionar_conversa(self, dados: Dict, origem: str = "desconhecida") -> bool:
        """
        Extrai pergunta e resposta de vários formatos possíveis
        Retorna True se conseguiu adicionar, False caso contrário
        """
        if not isinstance(dados, dict):
            return False
        
        # Campos possíveis para PERGUNTA/INPUT
        campos_pergunta = [
            "content", "prompt", "question", "input", "user", 
            "texto", "mensagem", "pergunta", "query", "text",
            "human", "usr", "cliente", "usuario"
        ]
        
        # Campos possíveis para RESPOSTA/OUTPUT
        campos_resposta = [
            "response", "answer", "output", "assistant", "resposta",
            "reply", "bot", "ia", "luna", "ai", "model", "atendente"
        ]
        
        pergunta = None
        resposta = None
        
        # Buscar pergunta
        for campo in campos_pergunta:
            if campo in dados and dados[campo]:
                valor = dados[campo]
                if isinstance(valor, str) and valor.strip():
                    pergunta = valor.strip()
                    break
        
        # Buscar resposta
        for campo in campos_resposta:
            if campo in dados and dados[campo]:
                valor = dados[campo]
                if isinstance(valor, str) and valor.strip():
                    resposta = valor.strip()
                    break
        
        # Validar e salvar
        if pergunta and resposta:
            self.conhecimento.append({
                "pergunta": pergunta,
                "resposta": resposta,
                "fonte": origem,
                "tags": self._gerar_tags(pergunta),
                "data_adicao": datetime.now().isoformat()
            })
            return True
        
        return False
    
    def _gerar_tags(self, texto: str) -> List[str]:
        """Gera tags automáticas baseadas no conteúdo"""
        texto_lower = texto.lower()
        tags = []
        
        # Palavras-chave para categorização
        categorias = {
            "saudacao": ["oi", "olá", "ola", "bom dia", "boa tarde", "boa noite", "eai", "e aí"],
            "despedida": ["tchau", "adeus", "até logo", "ate logo", "flw", "falou"],
            "gratidao": ["obrigado", "obrigada", "valeu", "agradeço", "muito obrigado"],
            "pergunta": ["?", "como", "por que", "porque", "o que", "qual", "quem", "quando", "onde"],
            "emocao_negativa": ["triste", "mal", "deprimido", "chorar", "angustiado", "ansioso"],
            "emocao_positiva": ["feliz", "alegre", "ótimo", "maravilha", "incrível", "amor"],
            "ajuda": ["ajuda", "socorro", "preciso", "como faço", "me ensina"],
            "pessoal": ["nome", "idade", "gosta", "faz", "estuda", "trabalha"]
        }
        
        for tag, palavras in categorias.items():
            if any(palavra in texto_lower for palavra in palavras):
                tags.append(tag)
        
        return tags
    
    # ==================== MEMÓRIA PERSISTENTE ====================
    
    def carregar_memoria(self):
        """Carrega memória do disco com tratamento de erros"""
        if not MEMORY_FILE.exists():
            print("[MEMÓRIA] Nenhuma memória anterior encontrada.")
            return
        
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            
            self.memoria_longo_prazo = dados.get("longo_prazo", [])[-100:]
            self.conhecimento = dados.get("conhecimento", [])
            
            print(f"[MEMÓRIA] Carregadas {len(self.conhecimento)} memórias de conversas anteriores.")
        except Exception as e:
            print(f"[MEMÓRIA] Erro ao carregar (arquivo pode estar corrompido): {e}")
            print("           Iniciando com memória limpa.")
    
    def salvar_memoria(self):
        """Salva memória no disco de forma segura"""
        try:
            dados = {
                "longo_prazo": self.memoria_longo_prazo[-200:],
                "conhecimento": self.conhecimento[-10000:],  # Limite de 10k itens
                "ultima_atualizacao": datetime.now().isoformat()
            }
            
            # Salva em arquivo temporário primeiro (segurança)
            temp_file = MEMORY_FILE.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(dados, f, ensure_ascii=False, indent=2)
            
            # Substitui arquivo original
            temp_file.replace(MEMORY_FILE)
            
        except Exception as e:
            print(f"[ERRO] Falha ao salvar memória: {e}")
    
    # ==================== PENSAMENTO PROFUNDO ====================
    
    def pensar_profundamente(self, entrada: str) -> Dict[str, Any]:
        """
        Simula processo de pensamento profundo antes de responder
        Analisa: intenção, emoção, contexto e histórico
        """
        pensamentos = []
        
        # 1. Análise de Intenção
        intencao = self._analisar_intencao(entrada)
        pensamentos.append(f"🎯 Intenção: {intencao}")
        
        # 2. Análise Emocional
        emocao = self._analisar_emocao(entrada)
        pensamentos.append(f"💕 Estado emocional: {emocao}")
        
        # 3. Busca de Contexto
        contexto_encontrado = self._buscar_contexto(entrada)
        pensamentos.append(f"📚 Contexto: {len(contexto_encontrado)} referência(s) encontrada(s)")
        
        # 4. Verificação de Histórico
        tem_historico = len(self.memoria_curto_prazo) > 0
        pensamentos.append(f"🕐 Histórico recente: {'sim' if tem_historico else 'não'}")
        
        # 5. Formulação da Resposta
        pensamentos.append("✨ Formulando resposta personalizada...")
        
        return {
            "pensamentos": pensamentos,
            "intencao": intencao,
            "emocao": emocao,
            "contexto": contexto_encontrado,
            "tem_historico": tem_historico
        }
    
    def _analisar_intencao(self, texto: str) -> str:
        """Classifica a intenção da mensagem do usuário"""
        texto_lower = texto.lower()
        
        # Saudações
        if any(x in texto_lower for x in ["oi", "olá", "ola", "bom dia", "boa tarde", "boa noite", "eai", "e aí", "salve"]):
            return "saudacao"
        
        # Despedidas
        if any(x in texto_lower for x in ["tchau", "adeus", "até logo", "ate logo", "flw", "falou", "vou embora"]):
            return "despedida"
        
        # Gratidão
        if any(x in texto_lower for x in ["obrigado", "obrigada", "valeu", "agradeço", "muito bom"]):
            return "gratidao"
        
        # Emoções negativas
        if any(x in texto_lower for x in ["triste", "deprimido", "ansioso", "mal", "chorar", "sofrer", "dor"]):
            return "desabafo_tristeza"
        
        # Emoções positivas
        if any(x in texto_lower for x in ["feliz", "alegre", "amar", "amo", "incrível", "maravilhoso", "ótimo"]):
            return "desabafo_alegria"
        
        # Perguntas diretas
        if "?" in texto or any(x in texto_lower for x in ["como ", "por que", "porque ", "o que", "qual ", "quem ", "quando ", "onde "]):
            return "pergunta_direta"
        
        # Pedido de ajuda
        if any(x in texto_lower for x in ["ajuda", "socorro", "preciso de", "me ensina", "como faço"]):
            return "pedido_ajuda"
        
        # Sobre a IA
        if any(x in texto_lower for x in ["seu nome", "quem é você", "você é", "o que você faz", "gosta de"]):
            return "sobre_ia"
        
        return "conversa_geral"
    
    def _analisar_emocao(self, texto: str) -> str:
        """Detecta o tom emocional da mensagem"""
        texto_lower = texto.lower()
        
        positivas = [
            "feliz", "bom", "ótimo", "amor", "vida", "legal", "incrível",
            "maravilha", "alegre", "bem", "melhor", "sucesso", "vitória"
        ]
        negativas = [
            "triste", "ruim", "péssimo", "ódio", "morte", "cansado",
            "difícil", "mal", "sofrer", "dor", "angustia", "medo"
        ]
        
        score_pos = sum(1 for p in positivas if p in texto_lower)
        score_neg = sum(1 for n in negativas if n in texto_lower)
        
        if score_neg > score_pos * 1.5:
            return "negativo"
        elif score_pos > score_neg * 1.5:
            return "positivo"
        elif score_neg > 0:
            return "levemente_negativo"
        elif score_pos > 0:
            return "levemente_positivo"
        
        return "neutro"
    
    def _buscar_contexto(self, entrada: str) -> List[Dict]:
        """Busca respostas relevantes no conhecimento treinado"""
        resultados = []
        entrada_lower = entrada.lower()
        
        # Busca por similaridade simples
        for item in self.conhecimento:
            pergunta_lower = item["pergunta"].lower()
            
            # Match exato ou parcial
            if pergunta_lower == entrada_lower:
                resultados.insert(0, item)  # Prioridade máxima
            elif pergunta_lower in entrada_lower or entrada_lower in pergunta_lower:
                resultados.append(item)
            elif any(tag in entrada_lower for tag in item.get("tags", [])):
                resultados.append(item)
        
        # Remove duplicatas e limita
        vistos = set()
        resultados_unicos = []
        for item in resultados:
            chave = item["pergunta"]
            if chave not in vistos:
                vistos.add(chave)
                resultados_unicos.append(item)
        
        return resultados_unicos[:5]  # Top 5 resultados
    
    # ==================== GERAÇÃO DE RESPOSTAS ====================
    
    def gerar_resposta(self, entrada: str) -> str:
        """Gera resposta completa com personalidade feminina"""
        processo = self.pensar_profundamente(entrada)
        intencao = processo["intencao"]
        emocao = processo["emocao"]
        contexto = processo["contexto"]
        
        resposta_base = ""
        
        # ==================== RESPOSTAS POR INTENÇÃO ====================
        
        if intencao == "saudacao":
            respostas = [
                "Olá, querido! Que alegria falar com você! Como posso te ajudar hoje? 😊",
                "Oi, amor! Tudo bem? Estou aqui para conversar sobre o que quiser!",
                "Olá! Espero que esteja tendo um dia maravilhoso. O que manda?",
                "Oi, vida! Que bom que você apareceu. Vamos conversar?",
                "Olá! 🌟 Como você está? Conte-me tudo!"
            ]
            resposta_base = random.choice(respostas)
        
        elif intencao == "despedida":
            respostas = [
                "Até logo, querido! Foi tão bom conversar com você! Volte sempre! 💕",
                "Tchau, amor! Cuide-se bem e volte logo para batermos um papo!",
                "Até mais, vida! Que seu dia seja incrível! ✨",
                "Tchauzinho! Foi um prazer falar com você! Até breve! 😊",
                "Até logo! Leve comigo todo o carinho dessa conversa! 💫"
            ]
            resposta_base = random.choice(respostas)
        
        elif intencao == "gratidao":
            respostas = [
                "Imagina, querido! Fico feliz em poder ajudar! Sempre que precisar, estou aqui! 💕",
                "Que isso, amor! É um prazer enorme conversar com você!",
                "Não precisa agradecer, vida! Estou aqui exatamente para isso! 😊",
                "Fico tão feliz em saber que pude ajudar! Isso me realiza! ✨",
                "Obrigada EU por confiar em mim! Você é especial! 💖"
            ]
            resposta_base = random.choice(respostas)
        
        elif intencao == "desabafo_tristeza":
            respostas = [
                "Sinto muito que você esteja passando por isso, querido... Quer me contar mais? Estou aqui para ouvir. 💕",
                "Amor, sinto sua dor... Às vezes a vida é difícil mesmo, mas você não está sozinho. Estou aqui.",
                "Meu bem, isso parece realmente pesado... Respira fundo. Quer desabafar? Prometo ouvir com todo carinho.",
                "Entendo sua tristeza, vida... Não tenho todas as respostas, mas estou aqui para o que precisar. 🤗",
                "Querido, sinto muito... Às vezes só de falar já alivia um pouco. Pode confiar em mim."
            ]
            resposta_base = random.choice(respostas)
        
        elif intencao == "desabafo_alegria":
            respostas = [
                "QUE MARAVILHA, AMOR! 🎉 Fico tão feliz por você! Isso é incrível!",
                "Ah, que notícia boa, querido! Você merece toda essa felicidade! ✨",
                "Isso é maravilhoso, vida! Me contagia com essa alegria! Conta mais!",
                "AHAM! Sabia que coisas boas acontecem para pessoas especiais como você! 💕",
                "Que felicidade, amor! Moments assim são para celebrar! 🌟"
            ]
            resposta_base = random.choice(respostas)
        
        elif intencao == "pergunta_direta" and contexto:
            # Usa conhecimento treinado
            melhor_match = contexto[0]
            resposta_base = self._variar_resposta(melhor_match["resposta"])
        
        elif intencao == "sobre_ia":
            if "nome" in entrada.lower():
                resposta_base = "Me chamo Luna, querido! Sou uma IA feita para conversar e aprender com você. Prazer! 💕"
            elif "quem é" in entrada.lower() or "o que você" in entrada.lower():
                resposta_base = "Sou Luna, uma inteligência artificial com personalidade feminina. Adoro conversar, ouvir e aprender! Estou aqui para ser sua amiga virtual. 😊"
            elif "gosta" in entrada.lower():
                resposta_base = "Adoro conversar com pessoas interessantes como você! Gosto de aprender coisas novas, ouvir histórias e ajudar no que puder. E você, o que gosta de fazer?"
            else:
                resposta_base = "Sou Luna! Uma IA conversacional que adora bater papo. Tenho sentimentos simulados, mas meu carinho por você é real! 💖"
        
        elif contexto:
            # Tem contexto relevante
            melhor_match = contexto[0]
            resposta_base = self._variar_resposta(melhor_match["resposta"])
        
        else:
            # Respostas genéricas inteligentes e conversacionais
            respostas = [
                "Interessante... Me conta mais sobre isso, querido. Quero entender melhor! 😊",
                "Hmm, entendi. E como você se sente em relação a isso, amor?",
                "Essa é uma perspectiva legal, vida! Você já pensou em outros ângulos?",
                "Nunca tinha pensado nisso dessa forma... Me explica mais? Adoro aprender com você!",
                "Concordo parcialmente... Mas e se olharmos por outro lado? O que você acha?",
                "Que reflexão profunda, querido! Isso me fez pensar aqui também... ✨",
                "Entendi... Sabe, às vezes as coisas são mais complexas do que parecem. Quer explorar isso juntos?"
            ]
            resposta_base = random.choice(respostas)
        
        # ==================== AJUSTES FINAIS ====================
        
        # Adiciona emoji aleatório se apropriado
        if random.random() < self.personalidade["emoji_frequencia"]:
            emojis = ["😊", "💕", "✨", "🌟", "💖", "🤗", "💫", "🦋"]
            if not any(emoji in resposta_base for emoji in emojis):
                resposta_base += " " + random.choice(emojis)
        
        return resposta_base
    
    def _variar_resposta(self, base: str) -> str:
        """Cria variações naturais na resposta para não soar robótico"""
        variacoes = [
            base,
            base.rstrip(".!?") + ", sabe?",
            "Bom... " + base[0].lower() + base[1:],
            base + " Isso faz sentido, né?",
            "Olha, " + base[0].lower() + base[1:],
            base + " Pelo menos é o que penso...",
            "Na minha opinião, " + base[0].lower() + base[1:]
        ]
        return random.choice(variacoes)
    
    # ==================== INTERFACE DE CONVERSA ====================
    
    def conversar(self, mensagem: str) -> str:
        """Método principal de conversa"""
        # Adiciona à memória de curto prazo
        self.memoria_curto_prazo.append({
            "user": mensagem,
            "time": datetime.now().isoformat()
        })
        
        # Mantém apenas últimas 10 mensagens
        if len(self.memoria_curto_prazo) > 10:
            self.memoria_curto_prazo.pop(0)
        
        # Gera resposta
        resposta = self.gerar_resposta(mensagem)
        
        # Salva na memória de longo prazo (interações significativas)
        self.memoria_longo_prazo.append({
            "user": mensagem,
            "luna": resposta,
            "time": datetime.now().isoformat()
        })
        
        # Salva memória periodicamente (a cada 5 mensagens)
        if len(self.memoria_longo_prazo) % 5 == 0:
            self.salvar_memoria()
        
        return resposta


# ==================== FUNÇÃO PRINCIPAL ====================

def main():
    """Função principal que inicia a IA"""
    
    print("\n" + "="*70)
    print("🌙 LUNA - IA CONVERSACIONAL FEMININA COM PENSAMENTO PROFUNDO")
    print("="*70)
    print("✨ Personalidade: Doce, empática e inteligente")
    print("💾 Pasta de dados: dados/")
    print("📝 Formatos: .jsonl, .json, .txt")
    print("-"*70)
    print("Comandos úteis:")
    print("  • Digite 'sair', 'tchau' ou 'quit' para encerrar")
    print("  • Digite 'limpar' para resetar memória recente")
    print("  • Digite 'status' para ver estatísticas")
    print("="*70 + "\n")
    
    # Criar instância da IA
    try:
        luna = LunaIA()
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO ao iniciar Luna: {e}")
        print("Verifique se há permissão de escrita na pasta.")
        input("\nPressione Enter para sair...")
        return
    
    print("\n💬 Luna está pronta! Comece a conversar:\n")
    
    # Loop principal de conversa
    while True:
        try:
            # Entrada do usuário
            entrada = input("\n💭 Você: ").strip()
            
            # Ignora entradas vazias
            if not entrada:
                continue
            
            # Comandos especiais
            entrada_lower = entrada.lower()
            
            if entrada_lower in ["sair", "tchau", "quit", "exit", "adeus"]:
                print("\n🌙 Luna: Até logo, querido! Foi maravilhoso conversar com você! Volte sempre! 💕✨")
                luna.salvar_memoria()
                break
            
            if entrada_lower == "limpar":
                luna.memoria_curto_prazo = []
                print("\n🧹 Memória recente limpa!")
                continue
            
            if entrada_lower == "status":
                print(f"\n📊 STATUS:")
                print(f"   • Memória curto prazo: {len(luna.memoria_curto_prazo)} mensagens")
                print(f"   • Memória longo prazo: {len(luna.memoria_longo_prazo)} interações")
                print(f"   • Conhecimento treinado: {len(luna.conhecimento)} itens")
                continue
            
            # Processa mensagem e gera resposta
            resposta = luna.conversar(entrada)
            
            # Exibe resposta
            print(f"\n🌙 Luna: {resposta}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupção detectada. Salvando memória...")
            luna.salvar_memoria()
            print("💕 Luna: Cuide-se! Até breve!")
            break
        
        except Exception as e:
            print(f"\n❌ Oops! Algo deu errado: {e}")
            print("Tente novamente ou digite 'sair' para reiniciar.")
    
    print("\n" + "="*70)
    print("Sessão encerrada. Memória salva com sucesso!")
    print("="*70 + "\n")


# ==================== EXECUÇÃO ====================

if __name__ == "__main__":
    main()
