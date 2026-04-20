"""
reasoner.py - Simulador de Chain of Thought para Grok-Advanced-BR-Base

Este módulo implementa um sistema de raciocínio simulado usando regras em Python.
Não usa redes neurais ou LLMs - apenas lógica programática para simular
o processo de pensamento passo a passo.

Autor: Grok-Advanced-BR-Base
Licença: MIT
"""

from typing import List, Tuple, Dict, Optional
from rag_system import RAGSystem, extrair_palavras_chave


class Reasoner:
    """
    Simula Chain of Thought (cadeia de pensamento) usando regras.
    
    O processo de raciocínio segue estes passos:
    1. Entender a pergunta (extrair palavras-chave e intenção)
    2. Analisar contexto do banco de dados
    3. Identificar se falta informação
    4. Pensar em caminhos alternativos (Tree of Thoughts simplificado)
    5. Criar rascunho da resposta
    6. Auto-crítica
    7. Versão final
    """
    
    def __init__(self, rag_system: RAGSystem):
        """
        Inicializa o reasoner.
        
        Args:
            rag_system: Instância do sistema RAG para buscar contexto
        """
        self.rag = rag_system
        self.historico_pensamentos: List[str] = []  # Registra o "pensamento"
    
    def pensar(self, pergunta: str) -> Dict:
        """
        Executa todo o processo de raciocínio sobre uma pergunta.
        
        Args:
            pergunta: Texto da pergunta do usuário
        
        Returns:
            Dicionário com:
            - pensamentos: Lista de passos do raciocínio
            - resposta_final: Resposta gerada
            - confianca: Nível de confiança na resposta (0-1)
            - documentos_usados: Documentos relevantes encontrados
        """
        self.historico_pensamentos = []
        
        # === PASSO 1: Entender a pergunta ===
        self._pensar("Entendendo a pergunta...")
        palavras_chave = extrair_palavras_chave(pergunta)
        intencao = self._detectar_intencao(pergunta)
        
        self._pensar(f"Palavras-chave identificadas: {', '.join(palavras_chave)}")
        self._pensar(f"Intenção detectada: {intencao}")
        
        # === PASSO 2: Analisar contexto do banco de dados ===
        self._pensar("Buscando informações no banco de dados...")
        documentos_relevantes = self.rag.buscar_relevantes(pergunta, top_k=3)
        
        if documentos_relevantes:
            self._pensar(f"Encontrados {len(documentos_relevantes)} documentos relevantes")
            for i, (doc, score) in enumerate(documentos_relevantes[:2], 1):
                self._pensar(f"  Documento {i} (similaridade: {score:.2f}): {doc[:60]}...")
        else:
            self._pensar("Nenhum documento relevante encontrado no banco de dados")
        
        # === PASSO 3: Identificar se falta informação ===
        self._pensar("Analisando se há informação suficiente...")
        informacao_suficiente = len(documentos_relevantes) > 0 and documentos_relevantes[0][1] > 0.1
        falta_info = not informacao_suficiente
        
        if falta_info:
            self._pensar("AVISO: Informação insuficiente nos documentos")
        else:
            self._pensar("Informação adequada encontrada")
        
        # === PASSO 4: Tree of Thoughts (2 caminhos alternativos) ===
        self._pensar("Considerando diferentes abordagens...")
        
        caminho_1 = self._gerar_caminho_resposta(pergunta, documentos_relevantes, tipo="direto")
        caminho_2 = self._gerar_caminho_resposta(pergunta, documentos_relevantes, tipo="contextual")
        
        self._pensar(f"Caminho 1 (direto): {caminho_1['estrategia']}")
        self._pensar(f"Caminho 2 (contextual): {caminho_2['estrategia']}")
        
        # Escolhe o melhor caminho baseado na confiança
        melhor_caminho = caminho_1 if caminho_1['confianca'] >= caminho_2['confianca'] else caminho_2
        self._pensar(f"Caminho escolhido: {melhor_caminho['estrategia']} (confiança: {melhor_caminho['confianca']:.2f})")
        
        # === PASSO 5: Criar rascunho da resposta ===
        self._pensar("Criando rascunho da resposta...")
        rascunho = self._criar_rascunho(pergunta, melhor_caminho, documentos_relevantes)
        self._pensar(f"Rascunho: {rascunho[:100]}...")
        
        # === PASSO 6: Auto-crítica rigorosa ===
        self._pensar("Realizando auto-crítica...")
        critica = self._auto_critica(rascunho, pergunta, documentos_relevantes)
        
        if critica['precisa_corrigir']:
            self._pensar(f"Problemas identificados: {', '.join(critica['problemas'])}")
            self._pensar("Corrigindo resposta...")
            resposta_final = self._corrigir_resposta(rascunho, critica, documentos_relevantes)
        else:
            self._pensar("Resposta aprovada na auto-crítica")
            resposta_final = rascunho
        
        # === PASSO 7: Versão final ===
        self._pensar("Gerando versão final da resposta")
        
        return {
            'pensamentos': self.historico_pensamentos.copy(),
            'resposta_final': resposta_final,
            'confianca': melhor_caminho['confianca'],
            'documentos_usados': documentos_relevantes,
            'palavras_chave': palavras_chave,
            'intencao': intencao
        }
    
    def _pensar(self, pensamento: str) -> None:
        """Registra um pensamento no histórico."""
        self.historico_pensamentos.append(pensamento)
    
    def _detectar_intencao(self, pergunta: str) -> str:
        """
        Detecta a intenção da pergunta usando regras simples.
        
        Retorna categorias como: definicao, comparacao, explicacao, etc.
        """
        pergunta_lower = pergunta.lower()
        
        # Palavras-chave para cada tipo de intenção
        if any(word in pergunta_lower for word in ['o que é', 'o que sao', 'defina', 'definicao', 'significa']):
            return 'definicao'
        elif any(word in pergunta_lower for word in ['como', 'de que forma', 'qual o modo']):
            return 'explicacao'
        elif any(word in pergunta_lower for word in ['qual a diferenca', 'diferente', 'compar', 'vs', 'versus']):
            return 'comparacao'
        elif any(word in pergunta_lower for word in ['porque', 'por que', 'qual a razao', 'motivo']):
            return 'causa'
        elif any(word in pergunta_lower for word in ['quem', 'qual pessoa', 'criador', 'autor']):
            return 'pessoa'
        elif any(word in pergunta_lower for word in ['quando', 'data', 'ano', 'epoca']):
            return 'tempo'
        elif any(word in pergunta_lower for word in ['onde', 'local', 'lugar']):
            return 'local'
        else:
            return 'geral'
    
    def _gerar_caminho_resposta(self, pergunta: str, docs: List[Tuple[str, float]], tipo: str) -> Dict:
        """
        Gera uma estratégia de resposta (Tree of Thoughts simplificado).
        
        Args:
            pergunta: Pergunta original
            docs: Documentos relevantes
            tipo: 'direto' ou 'contextual'
        
        Returns:
            Dicionário com estratégia e nível de confiança
        """
        if not docs:
            return {
                'estrategia': 'Responder que não há informação suficiente',
                'confianca': 0.1
            }
        
        doc_principal, score = docs[0]
        
        if tipo == 'direto':
            # Estratégia: responder diretamente com base no documento mais relevante
            estrategia = f"Usar documento principal (score: {score:.2f}) para resposta direta"
            confianca = score * 0.9  # Confiança baseada na similaridade
        else:
            # Estratégia: contextualizar com múltiplos documentos
            if len(docs) > 1:
                estrategia = f"Combinar {len(docs)} documentos para resposta contextualizada"
                confianca = (score + sum(s for _, s in docs[1:])) / len(docs) * 0.85
            else:
                estrategia = "Contextualizar com único documento disponível"
                confianca = score * 0.8
        
        return {
            'estrategia': estrategia,
            'confianca': min(confianca, 0.95),  # Limita confiança máxima
            'tipo': tipo
        }
    
    def _criar_rascunho(self, pergunta: str, caminho: Dict, docs: List[Tuple[str, float]]) -> str:
        """
        Cria um rascunho de resposta baseado nos documentos.
        
        Args:
            pergunta: Pergunta original
            caminho: Estratégia escolhida
            docs: Documentos relevantes
        
        Returns:
            Rascunho da resposta
        """
        if not docs:
            return "Desculpe, não encontrei informações suficientes no meu banco de dados para responder sua pergunta."
        
        # Pega o documento mais relevante
        doc_principal, score = docs[0]
        
        # Estratégias diferentes baseadas no tipo de caminho
        if caminho['tipo'] == 'direto':
            # Resposta direta
            rascunho = f"Baseado nas informações disponíveis: {doc_principal}"
        else:
            # Resposta contextualizada
            if len(docs) > 1:
                # Combina múltiplos documentos
                contextos = [doc for doc, _ in docs[:2]]
                rascunho = f"Segundo os documentos encontrados: {' '.join(contextos)}"
            else:
                rascunho = f"De acordo com o banco de dados: {doc_principal}"
        
        return rascunho
    
    def _auto_critica(self, rascunho: str, pergunta: str, docs: List[Tuple[str, float]]) -> Dict:
        """
        Realiza auto-crítica da resposta gerada.
        
        Args:
            rascunho: Resposta gerada
            pergunta: Pergunta original
            docs: Documentos usados
        
        Returns:
            Dicionário com problemas identificados e se precisa corrigir
        """
        problemas = []
        precisa_corrigir = False
        
        # Verificação 1: Resposta muito curta?
        if len(rascunho.split()) < 10:
            problemas.append("Resposta muito curta")
            precisa_corrigir = True
        
        # Verificação 2: Resposta genérica demais?
        if "não encontrei" in rascunho.lower() or "desculpe" in rascunho.lower():
            if len(docs) > 0 and docs[0][1] > 0.2:
                problemas.append("Resposta muito genérica apesar de ter documentos")
                precisa_corrigir = True
        
        # Verificação 3: Contém informações dos documentos?
        tem_info_doc = False
        for doc, _ in docs:
            if any(palavra in rascunho.lower() for palavra in doc.lower().split()[:5]):
                tem_info_doc = True
                break
        
        if docs and not tem_info_doc and len(rascunho) > 50:
            problemas.append("Pode não estar usando informações dos documentos corretamente")
        
        # Verificação 4: Responde à pergunta?
        palavras_pergunta = set(extrair_palavras_chave(pergunta))
        palavras_resposta = set(rascunho.lower().split())
        intersecao = palavras_pergunta.intersection(palavras_resposta)
        
        if len(intersecao) < 2 and len(palavras_pergunta) > 2:
            problemas.append("Pode não estar respondendo diretamente à pergunta")
        
        return {
            'precisa_corrigir': precisa_corrigir,
            'problemas': problemas
        }
    
    def _corrigir_resposta(self, rascunho: str, critica: Dict, docs: List[Tuple[str, float]]) -> str:
        """
        Corrige a resposta baseada na auto-crítica.
        
        Args:
            rascunho: Resposta original
            critica: Problemas identificados
            docs: Documentos para referência
        
        Returns:
            Resposta corrigida
        """
        # Se não há documentos, mantém a resposta original
        if not docs:
            return rascunho
        
        doc_principal, score = docs[0]
        
        # Correções baseadas nos problemas identificados
        if "Resposta muito curta" in critica['problemas']:
            # Expande a resposta
            rascunho = f"Analisando sua pergunta e consultando meu banco de dados: {doc_principal}"
        
        if "Resposta muito genérica" in critica['problemas']:
            # Torna mais específica
            rascunho = f"Especificamente sobre este tema: {doc_principal}"
        
        if "Pode não estar usando informações" in critica['problemas']:
            # Garante que usa informação do documento
            rascunho = f"Conforme registrado: {doc_principal}"
        
        return rascunho
    
    def get_pensamentos_formatados(self) -> str:
        """Retorna todos os pensamentos formatados para exibição."""
        return "\n".join([f"  • {p}" for p in self.historico_pensamentos])


# ==================== TESTE RÁPIDO ====================

if __name__ == "__main__":
    print("=== Teste do Reasoner ===\n")
    
    # Cria sistema RAG e adiciona documentos
    rag = RAGSystem()
    docs = [
        "Python é uma linguagem de programação criada por Guido van Rossum em 1991.",
        "Inteligência Artificial permite máquinas simularem inteligência humana.",
        "Machine Learning é um subcampo da IA que aprende com dados."
    ]
    rag.adicionar_documentos(docs)
    
    # Cria reasoner
    reasoner = Reasoner(rag)
    
    # Testa com uma pergunta
    pergunta = "O que é Python?"
    print(f"Pergunta: {pergunta}\n")
    
    resultado = reasoner.pensar(pergunta)
    
    print("\n=== Processo de Pensamento ===")
    print(reasoner.get_pensamentos_formatados())
    
    print(f"\n=== Resposta Final ===")
    print(resultado['resposta_final'])
    
    print(f"\nConfiança: {resultado['confianca']:.2f}")
    
    print("\n=== Teste concluído ===")
