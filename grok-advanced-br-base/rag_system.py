"""
rag_system.py - Sistema RAG leve para Grok-Advanced-BR-Base

Este módulo implementa um sistema de Retrieval-Augmented Generation (RAG)
super leve, usando TF-IDF e similaridade de cosseno do scikit-learn.
Suporta leitura de arquivos .txt, .md, .json, .csv, .zip da pasta data/

Autor: Grok-Advanced-BR-Base
Licença: MIT
"""

import os
import json
import csv
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configurações
TOP_K = 3  # Número máximo de documentos relevantes a retornar
DATA_FOLDER = "data"  # Pasta onde ficam os documentos


class RAGSystem:
    """
    Sistema RAG leve usando TF-IDF + Similaridade de Cosseno.
    
    Este sistema:
    1. Lê documentos de uma pasta (incluindo ZIPs)
    2. Indexa documentos usando TF-IDF
    3. Busca documentos relevantes usando similaridade de cosseno
    """
    
    def __init__(self, data_folder: str = DATA_FOLDER):
        """Inicializa o sistema RAG."""
        self.data_folder = Path(data_folder)
        self.documentos: List[str] = []  # Lista de documentos armazenados
        self.fontes: List[str] = []  # Lista de fontes (nomes dos arquivos)
        self.vectorizer: Optional[TfidfVectorizer] = None  # Vectorizador TF-IDF
        self.matriz_tfidf = None  # Matriz TF-IDF dos documentos
        
        # Cria pasta de dados se não existir
        self.data_folder.mkdir(parents=True, exist_ok=True)
        
        # Carrega documentos automaticamente da pasta
        self.carregar_documentos_da_pasta()
    
    def carregar_documentos_da_pasta(self) -> int:
        """
        Lê todos os arquivos suportados da pasta de dados.
        Suporta: .txt, .md, .json, .csv, .zip
        
        Returns:
            Número de documentos carregados
        """
        if not self.data_folder.exists():
            print(f"[RAG] Pasta {self.data_folder} não existe. Criando...")
            self.data_folder.mkdir(parents=True, exist_ok=True)
            return 0
        
        total_carregados = 0
        
        # Percorre todos os arquivos da pasta
        for arquivo in self.data_folder.iterdir():
            if arquivo.is_file():
                try:
                    docs = self._ler_arquivo(arquivo)
                    if docs:
                        for doc_texto in docs:
                            self.documentos.append(doc_texto)
                            self.fontes.append(str(arquivo.name))
                            total_carregados += 1
                        print(f"[RAG] ✓ {arquivo.name}: {len(docs)} documento(s)")
                except Exception as e:
                    print(f"[RAG] ⚠ Erro ao ler {arquivo.name}: {e}")
        
        # Reindexa após carregar tudo
        if total_carregados > 0:
            self._reindexar()
        
        print(f"[RAG] Total: {total_carregados} documento(s) carregado(s)")
        return total_carregados
    
    def _ler_arquivo(self, caminho: Path) -> List[str]:
        """
        Lê um arquivo e extrai texto.
        
        Args:
            caminho: Caminho do arquivo
            
        Returns:
            Lista de strings (documentos extraídos)
        """
        extensao = caminho.suffix.lower()
        
        if extensao == '.txt' or extensao == '.md':
            return self._ler_texto_simples(caminho)
        elif extensao == '.json':
            return self._ler_json(caminho)
        elif extensao == '.csv':
            return self._ler_csv(caminho)
        elif extensao == '.zip':
            return self._ler_zip(caminho)
        else:
            # Formato não suportado
            return []
    
    def _ler_texto_simples(self, caminho: Path) -> List[str]:
        """Lê arquivo .txt ou .md"""
        try:
            with open(caminho, 'r', encoding='utf-8') as f:
                conteudo = f.read()
            
            # Divide em parágrafos (cada parágrafo é um documento)
            paragrafos = [p.strip() for p in conteudo.split('\n\n') if p.strip()]
            
            # Se não tiver parágrafos claros, usa o texto inteiro
            if not paragrafos:
                paragrafos = [conteudo.strip()] if conteudo.strip() else []
            
            return paragrafos
        except Exception as e:
            print(f"Erro ao ler {caminho}: {e}")
            return []
    
    def _ler_json(self, caminho: Path) -> List[str]:
        """Lê arquivo .json e extrai campos de texto"""
        try:
            with open(caminho, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            
            textos = []
            
            # Se for lista de objetos
            if isinstance(dados, list):
                for item in dados:
                    if isinstance(item, dict):
                        # Extrai valores de texto do dicionário
                        for valor in item.values():
                            if isinstance(valor, str) and len(valor) > 20:
                                textos.append(valor)
                    elif isinstance(item, str) and len(item) > 20:
                        textos.append(item)
            # Se for objeto único
            elif isinstance(dados, dict):
                for valor in dados.values():
                    if isinstance(valor, str) and len(valor) > 20:
                        textos.append(valor)
                    elif isinstance(valor, list):
                        for item in valor:
                            if isinstance(item, str) and len(item) > 20:
                                textos.append(item)
            
            return textos if textos else [json.dumps(dados, ensure_ascii=False)]
        except Exception as e:
            print(f"Erro ao ler JSON {caminho}: {e}")
            return []
    
    def _ler_csv(self, caminho: Path) -> List[str]:
        """Lê arquivo .csv e converte linhas em documentos"""
        try:
            with open(caminho, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                textos = []
                
                for linha in reader:
                    # Concatena todos os valores da linha
                    partes = [str(v) for v in linha.values() if v and str(v).strip()]
                    if partes:
                        textos.append(' | '.join(partes))
                
                return textos
        except Exception as e:
            print(f"Erro ao ler CSV {caminho}: {e}")
            # Tenta ler como texto simples
            return self._ler_texto_simples(caminho)
    
    def _ler_zip(self, caminho: Path) -> List[str]:
        """
        Lê arquivo .zip e extrai texto de arquivos internos.
        Suporta: .txt, .md, .json, .csv dentro do ZIP
        """
        textos = []
        
        try:
            with zipfile.ZipFile(caminho, 'r') as zip_ref:
                # Lista todos os arquivos no ZIP
                for nome_arquivo in zip_ref.namelist():
                    extensao = Path(nome_arquivo).suffix.lower()
                    
                    # Processa apenas formatos suportados
                    if extensao in ['.txt', '.md', '.json', '.csv']:
                        try:
                            with zip_ref.open(nome_arquivo) as f:
                                conteudo = f.read().decode('utf-8')
                            
                            # Processa conforme o tipo
                            if extensao in ['.txt', '.md']:
                                paragrafos = [p.strip() for p in conteudo.split('\n\n') if p.strip()]
                                textos.extend(paragrafos if paragrafos else [conteudo])
                            elif extensao == '.json':
                                dados = json.loads(conteudo)
                                if isinstance(dados, list):
                                    for item in dados:
                                        if isinstance(item, str) and len(item) > 20:
                                            textos.append(item)
                                elif isinstance(dados, dict):
                                    for valor in dados.values():
                                        if isinstance(valor, str) and len(valor) > 20:
                                            textos.append(valor)
                            elif extensao == '.csv':
                                import io
                                reader = csv.DictReader(io.StringIO(conteudo))
                                for linha in reader:
                                    partes = [str(v) for v in linha.values() if v and str(v).strip()]
                                    if partes:
                                        textos.append(' | '.join(partes))
                        except Exception as e:
                            print(f"  ⚠ Erro ao extrair {nome_arquivo}: {e}")
                
                print(f"  📦 ZIP {caminho.name}: {len(textos)} documento(s) extraído(s)")
                return textos
                
        except Exception as e:
            print(f"Erro ao abrir ZIP {caminho}: {e}")
            return []
    
    def adicionar_documento(self, texto: str, fonte: str = "manual") -> None:
        """
        Adiciona um documento manualmente.
        
        Args:
            texto: Texto do documento
            fonte: Nome da fonte (opcional)
        """
        if not texto.strip():
            return
        
        self.documentos.append(texto.strip())
        self.fontes.append(fonte)
        self._reindexar()
    
    def _reindexar(self, incremental: bool = False) -> None:
        """
        Recria a matriz TF-IDF com todos os documentos.
        
        Args:
            incremental: Se True, tenta adicionar novos documentos sem reindexar tudo
                        (útil para grandes bases de dados)
        """
        if not self.documentos:
            return
        
        # Configurações otimizadas para grandes volumes de dados
        max_features = min(5000, len(self.documentos) * 2)  # Escala com o tamanho da base
        
        # Cria ou atualiza o vectorizador TF-IDF
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,  # Pode mudar para 'portuguese' se tiver nltk
            max_features=max_features,  # Ajusta dinamicamente baseado no tamanho da base
            ngram_range=(1, 3),  # Adiciona trigrams para melhor captura de frases
            sublinear_tf=True,  # Usa log(1+tf) em vez de tf (melhor para perguntas)
            min_df=2 if len(self.documentos) > 1000 else 1,  # Ignora termos muito raros em bases grandes
            max_df=0.95,  # Ignora termos muito comuns (top 5%)
            norm='l2',  # Normalização L2 para melhor similaridade de cosseno
            dtype=np.float32  # Usa float32 para economizar memória
        )
        
        # Transforma documentos em matriz TF-IDF
        print(f"[RAG] Indexando {len(self.documentos):,} documentos com {max_features:,} features...")
        self.matriz_tfidf = self.vectorizer.fit_transform(self.documentos)
        print(f"[RAG] Matriz criada: {self.matriz_tfidf.shape[0]:,} x {self.matriz_tfidf.shape[1]:,}")
    
    def buscar_relevantes(self, consulta: str, top_k: int = None, min_score: float = 0.05) -> List[Tuple[str, float]]:
        """
        Busca documentos relevantes para uma consulta.
        
        Args:
            consulta: Texto da pergunta/consulta
            top_k: Número de documentos a retornar (usa TOP_K se None)
            min_score: Score mínimo para considerar documento relevante
        
        Returns:
            Lista de tuplas (documento, score_de_similaridade)
        """
        if not self.documentos or self.matriz_tfidf is None:
            return []
        
        top_k = top_k or TOP_K
        
        # Usa TF-IDF puro (mais leve - sem dependência de sentence-transformers)
        resultados = self._buscar_com_tfidf(consulta, top_k * 2)  # Busca mais para filtrar por score
        
        # Filtra resultados por score mínimo
        resultados_filtrados = [(doc, score) for doc, score in resultados if score >= min_score]
        
        # Retorna apenas top_k após filtragem
        return resultados_filtrados[:top_k]
    
    def _buscar_com_tfidf(self, consulta: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Busca usando TF-IDF + similaridade de cosseno.
        
        Método super leve que não requer modelos pré-treinados.
        Usa busca esparsa otimizada para grandes volumes de dados.
        """
        # Transforma a consulta em vetor TF-IDF
        try:
            vetor_consulta = self.vectorizer.transform([consulta])
        except Exception:
            # Se houver palavras novas, retorna vazio
            return []
        
        # Calcula similaridade de cosseno entre consulta e todos os documentos
        # Para bases muito grandes, usa cálculo esparsificado
        similaridades = cosine_similarity(vetor_consulta, self.matriz_tfidf)[0]
        
        # Cria lista de TODOS os documentos com seus scores (mesmo zero)
        todos_resultados = []
        for idx, score in enumerate(similaridades):
            score_float = float(score)
            if score_float > 0:  # Ignora documentos com score zero para economizar memória
                todos_resultados.append((self.documentos[idx], score_float))
        
        # Ordena por score decrescente (melhor primeiro)
        todos_resultados.sort(key=lambda x: x[1], reverse=True)
        
        # Retorna apenas top_k
        return todos_resultados[:top_k]
    
    def get_total_documentos(self) -> int:
        """Retorna o número total de documentos armazenados."""
        return len(self.documentos)
    
    def limpar_documentos(self) -> None:
        """Limpa todos os documentos do banco de dados."""
        self.documentos = []
        self.vectorizer = None
        self.matriz_tfidf = None
        print("[RAG] Documentos limpos")


# ==================== FUNÇÕES DE UTILIDADE ====================

def criar_documento_resumo(texto: str, max_palavras: int = 50) -> str:
    """
    Cria um resumo simples de um texto.
    
    Args:
        texto: Texto original
        max_palavras: Número máximo de palavras no resumo
    
    Returns:
        Texto resumido
    """
    palavras = texto.split()
    if len(palavras) <= max_palavras:
        return texto
    
    return ' '.join(palavras[:max_palavras]) + "..."


def extrair_palavras_chave(texto: str, max_palavras: int = 5) -> List[str]:
    """
    Extrai palavras-chave simples de um texto.
    
    Args:
        texto: Texto para extrair palavras-chave
        max_palavras: Número máximo de palavras-chave
    
    Returns:
        Lista de palavras-chave
    """
    # Palavras comuns para remover (stop words básicas em português)
    stop_words = {
        'de', 'do', 'da', 'dos', 'das', 'em', 'um', 'uma', 'uns', 'umas',
        'o', 'a', 'os', 'as', 'que', 'e', 'ou', 'se', 'na', 'no', 'ao',
        'para', 'com', 'por', 'como', 'foi', 'é', 'são', 'ser', 'foi',
        'tem', 'ter', 'uma', 'isto', 'isso', 'aquilo', 'qual', 'quais'
    }
    
    # Remove pontuação e converte para minúsculas
    texto_limpo = ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in texto)
    
    # Divide em palavras e remove stop words
    palavras = [p for p in texto_limpo.split() if p not in stop_words and len(p) > 2]
    
    # Retorna as primeiras max_palavras
    return palavras[:max_palavras]


# ==================== TESTE RÁPIDO ====================

if __name__ == "__main__":
    # Teste rápido do sistema RAG
    print("=== Teste do Sistema RAG ===\n")
    
    # Cria instância do RAG (vai carregar da pasta data/)
    rag = RAGSystem()
    
    # Se não tiver documentos, adiciona alguns de exemplo
    if rag.get_total_documentos() == 0:
        print("\nNenhum documento na pasta. Adicionando exemplos...\n")
        docs_exemplo = [
            "Python é uma linguagem de programação de alto nível criada por Guido van Rossum em 1991.",
            "Inteligência Artificial é a simulação de processos de inteligência humana por máquinas.",
            "Machine Learning é um subcampo da IA que permite aos sistemas aprenderem com dados.",
            "Redes neurais são algoritmos inspirados no funcionamento do cérebro humano.",
            "Processamento de Linguagem Natural permite computadores entenderem linguagem humana."
        ]
        
        for doc in docs_exemplo:
            rag.adicionar_documento(doc, fonte="exemplo")
    
    # Faz uma busca de teste
    consulta = "O que é Python?"
    print(f"\nConsulta: {consulta}")
    
    resultados = rag.buscar_relevantes(consulta, top_k=2)
    
    print(f"\nDocumentos encontrados ({len(resultados)}):")
    for doc, score in resultados:
        print(f"  Score: {score:.3f} - {doc[:80]}...")
    
    print(f"\nTotal de documentos: {rag.get_total_documentos()}")
    print("\n=== Teste concluído ===")
