"""
Tokenizer Byte Pair Encoding (BPE) implementado do zero
Treina um tokenizer no seu dataset e salva o vocabulário
"""

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple
import pickle

# Tenta importar config, senão usa valores padrão
try:
    from config import VOCAB_SIZE, DATASET_PATH, DATASET_FORMAT, JSONL_TEXT_COLUMNS
except ImportError:
    VOCAB_SIZE = 8000
    DATASET_PATH = r"C:\Users\SEU_USUARIO\Downloads\chat_pt.jsonl"
    DATASET_FORMAT = 'jsonl'
    JSONL_TEXT_COLUMNS = ['text', 'content', 'prompt', 'response', 'message', 'conversation']


class BPETokenizer:
    """
    Tokenizador Byte Pair Encoding implementado do zero
    
    Funcionamento:
    1. Começa com vocabulário de caracteres individuais
    2. Iterativamente funde os pares de tokens mais frequentes
    3. Repete até atingir o tamanho desejado do vocabulário
    """
    
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}  # token -> id
        self.id_to_token: Dict[int, str] = {}  # id -> token
        self.merges: List[Tuple[str, str]] = []  # lista de merges aprendidos
        
        # Tokens especiais
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
    def _get_stats(self, corpus: List[str]) -> Dict[Tuple[str, str], int]:
        """
        Conta a frequência de pares de tokens adjacentes
        
        Matemática:
        Para cada palavra representada como sequência de tokens,
        conta quantas vezes cada par (token_i, token_{i+1}) aparece
        ponderado pela frequência da palavra
        """
        pairs = defaultdict(int)
        
        for word_freq in corpus:
            symbols = word_freq[0]
            freq = word_freq[1]
            
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        
        return pairs
    
    def _merge_pair(self, pair: Tuple[str, str], corpus: List) -> List:
        """
        Funde todas as ocorrências de um par de tokens no corpus
        
        Exemplo: Se pair = ('h', 'e') e palavra = ['h', 'e', 'l', 'l', 'o']
        Resultado: ['he', 'l', 'l', 'o']
        """
        new_corpus = []
        bigram = pair[0] + pair[1]
        
        for word_freq in corpus:
            symbols = word_freq[0]
            freq = word_freq[1]
            
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i+1] == pair[1]:
                    new_symbols.append(bigram)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            
            new_corpus.append((new_symbols, freq))
        
        return new_corpus
    
    def _word_to_freq(self, text: str) -> Dict[List[str], int]:
        """
        Converte texto em lista de símbolos com frequências
        
        Exemplo: "hello hello world" -> 
        [(['h','e','l','l','o'], 2), (['w','o','r','l','d'], 1)]
        """
        words = text.split()
        word_freq = defaultdict(int)
        
        for word in words:
            word_freq[tuple(word)] += 1
        
        return [(list(word), freq) for word, freq in word_freq.items()]
    
    def train(self, text: str, verbose: bool = True):
        """
        Treina o tokenizer BPE no texto fornecido
        
        Algoritmo:
        1. Inicializa vocabulário com caracteres únicos + tokens especiais
        2. Enquanto |vocab| < vocab_size:
           a. Conta frequência de todos os pares adjacentes
           b. Seleciona o par mais frequente
           c. Funde esse par em todo o corpus
           d. Adiciona novo token ao vocabulário
        """
        print("🔄 Iniciando treinamento do tokenizer BPE...")
        
        # Step 1: Converter texto em formato de frequência
        word_freq = self._word_to_freq(text)
        
        if verbose:
            print(f"   📊 Palavras únicas: {len(word_freq)}")
            total_chars = sum(len(w[0]) * w[1] for w in word_freq)
            print(f"   📊 Caracteres totais: {total_chars:,}")
        
        # Step 2: Inicializar vocabulário com caracteres únicos
        char_vocab = set()
        for word_freq_item in word_freq:
            for char in word_freq_item[0]:
                char_vocab.add(char)
        
        # Adiciona tokens especiais primeiro (IDs 0-3)
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.id_to_token[i] = token
        
        # Adiciona caracteres ao vocabulário
        next_id = len(self.special_tokens)
        for char in sorted(char_vocab):
            if char not in self.vocab:
                self.vocab[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1
        
        if verbose:
            print(f"   ✓ Vocabulário inicial: {len(self.vocab)} tokens (caracteres + especiais)")
        
        # Step 3: Iterativamente fundir pares
        num_merges = self.vocab_size - len(self.vocab)
        
        for i in range(num_merges):
            # Conta pares
            pairs = self._get_stats(word_freq)
            
            if not pairs:
                break
            
            # Seleciona par mais frequente
            best_pair = max(pairs, key=pairs.get)
            
            # Faz o merge
            word_freq = self._merge_pair(best_pair, word_freq)
            
            # Adiciona novo token
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.vocab:
                self.vocab[new_token] = next_id
                self.id_to_token[next_id] = new_token
                self.merges.append(best_pair)
                next_id += 1
            
            if verbose and (i + 1) % 500 == 0:
                print(f"   Progresso: {i+1}/{num_merges} merges, vocab={len(self.vocab)}")
        
        if verbose:
            print(f"   ✓ Treinamento completo! Vocabulário final: {len(self.vocab)} tokens")
            print(f"   ✓ Merges aprendidos: {len(self.merges)}")
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokeniza um texto usando o vocabulário treinado
        
        Processo:
        1. Divide em palavras
        2. Para cada palavra, aplica merges na ordem aprendida
        3. Converte tokens finais para IDs
        4. Tokens desconhecidos viram <unk>
        """
        tokens = []
        words = text.split()
        
        for word in words:
            # Converte palavra em lista de caracteres
            word_tokens = list(word)
            
            # Aplica todos os merges na ordem
            for merge in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == merge[0] and word_tokens[i+1] == merge[1]:
                        word_tokens[i:i+2] = [merge[0] + merge[1]]
                    else:
                        i += 1
            
            # Converte para IDs
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.vocab[self.unk_token])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Converte lista de IDs de volta para texto
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append(self.unk_token)
        
        return ''.join(tokens)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Codifica texto adicionando tokens especiais
        Formato: <bos> + tokens + <eos>
        """
        token_ids = self.tokenize(text)
        
        if add_special_tokens:
            token_ids = [self.vocab[self.bos_token]] + token_ids + [self.vocab[self.eos_token]]
        
        return token_ids
    
    def save(self, filepath: str):
        """Salva o tokenizer em arquivo"""
        data = {
            'vocab': self.vocab,
            'id_to_token': self.id_to_token,
            'merges': self.merges,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Tokenizer salvo em: {filepath}")
    
    def load(self, filepath: str):
        """Carrega tokenizer de arquivo"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.vocab = data['vocab']
        self.id_to_token = data['id_to_token']
        self.merges = data['merges']
        self.vocab_size = data['vocab_size']
        self.special_tokens = data.get('special_tokens', self.special_tokens)
        
        # Recria tokens especiais
        self.pad_token = self.special_tokens[0]
        self.unk_token = self.special_tokens[1]
        self.bos_token = self.special_tokens[2]
        self.eos_token = self.special_tokens[3]
        
        print(f"✓ Tokenizer carregado de: {filepath}")
        print(f"  Vocabulário: {len(self.vocab)} tokens")


def load_dataset(filepath: str, format: str = 'jsonl') -> str:
    """
    Carrega dataset e extrai texto como string única
    """
    print(f"📖 Carregando dataset: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset não encontrado: {filepath}")
    
    all_text = []
    
    if format == 'jsonl':
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    
                    # Tenta extrair texto das colunas configuradas
                    text_found = False
                    for col in JSONL_TEXT_COLUMNS:
                        if col in data:
                            value = data[col]
                            if isinstance(value, str):
                                all_text.append(value)
                                text_found = True
                            elif isinstance(value, list):
                                # Pode ser uma lista de mensagens
                                for item in value:
                                    if isinstance(item, str):
                                        all_text.append(item)
                                    elif isinstance(item, dict) and 'text' in item:
                                        all_text.append(item['text'])
                                text_found = True
                            break
                    
                    # Se nenhuma coluna configurada encontrada, tenta qualquer campo de texto
                    if not text_found and isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, str) and len(value) > 10:
                                all_text.append(value)
                                break
                
                except json.JSONDecodeError:
                    continue
                
                if (i + 1) % 10000 == 0:
                    print(f"   Processadas {i+1} linhas...")
    
    elif format == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            all_text = f.readlines()
    
    else:
        raise ValueError(f"Formato não suportado: {format}")
    
    full_text = '\n'.join(all_text)
    print(f"✓ Dataset carregado: {len(full_text):,} caracteres, {len(all_text)} amostras")
    
    return full_text


def main():
    print("=" * 60)
    print("🔤 TREINAMENTO DE TOKENIZER BPE")
    print("=" * 60)
    
    # Carrega dataset
    try:
        text = load_dataset(DATASET_PATH, DATASET_FORMAT)
    except Exception as e:
        print(f"\n❌ Erro ao carregar dataset: {e}")
        print("\n💡 Solução:")
        print("  1. Verifique se o caminho em config.py está correto")
        print("  2. Execute: python validate_dataset_windows.py primeiro")
        return
    
    # Limita texto se for muito grande (para treinamento rápido do tokenizer)
    max_chars = 50_000_000  # 50M caracteres
    if len(text) > max_chars:
        print(f"\n⚠️  Texto muito grande ({len(text)/1e6:.1f}M chars). Usando amostra de {max_chars/1e6:.0f}M chars")
        text = text[:max_chars]
    
    # Treina tokenizer
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.train(text, verbose=True)
    
    # Testa tokenização
    print("\n🧪 Testando tokenização:")
    test_texts = [
        "Olá, como você está?",
        "Machine learning é fascinante!",
        "A IA vai transformar o mundo."
    ]
    
    for test_text in test_texts:
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens[1:-1])  # Remove BOS e EOS
        print(f"  Original: {test_text}")
        print(f"  Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"  Decodificado: {decoded}")
        print()
    
    # Salva tokenizer
    output_path = "tokenizer.pkl"
    tokenizer.save(output_path)
    
    # Salva também como JSON para inspeção
    vocab_json = "vocab.json"
    with open(vocab_json, 'w', encoding='utf-8') as f:
        json.dump(tokenizer.vocab, f, ensure_ascii=False, indent=2)
    print(f"✓ Vocabulário salvo em: {vocab_json}")
    
    print("\n" + "=" * 60)
    print("✅ TOKENIZER PRONTO!")
    print("=" * 60)
    print(f"\nPróximos passos:")
    print(f"  1. Execute: python train.py (para treinar o modelo)")
    print(f"\nArquivos gerados:")
    print(f"  - {output_path} (tokenizer completo)")
    print(f"  - {vocab_json} (vocabulário para inspeção)")


if __name__ == "__main__":
    main()
