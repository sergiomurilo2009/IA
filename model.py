"""
Arquitetura Transformer Decoder-only implementada do zero em PyTorch
Modelo para geração de texto e diálogo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Adiciona informação posicional aos embeddings
    
    Matemática:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Onde:
    - pos: posição na sequência
    - i: índice da dimensão
    - d_model: dimensão do embedding
    
    Isso permite que o modelo saiba a ordem dos tokens,
    já que a atenção por si só é invariante à permutação.
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Cria matriz de positional encoding [max_seq_length, d_model]
        pe = torch.zeros(max_seq_length, d_model)
        
        # Calcula posições e frequências
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Aplica sin nas posições pares e cos nas ímpares
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Adiciona dimensão de batch
        pe = pe.unsqueeze(0)
        
        # Registra como buffer (não é atualizado no treino)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_length, d_model]
        """
        # Soma positional encoding ao embedding
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Mecanismo de Multi-Head Attention
    
    Matemática:
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    Multi-Head:
    - Divide Q, K, V em 'h' heads de dimensão d_k = d_model / h
    - Aplica attention independentemente em cada head
    - Concatena os resultados e aplica transformação linear
    
    Isso permite que o modelo preste atenção em diferentes
    aspectos da sequência simultaneamente.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Transformações lineares para Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # Transformação final
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)  # Fator de escala para evitar gradientes pequenos
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        query, key, value: [batch_size, seq_length, d_model]
        mask: [batch_size, 1, seq_length] ou [batch_size, seq_length, seq_length]
        """
        batch_size = query.size(0)
        
        # Aplica transformações lineares e divide em heads
        # Shape: [batch_size, num_heads, seq_length, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calcula atenção: softmax(QK^T / √d_k) V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Aplica máscara (para decoder: impede ver tokens futuros)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Aplica atenção aos values
        # Shape: [batch_size, num_heads, seq_length, d_k]
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatena heads e aplica transformação final
        # Shape: [batch_size, seq_length, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        
        return output


class FeedForwardNetwork(nn.Module):
    """
    Rede Feed-Forward aplicada a cada posição
    
    Arquitetura:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Ou seja: Linear -> ReLU -> Linear
    
    Expande para dimensão maior (ff_dim) e depois projeta
    de volta para d_model. Isso adiciona capacidade de
    representação não-linear.
    """
    
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_length, d_model]
        """
        # Linear -> ReLU -> Linear
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Um bloco do Transformer Decoder
    
    Estrutura:
    1. Multi-Head Attention (com máscara causal)
    2. Add & Norm
    3. Feed-Forward Network
    4. Add & Norm
    
    "Add" refere-se à conexão residual
    "Norm" refere-se à Layer Normalization
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, ff_dim, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: [batch_size, seq_length, d_model]
        mask: máscara causal para impedir ver tokens futuros
        """
        # Multi-Head Attention com conexão residual e layer norm
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-Forward com conexão residual e layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder completo (Decoder-only para geração de texto)
    
    Arquitetura:
    1. Token Embedding
    2. Positional Encoding
    3. N camadas de Transformer Blocks
    4. Layer Normalization final
    5. Projeção para vocabulário (Linear)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Embedding dos tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Pilha de blocos transformer
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Normalização final
        self.final_norm = nn.LayerNorm(d_model)
        
        # Projeção para vocabulário
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Inicialização de pesos (importante para convergência)
        self._init_weights()
    
    def _init_weights(self):
        """Inicializa pesos com Xavier uniforme"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """
        Cria máscara causal triangular inferior
        
        Impede que cada posição atenda a posições futuras,
        essencial para geração de texto autoregressiva.
        
        Retorna: [1, seq_length, seq_length] com 1s na parte inferior triangular
        """
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device)).unsqueeze(0)
        return mask
    
    def forward(
        self, 
        token_ids: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        token_ids: [batch_size, seq_length] - IDs dos tokens de entrada
        targets: [batch_size, seq_length] - IDs dos tokens alvo (para treino)
        
        Retorna:
        - logits: [batch_size, seq_length, vocab_size] - predições
        - loss: scalar (se targets fornecido)
        """
        batch_size, seq_length = token_ids.size()
        device = token_ids.device
        
        # Cria máscara causal
        causal_mask = self.create_causal_mask(seq_length, device)
        
        # Embedding + Positional Encoding
        x = self.token_embedding(token_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Passa por todas as camadas
        for layer in self.layers:
            x = layer(x, mask=causal_mask)
        
        # Normalização final
        x = self.final_norm(x)
        
        # Projeção para vocabulário
        logits = self.output_projection(x)
        
        # Calcula loss se targets fornecido
        loss = None
        if targets is not None:
            # Reshape para [batch_size * seq_length, vocab_size]
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            
            # Cross-Entropy Loss
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.pad_token_id)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Gera texto de forma autoregressiva
        
        Algoritmo:
        1. Para cada novo token:
           a. Passa sequência atual pelo modelo
           b. Pega logits do último token
           c. Aplica temperatura e sampling (top-k/top-p)
           d. Adiciona token gerado à sequência
        2. Repete até max_new_tokens ou EOS
        """
        self.eval()
        
        current_sequence = prompt_ids.clone()
        
        for _ in range(max_new_tokens):
            # Pega apenas últimos max_seq_length tokens
            current_sequence = current_sequence[:, -512:]
            
            # Forward pass
            logits, _ = self.forward(current_sequence)
            
            # Pega logits do último token: [batch_size, vocab_size]
            next_token_logits = logits[:, -1, :]
            
            # Aplica temperatura
            next_token_logits = next_token_logits / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens com probabilidade cumulativa > top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Adiciona à sequência
            current_sequence = torch.cat([current_sequence, next_token], dim=1)
            
            # Verifica EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        
        return current_sequence


def create_model(
    vocab_size: int,
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    ff_dim: int = 2048,
    max_seq_length: int = 512,
    dropout: float = 0.1,
    pad_token_id: int = 0
) -> TransformerDecoder:
    """
    Factory function para criar o modelo
    """
    model = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        max_seq_length=max_seq_length,
        dropout=dropout,
        pad_token_id=pad_token_id
    )
    
    # Conta parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🧠 Modelo criado:")
    print(f"   Parâmetros totais: {total_params:,}")
    print(f"   Parâmetros treináveis: {trainable_params:,}")
    print(f"   Config: {num_layers} camadas, {num_heads} heads, d_model={d_model}")
    
    return model


if __name__ == "__main__":
    # Teste rápido
    print("Testando arquitetura do modelo...")
    
    vocab_size = 1000
    batch_size = 2
    seq_length = 32
    
    model = create_model(vocab_size=vocab_size)
    
    # Cria input dummy
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    targets = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass
    logits, loss = model(input_ids, targets)
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Testa geração
    prompt = torch.randint(0, vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"\nGeração teste: {generated.shape}")
    
    print("\n✅ Arquitetura testada com sucesso!")
