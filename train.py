"""
Script de Treinamento do Modelo Transformer
Treina o modelo no seu dataset com suporte a checkpoints e logging
"""

import os
import sys
import json
import time
import math
from datetime import datetime
from typing import Optional, Dict, List
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Imports locais
from model import create_model, TransformerDecoder
from tokenizer import BPETokenizer

# Configurações
try:
    from config import (
        DATASET_PATH, DATASET_FORMAT, JSONL_TEXT_COLUMNS,
        VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS, FF_DIM,
        MAX_SEQ_LENGTH, DROPOUT,
        BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WARMUP_STEPS,
        DEVICE, CHECKPOINT_DIR, LOG_DIR,
        SAVE_EVERY_N_STEPS, LOG_EVERY_N_STEPS
    )
except ImportError:
    # Valores padrão se config não existir
    DATASET_PATH = r"C:\Users\SEU_USUARIO\Downloads\chat_pt.jsonl"
    DATASET_FORMAT = 'jsonl'
    JSONL_TEXT_COLUMNS = ['text', 'content', 'prompt', 'response']
    VOCAB_SIZE = 8000
    D_MODEL = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    FF_DIM = 2048
    MAX_SEQ_LENGTH = 512
    DROPOUT = 0.1
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WARMUP_STEPS = 1000
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CHECKPOINT_DIR = r".\checkpoints"
    LOG_DIR = r".\logs"
    SAVE_EVERY_N_STEPS = 500
    LOG_EVERY_N_STEPS = 50


class TextDataset(Dataset):
    """
    Dataset para carregar texto e criar sequências de treinamento
    
    Estratégia:
    - Carrega todo o texto como uma string longa
    - Tokeniza tudo previamente
    - Cria janelas deslizantes de tamanho MAX_SEQ_LENGTH
    """
    
    def __init__(self, text: str, tokenizer: BPETokenizer, seq_length: int):
        self.seq_length = seq_length
        
        # Tokeniza todo o texto
        print("   Tokenizando dataset...")
        all_tokens = tokenizer.tokenize(text)
        
        # Adiciona tokens especiais entre amostras (opcional)
        bos_id = tokenizer.vocab.get(tokenizer.bos_token, 0)
        eos_id = tokenizer.vocab.get(tokenizer.eos_token, 0)
        
        # Prepara sequências
        self.sequences = []
        
        # Cria janelas deslizantes
        for i in range(0, len(all_tokens) - seq_length, seq_length // 2):
            seq = all_tokens[i:i + seq_length + 1]  # +1 para target shift
            if len(seq) == seq_length + 1:
                self.sequences.append(seq)
        
        print(f"   ✓ {len(self.sequences):,} sequências criadas")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        return input_ids, target_ids


def load_dataset_text(filepath: str, format: str = 'jsonl') -> str:
    """Carrega dataset como texto único"""
    import json
    
    print(f"📖 Carregando dataset: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset não encontrado: {filepath}")
    
    all_text = []
    
    # Chaves possíveis para conteúdo de mensagem
    content_keys = ['content', 'text', 'value', 'body']
    
    if format == 'jsonl':
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    
                    text_found = False
                    
                    # Caso 1: Chaves diretas na raiz (ex: {"text": "..."})
                    for col in JSONL_TEXT_COLUMNS:
                        if col in data:
                            value = data[col]
                            if isinstance(value, str):
                                all_text.append(value)
                                text_found = True
                                break
                    
                    # Caso 2: Estrutura com 'messages' (ex: {"messages": [{"role": "...", "content": "..."}]})
                    if not text_found and 'messages' in data and isinstance(data['messages'], list):
                        conversation_parts = []
                        for msg in data['messages']:
                            if isinstance(msg, dict):
                                # Tenta encontrar conteúdo em chaves comuns
                                for key in content_keys:
                                    if key in msg and isinstance(msg[key], str):
                                        conversation_parts.append(msg[key])
                                        text_found = True
                                        break
                                # Fallback: se não achou chaves padrão, pega qualquer string
                                if not text_found:
                                    for v in msg.values():
                                        if isinstance(v, str) and len(v) > 5:
                                            conversation_parts.append(v)
                                            text_found = True
                                            break
                        
                        if conversation_parts:
                            all_text.append(" ".join(conversation_parts))
                    
                    # Caso 3: Fallback genérico - tenta achar qualquer string longa no dict
                    if not text_found and isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, str) and len(value) > 10:
                                all_text.append(value)
                                text_found = True
                                break
                            # Também verifica listas dentro do dict
                            elif isinstance(value, list):
                                for item in value:
                                    if isinstance(item, str) and len(item) > 10:
                                        all_text.append(item)
                                        text_found = True
                                        break
                                if text_found:
                                    break
                
                except json.JSONDecodeError:
                    continue
                
                if (i + 1) % 50000 == 0:
                    print(f"   Processadas {i+1:,} linhas...")
    
    elif format == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            all_text = f.readlines()
    
    full_text = '\n'.join(all_text)
    print(f"✓ Dataset carregado: {len(full_text):,} caracteres, {len(all_text):,} exemplos")
    
    return full_text


class Trainer:
    """
    Classe responsável pelo treinamento do modelo
    """
    
    def __init__(
        self,
        model: TransformerDecoder,
        tokenizer: BPETokenizer,
        device: str,
        learning_rate: float,
        warmup_steps: int,
        checkpoint_dir: str,
        log_dir: str
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.global_step = 0
        
        # Optimizer AdamW
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Scheduler será configurado após saber o total de steps
        self.scheduler = None
        self.warmup_steps = warmup_steps
        
        # Diretórios
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Logging
        self.training_history = {
            'loss': [],
            'learning_rate': [],
            'steps': []
        }
    
    def get_lr_schedule(self, step: int, total_steps: int) -> float:
        """
        Learning rate schedule com warmup e cosine decay
        
        Warmup: LR aumenta linearmente de 0 até lr_max em warmup_steps
        Decay: LR diminui seguindo cosseno até 0 no final
        """
        if step < self.warmup_steps:
            # Warmup linear
            return self.optimizer.defaults['lr'] * (step + 1) / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (total_steps - self.warmup_steps)
            return self.optimizer.defaults['lr'] * 0.5 * (1 + math.cos(math.pi * progress))
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        total_epochs: int,
        log_every: int,
        save_every: int
    ):
        """Treina uma época completa"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            # Move para GPU
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, loss = self.model(input_ids, target_ids)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (evita exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update
            self.optimizer.step()
            
            # Atualiza learning rate
            total_steps = len(dataloader) * total_epochs
            new_lr = self.get_lr_schedule(self.global_step, total_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            # Acumula métricas
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if self.global_step % log_every == 0:
                avg_loss = total_loss / num_batches
                elapsed = time.time() - start_time
                steps_per_sec = self.global_step / max(elapsed, 0.001)
                
                log_msg = (
                    f"Epoch {epoch+1}/{total_epochs} | "
                    f"Step {self.global_step:,} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {new_lr:.2e} | "
                    f"Speed: {steps_per_sec:.1f} steps/s"
                )
                print(log_msg)
                
                # Salva no histórico
                self.training_history['loss'].append(avg_loss)
                self.training_history['learning_rate'].append(new_lr)
                self.training_history['steps'].append(self.global_step)
            
            # Save checkpoint
            if self.global_step % save_every == 0:
                self.save_checkpoint(epoch, batch_idx, avg_loss)
        
        # Final da época
        avg_loss = total_loss / max(num_batches, 1)
        print(f"\n✓ Epoch {epoch+1}/{total_epochs} completada | Loss médio: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, batch: int, loss: float):
        """Salva checkpoint do treinamento"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': epoch,
            'batch': batch,
            'loss': loss,
            'training_history': self.training_history
        }
        
        filepath = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_step_{self.global_step}.pt"
        )
        
        torch.save(checkpoint, filepath)
        print(f"   💾 Checkpoint salvo: {filepath}")
        
        # Salva também o último checkpoint separadamente
        last_filepath = os.path.join(self.checkpoint_dir, "checkpoint_last.pt")
        torch.save(checkpoint, last_filepath)
    
    def save_training_logs(self):
        """Salva logs de treinamento em JSON"""
        log_file = os.path.join(
            self.log_dir,
            f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(log_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"📊 Logs salvos em: {log_file}")
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        log_every: int = 50,
        save_every: int = 500
    ):
        """Loop principal de treinamento"""
        print("\n" + "=" * 60)
        print("🚀 INICIANDO TREINAMENTO")
        print("=" * 60)
        print(f"Dispositivo: {self.device}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Sequências por época: {len(train_loader):,}")
        print(f"Total de steps: {len(train_loader) * num_epochs:,}")
        print("=" * 60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            avg_loss = self.train_epoch(
                train_loader,
                epoch,
                num_epochs,
                log_every,
                save_every
            )
            
            epoch_time = time.time() - epoch_start
            print(f"Tempo da época: {epoch_time/60:.1f} minutos\n")
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ TREINAMENTO COMPLETO!")
        print(f"Tempo total: {total_time/3600:.2f} horas")
        print(f"{'='*60}")
        
        # Salva logs finais
        self.save_training_logs()
        
        # Salva modelo final
        self.save_final_model()
    
    def save_final_model(self):
        """Salva o modelo final treinado"""
        # Salva estado do modelo
        model_path = os.path.join(self.checkpoint_dir, "model_final.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"🎯 Modelo final salvo: {model_path}")
        
        # Salva config do modelo
        config = {
            'vocab_size': VOCAB_SIZE,
            'd_model': D_MODEL,
            'num_layers': NUM_LAYERS,
            'num_heads': NUM_HEADS,
            'ff_dim': FF_DIM,
            'max_seq_length': MAX_SEQ_LENGTH,
            'dropout': DROPOUT
        }
        
        config_path = os.path.join(self.checkpoint_dir, "model_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"📋 Config salva: {config_path}")


def main():
    print("=" * 60)
    print("🤖 TREINAMENTO DE MODELO TRANSFORMER")
    print("=" * 60)
    
    # Verifica se tem GPU
    if DEVICE == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA não disponível! Usando CPU...")
        print("   O treinamento será lento. Considere reduzir batch_size.")
        device = 'cpu'
    else:
        device = DEVICE
    
    if device == 'cuda':
        print(f"🎮 GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"   Memória: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Carrega tokenizer
    tokenizer_path = "tokenizer.pkl"
    if not os.path.exists(tokenizer_path):
        print(f"\n❌ Tokenizer não encontrado: {tokenizer_path}")
        print("   Execute primeiro: python tokenizer.py")
        return
    
    print(f"\n🔤 Carregando tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    
    # Carrega dataset
    print(f"\n📖 Carregando dataset...")
    try:
        text = load_dataset_text(DATASET_PATH, DATASET_FORMAT)
    except Exception as e:
        print(f"\n❌ Erro ao carregar dataset: {e}")
        print("   Verifique se o caminho em config.py está correto")
        return
    
    # Limita dataset se for muito grande (para caber na memória)
    max_chars = 100_000_000  # 100M caracteres
    if len(text) > max_chars:
        print(f"\n⚠️  Dataset muito grande ({len(text)/1e6:.1f}M chars). Usando {max_chars/1e6:.0f}M chars")
        text = text[:max_chars]
    
    # Cria dataset
    print(f"\n📦 Criando dataset...")
    dataset = TextDataset(text, tokenizer, MAX_SEQ_LENGTH)
    
    # Cria dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows tem issues com multiprocessing
        pin_memory=True if device == 'cuda' else False
    )
    
    # Cria modelo
    print(f"\n🧠 Criando modelo...")
    model = create_model(
        vocab_size=len(tokenizer.vocab),
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout=DROPOUT,
        pad_token_id=tokenizer.vocab.get(tokenizer.pad_token, 0)
    )
    
    # Cria trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        checkpoint_dir=CHECKPOINT_DIR,
        log_dir=LOG_DIR
    )
    
    # Inicia treinamento
    trainer.train(
        train_loader=train_loader,
        num_epochs=NUM_EPOCHS,
        log_every=LOG_EVERY_N_STEPS,
        save_every=SAVE_EVERY_N_STEPS
    )
    
    print("\n" + "=" * 60)
    print("🎉 TUDO PRONTO!")
    print("=" * 60)
    print("\nPróximos passos:")
    print("  1. Execute: python generate.py (para testar o modelo)")
    print("\nArquivos gerados:")
    print(f"  - {CHECKPOINT_DIR}/model_final.pt (modelo treinado)")
    print(f"  - {CHECKPOINT_DIR}/checkpoint_*.pt (checkpoints)")
    print(f"  - {LOG_DIR}/training_log_*.json (logs)")


if __name__ == "__main__":
    main()
