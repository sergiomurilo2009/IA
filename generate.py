"""
Script de Geração de Texto (Inference)
Carrega o modelo treinado e gera texto interativamente
"""

import os
import sys
import json
import torch
from typing import Optional, List

# Imports locais
from model import TransformerDecoder
from tokenizer import BPETokenizer

# Configurações
try:
    from config import (
        TEMPERATURE, TOP_K, TOP_P, MAX_GENERATION_LENGTH,
        START_TOKEN, END_TOKEN, CHECKPOINT_DIR
    )
except ImportError:
    TEMPERATURE = 0.7
    TOP_K = 50
    TOP_P = 0.9
    MAX_GENERATION_LENGTH = 256
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    CHECKPOINT_DIR = r".\checkpoints"


class ChatBot:
    """
    Interface conversacional para o modelo treinado
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        config_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        
        # Carrega tokenizer
        print("🔤 Carregando tokenizer...")
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(tokenizer_path)
        
        # Carrega config do modelo
        print("📋 Carregando configuração...")
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)
        else:
            # Tenta encontrar config no mesmo diretório
            config_dir = os.path.dirname(model_path)
            default_config = os.path.join(config_dir, "model_config.json")
            if os.path.exists(default_config):
                with open(default_config, 'r') as f:
                    self.model_config = json.load(f)
            else:
                print("⚠️  Config não encontrada, usando padrões")
                self.model_config = {
                    'vocab_size': len(self.tokenizer.vocab),
                    'd_model': 512,
                    'num_layers': 6,
                    'num_heads': 8,
                    'ff_dim': 2048,
                    'max_seq_length': 512,
                    'dropout': 0.1
                }
        
        # Cria modelo
        print("🧠 Criando modelo...")
        self.model = TransformerDecoder(
            vocab_size=self.model_config['vocab_size'],
            d_model=self.model_config['d_model'],
            num_layers=self.model_config['num_layers'],
            num_heads=self.model_config['num_heads'],
            ff_dim=self.model_config['ff_dim'],
            max_seq_length=self.model_config['max_seq_length'],
            dropout=self.model_config['dropout'],
            pad_token_id=self.tokenizer.vocab.get(self.tokenizer.pad_token, 0)
        )
        
        # Carrega pesos treinados
        print(f"💾 Carregando pesos: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval()
        
        print(f"✅ Modelo carregado no dispositivo: {device}")
        
        # IDs especiais
        self.bos_id = self.tokenizer.vocab.get(self.tokenizer.bos_token, 0)
        self.eos_id = self.tokenizer.vocab.get(self.tokenizer.eos_token, 0)
        self.pad_id = self.tokenizer.vocab.get(self.tokenizer.pad_token, 0)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """
        Gera texto a partir de um prompt
        """
        # Tokeniza prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Gera
        with torch.no_grad():
            output_ids = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=self.eos_id
            )
        
        # Decodifica
        generated_ids = output_ids[0].tolist()
        
        # Remove tokens especiais do início e fim
        if generated_ids[0] == self.bos_id:
            generated_ids = generated_ids[1:]
        
        # Trunca no EOS se existir
        try:
            eos_index = generated_ids.index(self.eos_id)
            generated_ids = generated_ids[:eos_index]
        except ValueError:
            pass
        
        generated_text = self.tokenizer.decode(generated_ids)
        
        return generated_text
    
    def chat(self, system_prompt: str = ""):
        """
        Inicia sessão de chat interativo
        """
        print("\n" + "=" * 60)
        print("🤖 CHATBOT PRONTO!")
        print("=" * 60)
        print("Digite sua mensagem e pressione Enter")
        print("Comandos:")
        print("  /quit - Sair")
        print("  /clear - Limpar histórico")
        print("  /temp <valor> - Ajustar temperatura (0.1-2.0)")
        print("  /max <valor> - Ajustar máximo de tokens")
        print("=" * 60 + "\n")
        
        history = []
        current_temp = TEMPERATURE
        current_max = MAX_GENERATION_LENGTH
        
        if system_prompt:
            history.append(f"Sistema: {system_prompt}")
        
        while True:
            try:
                user_input = input("👤 Você: ").strip()
                
                if not user_input:
                    continue
                
                # Comandos
                if user_input.lower() == '/quit':
                    print("\n👋 Até logo!")
                    break
                
                if user_input.lower() == '/clear':
                    history = []
                    print("✓ Histórico limpo\n")
                    continue
                
                if user_input.lower().startswith('/temp'):
                    parts = user_input.split()
                    if len(parts) > 1:
                        try:
                            current_temp = float(parts[1])
                            print(f"✓ Temperatura ajustada para {current_temp}\n")
                        except:
                            print("⚠️  Use: /temp <valor> (ex: /temp 0.8)\n")
                    continue
                
                if user_input.lower().startswith('/max'):
                    parts = user_input.split()
                    if len(parts) > 1:
                        try:
                            current_max = int(parts[1])
                            print(f"✓ Máximo de tokens ajustado para {current_max}\n")
                        except:
                            print("⚠️  Use: /max <valor> (ex: /max 200)\n")
                    continue
                
                # Prepara prompt com contexto
                context = "\n".join(history[-5:])  # Últimas 5 mensagens
                full_prompt = f"{context}\nVocê: {user_input}\nIA:" if context else f"Você: {user_input}\nIA:"
                
                # Gera resposta
                print("🤖 IA: ", end="", flush=True)
                response = self.generate(
                    full_prompt,
                    max_new_tokens=current_max,
                    temperature=current_temp
                )
                print(response + "\n")
                
                # Adiciona ao histórico
                history.append(f"Você: {user_input}")
                history.append(f"IA: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Até logo!")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}\n")


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Encontra o checkpoint mais recente"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Tenta encontrar model_final.pt primeiro
    final_model = os.path.join(checkpoint_dir, "model_final.pt")
    if os.path.exists(final_model):
        return final_model
    
    # Procura checkpoints
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
    
    if not checkpoints:
        return None
    
    # Ordena por nome (assumindo que step maior vem depois)
    checkpoints.sort(reverse=True)
    
    return os.path.join(checkpoint_dir, checkpoints[0])


def main():
    print("=" * 60)
    print("🚀 CARREGANDO MODELO PARA GERAÇÃO")
    print("=" * 60)
    
    # Encontra modelo
    model_path = find_latest_checkpoint(CHECKPOINT_DIR)
    
    if not model_path:
        print(f"\n❌ Nenhum modelo encontrado em {CHECKPOINT_DIR}")
        print("\nPré-requisitos:")
        print("  1. Execute: python tokenizer.py")
        print("  2. Execute: python train.py")
        return
    
    print(f"✓ Modelo encontrado: {model_path}")
    
    # Encontra tokenizer
    tokenizer_path = "tokenizer.pkl"
    if not os.path.exists(tokenizer_path):
        print(f"\n❌ Tokenizer não encontrado: {tokenizer_path}")
        print("   Execute: python tokenizer.py")
        return
    
    # Encontra config
    config_path = os.path.join(CHECKPOINT_DIR, "model_config.json")
    if not os.path.exists(config_path):
        config_path = None
    
    # Detecta dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Usando CPU (será mais lento)")
    
    # Cria chatbot
    chatbot = ChatBot(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        config_path=config_path,
        device=device
    )
    
    # Inicia chat
    system_prompt = "Você é um assistente prestativo que responde em português."
    chatbot.chat(system_prompt)


if __name__ == "__main__":
    main()
