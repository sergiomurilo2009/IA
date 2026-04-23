"""
Configurações do Projeto - Ajuste aqui o caminho do seu dataset
"""
import os

# =============================================================================
# CONFIGURAÇÃO DO DATASET
# =============================================================================
# Caminho para o seu arquivo chat_pt.jsonl
# Exemplos:
#   - Se estiver na pasta Downloads: r"C:\Users\SEU_USUARIO\Downloads\chat_pt.jsonl"
#   - Se estiver numa pasta do projeto: r".\data\chat_pt.jsonl"
DATASET_PATH = r"C:\Users\sergi\Downloads\chat_pt.jsonl"  # <--- CAMINHO ATUALIZADO

# Formato do dataset: 'jsonl' ou 'txt'
DATASET_FORMAT = 'jsonl'

# Colunas do JSONL que contêm o texto (para jsonl)
# O script vai tentar extrair texto dessas colunas na ordem
JSONL_TEXT_COLUMNS = ['text', 'content', 'prompt', 'response', 'message', 'conversation']

# =============================================================================
# CONFIGURAÇÕES DO MODELO (Ajustadas para hardware doméstico)
# =============================================================================
# Tamanho do vocabulário do tokenizer
VOCAB_SIZE = 8000

# Dimensão do embedding
D_MODEL = 512

# Número de camadas do transformer
NUM_LAYERS = 6

# Número de heads de atenção
NUM_HEADS = 8

# Dimensão do feed-forward
FF_DIM = 2048

# Tamanho máximo do contexto (sequência)
MAX_SEQ_LENGTH = 512

# Dropout rate
DROPOUT = 0.1

# =============================================================================
# CONFIGURAÇÕES DE TREINAMENTO
# =============================================================================
# Batch size (ajuste conforme sua VRAM)
#   - 8GB VRAM: use 16-32
#   - 6GB VRAM: use 8-16
#   - 4GB VRAM: use 4-8
BATCH_SIZE = 16

# Número de épocas
NUM_EPOCHS = 10

# Learning rate
LEARNING_RATE = 1e-4

# Warmup steps (parte do treinamento onde LR aumenta gradualmente)
WARMUP_STEPS = 1000

# Dispositivo: 'cuda' (GPU NVIDIA), 'cpu' ou 'mps' (Mac)
DEVICE = 'cuda' if os.name != 'nt' else 'cuda'  # Tenta usar CUDA no Windows

# Pasta para salvar checkpoints
CHECKPOINT_DIR = r".\checkpoints"

# Pasta para salvar logs
LOG_DIR = r".\logs"

# Salvar checkpoint a cada N passos
SAVE_EVERY_N_STEPS = 500

# Logar métricas a cada N passos
LOG_EVERY_N_STEPS = 50

# =============================================================================
# CONFIGURAÇÕES DE GERAÇÃO (INFERENCE)
# =============================================================================
# Temperatura para amostragem (maior = mais criativo, menor = mais determinístico)
TEMPERATURE = 0.7

# Top-k sampling (considera apenas os k tokens mais prováveis)
TOP_K = 50

# Top-p sampling (nucleus sampling)
TOP_P = 0.9

# Máximo de tokens a gerar
MAX_GENERATION_LENGTH = 256

# Token de início de geração
START_TOKEN = "<start>"

# Token de fim de geração
END_TOKEN = "<end>"

# =============================================================================
# VALIDAÇÃO E PREPARAÇÃO
# =============================================================================
def validate_config():
    """Valida se as configurações estão corretas"""
    import os
    
    # Verifica se o dataset existe
    if not os.path.exists(DATASET_PATH):
        print(f"⚠️  AVISO: Dataset não encontrado em: {DATASET_PATH}")
        print("   Por favor, edite config.py e ajuste o caminho DATASET_PATH")
        print("   Exemplo: DATASET_PATH = r'C:\\Users\\SeuUsuario\\Downloads\\chat_pt.jsonl'")
        return False
    
    # Verifica se o formato é válido
    if DATASET_FORMAT not in ['jsonl', 'txt']:
        print(f"⚠️  AVISO: Formato inválido '{DATASET_FORMAT}'. Use 'jsonl' ou 'txt'")
        return False
    
    # Cria diretórios se não existirem
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print("✅ Configurações validadas com sucesso!")
    print(f"📁 Dataset: {DATASET_PATH}")
    print(f"📊 Formato: {DATASET_FORMAT}")
    print(f"🧠 Modelo: {NUM_LAYERS} camadas, {NUM_HEADS} heads, dim={D_MODEL}")
    print(f"📦 Batch Size: {BATCH_SIZE}, Seq Length: {MAX_SEQ_LENGTH}")
    print(f"💾 Checkpoints: {CHECKPOINT_DIR}")
    
    return True

if __name__ == "__main__":
    validate_config()
