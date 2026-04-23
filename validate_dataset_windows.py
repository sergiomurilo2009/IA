"""
Script de Validação e Preparação do Dataset para Windows
Analisa seu arquivo chat_pt.jsonl e prepara os dados para treinamento
"""

import json
import os
import sys
from pathlib import Path

# Adiciona o caminho atual ao sys.path para importar config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def get_windows_username():
    """Tenta obter o nome de usuário do Windows"""
    try:
        import getpass
        return getpass.getuser()
    except:
        return "SEU_USUARIO"

def find_dataset_paths():
    """Procura possíveis caminhos para o dataset no Windows"""
    username = get_windows_username()
    
    possible_paths = [
        rf"C:\Users\{username}\Downloads\chat_pt.jsonl",
        rf"C:\Users\{username}\chat_pt.jsonl",
        rf".\chat_pt.jsonl",
        rf".\data\chat_pt.jsonl",
        rf".\datasets\chat_pt.jsonl",
    ]
    
    found_paths = []
    for path in possible_paths:
        if os.path.exists(path):
            found_paths.append(path)
    
    return found_paths

def analyze_jsonl_file(filepath, max_lines=1000):
    """Analisa um arquivo JSONL e extrai informações"""
    print(f"\n🔍 Analisando arquivo: {filepath}")
    print("=" * 60)
    
    # Tamanho do arquivo
    file_size = os.path.getsize(filepath)
    file_size_mb = file_size / (1024 * 1024)
    file_size_gb = file_size / (1024 * 1024 * 1024)
    
    print(f"📊 Tamanho do arquivo: {file_size_mb:.2f} MB ({file_size_gb:.3f} GB)")
    
    # Contar linhas e analisar estrutura
    total_lines = 0
    valid_lines = 0
    sample_data = []
    all_keys = set()
    
    print("\n⏳ Lendo amostra do arquivo...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                
                total_lines += 1
                
                try:
                    data = json.loads(line.strip())
                    valid_lines += 1
                    
                    # Coleta todas as chaves encontradas
                    if isinstance(data, dict):
                        all_keys.update(data.keys())
                        
                        # Salva primeiras 5 amostras
                        if len(sample_data) < 5:
                            sample_data.append(data)
                    
                except json.JSONDecodeError:
                    print(f"⚠️  Linha {i+1} não é JSON válido")
                    continue
        
        print(f"✅ Linhas totais lidas: {total_lines}")
        print(f"✅ Linhas válidas (JSON): {valid_lines}")
        print(f"📋 Chaves encontradas: {sorted(all_keys)}")
        
        # Mostra amostras
        if sample_data:
            print("\n📝 Amostras dos dados:")
            print("-" * 60)
            for i, sample in enumerate(sample_data, 1):
                print(f"\nAmostra {i}:")
                if isinstance(sample, dict):
                    for key, value in list(sample.items())[:3]:  # Mostra até 3 campos
                        val_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                        print(f"  {key}: {val_str}")
                else:
                    print(f"  {sample[:200]}...")
        
        # Estima quantidade total de linhas baseado no tamanho
        if total_lines > 0:
            avg_line_size = file_size / total_lines
            estimated_total_lines = int(file_size / avg_line_size)
            print(f"\n📈 Estimativa total de linhas: ~{estimated_total_lines:,}")
            
            # Estima tokens (aproximadamente 4 caracteres por token em português)
            estimated_tokens = estimated_total_lines * (avg_line_size / 4)
            print(f"📈 Estimativa de tokens: ~{int(estimated_tokens):,}")
        
        return {
            'file_size': file_size,
            'total_lines': total_lines,
            'valid_lines': valid_lines,
            'keys': all_keys,
            'samples': sample_data
        }
    
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")
        return None

def suggest_text_columns(keys):
    """Sugere quais colunas usar para extrair texto"""
    priority_keys = ['text', 'content', 'prompt', 'response', 'message', 'conversation', 'input', 'output']
    
    suggested = []
    for key in priority_keys:
        if key in keys:
            suggested.append(key)
    
    # Adiciona outras chaves que parecem ser texto
    for key in keys:
        if key not in suggested and ('text' in key.lower() or 'message' in key.lower() or 'content' in key.lower()):
            suggested.append(key)
    
    return suggested

def create_config_template(found_path):
    """Cria um template de configuração com o caminho correto"""
    username = get_windows_username()
    
    config_content = f'''"""
Configurações do Projeto - ATUALIZADO PARA SEU SISTEMA
"""
import os

# =============================================================================
# CONFIGURAÇÃO DO DATASET
# =============================================================================
DATASET_PATH = r"{found_path}"  # Caminho encontrado automaticamente
DATASET_FORMAT = 'jsonl'

# Colunas sugeridas para extração de texto (ajuste conforme necessário)
# Execute validate_dataset.py primeiro para ver quais colunas estão disponíveis
JSONL_TEXT_COLUMNS = ['text', 'content', 'prompt', 'response', 'message', 'conversation']

# =============================================================================
# CONFIGURAÇÕES DO MODELO (Ajustadas para hardware doméstico)
# =============================================================================
VOCAB_SIZE = 8000
D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
FF_DIM = 2048
MAX_SEQ_LENGTH = 512
DROPOUT = 0.1

# =============================================================================
# CONFIGURAÇÕES DE TREINAMENTO
# =============================================================================
BATCH_SIZE = 16  # Ajuste: 8-32 dependendo da sua VRAM
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000

# Detecta automaticamente se tem GPU NVIDIA
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except:
    DEVICE = 'cpu'

CHECKPOINT_DIR = r".\\checkpoints"
LOG_DIR = r".\\logs"
SAVE_EVERY_N_STEPS = 500
LOG_EVERY_N_STEPS = 50

# =============================================================================
# CONFIGURAÇÕES DE GERAÇÃO
# =============================================================================
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.9
MAX_GENERATION_LENGTH = 256
START_TOKEN = "<start>"
END_TOKEN = "<end>"

def validate_config():
    """Valida configurações"""
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset não encontrado: {{DATASET_PATH}}")
        return False
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print("✅ Configurações validadas!")
    print(f"📁 Dataset: {{DATASET_PATH}} ({{os.path.getsize(DATASET_PATH) / (1024*1024):.2f}} MB)")
    print(f"🧠 Modelo: {{NUM_LAYERS}} camadas, {{NUM_HEADS}} heads")
    print(f"💻 Dispositivo: {{DEVICE}}")
    
    return True

if __name__ == "__main__":
    validate_config()
'''
    
    return config_content

def main():
    print("=" * 60)
    print("🔍 VALIDADOR DE DATASET PARA WINDOWS")
    print("=" * 60)
    
    # Procura arquivos dataset
    found_paths = find_dataset_paths()
    
    if not found_paths:
        print("\n❌ Arquivo chat_pt.jsonl não encontrado nos locais padrão!")
        print("\nLocais procurados:")
        username = get_windows_username()
        print(f"  - C:\\Users\\{username}\\Downloads\\chat_pt.jsonl")
        print(f"  - C:\\Users\\{username}\\chat_pt.jsonl")
        print(f"  - .\\chat_pt.jsonl")
        print(f"  - .\\data\\chat_pt.jsonl")
        print("\n💡 Soluções:")
        print("  1. Mova o arquivo para uma dessas pastas")
        print("  2. Ou edite config.py manualmente com o caminho completo")
        print("     Exemplo: DATASET_PATH = r'D:\\MeusDados\\chat_pt.jsonl'")
        return
    
    print(f"\n✅ Arquivo(s) encontrado(s): {len(found_paths)}")
    for path in found_paths:
        print(f"  📁 {path}")
    
    # Analisa o primeiro arquivo encontrado
    filepath = found_paths[0]
    result = analyze_jsonl_file(filepath)
    
    if result:
        # Sugere colunas
        suggested_cols = suggest_text_columns(result['keys'])
        print(f"\n💡 Colunas sugeridas para extração de texto: {suggested_cols}")
        
        # Oferece criar config atualizado
        print("\n" + "=" * 60)
        print("📝 GERAR CONFIGURAÇÃO AUTOMÁTICA?")
        print("=" * 60)
        
        config_path = os.path.join(os.path.dirname(filepath), 'config_atualizado.py')
        
        try:
            config_content = create_config_template(filepath)
            
            # Salva na pasta atual do projeto
            local_config = 'config_windows.py'
            with open(local_config, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            print(f"\n✅ Configuração gerada: {os.path.abspath(local_config)}")
            print("\n📋 Próximos passos:")
            print(f"  1. Copie o conteúdo de {local_config} para config.py")
            print("     OU renomeie este arquivo para config.py")
            print("  2. Execute: python tokenizer.py (para criar o vocabulário)")
            print("  3. Execute: python train.py (para iniciar treinamento)")
            
            # Mostra preview das colunas importantes
            if 'prompt' in result['keys'] and 'response' in result['keys']:
                print("\n✨ Formato ideal detectado! Seu dataset tem 'prompt' e 'response'")
                print("   Isso é perfeito para treinar um modelo conversacional.")
            
        except Exception as e:
            print(f"⚠️  Não foi possível gerar config: {e}")
            print("   Edite config.py manualmente")

if __name__ == "__main__":
    main()
