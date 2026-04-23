"""
Script para diagnosticar problemas comuns no dataset e configuração
Execute este script antes de treinar para identificar problemas
"""

import os
import json
import sys

def main():
    print("=" * 60)
    print("🔍 DIAGNÓSTICO DO SISTEMA")
    print("=" * 60)
    
    # 1. Verificar config.py
    print("\n1️⃣  Verificando config.py...")
    try:
        from config import DATASET_PATH, DATASET_FORMAT, JSONL_TEXT_COLUMNS
        print(f"   ✓ config.py encontrado")
        print(f"   📁 Dataset path: {DATASET_PATH}")
        print(f"   📊 Formato: {DATASET_FORMAT}")
        print(f"   🔑 Colunas: {JSONL_TEXT_COLUMNS}")
    except ImportError as e:
        print(f"   ❌ Erro ao importar config.py: {e}")
        return
    
    # 2. Verificar se dataset existe
    print(f"\n2️⃣  Verificando dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"   ❌ Dataset NÃO encontrado em: {DATASET_PATH}")
        print(f"\n💡 SOLUÇÃO:")
        print(f"   - Edite config.py e ajuste DATASET_PATH para o caminho correto")
        print(f"   - Exemplo: DATASET_PATH = r'C:\\\\Users\\\\sergi\\\\Downloads\\\\chat_pt.jsonl'")
        print(f"   - Ou copie seu arquivo para: {DATASET_PATH}")
        return
    else:
        print(f"   ✓ Dataset encontrado!")
        file_size = os.path.getsize(DATASET_PATH) / (1024 * 1024)
        print(f"   📦 Tamanho: {file_size:.2f} MB")
    
    # 3. Analisar estrutura do dataset
    print(f"\n3️⃣  Analisando estrutura do dataset...")
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            sample_lines = []
            for i, line in enumerate(f):
                if i >= 10:
                    break
                try:
                    data = json.loads(line.strip())
                    sample_lines.append(data)
                except json.JSONDecodeError:
                    print(f"   ⚠️ Linha {i} não é JSON válido")
                    continue
            
            if not sample_lines:
                print(f"   ❌ Nenhuma linha JSON válida encontrada!")
                return
            
            print(f"   ✓ Amostras lidas com sucesso")
            
            # Analisar primeira amostra
            first = sample_lines[0]
            print(f"\n   📄 Estrutura da primeira linha:")
            if isinstance(first, dict):
                print(f"      Chaves: {list(first.keys())}")
                for key, value in list(first.items())[:5]:
                    val_str = str(value)[:100]
                    print(f"      - {key}: {val_str}...")
            else:
                print(f"      Tipo: {type(first)}")
                print(f"      Conteúdo: {str(first)[:200]}")
            
            # Verificar colunas configuradas
            print(f"\n   🔍 Verificando colunas configuradas ({JSONL_TEXT_COLUMNS}):")
            found_cols = []
            for col in JSONL_TEXT_COLUMNS:
                if col in first:
                    val = first[col]
                    if isinstance(val, str) and len(val) > 0:
                        found_cols.append(col)
                        print(f"      ✓ '{col}' encontrada ({len(val)} chars)")
                    elif isinstance(val, list):
                        found_cols.append(col)
                        print(f"      ✓ '{col}' encontrada (lista com {len(val)} itens)")
                    else:
                        print(f"      ⚠️ '{col}' encontrada mas vazia ou tipo inesperado")
            
            if not found_cols:
                print(f"      ❌ Nenhuma das colunas configuradas foi encontrada!")
                print(f"\n💡 SOLUÇÃO:")
                print(f"   - Edite config.py e ajuste JSONL_TEXT_COLUMNS")
                print(f"   - Use as chaves encontradas acima: {list(first.keys()) if isinstance(first, dict) else 'N/A'}")
            
    except Exception as e:
        print(f"   ❌ Erro ao analisar dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Verificar tokenizer
    print(f"\n4️⃣  Verificando tokenizer...")
    tokenizer_path = "tokenizer.pkl"
    if os.path.exists(tokenizer_path):
        print(f"   ✓ tokenizer.pkl encontrado")
        try:
            import pickle
            with open(tokenizer_path, 'rb') as f:
                data = pickle.load(f)
                vocab_size = len(data.get('vocab', {}))
                merges_count = len(data.get('merges', []))
                print(f"   📊 Vocabulário: {vocab_size} tokens")
                print(f"   🔗 Merges: {merges_count}")
                
                if vocab_size <= 10:
                    print(f"   ⚠️  ATENÇÃO: Vocabulário muito pequeno!")
                    print(f"      Isso indica que o tokenizer não aprendeu nada útil")
                    print(f"      Possível causa: dataset vazio ou sem texto nas colunas corretas")
        except Exception as e:
            print(f"   ⚠️  Erro ao ler tokenizer: {e}")
    else:
        print(f"   ⚠️  tokenizer.pkl não encontrado")
        print(f"   💡 Execute: python tokenizer.py primeiro")
    
    # 5. Verificar checkpoints
    print(f"\n5️⃣  Verificando checkpoints...")
    checkpoint_dir = "./checkpoints"
    if os.path.exists(checkpoint_dir):
        files = os.listdir(checkpoint_dir)
        if files:
            print(f"   ✓ Checkpoints encontrados: {len(files)} arquivos")
            for f in files[:5]:
                size = os.path.getsize(os.path.join(checkpoint_dir, f)) / (1024 * 1024)
                print(f"      - {f} ({size:.2f} MB)")
        else:
            print(f"   ⚠️  Pasta checkpoints vazia")
            print(f"   💡 Execute: python train.py para treinar")
    else:
        print(f"   ⚠️  Pasta checkpoints não existe")
        print(f"   💡 Será criada automaticamente ao treinar")
    
    # 6. Resumo e recomendações
    print("\n" + "=" * 60)
    print("📋 RESUMO E RECOMENDAÇÕES")
    print("=" * 60)
    
    issues = []
    
    if not os.path.exists(DATASET_PATH):
        issues.append("❌ Dataset não encontrado - ajuste DATASET_PATH em config.py")
    
    if not found_cols:
        issues.append("❌ Colunas do dataset não correspondem - ajuste JSONL_TEXT_COLUMNS em config.py")
    
    if os.path.exists(tokenizer_path):
        try:
            import pickle
            with open(tokenizer_path, 'rb') as f:
                data = pickle.load(f)
                if len(data.get('vocab', {})) <= 10:
                    issues.append("❌ Vocabulário do tokenizer muito pequeno - execute tokenizer.py novamente após corrigir o dataset")
        except:
            pass
    
    if not issues:
        print("\n✅ Tudo parece estar correto!")
        print("\n🚀 Próximos passos:")
        if not os.path.exists(tokenizer_path):
            print("   1. python tokenizer.py (treinar tokenizer)")
        print("   2. python train.py (treinar modelo)")
        print("   3. python generate.py (gerar texto)")
    else:
        print("\n⚠️  Problemas encontrados:")
        for issue in issues:
            print(f"   {issue}")
        print("\n💡 Resolva os problemas acima antes de continuar!")

if __name__ == "__main__":
    main()
