import json
import os
from config import DATASET_PATH

def diagnosticar():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERRO: Arquivo não encontrado em: {DATASET_PATH}")
        return

    print(f"🔍 Analisando: {DATASET_PATH}")
    print("-" * 30)
    
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            # Ler as primeiras 5 linhas válidas
            linhas_lidas = 0
            exemplos = []
            
            for linha in f:
                if linhas_lidas >= 5:
                    break
                linha = linha.strip()
                if not linha:
                    continue
                
                try:
                    dado = json.loads(linha)
                    exemplos.append(dado)
                    linhas_lidas += 1
                except json.JSONDecodeError:
                    print(f"⚠️ Linha inválida (não é JSON): {linha[:50]}...")
                    continue

        if not exemplos:
            print("❌ O arquivo parece estar vazio ou não contém JSON válido.")
            return

        print(f"✅ Sucesso! {linhas_lidas} exemplos lidos.")
        print("\n📋 Estrutura detectada (Chaves disponíveis):")
        
        # Descobrir todas as chaves possíveis
        todas_chaves = set()
        for ex in exemplos:
            if isinstance(ex, dict):
                todas_chaves.update(ex.keys())
        
        print(f"Chaves encontradas: {list(todas_chaves)}")
        
        print("\n👀 Exemplo de conteúdo (primeira linha):")
        primeiro = exemplos[0]
        if isinstance(primeiro, dict):
            for chave, valor in primeiro.items():
                if isinstance(valor, str):
                    print(f"  - '{chave}': {valor[:100]}...")
                else:
                    print(f"  - '{chave}': {type(valor).__name__}")
        else:
            print(f"  Formato não é dicionário: {type(primeiro)}")
            print(f"  Conteúdo: {str(primeiro)[:200]}")

        print("\n💡 Recomendação:")
        if "text" in todas_chaves:
            print("   -> O código deve usar a chave 'text'")
        elif "content" in todas_chaves:
            print("   -> O código deve usar a chave 'content'")
        elif "prompt" in todas_chaves or "response" in todas_chaves:
            print("   -> Parece um dataset de instrução. O código deve juntar prompt + response")
        else:
            print("   -> Identifique manualmente qual chave contém o texto principal.")

    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")

if __name__ == "__main__":
    diagnosticar()
