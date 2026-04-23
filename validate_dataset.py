#!/usr/bin/env python3
"""
Script para validar e explorar datasets JSON/JSONL antes do treinamento.
Suporta arquivos grandes com processamento streaming.
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter

def validate_json_file(filepath):
    """Valida a estrutura do arquivo JSON/JSONL."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"❌ Arquivo não encontrado: {filepath}")
        return None
    
    file_size_gb = filepath.stat().st_size / (1024**3)
    print(f"\n📊 Informações do Arquivo:")
    print(f"   Nome: {filepath.name}")
    print(f"   Tamanho: {file_size_gb:.2f} GB ({filepath.stat().st_size:,} bytes)")
    print(f"   Extensão: {filepath.suffix}")
    
    # Detectar formato
    is_jsonl = filepath.suffix == '.jsonl' or filepath.suffix == '.json'
    
    sample_lines = []
    total_lines = 0
    valid_lines = 0
    keys_found = Counter()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            if is_jsonl:
                # Processar JSONL (uma linha = um JSON)
                print("\n🔍 Analisando formato JSONL...")
                for i, line in enumerate(f):
                    if i >= 1000:  # Amostrar primeiros 1000 lines
                        break
                    total_lines += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        valid_lines += 1
                        sample_lines.append(obj)
                        
                        # Contar chaves
                        if isinstance(obj, dict):
                            for key in obj.keys():
                                keys_found[key] += 1
                    except json.JSONDecodeError as e:
                        print(f"   ⚠️ Erro na linha {i}: {e}")
                        if i < 10:
                            print(f"      Conteúdo: {line[:200]}...")
                
                # Estimar total de linhas
                f.seek(0)
                buffer_size = 1024 * 1024  # 1MB
                sample_bytes = 0
                sample_count = 0
                for chunk in iter(lambda: f.read(buffer_size), ''):
                    sample_bytes += len(chunk.encode('utf-8'))
                    sample_count += chunk.count('\n')
                    if sample_bytes >= 10 * 1024 * 1024:  # 10MB amostra
                        break
                
                estimated_total = int((filepath.stat().st_size / sample_bytes) * sample_count) if sample_bytes > 0 else 0
                
            else:
                # Tentar JSON array
                print("\n🔍 Analisando formato JSON (array)...")
                data = json.load(f)
                if isinstance(data, list):
                    total_lines = len(data)
                    valid_lines = total_lines
                    sample_lines = data[:min(10, len(data))]
                    
                    for obj in sample_lines:
                        if isinstance(obj, dict):
                            for key in obj.keys():
                                keys_found[key] += 1
                else:
                    print("⚠️ JSON não é um array. Tentando tratar como objeto único...")
                    sample_lines = [data]
                    valid_lines = 1
        
        print(f"\n✅ Validação Concluída!")
        print(f"   Linhas totais (estimadas): {estimated_total if is_jsonl else total_lines:,}")
        print(f"   Linhas válidas (amostra): {valid_lines}")
        
        if keys_found:
            print(f"\n🔑 Chaves encontradas (top 10):")
            for key, count in keys_found.most_common(10):
                pct = (count / max(valid_lines, 1)) * 100
                print(f"   - '{key}': {count} ocorrências ({pct:.1f}%)")
        
        if sample_lines:
            print(f"\n📄 Exemplo de dado (primeiro registro):")
            print(json.dumps(sample_lines[0], indent=2, ensure_ascii=False)[:1000])
            
            # Dicas de pré-processamento
            print(f"\n💡 Dicas para este dataset:")
            
            # Detectar possíveis campos de texto
            text_fields = []
            if sample_lines and isinstance(sample_lines[0], dict):
                for key in sample_lines[0].keys():
                    val = sample_lines[0][key]
                    if isinstance(val, str) and len(val) > 50:
                        text_fields.append(key)
            
            if text_fields:
                print(f"   Campos de texto identificados: {', '.join(text_fields)}")
                print(f"   Sugestão: Combine campos ou use o mais relevante para treino")
            
            if is_jsonl:
                print(f"   Formato ideal para treinamento: JSONL")
                print(f"   Extraia apenas o campo de texto e salve como .txt ou mantenha .jsonl")
        
        return {
            'path': str(filepath),
            'size_gb': file_size_gb,
            'format': 'jsonl' if is_jsonl else 'json',
            'total_lines': estimated_total if is_jsonl else total_lines,
            'sample': sample_lines[0] if sample_lines else None,
            'keys': list(keys_found.keys())
        }
    
    except Exception as e:
        print(f"\n❌ Erro ao processar arquivo: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_to_training_format(input_path, output_path, text_field=None):
    """Converte dataset para formato de treino (.txt simples)."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    print(f"\n🔄 Convertendo para formato de treinamento...")
    print(f"   Entrada: {input_path}")
    print(f"   Saída: {output_path}")
    
    count = 0
    errors = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        if input_path.suffix == '.jsonl':
            for line in infile:
                try:
                    obj = json.loads(line.strip())
                    if isinstance(obj, dict):
                        if text_field and text_field in obj:
                            text = obj[text_field]
                        else:
                            # Pegar primeiro campo string longo
                            text = next((v for v in obj.values() if isinstance(v, str) and len(v) > 20), '')
                        
                        if text:
                            outfile.write(text + '\n')
                            count += 1
                    elif isinstance(obj, str):
                        outfile.write(obj + '\n')
                        count += 1
                except:
                    errors += 1
        else:
            data = json.load(infile)
            if isinstance(data, list):
                for obj in data:
                    try:
                        if isinstance(obj, dict):
                            if text_field and text_field in obj:
                                text = obj[text_field]
                            else:
                                text = next((v for v in obj.values() if isinstance(v, str) and len(v) > 20), '')
                            
                            if text:
                                outfile.write(text + '\n')
                                count += 1
                        elif isinstance(obj, str):
                            outfile.write(obj + '\n')
                            count += 1
                    except:
                        errors += 1
    
    output_size = output_path.stat().st_size / (1024**2)
    print(f"\n✅ Conversão concluída!")
    print(f"   Registros processados: {count:,}")
    print(f"   Erros: {errors}")
    print(f"   Tamanho do arquivo de saída: {output_size:.2f} MB")
    
    return count


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python validate_dataset.py <caminho_do_arquivo.json ou .jsonl> [campo_de_texto]")
        print("\nExemplos:")
        print("   python validate_dataset.py dataset.jsonl")
        print("   python validate_dataset.py dataset.jsonl text")
        print("   python validate_dataset.py dataset.json conversation")
        sys.exit(1)
    
    filepath = sys.argv[1]
    text_field = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = validate_json_file(filepath)
    
    if result and text_field:
        output_name = Path(filepath).stem + '_training.txt'
        convert_to_training_format(filepath, output_name, text_field)
