# ====================== load_chat_data.py ======================
"""
Script automático para carregar chat_pt.jsonl do banco de dados.

Este script lê automaticamente o arquivo chat_pt.jsonl da pasta 
"banco de dados" e adiciona todos os documentos ao vector store
para treinamento do sistema.

Uso:
    python load_chat_data.py

O script será executado automaticamente ao iniciar o main.py
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Adicionar src ao path para imports
sys.path.insert(0, str(Path(__file__).parent / "grok-advanced-br"))

from grok_advanced_br.src.config import Config
from grok_advanced_br.src.rag.document_loader import DocumentLoader
from grok_advanced_br.src.rag.vectorstore import VectorStoreManager


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Carrega um arquivo no formato JSONL (JSON Lines).
    
    Cada linha é um objeto JSON independente.
    
    Args:
        file_path: Caminho para o arquivo .jsonl
        
    Returns:
        Lista de documentos/dados
    """
    documents = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                documents.append(data)
            except json.JSONDecodeError as e:
                print(f"[LoadChatData] Erro na linha {line_num}: {e}")
                continue
    
    return documents


def process_chat_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Processa dados do chat para o formato de documentos do RAG.
    
    Espera-se que cada entrada tenha pelo menos um campo de texto
    para ser usado como conteúdo.
    
    Args:
        data: Lista de dados brutos do JSONL
        
    Returns:
        Lista de documentos formatados
    """
    documents = []
    loader = DocumentLoader()
    
    for idx, item in enumerate(data):
        # Tentar extrair conteúdo de vários campos possíveis
        content = None
        
        # Campos comuns em datasets de chat
        for field in ['content', 'text', 'message', 'prompt', 'response', 'answer']:
            if field in item and isinstance(item[field], str):
                content = item[field]
                break
        
        # Se não encontrou campos padrão, tentar concatenar todos os valores string
        if not content:
            text_parts = []
            for key, value in item.items():
                if isinstance(value, str):
                    text_parts.append(f"{key}: {value}")
            content = "\n".join(text_parts) if text_parts else str(item)
        
        # Extrair metadados
        metadata = {"source": "chat_pt.jsonl", "index": idx}
        
        # Adicionar campos adicionais como metadados
        for key, value in item.items():
            if key not in ['content', 'text', 'message', 'prompt', 'response', 'answer']:
                metadata[key] = value
        
        # Criar documento
        doc = loader.load_string(
            content=content,
            metadata=metadata,
            doc_id=f"chat_{idx:06d}"
        )
        
        documents.append(doc)
    
    return documents


def load_chat_dataset_to_vectorstore():
    """
    Carrega automaticamente o chat_pt.jsonl da pasta "banco de dados"
    e adiciona ao vector store.
    
    Returns:
        Número de documentos carregados, ou 0 se nenhum arquivo encontrado
    """
    # Caminho para a pasta "banco de dados"
    base_dir = Path(__file__).parent
    database_dir = base_dir / "banco de dados"
    
    # Verificar se a pasta existe
    if not database_dir.exists():
        print(f"[LoadChatData] Pasta 'banco de dados' não encontrada em {database_dir}")
        return 0
    
    # Procurar por chat_pt.jsonl
    jsonl_file = database_dir / "chat_pt.jsonl"
    
    if not jsonl_file.exists():
        print(f"[LoadChatData] Arquivo chat_pt.jsonl não encontrado em {jsonl_file}")
        print(f"[LoadChatData] Dica: Coloque seu arquivo chat_pt.jsonl na pasta 'banco de dados'")
        return 0
    
    print("\n" + "=" * 60)
    print("CARREGANDO DATASET DE CHAT PARA TREINAMENTO")
    print("=" * 60)
    print(f"Arquivo: {jsonl_file}")
    
    try:
        # Carregar dados brutos
        print("[LoadChatData] Lendo arquivo JSONL...")
        raw_data = load_jsonl_file(str(jsonl_file))
        print(f"[LoadChatData] {len(raw_data)} entradas encontradas no arquivo")
        
        if not raw_data:
            print("[LoadChatData] Nenhum dado válido encontrado no arquivo")
            return 0
        
        # Processar dados para formato de documentos
        print("[LoadChatData] Processando dados...")
        documents = process_chat_data(raw_data)
        print(f"[LoadChatData] {len(documents)} documentos processados")
        
        # Inicializar vector store
        print("[LoadChatData] Inicializando vector store...")
        vectorstore = VectorStoreManager()
        
        # Contar documentos existentes
        existing_count = vectorstore.get_document_count()
        print(f"[LoadChatData] Documentos existentes: {existing_count}")
        
        # Adicionar documentos
        print("[LoadChatData] Adicionando documentos ao vector store...")
        for i, doc in enumerate(documents, 1):
            vectorstore.add_document(
                doc_id=doc["id"],
                content=doc["content"],
                metadata=doc["metadata"]
            )
            
            # Progresso a cada 100 documentos
            if i % 100 == 0:
                print(f"[LoadChatData] Progresso: {i}/{len(documents)} documentos")
        
        # Contagem final
        final_count = vectorstore.get_document_count()
        
        print("\n" + "=" * 60)
        print("✅ CARREGAMENTO CONCLUÍDO!")
        print("=" * 60)
        print(f"Documentos adicionados: {len(documents)}")
        print(f"Total no banco: {final_count}")
        print("=" * 60 + "\n")
        
        return len(documents)
        
    except Exception as e:
        print(f"\n[LoadChatData] ❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    """Função principal."""
    print("\n" + "=" * 60)
    print("🤖 GROK-ADVANCED-BR - CARREGADOR DE DATASET AUTOMÁTICO")
    print("=" * 60)
    
    count = load_chat_dataset_to_vectorstore()
    
    if count > 0:
        print(f"✅ Sucesso! {count} documentos do chat_pt.jsonl foram carregados.")
        print("O sistema está pronto para usar esses dados no treinamento.")
    else:
        print("⚠️ Nenhum documento foi carregado.")
        print("Verifique se o arquivo chat_pt.jsonl está na pasta 'banco de dados'.")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
