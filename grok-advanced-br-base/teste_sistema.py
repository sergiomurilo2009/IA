"""
teste_sistema.py - Teste automatizado do Grok-Advanced-BR-Base

Este script testa o sistema sem necessidade de interação manual.
"""

from main import GrokAdvancedBR

def testar_sistema():
    """Executa testes automatizados."""
    
    print("=" * 70)
    print(" TESTE AUTOMATIZADO - Grok-Advanced-BR-Base")
    print("=" * 70)
    print()
    
    # Inicializa o sistema
    grok = GrokAdvancedBR()
    
    print(f"\n📊 Estatísticas iniciais:")
    print(f"   Total de documentos: {grok.listar_documentos()}")
    print()
    
    # Lista de perguntas para testar
    perguntas_teste = [
        "O que é Python?",
        "Explique Inteligência Artificial",
        "O que é RAG?",
        "Como funciona Machine Learning?"
    ]
    
    print("=" * 70)
    print(" EXECUTANDO PERGUNTAS DE TESTE")
    print("=" * 70)
    
    for i, pergunta in enumerate(perguntas_teste, 1):
        print(f"\n{'='*70}")
        print(f" TESTE {i}: {pergunta}")
        print('='*70)
        
        resposta = grok.perguntar(pergunta, mostrar_pensamento=True)
        print("\n" + resposta)
        print()
    
    # Testa adição de documento
    print("=" * 70)
    print(" TESTE: Adicionar novo documento")
    print("=" * 70)
    
    novo_doc = "Java é uma linguagem de programação orientada a objetos, criada por James Gosling em 1995."
    if grok.adicionar_documento(novo_doc):
        print(f"✓ Documento adicionado com sucesso!")
        print(f"📊 Novo total: {grok.listar_documentos()} documentos")
    else:
        print("✗ Falha ao adicionar documento")
    
    # Testa pergunta sobre documento recém-adicionado
    print("\n" + "=" * 70)
    print(" TESTE: Pergunta sobre documento novo")
    print("=" * 70)
    
    resposta = grok.perguntar("O que é Java?", mostrar_pensamento=False)
    print(f"\nPergunta: O que é Java?")
    print(f"Resposta: {resposta}")
    
    print("\n" + "=" * 70)
    print(" ✅ TODOS OS TESTES CONCLUÍDOS!")
    print("=" * 70)


if __name__ == "__main__":
    testar_sistema()
