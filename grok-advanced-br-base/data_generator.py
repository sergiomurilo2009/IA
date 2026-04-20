"""
data_generator.py - Gerador de Dados em Massa para Grok-Advanced-BR-Base

Este módulo gera milhões de documentos sintéticos de forma otimizada e organizada
para popular o banco de dados do sistema RAG.

Recursos:
- Geração paralela e eficiente
- Dados categorizados e organizados
- Suporte a múltiplos formatos (JSON, TXT, CSV)
- Compressão automática em ZIP
- Progresso em tempo real

Autor: Grok-Advanced-BR-Base
Licença: MIT
"""

import os
import json
import csv
import random
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import hashlib


# ==================== TEMPLATES DE DADOS ====================

# Templates para Python e Programação
PYTHON_TEMPLATES = [
    "Python é uma linguagem de programação {adjetivo} criada por {criador} em {ano}.",
    "A sintaxe do Python é {caracteristica}, tornando-o ideal para {uso}.",
    "Em Python, {conceito} é usado para {finalidade}.",
    "O framework {framework} é amplamente utilizado para {aplicacao} em Python.",
    "A biblioteca {biblioteca} fornece funcionalidades para {funcionalidade}.",
    "Listas em Python são {propriedade} e podem armazenar {tipo_dado}.",
    "Dicionários Python usam {estrutura} para armazenar pares {par}.",
    "Funções em Python são definidas usando a palavra-chave {keyword}.",
    "Classes em Python seguem o paradigma {paradigma} de programação.",
    "Decoradores em Python permitem {capacidade} de funções.",
    "Generators em Python usam {sintaxe} para criar iteradores eficientes.",
    "Context managers em Python implementam {protocolo} para gerenciamento de recursos.",
    "Type hints em Python foram introduzidos na versão {versao} para {beneficio}.",
    "Async/await em Python permite programação {tipo_programacao} eficiente.",
    "O GIL (Global Interpreter Lock) no CPython {efeito_gil}.",
]

# Templates para Inteligência Artificial
IA_TEMPLATES = [
    "{tecnologia} é um campo da IA que foca em {foco}.",
    "Redes neurais {tipo_rede} são usadas para {aplicacao_rede}.",
    "O algoritmo {algoritmo} foi desenvolvido por {pesquisador} em {ano_algoritmo}.",
    "Deep Learning revolucionou {area} através de {innovacao}.",
    "Transfer learning permite {capacidade_tl} em modelos de IA.",
    "Attention mechanisms melhoram {melhoria} em modelos de NLP.",
    "Transformers são arquiteturas baseadas em {base_transformer} para {tarefa}.",
    "GANs (Generative Adversarial Networks) consistem em {componentes_gan}.",
    "Reinforcement Learning usa {metodo_rl} para aprender {objetivo_rl}.",
    "Batch normalization em redes neurais {efeito_bn}.",
    "Dropout é uma técnica de {tipo_tecnica} que previne {problema}.",
    "Optimizer {optimizer} é comumente usado para {finalidade_opt}.",
    "Loss functions medem {medida} entre previsões e valores reais.",
    "CNNs (Convolutional Neural Networks) são especializadas em {tipo_dado_cnn}.",
    "RNNs (Recurrent Neural Networks) processam {tipo_sequencia} temporal.",
]

# Templates para Ciência de Dados
DATA_SCIENCE_TEMPLATES = [
    "Pandas é uma biblioteca Python para {funcionalidade_pandas}.",
    "NumPy fornece suporte para {recurso_numpy} em Python.",
    "Matplotlib é usado para {tipo_visualizacao} de dados.",
    "Scikit-learn implementa algoritmos de {tipo_ml} para machine learning.",
    "Feature engineering envolve {processo_fe} para melhorar modelos.",
    "Cross-validation é uma técnica para {objetivo_cv}.",
    "Overfitting ocorre quando o modelo {comportamento_of}.",
    "Regularização L1/L2 ajuda a {objetivo_reg}.",
    "PCA (Principal Component Analysis) reduz {o_que_pca}.",
    "Clustering é um tipo de aprendizado {tipo_clustering}.",
    "Random Forest combina {componentes_rf} para previsão.",
    "Gradient Boosting funciona através de {mecanismo_gb}.",
    "SVM (Support Vector Machines) encontra {objetivo_svm}.",
    "K-means é um algoritmo de clustering que {como_kmeans}.",
    "Time series analysis lida com dados {caracteristica_ts}.",
]

# Templates para Web Development
WEB_TEMPLATES = [
    "Django é um framework web {caracteristica_django} para Python.",
    "Flask é um micro-framework {tipo_flask} para aplicações web.",
    "FastAPI é conhecido por {vantagem_fastapi} e documentação automática.",
    "REST APIs seguem princípios {principio_rest} para comunicação.",
    "GraphQL permite {vantagem_graphql} em comparação com REST.",
    "WebSockets fornecem comunicação {tipo_comunicacao} bidirecional.",
    "Middleware em frameworks web {funcao_middleware}.",
    "ORM (Object-Relational Mapping) abstrai {abstracao_orm}.",
    "JWT (JSON Web Tokens) são usados para {uso_jwt}.",
    "OAuth 2.0 é um protocolo para {objetivo_oauth}.",
    "Docker containeriza aplicações para {beneficio_docker}.",
    "Kubernetes orquestra containers em {escala_k8s}.",
    "CI/CD pipelines automatizam {processo_cicd}.",
    "Microservices arquitetura divide aplicações em {caracteristica_ms}.",
    "Serverless computing executa código sem {sem_o_que_serverless}.",
]

# Templates para Banco de Dados
DB_TEMPLATES = [
    "SQL é uma linguagem para {uso_sql} em bancos relacionais.",
    "NoSQL databases são otimizadas para {caso_uso_nosql}.",
    "PostgreSQL é conhecido por {caracteristica_pg}.",
    "MongoDB armazena dados em formato {formato_mongo}.",
    "Redis é um banco {tipo_redis} usado para cache.",
    "Índices em bancos de dados melhoram {melhoria_indices}.",
    "Transações ACID garantem {propriedade_acid}.",
    "Sharding é uma técnica de {objetivo_sharding}.",
    "Replication aumenta {beneficio_replication} do sistema.",
    "Normalização de banco de dados visa {objetivo_normalizacao}.",
    "Query optimization melhora {o_que_otimizacao}.",
    "CAP theorem descreve trade-offs entre {captradeoffs}.",
    "Database migrations gerenciam {gerencia_migrations}.",
    "Connection pooling reutiliza {recurso_pool} para eficiência.",
    "Full-text search permite {capacidade_fts} em textos.",
]

# Templates para DevOps e Cloud
DEVOPS_TEMPLATES = [
    "AWS oferece serviços de cloud como {servico_aws}.",
    "Azure é a plataforma cloud da {empresa_azure}.",
    "GCP fornece ferramentas de {tipo_ferramenta_gcp}.",
    "Terraform é uma ferramenta de {tipo_terraform}.",
    "Ansible automatiza {automacao_ansible} sem agentes.",
    "Prometheus é usado para {uso_prometheus} e alertas.",
    "Grafana cria dashboards para {tipo_dashboard}.",
    "ELK Stack consiste em {componentes_elk}.",
    "Git flow é uma estratégia de {estrategia_git}.",
    "Infrastructure as Code permite {beneficio_iac}.",
    "Blue-green deployment reduz {risco_bg}.",
    "Canary releases lançam features para {publico_canary}.",
    "Service mesh gerencia {gerencia_sm} entre microservices.",
    "Observability inclui logs, métricas e {terceiro_pilar}.",
    "Chaos engineering testa {teste_ce} em sistemas.",
]

# Templates para Segurança
SECURITY_TEMPLATES = [
    "Criptografia {tipo_cripto} usa chaves para proteger dados.",
    "HTTPS garante {garantia_https} em comunicações web.",
    "Firewalls filtram {o_que_firewall} de rede.",
    "IDS/IPS detectam e previnem {ameaca_ids}.",
    "Penetration testing identifica {identifica_pentest}.",
    "Zero Trust é um modelo de segurança que {princípio_zt}.",
    "MFA (Multi-Factor Authentication) requer {requisitos_mfa}.",
    "OWASP Top 10 lista {lista_owasp} de vulnerabilidades.",
    "SIEM sistemas correlacionam {correlacao_siem}.",
    "Endpoint protection protege {protege_ep} contra malware.",
    "Data loss prevention previne {previne_dlp}.",
    "Security auditing verifica {verifica_audit}.",
    "Incident response lida com {lida_ir} de segurança.",
    "Threat intelligence fornece {fornece_ti} sobre ameaças.",
    "Compliance GDPR regula {regula_gdpr} de dados.",
]

# Templates para Hardware e Sistemas
HARDWARE_TEMPLATES = [
    "CPU é o componente responsável por {responsabilidade_cpu}.",
    "GPU é otimizada para {otimizacao_gpu} paralela.",
    "RAM armazena dados {tipo_armazenamento_ram} temporariamente.",
    "SSDs são mais rápidos que HDDs porque {razao_ssd}.",
    "TPU é um acelerador de IA desenvolvido por {desenvolvedor_tpu}.",
    "Quantum computing usa {usos_qc} para computação.",
    "Edge computing processa dados {onde_edge}.",
    "IoT conecta dispositivos {tipo_dispositivos} à internet.",
    "5G oferece {beneficio_5g} em comunicações móveis.",
    "Neuromorphic chips imitam {imita_nc} biológicos.",
    "Optical computing usa {usa_oc} para processamento.",
    "ARM architecture é conhecida por {caracteristica_arm}.",
    "x86 é uma arquitetura {tipo_x86} dominante em PCs.",
    "Cache L1/L2/L3 melhora {melhora_cache} da CPU.",
    "Virtualização permite {permite_virt} múltiplos sistemas.",
]

# Dados variáveis para preenchimento
ADJETIVOS = ["poderosa", "versátil", "elegante", "eficiente", "moderna", "robusta", "flexível", "intuitiva"]
CRIADORES = ["Guido van Rossum", "James Gosling", "Brendan Eich", "Anders Hejlsberg", "Ryan Dahl"]
ANOS = ["1991", "1995", "2009", "2012", "2014", "2015", "2018", "2020"]
CARACTERISTICAS = ["limpa", "legível", "concisa", "expressiva", "minimalista", "poderosa"]
USOS = ["iniciantes", "ciência de dados", "web development", "automação", "machine learning", "scripting"]
CONCEITOS = ["list comprehension", "decorator", "generator", "context manager", "async/await", "type hint"]
FINALIDADES = ["iteração concisa", "modificação de funções", "criação de iteradores", "gerenciamento de recursos", "programação assíncrona", "documentação de tipos"]
FRAMEWORKS = ["Django", "Flask", "FastAPI", "Pyramid", "Tornado", "Bottle", "CherryPy"]
APLICACOES = ["desenvolvimento web", "APIs REST", "microservices", "aplicações em tempo real", "prototipagem rápida"]
BIBLIOTECAS = ["NumPy", "Pandas", "Scikit-learn", "TensorFlow", "PyTorch", "Matplotlib", "Requests"]
FUNCIONALIDADES = ["computação numérica", "análise de dados", "machine learning", "deep learning", "visualização", "HTTP requests"]


def gerar_documento_python(template_id: int, variacoes: Dict) -> str:
    """Gera um documento sobre Python."""
    template = PYTHON_TEMPLATES[template_id % len(PYTHON_TEMPLATES)]
    return template.format(
        adjetivo=random.choice(ADJETIVOS),
        criador=random.choice(CRIADORES),
        ano=random.choice(ANOS),
        caracteristica=random.choice(CARACTERISTICAS),
        uso=random.choice(USOS),
        conceito=random.choice(CONCEITOS),
        finalidade=random.choice(FINALIDADES),
        framework=random.choice(FRAMEWORKS),
        aplicacao=random.choice(APLICACOES),
        biblioteca=random.choice(BIBLIOTECAS),
        funcionalidade=random.choice(FUNCIONALIDADES),
        propriedade="ordenadas e mutáveis",
        tipo_dado="qualquer tipo de objeto",
        estrutura="chaves e valores",
        par="chave-valor",
        keyword="def",
        paradigma="orientado a objetos",
        capacidade="modificação dinâmica",
        sintaxe="yield",
        protocolo="__enter__ e __exit__",
        versao="3.5",
        beneficio="documentação e type checking",
        tipo_programacao="assíncrona",
        efeito_gil="limita execução paralela de threads"
    )


def gerar_documento_ia(template_id: int, variacoes: Dict) -> str:
    """Gera um documento sobre Inteligência Artificial."""
    template = IA_TEMPLATES[template_id % len(IA_TEMPLATES)]
    return template.format(
        tecnologia=random.choice(["Machine Learning", "Deep Learning", "Computer Vision", "NLP", "Reinforcement Learning"]),
        foco=random.choice(["aprendizado de padrões", "reconhecimento de imagens", "processamento de texto", "tomada de decisões"]),
        tipo_rede=random.choice(["convolucionais", "recorrentes", "transformers", "autoencoders"]),
        aplicacao_rede=random.choice(["classificação de imagens", "tradução automática", "generação de texto", "detecção de anomalias"]),
        algoritmo=random.choice(["Backpropagation", "Gradient Descent", "Q-Learning", "Adam Optimizer"]),
        pesquisador=random.choice(["Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio", "Andrew Ng"]),
        ano_algoritmo=random.choice(["1986", "1989", "2012", "2014", "2017"]),
        area=random.choice(["visão computacional", "NLP", "speech recognition", "recomendação"]),
        innovacao=random.choice(["CNNs", "Transformers", "Attention", "ResNet"]),
        capacidade_tl=random.choice(["reutilizar modelos pré-treinados", "adaptar para novos domínios"]),
        melhoria=random.choice(["performance em tarefas sequenciais", "captura de dependências de longo prazo"]),
        base_transformer=random.choice(["self-attention", "encoder-decoder"]),
        tarefa=random.choice(["tradução", "sumarização", "QA", "generação"]),
        componentes_gan=random.choice(["generator e discriminator", "duas redes competindo"]),
        metodo_rl=random.choice(["reward signals", "trial and error"]),
        objetivo_rl=random.choice(["políticas ótimas", "maximizar recompensas"]),
        efeito_bn=random.choice(["normaliza ativações", "acelera treinamento"]),
        tipo_tecnica="regularização",
        problema="overfitting",
        optimizer=random.choice(["Adam", "SGD", "RMSprop", "AdaGrad"]),
        finalidade_opt=random.choice(["minimizar loss", "atualizar pesos"]),
        medida=random.choice(["discrepância", "erro", "distância"]),
        tipo_dado_cnn=random.choice(["imagens", "dados grid-like"]),
        tipo_sequencia=random.choice(["séries temporais", "texto", "áudio"])
    )


def gerar_documento_data_science(template_id: int, variacoes: Dict) -> str:
    """Gera um documento sobre Ciência de Dados."""
    template = DATA_SCIENCE_TEMPLATES[template_id % len(DATA_SCIENCE_TEMPLATES)]
    return template.format(
        funcionalidade_pandas=random.choice(["manipulação de DataFrames", "análise de dados tabulares"]),
        recurso_numpy=random.choice(["arrays multidimensionais", "operações vetoriais"]),
        tipo_visualizacao=random.choice(["gráficos 2D/3D", "plots estatísticos"]),
        tipo_ml=random.choice(["classificação", "regressão", "clustering"]),
        processo_fe=random.choice(["seleção de features", "criação de novas variáveis"]),
        objetivo_cv=random.choice(["avaliar generalização", "prevenir overfitting"]),
        comportamento_of=random.choice(["memoriza dados de treino", "não generaliza bem"]),
        objetivo_reg=random.choice(["reduzir overfitting", "penalizar coeficientes"]),
        o_que_pca=random.choice(["dimensionalidade", "número de features"]),
        tipo_clustering="não supervisionado",
        componentes_rf=random.choice(["múltiplas decision trees", "ensemble de árvores"]),
        mecanismo_gb=random.choice(["correção iterativa de erros", "boosting sequencial"]),
        objetivo_svm=random.choice(["hiperplano ótimo", "margem máxima"]),
        como_kmeans=random.choice(["particiona em k clusters", "minimiza inércia"]),
        caracteristica_ts=random.choice(["temporalmente ordenados", "com dependência temporal"])
    )


def gerar_documento_web(template_id: int, variacoes: Dict) -> str:
    """Gera um documento sobre Desenvolvimento Web."""
    template = WEB_TEMPLATES[template_id % len(WEB_TEMPLATES)]
    return template.format(
        caracteristica_django=random.choice(["full-featured", "baterias inclusas", "maduro"]),
        tipo_flask=random.choice(["leve", "minimalista", "flexível"]),
        vantagem_fastapi=random.choice(["alta performance", "type hints", "async nativo"]),
        principio_rest=random.choice(["stateless", "client-server", "cacheable"]),
        vantagem_graphql=random.choice(["queries específicas", "evitar over-fetching"]),
        tipo_comunicacao="em tempo real",
        funcao_middleware=random.choice(["processa requests", "adiciona funcionalidades"]),
        abstracao_orm=random.choice(["banco de dados", "queries SQL"]),
        uso_jwt=random.choice(["autenticação stateless", "autorização"]),
        objetivo_oauth=random.choice(["autorização delegada", "single sign-on"]),
        beneficio_docker=random.choice(["portabilidade", "isolamento", "consistência"]),
        escala_k8s=random.choice(["grande escala", "clusters"]),
        processo_cicd=random.choice(["build, test, deploy", "entrega contínua"]),
        caracteristica_ms=random.choice(["serviços independentes", "domínios específicos"]),
        sem_o_que_serverless="gerenciar servidores"
    )


def gerar_documento_db(template_id: int, variacoes: Dict) -> str:
    """Gera um documento sobre Banco de Dados."""
    template = DB_TEMPLATES[template_id % len(DB_TEMPLATES)]
    return template.format(
        uso_sql=random.choice(["consultas", "manipulação de dados"]),
        caso_uso_nosql=random.choice(["grande volume", "baixa latência", "esquema flexível"]),
        caracteristica_pg=random.choice(["ACID compliance", "extensibilidade", "JSON support"]),
        formato_mongo="BSON",
        tipo_redis=random.choice(["in-memory", "key-value"]),
        melhoria_indices=random.choice(["performance de queries", "tempo de busca"]),
        propriedade_acid=random.choice(["atomicidade, consistência, isolamento, durabilidade"]),
        objetivo_sharding=random.choice(["distribuir dados", "escalabilidade horizontal"]),
        beneficio_replication=random.choice(["disponibilidade", "fault tolerance"]),
        objetivo_normalizacao=random.choice(["reduzir redundância", "melhorar integridade"]),
        o_que_otimizacao=random.choice(["tempo de execução", "uso de recursos"]),
        captradeoffs=random.choice(["consistência, disponibilidade, partição"]),
        gerencia_migrations=random.choice(["mudanças de schema", "versionamento"]),
        recurso_pool=random.choice(["conexões", "sockets"]),
        capacidade_fts=random.choice(["buscas full-text", "indexação de texto"])
    )


def gerar_documento_devops(template_id: int, variacoes: Dict) -> str:
    """Gera um documento sobre DevOps e Cloud."""
    template = DEVOPS_TEMPLATES[template_id % len(DEVOPS_TEMPLATES)]
    return template.format(
        servico_aws=random.choice(["EC2", "S3", "Lambda", "RDS", "DynamoDB"]),
        empresa_azure="Microsoft",
        tipo_ferramenta_gcp=random.choice(["ML", "analytics", "cloud storage"]),
        tipo_terraform=random.choice(["infrastructure as code", "provisionamento"]),
        automacao_ansible=random.choice(["configuração", "deploy", "orchestration"]),
        uso_prometheus=random.choice(["monitoring", "coleta de métricas"]),
        tipo_dashboard=random.choice(["métricas", "logs", "alertas"]),
        componentes_elk=random.choice(["Elasticsearch, Logstash, Kibana"]),
        estrategia_git=random.choice(["branching", "versionamento"]),
        beneficio_iac=random.choice(["reprodutibilidade", "automação"]),
        risco_bg=random.choice(["downtime", "risk of deployment"]),
        publico_canary=random.choice(["pequeno grupo", "usuários beta"]),
        gerencia_sm=random.choice(["comunicação", "tráfego"]),
        terceiro_pilar="traces",
        teste_ce=random.choice(["resiliência", "falhas"])
    )


def gerar_documento_security(template_id: int, variacoes: Dict) -> str:
    """Gera um documento sobre Segurança."""
    template = SECURITY_TEMPLATES[template_id % len(SECURITY_TEMPLATES)]
    return template.format(
        tipo_cripto=random.choice(["simétrica", "assimétrica", "hash"]),
        garantia_https=random.choice(["confidencialidade", "integridade"]),
        o_que_firewall=random.choice(["tráfego", "pacotes"]),
        ameaca_ids=random.choice(["intrusões", "ataques"]),
        identifica_pentest=random.choice(["vulnerabilidades", "fraquezas"]),
        princípio_zt=random.choice(["nunca confia, sempre verifica"]),
        requisitos_mfa=random.choice(["múltiplos fatores", "2+ credenciais"]),
        lista_owasp="principais riscos",
        correlacao_siem=random.choice(["eventos de segurança", "logs"]),
        protege_ep=random.choice(["dispositivos finais", "endpoints"]),
        previne_dlp=random.choice(["vazamento de dados", "data exfiltration"]),
        verifica_audit=random.choice(["conformidade", "controles"]),
        lida_ir=random.choice(["incidentes", "breaches"]),
        fornece_ti=random.choice(["informações", "intelligence"]),
        regula_gdpr=random.choice(["proteção de dados", "privacidade"])
    )


def gerar_documento_hardware(template_id: int, variacoes: Dict) -> str:
    """Gera um documento sobre Hardware."""
    template = HARDWARE_TEMPLATES[template_id % len(HARDWARE_TEMPLATES)]
    return template.format(
        responsabilidade_cpu=random.choice(["executar instruções", "processamento geral"]),
        otimizacao_gpu=random.choice(["processamento paralelo", "operações matriciais"]),
        tipo_armazenamento_ram="volátil e rápido",
        razao_ssd=random.choice(["não têm partes móveis", "usam flash memory"]),
        desenvolvedor_tpu="Google",
        usos_qc=random.choice(["qubits", "superposição quântica"]),
        onde_edge=random.choice(["na borda da rede", "localmente"]),
        tipo_dispositivos=random.choice(["inteligentes", "sensorizados"]),
        beneficio_5g=random.choice(["alta velocidade", "baixa latência"]),
        imita_nc=random.choice(["neurônios", "sistemas neurais"]),
        usa_oc=random.choice(["fótons", "luz"]),
        caracteristica_arm=random.choice(["baixo consumo", "eficiência energética"]),
        tipo_x86="CISC complexa",
        melhora_cache=random.choice(["latência de memória", "performance"]),
        permite_virt=random.choice(["execução de", "isolamento de"])
    )


# Mapeamento de categorias para geradores
GERADORES = {
    'python': gerar_documento_python,
    'ia': gerar_documento_ia,
    'data_science': gerar_documento_data_science,
    'web': gerar_documento_web,
    'db': gerar_documento_db,
    'devops': gerar_documento_devops,
    'security': gerar_documento_security,
    'hardware': gerar_documento_hardware
}

CATEGORIAS = list(GERADORES.keys())


def gerar_documento(categoria: str, template_id: int) -> Dict:
    """
    Gera um único documento estruturado.
    
    Args:
        categoria: Categoria do documento
        template_id: ID do template para variação
    
    Returns:
        Dicionário com id, categoria, conteudo e metadata
    """
    gerador = GERADORES.get(categoria, gerar_documento_python)
    conteudo = gerador(template_id, {})
    
    # Cria hash único para o documento
    doc_hash = hashlib.md5(f"{categoria}_{template_id}_{conteudo}".encode()).hexdigest()[:12]
    
    return {
        'id': doc_hash,
        'categoria': categoria,
        'conteudo': conteudo,
        'template_id': template_id,
        'timestamp': datetime.now().isoformat()
    }


def gerar_lote_documentos(categoria: str, inicio: int, quantidade: int) -> List[Dict]:
    """
    Gera um lote de documentos.
    
    Args:
        categoria: Categoria dos documentos
        inicio: ID inicial do template
        quantidade: Quantidade de documentos
    
    Returns:
        Lista de documentos
    """
    documentos = []
    for i in range(quantidade):
        doc = gerar_documento(categoria, inicio + i)
        documentos.append(doc)
    return documentos


def salvar_em_json(documentos: List[Dict], caminho: Path) -> None:
    """Salva documentos em arquivo JSON."""
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(documentos, f, ensure_ascii=False, indent=2)


def salvar_em_txt(documentos: List[Dict], caminho: Path) -> None:
    """Salva documentos em arquivo TXT (um por linha)."""
    with open(caminho, 'w', encoding='utf-8') as f:
        for doc in documentos:
            f.write(f"{doc['conteudo']}\n\n")


def salvar_em_csv(documentos: List[Dict], caminho: Path) -> None:
    """Salva documentos em arquivo CSV."""
    if not documentos:
        return
    
    with open(caminho, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'categoria', 'conteudo', 'template_id', 'timestamp'])
        writer.writeheader()
        writer.writerows(documentos)


def criar_zip_comprimido(pasta_origem: Path, caminho_saida: Path) -> None:
    """Cria um arquivo ZIP comprimido com os arquivos da pasta."""
    with zipfile.ZipFile(caminho_saida, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for arquivo in pasta_origem.iterdir():
            if arquivo.is_file():
                zipf.write(arquivo, arquivo.name)


def gerar_base_conhecimentos(
    total_documentos: int = 1000000,
    pasta_saida: str = "data/bulk",
    formato: str = "json",
    usar_compressao: bool = True,
    workers: int = None
) -> Dict:
    """
    Gera uma base de conhecimentos massiva.
    
    Args:
        total_documentos: Número total de documentos a gerar
        pasta_saida: Pasta para salvar os arquivos
        formato: Formato de saída (json, txt, csv, all)
        usar_compressao: Se True, cria ZIP comprimido
        workers: Número de workers para processamento paralelo
    
    Returns:
        Estatísticas da geração
    """
    print("=" * 70)
    print(f"  GERADOR DE BASE DE CONHECIMENTOS - {total_documentos:,} DOCUMENTOS")
    print("=" * 70)
    print()
    
    inicio_total = datetime.now()
    
    # Configura pasta de saída
    pasta = Path(pasta_saida)
    pasta.mkdir(parents=True, exist_ok=True)
    
    # Calcula documentos por categoria
    docs_por_categoria = total_documentos // len(CATEGORIAS)
    resto = total_documentos % len(CATEGORIAS)
    
    # Configura workers
    if workers is None:
        workers = multiprocessing.cpu_count()
    
    print(f"📊 Configuração:")
    print(f"   Total de documentos: {total_documentos:,}")
    print(f"   Categorias: {len(CATEGORIAS)} ({', '.join(CATEGORIAS)})")
    print(f"   Documentos por categoria: ~{docs_por_categoria:,}")
    print(f"   Workers: {workers}")
    print(f"   Formato: {formato}")
    print(f"   Compressão: {'Sim' if usar_compressao else 'Não'}")
    print()
    
    total_gerado = 0
    stats = {
        'categorias': {},
        'arquivos_criados': [],
        'tempo_total': 0
    }
    
    # Gera documentos por categoria
    for idx, categoria in enumerate(CATEGORIAS):
        print(f"📝 Gerando categoria: {categoria.upper()}")
        inicio_cat = datetime.now()
        
        # Ajusta quantidade para última categoria pegar o resto
        qtd = docs_por_categoria + (1 if idx < resto else 0)
        
        # Divide em lotes menores para processamento
        lote_size = min(10000, qtd)
        lotes = (qtd + lote_size - 1) // lote_size
        
        documentos_categoria = []
        
        for lote_idx in range(lotes):
            inicio_lote = lote_idx * lote_size
            
            # Gera lote (usando ThreadPool para I/O bound)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []
                for worker_id in range(workers):
                    worker_inicio = inicio_lote + (worker_id * lote_size // workers)
                    worker_qtd = min(lote_size // workers, qtd - len(documentos_categoria))
                    
                    if worker_qtd > 0:
                        future = executor.submit(
                            gerar_lote_documentos,
                            categoria,
                            worker_inicio,
                            worker_qtd
                        )
                        futures.append(future)
                
                # Coleta resultados
                for future in futures:
                    documentos_categoria.extend(future.result())
            
            # Progresso
            progresso = min(len(documentos_categoria), qtd)
            print(f"   Progresso: {progresso:,}/{qtd:,} ({progresso*100//qtd}%)", end='\r')
        
        print(f"   ✓ {categoria.upper()}: {len(documentos_categoria):,} documentos gerados")
        
        # Salva documentos da categoria
        if formato in ['json', 'all']:
            arquivo_json = pasta / f"{categoria}.json"
            salvar_em_json(documentos_categoria, arquivo_json)
            stats['arquivos_criados'].append(str(arquivo_json))
            print(f"   💾 Salvo: {arquivo_json.name}")
        
        if formato in ['txt', 'all']:
            arquivo_txt = pasta / f"{categoria}.txt"
            salvar_em_txt(documentos_categoria, arquivo_txt)
            stats['arquivos_criados'].append(str(arquivo_txt))
            print(f"   💾 Salvo: {arquivo_txt.name}")
        
        if formato in ['csv', 'all']:
            arquivo_csv = pasta / f"{categoria}.csv"
            salvar_em_csv(documentos_categoria, arquivo_csv)
            stats['arquivos_criados'].append(str(arquivo_csv))
            print(f"   💾 Salvo: {arquivo_csv.name}")
        
        # Stats da categoria
        tempo_cat = (datetime.now() - inicio_cat).total_seconds()
        stats['categorias'][categoria] = {
            'documentos': len(documentos_categoria),
            'tempo_segundos': tempo_cat,
            'docs_por_segundo': len(documentos_categoria) / tempo_cat if tempo_cat > 0 else 0
        }
        
        total_gerado += len(documentos_categoria)
        print(f"   ⏱️  Tempo: {tempo_cat:.2f}s ({stats['categorias'][categoria]['docs_por_segundo']:.0f} docs/s)")
        print()
    
    # Cria ZIP se solicitado
    if usar_compressao and stats['arquivos_criados']:
        print("📦 Criando arquivo ZIP comprimido...")
        zip_path = pasta / "bulk_data.zip"
        criar_zip_comprimido(pasta, zip_path)
        stats['arquivos_criados'].append(str(zip_path))
        tamanho_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"   ✓ ZIP criado: {zip_path.name} ({tamanho_mb:.2f} MB)")
        print()
    
    # Estatísticas finais
    tempo_total = (datetime.now() - inicio_total).total_seconds()
    stats['tempo_total'] = tempo_total
    stats['total_documentos'] = total_gerado
    stats['docs_por_segundo'] = total_gerado / tempo_total if tempo_total > 0 else 0
    
    print("=" * 70)
    print("  ✅ GERAÇÃO CONCLUÍDA!")
    print("=" * 70)
    print(f"📊 Estatísticas:")
    print(f"   Total de documentos: {total_gerado:,}")
    print(f"   Tempo total: {tempo_total:.2f} segundos ({tempo_total/60:.2f} minutos)")
    print(f"   Velocidade média: {stats['docs_por_segundo']:.0f} documentos/segundo")
    print(f"   Arquivos criados: {len(stats['arquivos_criados'])}")
    print()
    
    if stats['categorias']:
        print("📈 Performance por categoria:")
        for cat, cat_stats in stats['categorias'].items():
            print(f"   {cat.upper()}: {cat_stats['docs_por_segundo']:.0f} docs/s")
    
    print()
    print(f"📁 Pasta de saída: {pasta.absolute()}")
    print()
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gerador de Base de Conhecimentos em Massa")
    parser.add_argument('--total', '-t', type=int, default=100000,
                       help='Total de documentos (padrão: 100000)')
    parser.add_argument('--pasta', '-p', type=str, default='data/bulk',
                       help='Pasta de saída (padrão: data/bulk)')
    parser.add_argument('--formato', '-f', type=str, default='json',
                       choices=['json', 'txt', 'csv', 'all'],
                       help='Formato de saída (padrão: json)')
    parser.add_argument('--sem-zip', action='store_true',
                       help='Não criar arquivo ZIP')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Número de workers (padrão: CPUs disponíveis)')
    
    args = parser.parse_args()
    
    gerar_base_conhecimentos(
        total_documentos=args.total,
        pasta_saida=args.pasta,
        formato=args.formato,
        usar_compressao=not args.sem_zip,
        workers=args.workers
    )
