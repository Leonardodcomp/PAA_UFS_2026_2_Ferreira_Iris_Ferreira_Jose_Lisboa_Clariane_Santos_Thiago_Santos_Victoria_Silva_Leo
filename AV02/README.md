# Projeto e Análise de Algoritmos: Seleção de Contexto para RAG

Atividade 2 da disciplina de Projeto e Análise de Algoritmos (PAA) – 2026.2.

Nesta atividade, usamos programação dinâmica para selecionar artigos científicos que possam servir de contexto para sistemas RAG.

A ideia é escolher os artigos mais relevantes sem ultrapassar um limite de tokens. Para isso, comparamos três métodos: o ranking da AV1, uma abordagem gulosa e a programação dinâmica.

Os artigos utilizados são da ACL Anthology, com temas relacionados a LLMs, recuperação de informação e RAG.

## Equipe

- Clariane Maria Oliveira Lisboa
- Iris Azenate Ferreira
- José Nilton dos Santos Ferreira
- Leonardo Souza Silva
- Thiago de Jesus Santos
- Victória Cardoso dos Santos


1- INSTALAÇÃO E CONFIGURAÇÕES (Instala as Bibliotecas e define os parâmetros, as consultas e as pastas usadas nos experimentos )
Responsável: Íris

%pip -q install pandas numpy matplotlib nltk acl-anthology tiktoken sentence-transformers scikit-learn
Saída da execução
0KB
	Stream
		Note: you may need to restart the kernel to use updated packages.
		[notice] A new release of pip is available: 25.1.1 -> 26.2.2
		[notice] To update, run: python.exe -m pip install --upgrade pip

Célula de código <undefined>
# %% [code]
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path
import importlib.metadata as metadata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken
from IPython.display import display
from nltk.stem import PorterStemmer

SEMENTE = 42
SEED = SEMENTE  # nome usado nos testes
random.seed(SEMENTE)
np.random.seed(SEMENTE)

PASTA_DADOS = Path('dados')
PASTA_SAIDA = Path('resultados_av2')
PASTA_DADOS.mkdir(exist_ok=True)
PASTA_SAIDA.mkdir(exist_ok=True)
ARQUIVO_CORPUS = PASTA_DADOS / 'acl_corpus_av1.csv'

N_ARTIGOS_COLETA = 18000
ESCALAS_CORPUS = [500, 1000]  # duas quantidades de artigos
ESCALAS_CANDIDATOS = [20, 50]  # quantos resultados chegam à mochila
REPETICOES = 3
CONSULTAS_EXPERIMENTO = ['Q1', 'Q3', 'Q5']

TOKENIZADOR_NOME = 'cl100k_base'
ENC = tiktoken.get_encoding(TOKENIZADOR_NOME)
QUERY_TERMS = ['retrieval augmented generation', 'information retrieval', 'large language model']
QUERIES = {
    'Q1': 'What information retrieval methods are used in RAG systems?',
    'Q2': 'Which studies compare sparse retrieval and dense retrieval?',
    'Q3': 'What semantic search techniques are used to retrieve context for language models?',
    'Q4': 'What reranking methods are used in retrieval systems for LLMs?',
    'Q5': 'What metrics are used to evaluate Retrieval-Augmented Generation systems?',
    'Q6': 'Which papers use embeddings for document retrieval?',
    'Q7': 'What techniques are used to select the most relevant documents or excerpts for a language model?',
    'Q8': 'Which methods were compared for context retrieval?',
    'Q9': 'What limitations of RAG systems are reported in the articles?',
    'Q10': 'What chunking strategies are used in document retrieval?'
}

print('Sistema:', platform.platform())
print('Python:', platform.python_version())
try:
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
except Exception:
    commit = 'preencher após publicar no GitHub'
print('Commit:', commit)
Saída da execução
0KB
	Stream
		Sistema: Windows-11-10.0.26200-SP0
		Python: 3.13.5
		Commit: preencher após publicar no GitHub

Célula de texto <undefined>
# %% [markdown]
## 2. Corpus da AV1
Carrega os artigos a partir do CSV ou, se ele não existir, coleta os artigos na ACL Anthology.
Responsável: Iris

Célula de código <undefined>
# %% [code]
def contar_termos(texto, termos):
    """Pontuação para filtrar artigos na coleta inicial."""
    return sum(texto.count(termo) for termo in termos)


def baixar_artigos_acl(termos, quantidade):
    from acl_anthology import Anthology

    biblioteca = Anthology.from_repo()
    encontrados = []
    for artigo in biblioteca.papers():
        if artigo.title is None or artigo.abstract is None:
            continue

        titulo = str(artigo.title)
        resumo = str(artigo.abstract)
        pontos = contar_termos((titulo + ' ' + resumo).lower(), termos)
        if pontos > 0:
            encontrados.append({
                'id': str(artigo.full_id), 'title': titulo,
                'abstract': resumo, 'year': artigo.year,
                'doi': str(artigo.doi or ''), 'url': str(artigo.web_url),
                'pontos_coleta': pontos
            })

    encontrados.sort(key=lambda x: (x['pontos_coleta'], int(x['year'] or 0)), reverse=True)
    return pd.DataFrame(encontrados[:quantidade]).drop(columns='pontos_coleta')


if ARQUIVO_CORPUS.exists():
    df = pd.read_csv(ARQUIVO_CORPUS, dtype={'id': str})
    origem = 'CSV da AV1'
else:
    try:
        df = baixar_artigos_acl(QUERY_TERMS, N_ARTIGOS_COLETA)
    except Exception as erro:
        raise RuntimeError('Coloque o CSV real da AV1 em dados/acl_corpus_av1.csv') from erro
    df.to_csv(ARQUIVO_CORPUS, index=False)
    origem = 'Nova coleta da ACL Anthology'

colunas = {'id', 'title', 'abstract', 'year', 'url'}
assert colunas.issubset(df.columns), 'O CSV não contém todas as colunas necessárias.'
df = df.dropna(subset=['id', 'title', 'abstract'])
df = df.drop_duplicates(subset='id').reset_index(drop=True)
assert len(df) >= max(ESCALAS_CORPUS), 'São necessários pelo menos 1000 artigos reais.'

print('Origem:', origem, '| Artigos:', len(df))
display(df[['id', 'title', 'year']].head())
Saída da execução
2KB
	Stream
		Origem: CSV da AV1 | Artigos: 17990
	text/plain
		id                                              title  \
		0  2025.findings-emnlp.989  LlmFixer: Fix the Helpfulness of Defensive Lar...   
		1   2024.findings-naacl.43    Measuring Social Norms of Large Language Models   
		2     2025.emnlp-main.1006  Flaw or Artifact? Rethinking Prompt Sensitivit...   
		3   2025.findings-naacl.27  CLaMP 2: Multimodal Music Information Retrieva...   
		4   2025.findings-acl.1014  Evaluating the Long-Term Memory of Large Langu...   
		
		   year  
		0  2025  
		1  2024  
		2  2025  
		3  2025  
		4  2025
Célula de texto <undefined>
# %% [markdown]
## 3. Custos em tokens
Conta os tokens de cada artigo (título + resumo) com o tokenizador cl100k_base e mostra a distribuição dos custos.

Responsável: Iris

Célula de código <undefined>
# %% [code]
def custo_tokens(texto):
    """Conta tokens reais do texto original (título + resumo), não palavras."""
    return len(ENC.encode(str(texto)))


custos = np.array([custo_tokens(str(t) + ' ' + str(a)) for t, a in zip(df['title'], df['abstract'])], dtype=int)
assert (custos > 0).all()

estat_custos = pd.DataFrame([{
    'min': int(custos.min()), 'p25': float(np.percentile(custos, 25)),
    'mediana': float(np.median(custos)), 'p75': float(np.percentile(custos, 75)),
    'p90': float(np.percentile(custos, 90)), 'max': int(custos.max())
}])
display(estat_custos)

plt.figure(figsize=(8, 4))
plt.hist(custos, bins=30)
plt.xlabel('Tokens por artigo (título + resumo)')
plt.ylabel('Número de artigos')
plt.title('Distribuição dos custos em tokens')
plt.tight_layout()
plt.savefig(PASTA_SAIDA / 'histograma_tokens.png', dpi=150)
plt.show()
Saída da execução
29KB
	text/plain
		min    p25  mediana    p75    p90  max
		0   45  206.0    236.0  267.0  297.0  853
		<Figure size 800x400 with 1 Axes>

Célula de texto <undefined>
# %% [markdown]
## 4. Problema da mochila e orçamento W
Trata a seleção como mochila 0/1 (capacidade = W, peso = tokens do artigo, valor = score) e define os dois orçamentos W a partir da mediana dos custos.
