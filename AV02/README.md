# Projeto e Análise de Algoritmos: Seleção de Contexto para RAG com Programação Dinâmica

Atividade 2 da disciplina de Projeto e Análise de Algoritmos (PAA) – 2026.2.

Nesta atividade, modelamos a seleção ótima de contexto para sistemas RAG (Retrieval-Augmented Generation) como uma variação do **Problema da Mochila 0/1**. O objetivo é selecionar o subconjunto de artigos científicos mais relevantes para uma dada consulta sem ultrapassar o limite máximo de tokens ($W$) da janela de contexto de um LLM.

Para isso, comparamos três abordagens centrais:
1. **Ranking da AV1** (Baseline 1): Ordenação decrescente por relevância lexical até atingir o limite de tokens ($O(N \log N)$).
2. **Abordagem Gulosa** (Baseline 2): Heurística por densidade de valor ($\text{score} / \text{tokens}$) ($O(N \log N)$).
3. **Programação Dinâmica (DP)**: Solução exata que encontra o subconjunto ótimo global respeitando estritamente a capacidade ($O(N \cdot W)$, pseudo-polinomial).

Os artigos utilizados são provenientes da **ACL Anthology**, com temas relacionados a Modelos de Linguagem (LLMs), Recuperação de Informação (RI) e RAG.

## Link video apresentação



## Equipe (Discentes)

- Clariane Maria Oliveira Lisboa
- Iris Azenate Ferreira
- José Nilton dos Santos Oliveira
- Leonardo Souza Silva
- Thiago de Jesus Santos
- Victória Cardoso dos Santos

## Ambiente Recomendado

O projeto foi desenvolvido em formato Jupyter Notebook (`.ipynb`) e é altamente recomendado que seja executado no **Google Colab**. Uma vez que o notebook já contém as diretivas de ambiente necessárias para baixar os pacotes diretamente no runtime do Colab, eliminando a necessidade de configuração complexa de ambiente virtual local.

## Dependências

As bibliotecas externas necessárias para a execução do projeto foram deixadas prontas para instalação automática na primeira célula do notebook. São elas:

- `pandas` e `numpy` (manipulação de dados, matrizes e estruturas tabulares)
- `matplotlib` (geração de gráficos comparativos e análises assintóticas)
- `nltk` (tokenização, normalização e stemming com PorterStemmer)
- `acl-anthology` (coleta e estruturação do corpus de artigos científicos)
- `tiktoken` (contagem precisa de tokens reais via codificador `cl100k_base`)
- `sentence-transformers` (geração de embeddings com o modelo `all-MiniLM-L6-v2` para o gabarito semântico)
- `scikit-learn` (cálculo de similaridade de cosseno e métricas de avaliação)

## Parâmetros do Experimento

O notebook foi configurado para garantir a reprodutibilidade dos testes. Os principais parâmetros utilizados na coleta e na análise são:

- `SEED = 42`: Fixado para semente de aleatoriedade (`random` e `numpy`), assegurando consistência nos resultados.
- `N_ARTIGOS_COLETA = 18000`: Quantidade inicial de artigos pré-selecionados da ACL Anthology via pontuação lexical simples.
- `QUERY_TERMS`: Termos utilizados para a filtragem inicial (`"retrieval augmented generation"`, `"information retrieval"`, `"large language model"`).
- `ESCALAS_CORPUS = [500, 1000]`: Tamanhos do corpus de artigos avaliados no experimento de benchmark.
- `ESCALAS_CANDIDATOS = [20, 50]`: Quantidade de artigos candidatos ($N$) pré-selecionados pela recuperação lexical que chegam como itens para a mochila.
- `TOKENIZADOR_NOME = 'cl100k_base'`: Codificação do `tiktoken` para contagem real do custo de tokens de cada artigo (título + resumo).
- `ORCAMENTOS (W)`: Capacidades da mochila calculadas a partir da mediana dos custos de tokens ($3 \times$ e $6 \times$ a mediana empírica, e.g., $W_1 = 708$ e $W_2 = 1416$ tokens).
- `CONSULTAS_EXPERIMENTO = ['Q1', 'Q3', 'Q5']`: Consultas benchmark utilizadas para as medições comparativas.
- `REPETICOES = 3`: Número de repetições por cenário no benchmark (totalizando 216 execuções mensuráveis com monitoramento de tempo e consumo de memória com `tracemalloc`).
- `MODELO_SEMANTICO = 'sentence-transformers/all-MiniLM-L6-v2'`: Modelo para geração de embeddings e gabarito semântico para cálculo das métricas de precisão.

## Como Reproduzir

Para reproduzir este experimento, siga os passos abaixo:

1. Faça o clone deste repositório ou baixe o arquivo `PAA_AV2_Selecao_tokens_DP.ipynb`.
2. Acesse o [Google Colab](https://colab.research.google.com/).
3. Faça o upload do arquivo `PAA_AV2_Selecao_tokens_Mochila01_DP.ipynb` para a plataforma.
4. No menu superior, clique em **Ambiente de Execução** (Runtime) > **Executar tudo** (Run all).
5. A primeira célula cuidará da instalação das dependências. A execução seguirá o fluxo:
   - **Instalação e configurações**: Inicialização de diretórios (`dados/`, `resultados_av2/`) e configuração de sementes e hiperparâmetros.
   - **Carregamento do Corpus**: Leitura do CSV da AV1 (`dados/acl_corpus_av1.csv`) ou download direto dos artigos da ACL Anthology.
   - **Contagem e Custos em Tokens**: Tokenização via `tiktoken` e análise da distribuição estatística dos custos por documento.
   - **Definição da Mochila e Orçamentos ($W$)**: Cálculo das capacidades de corte baseadas na mediana dos custos em tokens.
   - **Validação de Entrada e Formato**: Testes prévios de consistência de dados e contrato de retorno dos algoritmos.
   - **Busca Lexical e Candidatos**: Normalização e índice invertido para filtrar os $N$ candidatos mais relevantes para cada consulta.
   - **Métodos de Seleção**: Execução do Ranking AV1, Heurística Gulosa (densidade de valor) e Programação Dinâmica (Mochila 0/1 com reconstrução de itens).
   - **Testes de Corretude e Casos de Borda**: Validação contra Busca Exaustiva em instâncias controladas e teste de contraexemplo do algoritmo guloso.
   - **Prova Formal de Corretude**: Justificativa por indução matemática da subestrutura ótima e sobreposição de subproblemas na DP.
   - **Benchmark Experimental**: Execução das 216 medições automatizadas coletando tempo de execução, pico de memória (`tracemalloc`), valor total, precisão e redundância (Jaccard).
   - **Análise Assintótica**: Medição prática da pseudo-polinomialidade aumentando $W$ com $N$ constante.
   - **Montagem do Contexto RAG**: Geração do prompt formatado com os artigos selecionados e análise das limitações práticas.
   - **Reprodutibilidade e Metadados**: Exportação dos gráficos gerados e do arquivo `metadados.json` com especificações do ambiente e hash do corpus.
