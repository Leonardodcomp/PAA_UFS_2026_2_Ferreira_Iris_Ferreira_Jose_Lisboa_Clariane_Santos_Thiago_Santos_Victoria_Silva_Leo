# Projeto e Análise de Algoritmos: Recuperacao-contexto_RAG-articles

Atividade proposta da disciplina de PAA 2026.2   


Este projeto realiza uma análise de algoritmos focada na recuperação eficiente de contexto em artigos científicos sobre Modelos de Linguagem (LLMs), Recuperação de Informação (RI) e RAG (Retrieval-Augmented Generation).

Utilizamos o repositório da ACL ANTHOLOGY.

## Link video apresentação

https://drive.google.com/file/d/1tM8qdJzN7ZpnDLzEfTQ7DrRIJ9Djf31t/view?usp=sharing

## Equipe (Discentes)

- Clariane Maria Oliveira Lisboa
- Iris Azenate Ferreira
- José Nilton dos Santos Ferreira
- Leonardo Souza Silva
- Thiago de Jesus Santos
- Victória Cardoso dos Santos

## Ambiente Recomendado

O projeto foi desenvolvido em formato Jupyter Notebook (`.ipynb`) e é altamente recomendado que seja executado no **Google Colab**. Uma vez que o notebook já contém as diretivas de ambiente necessárias para baixar os pacotes diretamente no runtime do Colab, eliminando a necessidade de configuração complexa de ambiente virtual local.

## Dependências

As bibliotecas externas necessárias para a execução do projeto foram deixadas prontas para instalação automática na primeira célula do notebook. São elas:

- `requests` (atualizado via pip)
- `acl-anthology` (coleta do corpus de artigos)
- `sentence-transformers` (geração de embeddings com o modelo `all-MiniLM-L6-v2` para o gabarito semântico)
- `nltk` (tokenização, normalização e stemming)
- `pandas` e `numpy` (manipulação de dados)
- `matplotlib` (visualização de gráficos)
- `scikit-learn` (cálculo de similaridade de cosseno)

##  Parâmetros do Experimento

O notebook foi configurado para garantir a reprodutibilidade dos testes. Os principais parâmetros utilizados na coleta e na análise são:
- `SEED = 42`: Fixado para garantir consistência nos resultados que dependem de aleatoriedade.
- `N_ARTIGOS_COLETA = 18000`: Quantidade inicial de artigos pré-selecionados da ACL Anthology via pontuação lexical simples.
- `QUERY_TERMS`: Termos utilizados para a filtragem inicial ("retrieval augmented generation", "information retrieval", "large language model").
- `TAMANHOS = [500, 1000]`: Tamanhos do corpus (número de chunks/artigos) testados durante o experimento de benchmark.
- `REPETICOES = 30`: Número de execuções mensuráveis para cada cenário (garantindo média estatística de tempo e comparações).
- `K = 5`: Número de documentos retornados para a métrica de avaliação (Precision@5).

## Como Reproduzir

Para reproduzir este experimento, siga os passos abaixo:

1. Faça o clone deste repositório ou baixe o arquivo `.ipynb`.
2. Acesse o [Google Colab](https://colab.research.google.com/).
3. Faça o upload do arquivo `.ipynb` para a plataforma.
4. No menu superior, clique em **Ambiente de Execução** (Runtime) > **Executar tudo** (Run all).
5. A primeira célula cuidará da instalação das dependências. A execução seguirá o fluxo:
   - Coleta de dados (ACL Anthology).
   - Normalização e Tokenização (Stemming).
   - Criação do gabarito semântico usando embeddings.
   - Execução dos testes automatizados de borda e corretude (Merge Sort).
   - Execução do benchmark (1800 execuções mensuráveis).
   - Geração das tabelas e gráficos comparativos de Tempo, Comparações e Memória.
