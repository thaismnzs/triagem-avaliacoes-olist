# Triagem de avaliações de clientes (Olist)

Classificador de texto em português que identifica **avaliações negativas** de clientes de e-commerce, para ajudar a priorizar o atendimento.

Projeto de estudo em Machine Learning, desenvolvido em Python com scikit-learn.

## Problema

Equipes de atendimento recebem muitas avaliações e mensagens, e as reclamações ficam misturadas com elogios. A pergunta deste projeto é: **dado o texto de uma avaliação, é possível identificar automaticamente se ela é negativa?**

Defini a avaliação como **negativa** quando a nota é 1 ou 2.

**Métrica principal: recall da classe negativa.** Deixar passar uma reclamação custa mais do que revisar uma avaliação a mais, então priorizei encontrar o máximo de reclamações reais. Também acompanho precision e F1.

## Dados

- **Fonte:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle)
- **Arquivo usado:** `olist_order_reviews_dataset.csv`
- **Amostra:** 40.977 avaliações com comentário escrito (linhas sem comentário foram removidas)
- **Distribuição das notas:**

| Nota | Quantidade |
|---|---|
| 1 | 8.745 |
| 2 | 2.145 |
| 3 | 3.557 |
| 4 | 5.976 |
| 5 | 20.554 |

As notas 1 e 2 somam 10.890 avaliações (cerca de 27%), então os dados são **desbalanceados**. Por isso a acurácia sozinha engana: um modelo que dissesse "ninguém reclama" acertaria cerca de 73% sem aprender nada.

> O dataset não está incluído neste repositório. Baixe pelo link acima.

## Abordagem

1. **Divisão treino/teste:** 80% / 20%, estratificada pela classe, com `random_state=42` (32.781 avaliações de treino e 8.196 de teste).
2. **Baseline:** TF-IDF com unigramas e bigramas (`ngram_range=(1, 2)`, `min_df=3`, `max_features=50000`) + Regressão Logística com `class_weight="balanced"`.
3. **Comparação:** o mesmo TF-IDF + Linear SVM (`LinearSVC`, `class_weight="balanced"`).
4. **Avaliação:** relatório de classificação no conjunto de teste e análise manual de erros.

Os bigramas importam porque, em textos curtos, "recebi" sozinha é positiva e "não recebi" é reclamação.

## Resultados

Desempenho na classe **negativa**, no conjunto de teste (8.196 avaliações, sendo 2.178 negativas):

| Modelo | Precision | Recall | F1 |
|---|---|---|---|
| **Regressão Logística (baseline)** | 0,73 | **0,93** | **0,82** |
| Linear SVM | 0,76 | 0,88 | 0,81 |

A acurácia ficou em 0,89 nos dois modelos.

**Modelo escolhido: Regressão Logística.** O SVM teve precision um pouco maior, mas deixou passar mais reclamações (recall 0,88 contra 0,93). Como a métrica principal foi definida antes dos experimentos, o modelo mais simples foi o escolhido.

## Análise de erros

Li 12 exemplos de cada tipo de erro do modelo escolhido. São amostras pequenas: elas indicam padrões, mas não permitem estimar quanto cada um pesa no total.

**Reclamações não detectadas** (nota 1 ou 2, modelo previu "não negativa"):

- **6 de 12 eram queixas reais** com poucas palavras negativas, como "a alça quebrou sem muito peso", "compramos 5 pufs mas só foram entregues 4" e "o relógio não veio com manual". Havia também um "não recebi" no meio de um texto longo e elogioso.
- **4 de 12 eram comentários positivos com nota baixa**, como "Gostei muito" e "Sai mto rapido", provavelmente notas marcadas por engano.
- **2 de 12 eram mistos**, como "mas poderia ter sido mais rápido".

**Falsos alarmes** (nota 3 a 5, modelo previu "negativa"):

- **11 de 12 eram reclamações reais** com nota média ou alta, como "O produto nunca apareceu em casa!", "Não chegou o meu produto" e "Preciso que vcs resolvam o meu problema de trocar o produto".
- **1 de 12 foi erro de fato do modelo**: um comentário positivo sobre um produto para uma criança.

**O que isso indica:** a nota usada como rótulo tem ruído nos dois sentidos. Na amostra lida, boa parte dos "falsos alarmes" eram reclamações genuínas, então a precision de 0,73 provavelmente **subestima** o desempenho real do modelo para identificar reclamações. Isso precisa ser confirmado com rotulagem manual (veja os próximos passos).

## Limitações

- A nota é um **rótulo aproximado** de reclamação: nem toda nota baixa é queixa do serviço, e nem toda queixa vem com nota baixa.
- A análise de erros usou amostras de 12 casos por tipo.
- Os dados vêm de um único marketplace brasileiro, em um período específico.
- O modelo não trata ironia, gírias e abreviações (por exemplo, "mto").
- O modelo diz apenas se a avaliação é negativa. Não identifica o tipo de problema (atraso, defeito, produto errado).

## Como reproduzir

1. Crie um notebook no [Kaggle](https://www.kaggle.com) e adicione o dataset **Brazilian E-Commerce Public Dataset by Olist** (autor: Olist) em *Add Input*.
2. Abra o arquivo `notebook22a893d03f.ipynb` deste repositório (ou copie as células para o notebook novo).
3. Ajuste o caminho do CSV se necessário. No Kaggle ele costuma ser `/kaggle/input/datasets/olistbr/brazilian-ecommerce/olist_order_reviews_dataset.csv`.
4. Execute as células em ordem.

**Bibliotecas:** `pandas`, `scikit-learn` (já instaladas no ambiente do Kaggle).

## Próximos passos

- [ ] Rotular manualmente cerca de 200 comentários para medir o desempenho sem o ruído da nota.
- [ ] Fine-tuning do BERTimbau (modelo de linguagem em português) e comparação com o baseline, usando a mesma divisão de dados.
- [ ] Publicar o modelo no Hugging Face com model card.
- [ ] Criar uma API com FastAPI e Docker para servir as previsões.


