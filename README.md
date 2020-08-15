# ml-laboratorio-02
Laboratório 02 da disciplina de Machine Learning do PPGINF.

## Objetivo

Dado um conjunto de amostras para treinamento (composto por 20 mil elementos) e um conjunto de testes para classificação (composto por aproximadamente 58 mil elementos), propõe-se experimentar o treinamento com quantidades diferentes de amostras (1000, 2000, ..., 20000) e testar todos os elementos do conjunto de testes, para cada um dos cinco classificadores lineares a seguir:

- *K Nearest Neighbor (KNN)*;
- *Naive Bayes*;
- *Linear Discriminant Analisys (LDA)*;
- *Logistic Regression* e;
- *Perceptron*.

O objetivo é comparar o desempenho e os resultados para cada arranjo de classificador e quantidade de amostras utilizadas.

## Enunciado

> ## Impactos da Base de Aprendizagem
>
> Para esse laboratório considere os seguintes classificadores:
>
> - KNN
> - Naïve Bayes
> - Linear Discriminant Analysis
> - Logistic Regression
> - Perceptron
>
> Considere também as base de treinamento (20000 exemplos) e teste (58646 exemplos), as quais contem 10 classes balanceadas e 132 características.
>
> Escreva um breve relatório que:
>
> 1. Compare o desempenho desses classificadores em função da disponibilidade de base de treinamento. Alimente os classificadores com blocos de 1000 exemplos e plote num gráfico o desempenho na base de testes. Analise em qual ponto o tamanho da base de treinamento deixa de ser relevante.
> 2. Indique qual é o classificador que tem o melhor desempenho com poucos dados < 1000 exemplos.
> 3. Indique o classificador que tem melhor desempenho com todos os dados.
> 4. Indique o classificador mais rápido para classificar os 58k exemplos de teste.
> 5. Analise as matrizes de confusão. Os erros são os mesmos para todos os classificadores quando todos eles utlizam toda a base de teste?
>
> O relatório reportando seus experimentos deve entregue em formato PDF.

## Estrutura de Arquivos e Pastas

- 📂dados
  - test.txt
  - train.txtx
- 📂out
  - [classificador]_[qtde_amostras].csv
  - resultados.csv
- main.py
- relatorio.pdf

A pasta **dados** contém a base de treinamento e testes, fornecidas pelo professor. Já está em arquivo texto os exemplos em forma de vetores com 132 características.

A pasta **out** armazena os resultados em formato CSV. Um arquivo csv para cada matriz de confusão e um arquivo resultados.csv com uma tabela contendo a identificação do experimento, tempo de treinamento, tempo de testes, acurácia e f1-score. O conteúdo da pasta é apagado e reescrito a cada nova execução.

O arquivo main.py contém o script em python.

## Dependências

- Biblioteca scikit-learn para execução dos treinos e validação;
- Biblioteca pandas, usada para facilmente para gerar arquivos csv;

## Modo de Funcionamento
Execute main.py e verifique as saídas na pasta out.
