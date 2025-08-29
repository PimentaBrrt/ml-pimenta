# Modelo de Machine Learning - Árvore de Decisões

Para esse projeto, foi utilizado um dataset obtido no [**Kaggle**](https://kaggle.com){:target='_blank'}.
Os dados usados podem ser baixados [**aqui**](https://www.kaggle.com/datasets/dalmacyali1905/game-of-thrones-classification-decision-tree?resource=download){:target='_blank'}.

## Objetivo

O dataset apresenta diversos dados relacionados à cada um dos personagens da série de livros [**A Song of Ice and Fire**](https://en.wikipedia.org/wiki/A_Song_of_Ice_and_Fire){:target='_blank'}, escrita por [**George R. R. Martin**](https://en.wikipedia.org/wiki/George_R._R._Martin){:target='_blank'}, inspiração para a famosa série [**Game of Thrones**](https://en.wikipedia.org/wiki/Game_of_Thrones){:target='_blank'}.
O objetivo dessa análise é o modelo fazer a predição da importância do personagem para a série no sentido de trama. Uma variável categórica será criada a partir das variáveis presentes no dataset, classificando a relevância do personagem. Essa variável será avaliada pelo modelo de Machine Learning.

## Workflow

Os pontos *"etapas"* são o passo-a-passo da realização do projeto.

### Etapa 1 - Exploração de Dados

O dataset escolhido é composto por **1946 linhas** e **30 colunas**, contendo um personagem distinto em cada linha e diversas informações sobre cada um.

#### Colunas do dataset

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| S.No | Inteiro | Identificador único do personagem |
| plod | Float | Valor não especificado |
| name | String | Nome do personagem |
| title | String | Alcunha atribuída ao personagem dentro do mundo |
| gender | Binário | Sexo do personagem: 0 = feminino, 1 = masculino |
| culture | String | Grupo social ao qual o personagem pertence |
| dateOfBirth | Inteiro | Data de nascimento. Valores positivos = depois do ano 0, negativos = antes do ano 0 |
| DateoFdeath | Inteiro | Data de morte. Valores positivos = depois do ano 0, negativos = antes do ano 0 |
| mother | String | Nome da mãe do personagem |
| father | String | Nome do pai do personagem |
| heir | String | Nome do herdeiro do personagem |
| house | String | Nome da casa à qual o personagem pertence |
| spouse | String | Nome do cônjuge do personagem |
| book1 | Binário | Indica se o personagem apareceu no primeiro livro |
| book2 | Binário | Indica se o personagem apareceu no segundo livro |
| book3 | Binário | Indica se o personagem apareceu no terceiro livro |
| book4 | Binário | Indica se o personagem apareceu no quarto livro |
| book5 | Binário | Indica se o personagem apareceu no quinto livro |
| isAliveMother | Binário | Indica se a mãe do personagem está viva |
| isAliveFather | Binário | Indica se o pai do personagem está vivo |
| isAliveHeir | Binário | Indica se o herdeiro do personagem está vivo |
| isAliveSpouse | Binário | Indica se o cônjuge do personagem está vivo |
| isMarried | Binário | Indica se o personagem é casado |
| isNoble | Binário | Indica se o personagem é nobre |
| age | Inteiro | Idade do personagem (referência: ano 305 D.C.) |
| numDeadRelations | Inteiro | Número de personagens mortos com os quais o personagem se relaciona |
| boolDeadRelations | Binário | Indica se há personagens mortos relacionados ao personagem |
| isPopular | Binário | Indica se o personagem é considerado popular |
| popularity | Float | Índice entre 0 e 1 que indica o quão popular é o personagem |
| isAlive | Binário | Indica se o personagem está vivo |

#### Estudo da coluna **`plod`**

No dataset, temos uma coluna que possui um índice que aponta algo não identificado: o **plod**. Para investigar seu significado, são necessárias algumas análises:

- **Inspeção dos valores:** Primeiro, foram realizadas algumas linhas de código para verificar os valores da coluna;

=== "Saída"

    ```python exec="1"
    --8<-- "docs\decision-tree\plod-teste1.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs\decision-tree\plod-teste1.py"
    ```

A análise da saída obtida nos permite observar que os valores estão sempre no intervalo [0,1], sugerindo que representam uma probabilidade ou índice normalizado.

- **Correlações entre `plod` e as outras colunas:** Levando isso em consideração, é necessário realizar um cálculo de correlações para descobrir a principal variável no cálculo do **plod**:

=== "Saída"

    ```python exec="1"
    --8<-- "docs\decision-tree\plod-teste2.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs\decision-tree\plod-teste2.py"
    ```

É possível observar que a correlação mais forte entre **plod** e qualquer outra coluna no dataset é com a coluna **isAlive**. Esse dado nos permite criar uma **hipótese** de que **plod** é a *estimativa da probabilidade de morte do personagem*.

- **Comparação com a coluna `isAlive`:** Em seguida, para verificar a hipótese estabelecida, será feito um gráfico de boxplot para analisar a relação de **plod** e **isAlive**;

=== "Gráfico"

    ```python exec="on" html="1"
    --8<-- "docs\decision-tree\plod-teste3.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs\decision-tree\plod-teste3.py"
    ```

No gráfico, observa-se que personagens vivos (isAlive = 1) tendem a possuir baixos valores de plod, enquanto personagens mortos (isAlive = 0) geralmente têm valores altos. Além disso, é possível observar diversos outliers dentre os personagens vivos, que são provavelmente personagens que aparecem pouco e/ou possuem informações incompletas. Essa ideia é fortalecida pelo fato de que a variável **popularity** (popularidade) também possui correlação moderada com **plod**.

Portanto, os padrões do gráfico indicam, novamente, que **plod** funciona como uma estimativa da probabilidade de morte do personagem, reforçando a hipótese inicial. Essa coluna provavelmente foi calculada com algum modelo preditivo anterior.

É necessário ressaltar que isso é uma **observação exploratória**, baseada nos dados disponíveis, e *será considerada* no pré-processamento e na escolha das features do modelo.

#### Exploração aprofundada da coluna **`popularity`**

- **Estatísticas descritivas:** Primeiramente, vamos calcular alguns valores essenciais dessa coluna;

=== "Saída"

    ```python exec="1"
    --8<-- "docs\decision-tree\pop-stats.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs\decision-tree\pop-stats.py"
    ```

Na saída, observa-se que **popularity** é um índice que indica a popularidade do personagem, variando entre 0 e 1, com o valor 0 para irrelevante e 1 para popular.

- **Gráfico de dispersão de `popularity`:** O gráfico relaciona o índice **popularity** com a soma das 5 variáveis **book**, que indicam a presença de um personagem em cada livro em binário. Os livros considerados nessas variáveis são apenas a narrativa principal da história, sem spin-offs e personagens que são apenas citados e referenciados.

=== "Gráfico"

    ```python exec="on" html="1"
    --8<-- "docs\decision-tree\pop-scatter.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs\decision-tree\pop-scatter.py"
    ```

A análise do gráfico indica que, de 1 até 5 aparições, o número de personagens populares **aumenta** de forma diretamente proporcional, com alguns out-liers.

Contudo, podemos observar que diversos personagens que não apareceram na *série principal de livros*, possuindo soma de aparições igual a 0, são extremamente populares. Isso acontece pois há personagens de spin-offs muito amados pela comunidade, além de outros personagens que são apenas citados ao longo da história, sem aparecer diretamente, e também adquirem alta popularidade.

### Etapa 2 - Pré-processamento

O objetivo do projeto é realizar uma predição da relevância dos personagens na trama principal, a variável categórica **relevance** que possuirá as seguintes categorias: Low, Medium, High e Very High. 

#### 1° Passo: Criação de **`book_freq`**

Primeiramente, é importante criar uma variável representante para a frequência em livros para cada personagem. Ao invés de utilizar 5 variáveis **book** diferentes, criaremos a variável **boof_freq**. Para isso, será feita a soma dos 5 valores das variáveis **book**, o que resultará em um intervalo de [0,5]. Contudo, para que esse valores sejam normalizados, e possuam um número entre 0 e 1, é feita a divisão desse resultado por 5.

``` python 
    
df["book_freq"] = df[["book1", "book2", "book3", "book4","book5"]].sum(axis=1) / 5

```

#### 2° Passo: Seleção de colunas

Em seguida, é necessário definir quais são as variáveis serão utilizadas para prever a relevância. Elas são as seguintes:

- **plod:** Se esse valor for alto, há maior chance do personagem ser irrelevante 

- **title:** Se o personagem tiver um título, qualquer que seja, já possui alguma relevância 

- **culture:** Se o personagem tiver alguma cultura, qualquer que seja, já possui alguma relevância 

- **mother:** Se o personagem tiver paretentesco revelado, provavelmente tem alguma importância 

- **father:** Se o personagem tiver paretentesco revelado, provavelmente tem alguma importância 

- **heir:** Se o personagem tiver paretentesco revelado, provavelmente tem alguma importância 

- **house:** Se o personagem tiver uma casa, deve ser mais relevante 

- **book_freq:** Se o personagem aparece frequentemente, deve ser relevante 

- **isNoble:** Se o personagem for um nobre, tem mais chances de ser importante 

- **popularity:** Se o personagem for popular, também aumenta sua chance de relevância 

A seleção das colunas foi feita, em código, da seguinte forma:
``` python 
    
cols = ["plod", "title", "culture", "mother", "father", "heir", "house",
"book_freq", "isNoble", "popularity"]

df = df[cols]

```

#### 3° Passo: Tratamento de valores faltantes

Precisamos garantir que não existam valores faltantes no dataframe. Por isso, será feita uma alteração em todas as linhas restantes que possuem valor *NA*. Contudo, temos que tratar diferentemente cada tipo de variável para o preenchimento dos vazios. As regras utilizadas serão as seguintes:

- Faltantes númericos serão preenchidos com a mediana da coluna - **plod**, **popularity**

- Faltantes categóricos nominais serão preenchidos com *"Unknown"* (Desconhecido) - **title**, **culture**, **mother**, **father**, **heir**, **house**

- Faltantes binários serão preenchidos com a moda da coluna (valor mais frequente) - **isNoble**

``` python

cols = ["plod", "popularity"]
for col in cols:
    df.fillna({col: df[col].median()}, inplace=True)

cols = ["title", "culture", "mother", "father", "heir", "house"]
for col in cols:
    df.fillna({col: "Unknown"}, inplace=True)

df.fillna({"isNoble": df["isNoble"].mode()[0]}, inplace=True)

```

#### 4° Passo: Binarização dos categóricos nominais

As variáveis categóricas no dataframe tem, simplesmente, muitas categorias para a realização de Label ou One-hot Encoding. Além disso, o único dado importante provindo dessas no modelo sendo criado é se existem ou não essas informações sobre o personagem. Portanto, as colunas **title**, **culture**, **mother**, **father**, **heir** e **house** serão binarizadas. Ou seja, se possuírem um valor, assumirão o valor 1. Caso contrário, 0. 

Além disso, os nomes das colunas serão alterados, adicionando um "has_" antes do nome original da variável.

``` python

cols = ["title", "culture", "mother", "father", "heir", "house"]

for col in cols:
    df[f"has_{col}"] = (df[col] != "Unknown").astype(int)
    df.drop(columns=[col], inplace=True)

```

#### 5° Passo: Inversão e renomeação de **`plod`**

A variável **plod**, que indica a probabilidade de morte, possui uma relação *inversamente proporcional* à relevância do personagem. Por isso, é necessária a inversão dessa variável. Além disso, renomear a variável para **survival_prob** deixará mais claro o seu propósito.

``` python

df["survival_prob"] = 1 - df["plod"]
df.drop(columns="plod", inplace=True)

```

#### 6° Passo: Criação da variável target **`relevance_category`** a partir do score **`relevance_score`**

Agora, precisamos criar a variável categórica que será avaliada pelo modelo. Utilizaremos a seguinte distribuição de pesos:

- **popularity:** Popularidade - 25%

- **book_freq:** Frequência de aparições - 25%

- **survival_prob:** Probabilidade de sobrevivência - 15%

- **isNoble:** É nobre - 10%

- **has_title:** Tem um título - 10%

- **has_house:** Possui uma casa - 5%

- **has_culture:** Tem uma cultura - 5%

- **has_mother** + **has_father** + **has_heir:** Possui parentesco - 5%

Com o **relevance_score** definido, criaremos a nova coluna, **relevance_category**, a partir dos seguintes valores de score:

- **x < 0.25:** *Low* (Baixa relevância)

- **0.25 <= x < 0.5:** *Medium* (Relevância média)

- **0.5 <= x < 0.75:** *High* (Alta relevância)

- **0.75 <= x <= 1:** *Very High* (Relevância muito alta)

#### Resultado final do pré-processamento

=== "Saída"

    ```python exec="1"
    --8<-- "docs\decision-tree\preprocessing.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs\decision-tree\preprocessing.py"
    ```

### Etapa 3 - Divisão de dados

Na etapa de divisão de dados, separaremos o conjunto de dados processado em dois grupos distintos:

- **Conjunto de Treino:** É utilizado para ensinar o modelo a reconhecer padrões

- **Conjunto de Teste:** É utilizado para avaliar o desempenho do modelo com dados ainda não vistos

Para realizar a divisão, utilizaremos a função `train_test_split()` do `scikit-learn`. Os parâmetros utilizados serão:

- **test_size=0.2:** Define que 20% dos dados serão utilizados para teste, enquanto o restante será usado para treino.

- **random_state=42:** Parâmetro que controla o gerador de número aleatórios utilizado para sortear os dados antes de separá-los. Garante reprodutibilidade.

- **stratify=y:** Esse atributo definido como *y* é essencial devido à natureza da coluna **relevance_category**. Com essa definição, será mantida a mesma proporção das categorias em ambos os conjuntos, reduzindo o viés.

=== "Saída"

    ```python exec="1"
    --8<-- "docs\decision-tree\division.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs\decision-tree\division.py"
    ```

Os dados, agora, estão devidamente divididos. Esta divisão adequada é de extrema importância, pois ajuda a evitar *overfitting* e garante que o modelo possa generalizar bem para novos personagens não vistos durante o treinamento.

### Etapa 4 - Treinamento do Modelo

Agora, será realizado o treinamento do modelo. O objetivo dessa etapa é ensinar o algoritmo a reconhecer padrões nos dados que são fornecidos, e determinar a importância narrativa de cada personagem na *série principal* de livros de [**A Song of Ice and Fire**](https://en.wikipedia.org/wiki/A_Song_of_Ice_and_Fire){:target='_blank'}.

=== "Gráfico"

    ```python exec="on" html="1"
    --8<-- "docs\decision-tree\training.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs\decision-tree\training.py"
    ```

### Etapa 5 - Avaliação do modelo

#### Acurácia do modelo

O modelo alcançou uma acurácia **impressionante** de 95,13% no conjunto teste, demonstrando uma ótima capacidade de previsão com personagens ainda não vistos com base nas features escolhidas.

#### Importância das features

A análise da importância das features revela quais foram as variáveis mais importantes para a previsão e decisões do modelo:

| Feature | Importância | Descrição |
|--------|------|-----------|
| `book_freq` | 43,98% | Frequência de aparição nos livros da série principal |
| `survival_prob` | 20,09% | Probabilidade de sobrevivência |
| `isNoble` | 14,62% | Status nobre |
| `popularity` | 10,23% | Índice de popularidade |
| `has_culture` | 5,41% | Possui cultura conhecida |
| `has_house` | 3,48% | Pertence a uma casa |
| `has_title` | 2,19% | Tem algum título |
| `has_mother` | 0,00% | Há informação sobre a mãe |
| `has_father` | 0,00% | Há informação sobre o pai |
| `has_heir` | 0,00% | Possui herdeiro |

#### Insights importantes sobre o modelo

- **Frequência em livros é determinante:** A feature `book_freq` responde à aproximadamente 44% da importância, confirmando que personagens com mais aparições em diferentes livros da série principal possuem maior importância.

- **Features desnecessárias:** O modelo também demonstrou que algumas features (has_mother, has_father, has_heir) não têm *nenhuma importância* na predição.

### Etapa 6 - Relatório Final

O projeto geral foi um sucesso, com a obtenção de um modelo com uma acurácia de 95,13%. O modelo, além de alta performance, possui features relevantes identificadas e bem estabelecidas: `book_freq`, `survival_prob` e `isNoble`.

#### Limitações do modelo

Contudo, há limitações no modelo:

- **Features Redudantes:** **has_mother**, **has_father** e **has_heir** possuem importância nula para a predição do sistema

- **Possível viés:** É possível que, pela variável **relevance_score** ter sido manualmente estabelecida, pode haver viés

#### Possíveis melhorias

- **Validação de `plod`:** Durante a primeira etapa, na exploração da base de dados, poderia ter sido feita uma *Regressão Linear Múltipla* completa para validar completamente a hipótese de que **plod** é a probabilidade de morte do personagem.

- **Remoção de features desnecessárias:** As features relacionadas à parentesco podem ser removidas do modelo sem nenhum impacto na predição.

#### Considerações finais

A árvore de decisão se mostrou muito capaz de fazer a predição de narrativas literárias complexas como [**A Song of Ice and Fire**](https://en.wikipedia.org/wiki/A_Song_of_Ice_and_Fire){:target='_blank'}. Além do excelente resultado de acurácia, foram providos insights importantes sobre a obra pelo modelo. 

Além disso, foi possível observar que tanto a *Etapa 1* quanto a *Etapa 2* foram muito mais longas do que as posteriores, demonstrando a importância de entender e limpar o dataset antes do uso.