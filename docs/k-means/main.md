# Modelo de Machine Learning - K-Means

Para esse projeto, foi utilizado um dataset obtido no [**Kaggle**](https://kaggle.com){:target='_blank'}.
Os dados usados podem ser baixados [**aqui**](https://www.kaggle.com/datasets/youssefaboelwafa/hotel-booking-cancellation-prediction/data){:target='_blank'}.

A mesma base de dados utilizada para o projeto de KNN está sendo utilizada aqui. Contudo, esse projeto utilizará apenas a lógica do *Modelo 2* estudada no projeto anterior. Ou seja, evitaremos **data leakage** e será aplicado o pré-processamento apenas no conjunto de treino. Mais um ponto importante de esclarecer é que, como a base de dados segue sendo a mesma, a exploração de dados, pré-processamento e divisão de dados serão basicamente a mesma coisa, mas agora, apenas com a lógica do *Modelo 2*.

## Objetivo

O dataset utilizado possui informações sobre reservas em um hotel, e foi criado justamente para a criação de modelos de machine learning com o objetivo de prever se um agendamento será, ou não, cancelado.

Os dados originais para a criação desse dataset foram obtidos em um artigo de dados, no site [**Science Direct**](https://www.sciencedirect.com/){:target='_blank'}. O [**artigo em questão**](https://www.sciencedirect.com/science/article/pii/S2352340918315191) foi escrito por *Nuno Antonio*, *Ana de Almeida* e *Luis Nunes*, e contém uma quantidade maior de dados do que a sua versão derivada do **Kaggle**, que estou utilizando para este projeto.

## Workflow

Os pontos *"etapas"* são o passo-a-passo da realização do projeto.

### Etapa 1 - Exploração de Dados

Primeiramente, deve ser feita a exploração dos dados da base, com o objetivo de compreender a forma como são estruturados os dados, sua natureza e possível significância para o modelo de predição.

O dataset é composto por **36285 linhas** e **17 colunas**, com cada linha representando uma reserva distinta. Essa verificação pôde ser feita com as linhas de código abaixo;

=== "Saída"

    ```python exec="1"
    --8<-- "docs/knn/exploring-knn.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs/knn/exploring-knn.py"
    ```

#### Colunas do dataset

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| Booking_ID | String | Identificador único da reserva |
| number of adults | Inteiro | Número de adultos presentes na reserva |
| number of children | Inteiro | Número de crianças presentes na reserva |
| number of weekend nights | Inteiro | Quantidade de noites em finais de semana reservadas |
| number of week nights | Inteiro | Quantidade de noites em dias de semana reservadas |
| type of meal | String | Plano de alimentação escolhido pelo cliente |
| car parking space | Inteiro | Variável binária que indica se um estacionamento de carro foi pedido ou incluso na reserva |
| room type | String | Tipo de quarto reservado |
| lead time | Inteiro | Número de dias entre a data da reserva e a data de chegada do cliente |
| market segment type | String | Tipo de segmento do mercado associado à reserva |
| repeated | Inteiro | Variável binária que indica se a reserva é, ou não, repetida |
| P-C | Inteiro | Número de reservas anteriores que foram canceladas pelo cliente antes do agendamento atual |
| P-not-C | Inteiro | Número de reservas anteriores que não foram canceladas pelo cliente antes do agendamento atual |
| average price | Float | Preço médio associado à reserva |
| special requests | Inteiro | Número de pedidos especiais feitos pelo convidado(a) |
| date of reservation | String | Data da reserva |
| booking status | String | Status da reserva (cancelada ou não cancelada) |

#### Visualizações das variáveis

Em seguida, é essencial realizar gráficos para visualizar como cada uma das variáveis se comportam, com o objetivo de entender melhor a base da dados.

Está seção será divida para cada tipo de variável, entre variáveis quantitativas discretas, quantitativas contínuas, qualitativas categóricas, binárias e, por fim, a variável alvo.

##### Variáveis Quantitativas Discretas

=== "number of adults"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/knn/visualizations/number_adults.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/knn/visualizations/number_adults.py"
        ```

=== "number of children"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/knn/visualizations/number_children.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/knn/visualizations/number_children.py"
        ```

=== "number of weekend nights"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/knn/visualizations/weekend_nights.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/knn/visualizations/weekend_nights.py"
        ```

=== "number of week nights"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/knn/visualizations/week_nights.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/knn/visualizations/week_nights.py"
        ```

=== "lead time"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/knn/visualizations/lead_time.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/knn/visualizations/lead_time.py"
        ```

=== "P-C"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/knn/visualizations/p_c.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/knn/visualizations/p_c.py"
        ```

=== "P-not-C"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/knn/visualizations/p_not_c.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/knn/visualizations/p_not_c.py"
        ```

=== "special requests"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/knn/visualizations/special_requests.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/knn/visualizations/special_requests.py"
        ```

##### Variável Quantitativa Contínua **`average price`**

=== "Gráfico"

    ``` python exec="1" html="1"
    --8<-- "docs/knn/visualizations/average_price.py"
    ```

=== "Código"

    ``` python exec="0"
    --8<-- "docs/knn/visualizations/average_price.py"
    ```

##### Variáveis Categóricas

=== "type of meal"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/knn/visualizations/type_meal.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/knn/visualizations/type_meal.py"
        ```

=== "room type"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/knn/visualizations/type_room.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/knn/visualizations/type_room.py"
        ```

=== "market segment type"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/knn/visualizations/type_market.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/knn/visualizations/type_market.py"
        ```

##### Variáveis Binárias

=== "car parking space"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/knn/visualizations/car_parking.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/knn/visualizations/car_parking.py"
        ```

=== "repeated"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/knn/visualizations/repeated.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/knn/visualizations/repeated.py"
        ```

##### Variável Alvo **`booking status`**

=== "Gráfico"

    ``` python exec="1" html="1"
    --8<-- "docs/knn/visualizations/booking_status.py"
    ```

=== "Código"

    ``` python exec="0"
    --8<-- "docs/knn/visualizations/booking_status.py"
    ```

Através das análises, foi possível alcançar uma compreensão mais aprofundada do funcionamento de cada uma das variáveis no dataset, além de haver insights valiosos nesses gráficos. Esses dados serão essenciais para a escolha das variáveis que serão utilizadas no modelo.

### Etapa 2 - Pré-processamento e Divisão de Dados

Nesta etapa, vamos fazer a divisão e pré-processamento dos dados. Como citado na introdução, estaremos aplicando apenas o modelo em que o pré-processamento é realizado a partir do conjunto de treino e depois aplicado no conjunto de teste. Com isso, estamos evitando **data leakage**.

#### 1° Passo: Identificação e tratamento de valores nulos

O primeiro passo para o pré-processamento é identificar e tratar valores nulos na base.

``` python exec="0"
print(df.isna().sum())
```

Executando a linha de código acima para o dataframe contendo os dados da base, foi possível identificar que não há valores nulos na base.

#### 2° Passo: Remoção de colunas desimportantes

Em seguida, colunas que não são importantes para a predição serão removidas do dataframe. Essas colunas são `Booking_ID` e `date of reservation`. A forma que essa exclusão foi feita está representada abaixo:

``` python
df = df.drop(columns=["Booking_ID", "date of reservation"])
```

#### 3° Passo: Codificação de variáveis categóricas

O terceiro passo se consiste na codificação das variáveis categóricas. Essas são: `type of meal`, `room type` e `market segment type`.
Considerando a forma como a técnica do KNN funciona, calculando a distância euclidiana entre pontos para predizer, a técnica de label encoding seria ruim, pois os valores numéricos arbitrários poderiam criar distâncias falsas entre as categorias. Por isso, utilizaremos a técnica de One-Hot Encoding para codificar essas variáveis, utilizando o *OneHotEncoder()* do `scikit-learn`.

``` python exec="0"

from sklearn.preprocessing import OneHotEncoder

categorical_cols = ["type of meal", "room type", "market segment type"]

X = df.drop("booking status", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

encoder = OneHotEncoder(drop="first", sparse_output=False)
encoder.fit(X_train[categorical_cols])

X_train_encoded = encoder.transform(X_train[categorical_cols])
X_test_encoded = encoder.transform(X_test[categorical_cols])

```

#### 4° Passo: Padronização das features numéricas 

Em seguida, é necessária a padronização das features numéricas na base. Ao invés da normalização, será utilizada a técnica de padronização devido aos outliers nas features numéricas, principalmente as variáveis `lead time` e `average price`, que desbalanceariam o cálculo de distâncias se apenas normalizadas.
Para a padronização, utilizaremos o *StandardScaler()* do `scikit-learn`.

``` python exec="0"

from sklearn.preprocessing import StandardScaler

numeric_cols = ["number of adults", "number of children", "number of weekend nights", 
                "number of week nights", "lead time", "P-C", "P-not-C", 
                "average price", "special requests"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

```

#### 5° Passo: Codificação da variável alvo

Por fim, vamos codificar a variável alvo `booking status` utilizando a técnica de label encoding. Ou seja, após esse passo, "Not_Canceled" vai assumir o valor 1 e "Canceled" o valor 0. Aqui, essa técnica pode ser utilizada, pois essa é a variável alvo, e não será utilizada no cálculo das distâncias. Para codificar, utilizaremos o *LabelEncoder()* do `scikit-learn`.

``` python exec="0"

from sklearn.preprocessing import LabelEncoder

l_encoder = LabelEncoder()
y = l_encoder.fit_transform(df["booking status"])

```

#### Divisão dos dados

Essa etapa será realizada antes do pré-processamento, e apenas o conjunto de treino resultante desta divisão será base para o pré-processamento.

- **Conjunto de Treino:** Utilizado para ensinar o modelo a reconhecer padrões

- **Conjunto de Teste:** Utilizado para avaliar o desempenho do modelo com dados ainda não vistos

Para realizar a divisão, foi utilizada a função *train_test_split()* do `scikit-learn`. Os parâmetros utilizados são:

- **test_size=0.2:** Define que 20% dos dados serão utilizados para teste, enquanto o restante será usado para treino.

- **random_state=42:** Parâmetro que controla o gerador de número aleatórios utilizado para sortear os dados antes de separá-los. Garante reprodutibilidade.

- **stratify=y:** Esse atributo definido como *y* é essencial devido à natureza da coluna `booking status`. Com essa definição, será mantida a mesma proporção das categorias em ambos os conjuntos, reduzindo o viés.

=== "Saída"

    ```python exec="1"
    --8<-- "docs/knn/division.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs/knn/division.py"
    ```

Esta divisão adequada é de extrema importância, pois ajuda a evitar *overfitting*.

### Etapa 3 - Treinamento dos Modelos

Agora, será realizado o treinamento do modelo. O objetivo dessa etapa é ensinar o algoritmo a reconhecer padrões nos dados que são fornecidos, e determinar se uma reserva será, ou não, cancelada de acordo com os dados das outras variáveis na base. A técnica utilizada, desta vez, será do K-Means, que se consiste na segregação dos dados em torno de centros, criando *clusters*.

#### Resultado dos treinamentos

=== "K-Means"

    

=== "Código"

    

### Etapa 4 - Avaliação dos modelos

#### Matriz de confusão

Agora, vamos observar a matriz de confusão do modelo. A matriz de confusão consegue nos oferecer diversas métricas de qualidade do modelo, como o número de previsões correta para positivos e negativos, os falso positivos, falso negativos, a precisão, especificidade, dentre outras métricas.

=== "Matriz de confusão - Modelo 2"



=== "Código"



### Etapa 5 - Relatório Final

