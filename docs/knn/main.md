# Modelo de Machine Learning - KNN

Para esse projeto, foi utilizado um dataset obtido no [**Kaggle**](https://kaggle.com){:target='_blank'}.
Os dados usados podem ser baixados [**aqui**](https://www.kaggle.com/datasets/youssefaboelwafa/hotel-booking-cancellation-prediction/data){:target='_blank'}.

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

### Etapa 2 - Pré-processamento

TODO list do pré-processamento:

- Identificar valores NA

- Tratamento dos valores ausentes

- Codificação de variáveis categóricas

- Normalização/Padronização das features numéricas 

### Etapa 3 - Divisão de dados



### Etapa 4 - Treinamento do Modelo



### Etapa 5 - Avaliação do modelo



### Etapa 6 - Relatório Final


