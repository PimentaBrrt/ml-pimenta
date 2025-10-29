# Modelo de Machine Learning - KNN

Para esse projeto, foi utilizado um dataset obtido no [**Kaggle**](https://kaggle.com){:target='_blank'}.
Os dados usados podem ser baixados [**aqui**](https://www.kaggle.com/datasets/prathamtripathi/drug-classification){:target='_blank'}.

## Objetivo

O dataset utilizado possui informações sobre pacientes sob efeito de drogas farmacêuticas. O objetivo da predição é prever o tipo de droga baseado nas features do modelo.

## Workflow

Os pontos *"etapas"* são o passo-a-passo da realização do projeto.

### Etapa 1 - Exploração de Dados

Primeiramente, deve ser feita a exploração dos dados da base, com o objetivo de compreender a forma como são estruturados os dados, sua natureza e possível significância para o modelo de predição.

O dataset é composto por **200 linhas** e **6 colunas**, com cada linha representando um paciente distinto. Essa verificação pôde ser feita com as linhas de código abaixo;

=== "Saída"

    ```python exec="1"
    --8<-- "docs/random-forest/exploring_rf.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs/random-forest/exploring_rf.py"
    ```

#### Colunas do dataset

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| Age | Inteiro | Idade do paciente |
| Sex | String | Gênero do paciente |
| BP | String | Níveis de pressão sanguínea |
| Cholesterol | String | Níveis de colesterol |
| Na_to_K | Float | Razão do sódio para o potássio no sangue |
| Drug | String | Tipo de droga |

#### Visualizações das variáveis

Em seguida, é essencial realizar gráficos para visualizar como cada uma das variáveis se comportam, com o objetivo de entender melhor a base da dados.

Está seção será divida para cada tipo de variável, entre variáveis quantitativas discretas, quantitativas contínuas, categóricas e, por fim, a variável alvo.

##### Variáveis Categóricas

=== "Sex"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/random-forest/visualizations/sex.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/random-forest/visualizations/sex.py"
        ```

=== "BP"

    === "Gráfico"

        ``` python exec="1" html="1"
        --8<-- "docs/random-forest/visualizations/bp.py"
        ```

    === "Código"

        ``` python exec="0"
        --8<-- "docs/random-forest/visualizations/bp.py"
        ```

##### Variável Quantitativa Discreta **`Age`**

=== "Gráfico"

    ``` python exec="1" html="1"
    --8<-- "docs/random-forest/visualizations/age.py"
    ```

=== "Código"

    ``` python exec="0"
    --8<-- "docs/random-forest/visualizations/age.py"
    ```

##### Variável Quantitativa Contínua **`Na_to_K`**

=== "Gráfico"

    ``` python exec="1" html="1"
    --8<-- "docs/random-forest/visualizations/natok.py"
    ```

=== "Código"

    ``` python exec="0"
    --8<-- "docs/random-forest/visualizations/natok.py"
    ```

##### Variável Alvo **`Drug`**

=== "Gráfico"

    ``` python exec="1" html="1"
    --8<-- "docs/random-forest/visualizations/drug.py"
    ```

=== "Código"

    ``` python exec="0"
    --8<-- "docs/random-forest/visualizations/drug.py"
    ```

Através das análises, foi possível alcançar uma compreensão mais aprofundada do funcionamento de cada uma das variáveis no dataset, além de haver insights valiosos nesses gráficos.

### Etapa 2 - Pré-processamento

Nessa etapa, vamos tratar a base para uso no treinamento do modelo.

#### 1° Passo: Identificação e tratamento de valores nulos

O primeiro passo para o pré-processamento é identificar e tratar valores nulos na base.

``` python exec="0"
print(df.isna().sum())
```

Executando a linha de código acima para o dataframe contendo os dados da base, foi possível identificar que não há valores nulos na base.

#### 2° Passo: Codificação de variáveis categóricas

O segundo passo se consiste na codificação das variáveis categóricas. Essas são: `Sex`, `BP` e `Cholesterol`.
Utilizaremos a técnica de One-Hot Encoding para codificar essas variáveis, utilizando o *OneHotEncoder()* do `scikit-learn`.

``` python exec="0"

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
categorical_cols = ["Sex", "BP", "Cholesterol"]

X = df.drop("Drug", axis=1)

X_encoded = encoder.fit_transform(X[categorical_cols])
encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols), index=X.index)

X = pd.concat([X.drop(columns=categorical_cols), encoded_df], axis=1)

```

#### 3° Passo: Padronização das features numéricas

Em seguida, é necessária a padronização das features numéricas na base. Utilizaremos o *StandardScaler()* do `scikit-learn` para padronizar as variáveis `Na_to_K` e `Age`.

``` python exec="0"

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numeric_cols = ["Age", "Na_to_K"]

X = df.drop("Drug", axis=1)

X_scaled = scaler.fit_transform(X[numeric_cols])
scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols, index=X.index)

X = pd.concat([X.drop(columns=numeric_cols), scaled_df], axis=1)

```

#### 4° Passo: Codificação da variável alvo

Por fim, vamos codificar a variável alvo `Drug` utilizando a técnica de label encoding. Para codificar, utilizaremos o *LabelEncoder()* do `scikit-learn`.

``` python exec="0"

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(df["Drug"])

```

### Etapa 3 - Divisão dos dados

Em seguida, vamos realizar a divisão dos dados em conjuntos de *treino* e *teste*.

- **Conjunto de Treino:** Utilizado para ensinar o modelo a reconhecer padrões

- **Conjunto de Teste:** Utilizado para avaliar o desempenho do modelo com dados ainda não vistos

Para realizar a divisão, foi utilizada a função *train_test_split()* do `scikit-learn`. Os parâmetros utilizados são:

- **test_size=0.2:** Define que 20% dos dados serão utilizados para teste, enquanto o restante será usado para treino.

- **random_state=42:** Parâmetro que controla o gerador de número aleatórios utilizado para sortear os dados antes de separá-los. Garante reprodutibilidade.

- **stratify=y:** Esse atributo definido como *y* é essencial devido à natureza da coluna `Drug`. Com essa definição, será mantida a mesma proporção das categorias em ambos os conjuntos, reduzindo o viés.

=== "Saída"

    ```python exec="1"
    --8<-- "docs/random-forest/division.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs/random-forest/division.py"
    ```

Esta divisão adequada é de extrema importância, pois ajuda a evitar *overfitting*.

### Etapa 4 - Treinamento do Modelo

Agora, será realizado o treinamento do modelo. O objetivo dessa etapa é ensinar o algoritmo a reconhecer padrões nos dados que são fornecidos, e prever o tipo de droga presente no sangue dos pacientes através das features do modelo.

=== "Saída"

    ```python exec="1" html="1"
    --8<-- "docs/random-forest/training.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs/random-forest/training.py"
    ```

### Etapa 5 - Avaliação do modelo



#### Matriz de confusão



#### Acurácia do modelo



#### Análise da visualização



### Etapa 6 - Relatório Final



#### Recomendações e Conclusões



#### Pontos Importantes Observados



#### Possíveis próximos passos e melhorias



#### Conclusão Final
