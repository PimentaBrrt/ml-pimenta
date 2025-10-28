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



### Etapa 3 - Divisão dos dados



### Etapa 4 - Treinamento dos Modelos



#### Resultado do treinamento



### Etapa 5 - Avaliação do modelo



#### Matriz de confusão



#### Acurácia do modelo



#### Análise da visualização



### Etapa 6 - Relatório Final



#### Recomendações e Conclusões



#### Pontos Importantes Observados



#### Possíveis próximos passos e melhorias



#### Conclusão Final
