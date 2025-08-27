# Modelo de Machine Learning - Árvore de Decisões

Para esse projeto, foi utilizado um dataset obtido no [**Kaggle**](https://kaggle.com){:target='_blank'}.
Os dados usados podem ser baixados [**aqui**](https://www.kaggle.com/datasets/dalmacyali1905/game-of-thrones-classification-decision-tree?resource=download){:target='_blank'}.

## Objetivo

O dataset apresenta diversos dados relacionados à cada um dos personagens da série de livros [**A Song of Ice and Fire**](https://en.wikipedia.org/wiki/A_Song_of_Ice_and_Fire){:target='_blank'}, escrita por [**George R. R. Martin**](https://en.wikipedia.org/wiki/George_R._R._Martin){:target='_blank'}, inspiração para a famosa série [**Game of Thrones**](https://en.wikipedia.org/wiki/Game_of_Thrones){:target='_blank'}.
O objetivo dessa análise é o modelo fazer a predição da importância do personagem para a série no sentido de trama. Um índice será criado a partir das variáveis presentes no dataset, variando entre 0 para completamente irrelevante e 1 para muito relevante.

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

#### Observação sobre a coluna **plod**

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

- **Correlações entre *plod* e as outras colunas:** Levando isso em consideração, é necessário realizar um cálculo de correlações para descobrir a principal variável no cálculo do **plod**:

=== "Saída"

    ```python exec="1"
    --8<-- "docs\decision-tree\plod-teste2.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs\decision-tree\plod-teste2.py"
    ```

É possível observar que a correlação mais forte entre **plod** e qualquer outra coluna no dataset é com a coluna **isAlive**. Esse dado nos permite criar uma **hipótese** de que **plod** é a *estimativa da probabilidade de morte do personagem*.

- **Comparação com a coluna *isAlive*:** Em seguida, para verificar a hipótese estabelecida, será feito um gráfico de boxplot para analisar a relação de **plod** e **isAlive**;

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

#### Exploração aprofundada da coluna **popularity**

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

- **Gráfico de dispersão de *popularity*:** O gráfico relaciona o índice **popularity** com a soma das 5 variáveis **book**, que indicam a presença de um personagem em cada livro em binário. Os livros considerados nessas variáveis são apenas a narrativa principal da história, sem spin-offs e personagens que são apenas citados e referenciados.

=== "Gráfico"

    ```python exec="on" html="1"
    --8<-- "docs\decision-tree\pop-scatter.py"
    ```

=== "Código"

    ```python exec="0"
    --8<-- "docs\decision-tree\pop-scatter.py"
    ```

A análise do gráfico indica que, de 1 até 5 aparições, o número de personagens populares **aumenta** de forma diretamente proporcional, com alguns out-liers.

Contudo, podemos observar que diversos personagens que não apareceram na *série principal de livros* são extremamente populares. Isso acontece pois há personagens de spin-offs muito amados pela comunidade, além de outros personagens que são apenas citados ao longo da história, sem aparecer diretamente, e também adquirem alta popularidade.

### Etapa 2 - Pré-processamento

