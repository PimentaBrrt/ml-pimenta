# Github Pages

O site está rodando em: [https://pimentabrrt.github.io/ml-pimenta/decision-tree/main/](https://pimentabrrt.github.io/ml-pimenta/decision-tree/main/)

## Setup

Para utilizar o código deste repositório, siga as instruções a seguir:

Crie um ambiente virtual do Python:

``` shell
python -m venv env
```

Ative o ambiente virtual (**você deve fazer isso sempre que for executar algum script deste repositório**):

``` cmd
.\env\Scripts\activate
```

Instale as dependências com:

``` cmd
python -m pip install -r requirements.txt --upgrade
```

## Deployment

O material utiliza o [mkdocs](https://www.mkdocs.org/) para gerar a documentação. Para visualizar a documentação, execute o comando:

``` shell
mkdocs serve -o
```

Para subir ao GitHub Pages, execute o comando:

``` shell
mkdocs gh-deploy
```
