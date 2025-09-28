import pandas as pd

df = pd.read_csv("docs/decision-tree/dados.csv", sep=",", encoding="UTF-8")

print(f"Tipo de dado: {df["plod"].dtype}\n")
print(f"Valor mínimo: {df["plod"].min()}\n")
print(f"Valor máximo: {df["plod"].max()}\n")
print(f"Valor médio: {format(df["plod"].mean(), ".3f")}\n")
print(f"Exemplo de valor: {df.loc[0, "plod"]}")