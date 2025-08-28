import pandas as pd

df = pd.read_csv("docs\decision-tree\dados.csv", sep=",", encoding="UTF-8")

print(df["popularity"].describe().to_markdown())