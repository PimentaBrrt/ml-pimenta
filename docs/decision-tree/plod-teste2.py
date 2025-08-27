import pandas as pd

df = pd.read_csv("docs\decision-tree\dados.csv", sep=",", encoding="UTF-8")

df_numerico = df.select_dtypes(include=["number"])

correlacoes = df_numerico.corr()["plod"].sort_values(ascending=False)

for col, corr in correlacoes.items():
    if (corr > 0.3 or corr < -0.3) and corr != 1:
        print(f"{col}: {corr}\n")
    
