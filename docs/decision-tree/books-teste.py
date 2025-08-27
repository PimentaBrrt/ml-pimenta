import pandas as pd

df = pd.read_csv("docs/decision-tree/dados.csv", sep=",", encoding="UTF-8")

df["book_freq"] = df[["book1", "book2", "book3", "book4", "book5"]].sum(axis=1)

print(df[df["book_freq"]==0][["name", "book1","book2","book3","book4","book5","popularity"]])

