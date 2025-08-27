import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("docs/decision-tree/dados.csv", sep=",", encoding="UTF-8")

df["book_freq"] = df[["book1", "book2", "book3", "book4", "book5"]].sum(axis=1)

plt.rcParams["figure.figsize"] = (10, 5)
fig, ax = plt.subplots(facecolor="white")
ax.set_facecolor("white")

ax.scatter(df["book_freq"], df["popularity"], alpha=0.7, color="red", edgecolor="black")

ax.set_title("Popularidade X Frequência nos livros", color="black")
ax.set_xlabel("Frequência em livros (soma)", color="black")
ax.set_ylabel("Popularidade", color="black")
ax.grid(axis="y", linestyle="--", alpha=0.7, color="gray")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("black")
ax.spines["bottom"].set_color("black")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())
