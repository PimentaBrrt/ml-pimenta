import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

df = pd.read_csv("docs/decision-tree/dados.csv", sep=",", encoding="UTF-8")

plod_alive = df[df["isAlive"] == 1]["plod"]
plod_dead = df[df["isAlive"] == 0]["plod"]

plt.rcParams["figure.figsize"] = (10, 5)
fig, ax = plt.subplots(facecolor="white")
ax.set_facecolor("white")

ax.boxplot([plod_alive, plod_dead], tick_labels=["Vivo", "Morto"],
           patch_artist=True,
           boxprops=dict(facecolor="lightblue", color="black"),
           medianprops=dict(color="red"))

ax.set_title("Distribuição de plod por estado de vida", color="black")
ax.set_ylabel("plod (probabilidade de morte)", color="black")
ax.set_xlabel("Estado de vida (isAlive)", color="black")
ax.grid(axis="y", linestyle="--", alpha=0.7, color="gray")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("black")
ax.spines["bottom"].set_color("black")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())