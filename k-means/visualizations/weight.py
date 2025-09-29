import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.read_csv("docs/k-means/fish_data.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(df["weight"], bins=20, edgecolor="black", alpha=0.7)
ax1.set_title("Distribuição do Peso")
ax1.set_xlabel("Peso (g)")
ax1.set_ylabel("Frequência")

bplot = ax2.boxplot(df["weight"], patch_artist=True)
ax2.set_title("Boxplot - Peso")
ax2.set_ylabel("Peso (g)")

colors = ["lightblue"]

for patch, color in zip(bplot["boxes"], colors):
    patch.set_facecolor(color)

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())
plt.close()