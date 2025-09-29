import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.read_csv("docs/k-means/fish_data.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(df["length"], bins=20, edgecolor="black", alpha=0.7)
ax1.set_title("Distribuição do Comprimento")
ax1.set_xlabel("Comprimento (cm)")
ax1.set_ylabel("Frequência")

bplot = ax2.boxplot(df["length"], patch_artist=True)
ax2.set_title("Boxplot - Comprimento")
ax2.set_ylabel("Comprimento (cm)")

colors = ["peachpuff"]

for patch, color in zip(bplot["boxes"], colors):
    patch.set_facecolor(color)

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())
plt.close()