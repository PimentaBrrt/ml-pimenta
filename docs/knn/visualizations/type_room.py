import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.read_csv("docs/knn/booking.csv")

plt.figure(figsize=(10, 8))
df["room type"].value_counts().plot(kind="bar", color="mediumpurple", edgecolor="black")
plt.title("Distribuição dos Tipos de Quarto - Barras")
plt.xlabel("Tipo de Quarto")
plt.ylabel("Frequência")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)

ax = plt.gca()
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", 
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha="center", va="bottom")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())
plt.close()