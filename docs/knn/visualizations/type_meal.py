import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.read_csv("docs/knn/booking.csv")

plt.figure(figsize=(10, 8))
df["type of meal"].value_counts().plot(kind="bar", color="lightseagreen", edgecolor="black")
plt.title("Distribuição dos Tipos de Refeição - Barras")
plt.xlabel("Tipo de Refeição")
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