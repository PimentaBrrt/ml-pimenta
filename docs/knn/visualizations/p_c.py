import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.read_csv("docs/knn/booking.csv")

plt.figure(figsize=(10, 6))
df["P-C"].value_counts().sort_index().plot(kind="bar", color="orange", edgecolor="black")
plt.title("Distribuição de Reservas Anteriormente Canceladas (P-C) - Barras")
plt.xlabel("Número de Reservas Anteriormente Canceladas")
plt.ylabel("Frequência")
plt.xticks(rotation=0)
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