import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.read_csv("docs/knn/booking.csv")

plt.figure(figsize=(10, 6))
lead_time_bins = pd.cut(df["lead time"], bins=8)
lead_time_bins.value_counts().sort_index().plot(kind="bar", color="coral", edgecolor="black")
plt.title("Distribuição do Lead Time (dias entre reserva e chegada) - Barras")
plt.xlabel("Intervalo de Dias")
plt.ylabel("Frequência")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

ax = plt.gca()
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", 
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha="center", va="bottom")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())