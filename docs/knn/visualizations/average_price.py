import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.read_csv("docs/knn/booking.csv")

plt.figure(figsize=(10, 6))
plt.hist(df["average price"], bins=30, edgecolor="black", alpha=0.7, color="red")
plt.title("Distribuição do Preço Médio das Reservas - Histograma")
plt.xlabel("Preço Médio")
plt.ylabel("Frequência")
plt.grid(axis="y", alpha=0.3)

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())
plt.close()