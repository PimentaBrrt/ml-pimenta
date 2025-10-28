import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.read_csv("docs/random-forest/drug200.csv")

plt.figure(figsize=(10, 6))
plt.hist(df["Age"], bins=30, edgecolor="black", alpha=0.7, color="red")
plt.title("Idade dos Pacientes - Histograma")
plt.xlabel("Idade")
plt.ylabel("Frequência")
plt.grid(axis="y", alpha=0.3)

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())
plt.close()