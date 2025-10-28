import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.read_csv("docs/random-forest/drug200.csv")

plt.figure(figsize=(10, 6))
plt.hist(df["Na_to_K"], bins=30, edgecolor="black", alpha=0.7, color="lightblue")
plt.title("Distribuição da Razão de Sódio para Potássio no Sangue dos Pacientes - Histograma")
plt.xlabel("Razão de Sódio para Potássio no Sangue")
plt.ylabel("Frequência")
plt.grid(axis="y", alpha=0.3)

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())
plt.close()