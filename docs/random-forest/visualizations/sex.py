import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.read_csv("docs/random-forest/drug200.csv")

plt.figure(figsize=(10, 6))
df["Sex"].value_counts().sort_index().plot(kind="bar", color="lightyellow", edgecolor="black")
plt.title("Gênero dos Pacientes - Barras")
plt.xlabel("Gênero")
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