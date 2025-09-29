import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.read_csv("docs/k-means/fish_data.csv")

plt.figure(figsize=(10, 8))
df["species"].value_counts().plot(kind="bar")
plt.title("Distribuição das Espécies de Peixe")
plt.xlabel("Espécie")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)

ax = plt.gca()
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", 
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha="center", va="bottom")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())
plt.close()