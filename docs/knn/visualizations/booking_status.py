import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.read_csv("docs/knn/booking.csv")

labels = ["Cancelado", "Não Cancelado"]  
values = df["booking status"].value_counts()

plt.figure(figsize=(10, 8))
plt.pie(values, labels=labels, autopct=lambda p: f"{int(p * sum(values) / 100)}", colors=["lightcoral", "lightgreen"])
plt.title("Distribuição do Status das Reservas - Pizza")

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
print(buffer.getvalue())