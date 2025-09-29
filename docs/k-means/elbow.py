import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df = pd.read_csv("docs/k-means/fish_data.csv")

X = df.drop(columns=["species"])

for col in X:
    X[col] = scaler.fit_transform(X[[col]])

# Elbow Method

wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, "bo-", markersize=8, linewidth=2)
plt.xlabel("Número de Clusters (K)")
plt.ylabel("WCSS (Within-Cluster Sum of Square)")
plt.title("Elbow Method - Determinando o K ideal")
plt.grid(True, alpha=0.3)
plt.xticks(k_range)

plt.axvline(x=3, color="red", linestyle="--", alpha=0.7, label="Possível cotovelo K=3")

plt.legend()
plt.savefig("docs/images/elbow.svg", format="svg", transparent=True)
plt.close()