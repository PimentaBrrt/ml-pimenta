import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df = pd.read_csv("docs/k-means/fish_data.csv")

X = df.drop(columns=["species"])

for col in X:
    X[col] = scaler.fit_transform(X[[col]])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

print(f"Tamanho dos clusters: {np.bincount(cluster_labels)}")