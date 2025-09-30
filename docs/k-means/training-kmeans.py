import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

scaler = StandardScaler()

df = pd.read_csv("docs/k-means/fish_data.csv")

X = df.drop(columns=["species"])

X = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

silhouette_avg = silhouette_score(X, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis", alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="*", s=300, c="red", label="Centroids")

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var.)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var.)")
plt.title("Clusters de Peixes - K-means (K=3)")
plt.legend()
plt.colorbar(scatter, label="Cluster")

plt.savefig("docs/images/k-means.svg", format="svg", transparent=True)
plt.close()