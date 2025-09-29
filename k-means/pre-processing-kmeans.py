import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df = pd.read_csv("docs/k-means/fish_data.csv")

X = df.drop(columns=["species"])

for col in X:
    X[col] = scaler.fit_transform(X[[col]])