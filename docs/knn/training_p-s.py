import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

encoder = OneHotEncoder()
scaler = StandardScaler()

df = pd.read_csv("docs/knn/booking.csv")

df = df.drop(columns=["Booking_ID", "date of reservation"])

numeric_cols = ['number of adults', 'number of children', 'number of weekend nights', 
                'number of week nights', 'lead time', 'P-C', 'P-not-C', 
                'average price', 'special requests']

categorical_cols = ['type of meal', 'room type', 'market segment type']

X = df.drop("booking status", axis=1)
y = df["booking status"]

for col in numeric_cols:
    X[col] = scaler.fit_transform(X[[col]])

X_encoded = encoder.fit_transform(X[categorical_cols])

encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols), index=X.index)

X = pd.concat([X.drop(columns=categorical_cols), encoded_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}<br>")

# Visualização do KNN 

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("Variance explained by each component:", pca.explained_variance_ratio_)

# buffer = StringIO()
# plt.savefig(buffer, format="svg", transparent=True)
# print(buffer.getvalue())