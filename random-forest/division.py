import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("docs/random-forest/drug200.csv")

encoder = OneHotEncoder()
l_encoder = LabelEncoder()
scaler = StandardScaler()

numeric_cols = ["Age", "Na_to_K"]
categorical_cols = ["Sex", "BP", "Cholesterol"]

X = df.drop("Drug", axis=1)

X_encoded = encoder.fit_transform(X[categorical_cols])
encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols), index=X.index)

X_scaled = scaler.fit_transform(X[numeric_cols])
scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols, index=X.index)

X = pd.concat([scaled_df, encoded_df], axis=1)

y = l_encoder.fit_transform(df["Drug"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Treino: {X_train.shape[0]} amostras\n")
print(f"Teste: {X_test.shape[0]} amostras\n")
print(f"Proporção: {X_train.shape[0]/X.shape[0]*100:.1f}% treino, {X_test.shape[0]/X.shape[0]*100:.1f}% teste\n")

print("Distribuição das classes - \n")
print("Treino:\n")
print(pd.Series(y_train).value_counts().to_markdown(), "\n")
print("Teste:\n")
print(pd.Series(y_test).value_counts().to_markdown(), "\n")