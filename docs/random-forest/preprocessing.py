import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

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