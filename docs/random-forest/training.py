import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

rf = RandomForestClassifier(n_estimators=100,
                            max_depth=5,
                            max_features='sqrt', 
                            random_state=42)

rf.fit(X_train, y_train)

predictions = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")

feature_importance = pd.DataFrame({
    "Feature": rf.feature_names_in_,
    "Importância": rf.feature_importances_
})
print("<br>Importância das Features:")
print(feature_importance.sort_values(by="Importância", ascending=False).to_html() + "<br>")