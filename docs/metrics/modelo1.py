import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

encoder = OneHotEncoder()
scaler = StandardScaler()
l_encoder = LabelEncoder()

df = pd.read_csv("docs/knn/booking.csv")

df = df.drop(columns=["Booking_ID", "date of reservation"])

numeric_cols = ["number of adults", "number of children", "number of weekend nights", 
                "number of week nights", "lead time", "P-C", "P-not-C", 
                "average price", "special requests"]

categorical_cols = ["type of meal", "room type", "market segment type"]

X = df.drop("booking status", axis=1)
y = l_encoder.fit_transform(df["booking status"])

for col in numeric_cols:
    X[col] = scaler.fit_transform(X[[col]])

X_encoded = encoder.fit_transform(X[categorical_cols])

encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols), index=X.index)

X = pd.concat([X.drop(columns=categorical_cols), encoded_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

cm = confusion_matrix(y_test, predictions)
tn, fp, fn, tp = cm.ravel()

print("<b>Matriz de confusão detalhada:</b><br>")
print(f"Verdadeiros Negativos (TN): {tn}<br>")
print(f"Falsos Positivos (FP): {fp}<br>")
print(f"Falsos Negativos (FN): {fn}<br>")
print(f"Verdadeiros Positivos (TP): {tp}<br>")

acuracia = (tn + tp) / (tn + fp + fn + tp)
sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0
especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0
precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
valor_preditivo_negativo = tn / (tn + fn) if (tn + fn) > 0 else 0
falsos_positivos_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
falsos_negativos_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

print("<br><b>Métricas:</b><br>")
print(f"Acurácia: {acuracia:.4f}<br>")
print(f"Sensibilidade (Recall): {sensibilidade:.4f}<br>")
print(f"Especificidade: {especificidade:.4f}<br>")
print(f"Precisão: {precisao:.4f}<br>")
print(f"Valor Preditivo Negativo: {valor_preditivo_negativo:.4f}<br>")
print(f"Taxa de Falsos Positivos: {falsos_positivos_rate:.4f}<br>")
print(f"Taxa de Falsos Negativos: {falsos_negativos_rate:.4f}")