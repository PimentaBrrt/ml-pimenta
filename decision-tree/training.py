import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from io import StringIO

df = pd.read_csv("dados_processado.csv")

features = [
    "book_freq", "popularity", "survival_prob", "isNoble",
    "has_title", "has_culture", "has_mother", "has_father", 
    "has_heir", "has_house"
]

target = "relevance_category"

x = df[features]
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

classifier = tree.DecisionTreeClassifier(random_state=42)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisão do Modelo: {accuracy:.4f}")

feature_importance = pd.DataFrame({
    "Feature": classifier.feature_names_in_,
    "Importância": classifier.feature_importances_
})
print("<br>Importância das Features:")
print(feature_importance.sort_values(by="Importância", ascending=False).to_html() + "<br>")

plt.figure(figsize=(20, 10))
tree.plot_tree(
    classifier, 
    feature_names=features,
    class_names=classifier.classes_,
    filled=True,
    rounded=True,
    max_depth=3, 
    fontsize=10
)

buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())