import pandas as pd
from sklearn.model_selection import train_test_split

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

print(f"Treino: {x_train.shape[0]} amostras\n")
print(f"Teste: {x_test.shape[0]} amostras\n")
print(f"Proporção: {x_train.shape[0]/x.shape[0]*100:.1f}% treino, {x_test.shape[0]/x.shape[0]*100:.1f}% teste\n")

print("Distribuição das classes - \n")
print("Treino:\n")
print(y_train.value_counts().to_markdown(), "\n")
print("Teste:\n")
print(y_test.value_counts().to_markdown(), "\n")