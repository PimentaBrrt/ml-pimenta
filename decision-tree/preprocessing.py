import pandas as pd

df = pd.read_csv("docs\decision-tree\dados.csv", sep=",", encoding="UTF-8")

# 1° Passo: Criando a coluna "book_freq", já normalizando-a

df["book_freq"] = df[["book1", "book2", "book3", "book4", "book5"]].sum(axis=1) / 5

# 2° Passo: Dropando colunas que não serão utilizadas

cols = [
    "plod", "title", "culture", "mother", "father", "heir", "house", "book_freq", "isNoble", "popularity"
]

df = df[cols]

# 3° Passo: Tratamento de valores faltantes

cols = ["plod", "popularity"]
for col in cols:
    df.fillna({col: df[col].median()}, inplace=True)

cols = ["title", "culture", "mother", "father", "heir", "house"]
for col in cols:
    df.fillna({col: "Unknown"}, inplace=True)

df.fillna({"isNoble": df["isNoble"].mode()[0]}, inplace=True)

# 4° Passo: Binarização das variáveis categóricas nominais

cols = ["title", "culture", "mother", "father", "heir", "house"]

for col in cols:
    df[f"has_{col}"] = (df[col] != "Unknown").astype(int)
    df.drop(columns=[col], inplace=True)
    
# 5° Passo: Inversão da variável "plod" e renomeação para "survival_prob"

df["survival_prob"] = 1 - df["plod"]
df.drop(columns="plod", inplace=True)

# 6° Passo: Criar variável target "relevance_category" a partir do score "relevance_score"

def calculate_relevance_score(row):
    
    score = (
        row["popularity"] * 0.25 +
        row["book_freq"] * 0.25 +
        row["survival_prob"] * 0.15 +
        row["isNoble"] * 0.10 +
        row["has_title"] * 0.10 +
        row["has_house"] * 0.05 +
        row["has_culture"] * 0.05 +
        (row["has_mother"] + row["has_father"] + row["has_heir"]) * 0.05 / 3
    )
    
    return min(max(score, 0), 1)

def categorize_relevance(score):
    if score < 0.25:
        return "Low"
    elif score < 0.5:
        return "Medium"
    elif score < 0.75:
        return "High"
    else:
        return "Very High"

df["relevance_score"] = df.apply(calculate_relevance_score, axis=1)
df["relevance_category"] = df["relevance_score"].apply(categorize_relevance)

features = [
    "book_freq", "popularity", "survival_prob", "isNoble",    
    "has_title", "has_culture", "has_mother", "has_father", 
    "has_heir", "has_house",
]

target = "relevance_category"

# df.to_csv("dados_processado.csv", index=False)

print(f"Colunas após tratamento: {df.columns.tolist()}\n")
print(f"Valores ausentes após pré-processamento: {df.isnull().sum().sum()}\n") 
print(f"Formato do dataset final: {df.shape}\n")
print(f"Features: {len(features)}\n")
print(f"Target: {target}\n")
print(f"Distribuição da variável target:\n")
print(df[target].value_counts().to_markdown())