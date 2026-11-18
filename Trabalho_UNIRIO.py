# ============================================================
# TRABALHO 18 - GRUPO 18 - MACHINE LEARNING
# EVASÃO NO ENSINO SUPERIOR: QUEM DESISTE DO CURSO?
# (versão atualizada com um dataset maior: 4.424 alunos)
# ============================================================
#
# Roteiro da apresentação:
#   1. Qual o dataset
#   2. Análise exploratória
#   3. Pré-processamento
#   4. O que queremos prever
#   5. Qual o modelo escolhido
#   6. Qual a métrica de avaliação
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)

sns.set_theme(style="whitegrid")

# Se SALVAR_FIGURAS=1, os gráficos são salvos na pasta "figs"
SALVAR = os.environ.get("SALVAR_FIGURAS") == "1"
if SALVAR:
    os.makedirs("figs", exist_ok=True)


def mostrar(nome):
    plt.tight_layout()
    if SALVAR:
        plt.savefig(f"figs/{nome}.png", dpi=130, bbox_inches="tight")
    plt.show()
    plt.close()


# ============================================================
# 1. DATASET
# ============================================================
# Fonte: "Predict Students' Dropout and Academic Success"
# (Realinho et al., 2021, Instituto Politécnico de Portalegre,
# Portugal - UCI Machine Learning Repository, licença CC BY 4.0).
# Versão em português usada no curso de classificação da Alura.

url = ("https://raw.githubusercontent.com/alura-cursos/"
       "classificacao_multiclasse/main/Dados/dados_estudantes.csv")

df = pd.read_csv(url)

print("=" * 60)
print("1. DATASET")
print("=" * 60)
print("Dimensão da base:", df.shape)
print("\nPrimeiras linhas:")
print(df.head())
print("\nTipos das variáveis:")
print(df.dtypes)
print("\nValores ausentes:", int(df.isnull().sum().sum()))
print("\nDistribuição original do Target:")
print(df["Target"].value_counts())

# ============================================================
# 4. O QUE QUEREMOS PREVER
# ============================================================
# Problema: classificação binária.
# Alvo: "Evasao" -> Desistente (1) ou Graduado (0).
# Os alunos "Matriculado" ficam de fora, porque ainda não
# têm um desfecho definido (nem desistiram, nem se formaram).

df = df[df["Target"] != "Matriculado"].copy()
df["Evasao"] = df["Target"].map({"Graduado": "Graduado",
                                 "Desistente": "Desistente"})
alvo = "Evasao"

print("\n" + "=" * 60)
print("4. O QUE QUEREMOS PREVER")
print("=" * 60)
print("Registros usados (Graduado ou Desistente):", df.shape[0])
print(df[alvo].value_counts())
print((df[alvo].value_counts(normalize=True) * 100).round(2))

# ============================================================
# 2. ANÁLISE EXPLORATÓRIA
# ============================================================

numericas_eda = [
    "Idade na matrícula",
    "disciplinas 1º semestre (aprovadas)",
    "disciplinas 1º semestre (notas)",
    "disciplinas 2º semestre (notas)",
]
ordem = ["Graduado", "Desistente"]

# 2.1 Distribuição da classe
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x=alvo, hue=alvo, order=ordem,
              hue_order=ordem, palette="Set2", legend=False)
plt.title("Alunos graduados e desistentes")
plt.xlabel("Situação")
plt.ylabel("Quantidade")
mostrar("classes")

# 2.2 Histogramas
df[numericas_eda].hist(figsize=(12, 8), bins=15, edgecolor="black")
plt.suptitle("Distribuição das variáveis numéricas")
mostrar("hist")

# 2.3 Boxplots por situação
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for coluna, ax in zip(numericas_eda, axes.flatten()):
    sns.boxplot(data=df, x=alvo, y=coluna, hue=alvo, order=ordem,
                hue_order=ordem, palette="Set2", legend=False, ax=ax)
    ax.set_title(coluna)
    ax.set_xlabel("")
mostrar("box")

# 2.4 Correlação
plt.figure(figsize=(9, 6))
sns.heatmap(df[numericas_eda].corr(), annot=True,
            cmap="coolwarm", fmt=".2f")
plt.title("Correlação entre variáveis numéricas")
mostrar("corr")

# 2.5 Estatísticas
print("\n" + "=" * 60)
print("2. ANÁLISE EXPLORATÓRIA")
print("=" * 60)
print(df[numericas_eda].describe().round(2))
print("\nMédias por situação:")
print(df.groupby(alvo)[numericas_eda].mean().round(2))

for col in ["Bolsista", "Devedor", "Taxas de matrícula em dia", "Período"]:
    taxa = (df.groupby(col)[alvo]
              .apply(lambda s: (s == "Desistente").mean() * 100)
              .round(1))
    print(f"\n% de desistentes por '{col}':")
    print(taxa)

# ============================================================
# 3. PRÉ-PROCESSAMENTO
# ============================================================

X = df.drop(columns=["Target", alvo])
y = df[alvo]

numericas = X.select_dtypes(include="number").columns.tolist()
categoricas = [c for c in X.columns if c not in numericas]

print("\n" + "=" * 60)
print("3. PRÉ-PROCESSAMENTO")
print("=" * 60)
print("Variáveis numéricas:", len(numericas))
print("Variáveis categóricas:", len(categoricas))

processamento_numerico = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

processamento_categorico = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessador = ColumnTransformer([
    ("num", processamento_numerico, numericas),
    ("cat", processamento_categorico, categoricas)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Treinamento:", X_train.shape)
print("Teste:", X_test.shape)

# ============================================================
# 5. MODELOS
# ============================================================

modelo_lr = Pipeline([
    ("preprocessamento", preprocessador),
    ("modelo", LogisticRegression(max_iter=1000))
])

modelo_arvore = Pipeline([
    ("preprocessamento", preprocessador),
    ("modelo", DecisionTreeClassifier(max_depth=4, random_state=42))
])

modelo_lr.fit(X_train, y_train)
modelo_arvore.fit(X_train, y_train)

pred_lr = modelo_lr.predict(X_test)
pred_arvore = modelo_arvore.predict(X_test)

# ============================================================
# 6. MÉTRICAS DE AVALIAÇÃO
# ============================================================
# Métrica principal: acurácia.
# Como a classe de interesse é "Desistente" (minoria), também
# olhamos precision, recall e F1-score dessa classe.

def metricas(y_real, y_pred):
    return {
        "Acurácia": accuracy_score(y_real, y_pred),
        "Precision (Desistente)": precision_score(y_real, y_pred, pos_label="Desistente"),
        "Recall (Desistente)": recall_score(y_real, y_pred, pos_label="Desistente"),
        "F1 (Desistente)": f1_score(y_real, y_pred, pos_label="Desistente"),
    }

tabela = pd.DataFrame({
    "Regressão Logística": metricas(y_test, pred_lr),
    "Árvore de Decisão": metricas(y_test, pred_arvore),
}).T

print("\n" + "=" * 60)
print("6. MÉTRICAS (random_state = 42)")
print("=" * 60)
print((tabela * 100).round(2))

print("\nREGRESSÃO LOGÍSTICA")
print(classification_report(y_test, pred_lr, zero_division=0))
print("ÁRVORE DE DECISÃO")
print(classification_report(y_test, pred_arvore, zero_division=0))

# Matrizes de confusão
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred_lr, labels=ordem, ax=axes[0], cmap="Blues")
axes[0].set_title("Regressão Logística")
ConfusionMatrixDisplay.from_predictions(
    y_test, pred_arvore, labels=ordem, ax=axes[1], cmap="Greens")
axes[1].set_title("Árvore de Decisão")
mostrar("confusao")

# Validação cruzada (5 partes)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_lr = cross_val_score(modelo_lr, X, y, cv=cv, scoring="accuracy")
cv_arvore = cross_val_score(modelo_arvore, X, y, cv=cv, scoring="accuracy")

print("\nValidação cruzada (5 partes) - acurácia média:")
print("Regressão Logística:", round(cv_lr.mean() * 100, 2), "%",
      "(+/-", round(cv_lr.std() * 100, 2), ")")
print("Árvore de Decisão:  ", round(cv_arvore.mean() * 100, 2), "%",
      "(+/-", round(cv_arvore.std() * 100, 2), ")")

# Diferentes valores de random_state
estados = [1, 10, 20, 42, 50, 100, 123, 200]
resultados = []

for estado in estados:
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=estado, stratify=y
    )
    for nome, modelo in [("Regressão Logística", modelo_lr),
                         ("Árvore de Decisão", modelo_arvore)]:
        modelo.fit(Xtr, ytr)
        resultados.append({
            "Random State": estado,
            "Modelo": nome,
            "Acurácia": accuracy_score(yte, modelo.predict(Xte))
        })

resultado_df = pd.DataFrame(resultados)
print("\nAcurácia por random_state:")
print(resultado_df.pivot(index="Random State", columns="Modelo",
                         values="Acurácia").round(4))

plt.figure(figsize=(11, 5))
sns.barplot(data=resultado_df, x="Random State", y="Acurácia",
            hue="Modelo", palette=["cornflowerblue", "seagreen"])
plt.title("Acurácia por random_state")
plt.ylim(0, 1)
mostrar("random_state")

resumo = resultado_df.groupby("Modelo")["Acurácia"].agg(["mean", "min", "max"])
print("\nResumo dos 8 random_state:")
print((resumo * 100).round(2))

# ============================================================
# CONCLUSÃO
# ============================================================

melhor = tabela["Acurácia"].idxmax()
proporcao_graduado = (y == "Graduado").mean() * 100

print("\n" + "=" * 60)
print("CONCLUSÃO")
print("=" * 60)
print("Modelo escolhido (maior acurácia):", melhor)
print("Referência (sempre prever 'Graduado'):",
      round(proporcao_graduado, 2), "%")
print("""
Foram feitos: análise exploratória, pré-processamento com
Pipeline, treino de dois modelos e avaliação com acurácia,
precision, recall, F1-score, matriz de confusão e validação
cruzada.

Atenção: as variáveis do 1º e 2º semestres só existem depois
que o aluno cursou esses períodos. O modelo funciona como um
alerta de evasão após o 1º/2º semestre, e não na matrícula.
""")
print("========== FIM DO TRABALHO ==========")