import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Leemos los datos
data = pd.read_csv("./bank+marketing/bank/bank-full.csv", sep=";", encoding="utf-8")
data.head()

# Imprimimos la forma del dataframe
print(data.shape)

# Hacemos un histograma de los datos
for col in data.columns:
    plt.hist(data[col])
    plt.title(col)
    plt.show()

# Para variables categóricas, hacemos gráficas de barras como sigue:
job_counts = data["job"].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(job_counts.index, job_counts.values)
plt.xticks(rotation="vertical")
plt.title("Distribución de Trabajos")
plt.xlabel("Trabajo")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

# Generamos un dataframe de los datos correspondientes a yes
datayes = data.loc[data["y"] == "yes"]

# Graficamos los trabajos de los que dijeron que sí
job_counts = datayes["job"].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(job_counts.index, job_counts.values)
plt.xticks(rotation="vertical")
plt.title("Distribución de Trabajos de los Sí")
plt.xlabel("Trabajo")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

# Contamos cuántas filas tienen un valor de 'previous' mayor a 0
zero_previous_count = len(data.loc[data["previous"] > 0])

print(f"Número de filas en donde 'previous' es > 0: {zero_previous_count}")

# Hacemos histogramas para las personas que dijeron que sí
for col in datayes.columns:
    plt.hist(datayes[col])
    plt.title(col)
    plt.show()

# Calculamos las correlaciones entre las variables numéricas
import seaborn as sns

# Seleccionamos las columnas numéricas
num_features = data.select_dtypes(include=np.number).columns

# Calculamos la matriz de correlación
corr_matrix = data[num_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Calor de la Matriz de Correlación")
plt.show()

# Imputamos educación con una muestra aleatoria de los valores conocidos
unknown_education_mask = data["education"] == "unknown"
known_education_values = data.loc[~unknown_education_mask, "education"]
data.loc[unknown_education_mask, "education"] = np.random.choice(
    known_education_values, size=unknown_education_mask.sum(), replace=True
)
plt.hist(data["education"])
plt.title("Distribución de Educación Imputada")
plt.show()

# Entrenamos y validamos clasificadores
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
)
from sklearn.linear_model import LogisticRegression

df = data
X = df.drop(columns=["y"])
y = df["y"].map({"yes": 1, "no": 0})

# Variables numéricas
num_features = data.select_dtypes(include=np.number).columns
# Variables categóricas
cat_features = [c for c in X.columns if c not in num_features]

# Escalamos las variables numéricas y codificamos las categóricas con one hot
preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

# Dividimos los datos en entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Modelos a usar
models = {
    "kNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "Naive Bayes": GaussianNB(),
    "Fisher Criterion": Pipeline(
        [
            ("preproc", preprocessor),
            ("fisher", LinearDiscriminantAnalysis(n_components=1)),
            ("clf", LogisticRegression()),
        ]
    ),
}

# Hacemos el entrenamiento y evaluación
for name, model in models.items():
    if name == "Fisher Criterion":
        pipe = model
    else:
        pipe = Pipeline([("preproc", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Calculamos la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Calculamos métricas
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_test, y_pred)

    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = None
    # Imprimimos resultados
    print(f"\n=== {name} ===")
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", accuracy)
    print("Specificity:", specificity)
    print(classification_report(y_test, y_pred))
    if auc is not None:
        print("ROC AUC:", auc)

    # Graficamos la matriz de confusión
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred: No", "Pred: Yes"],
        yticklabels=["True: No", "True: Yes"],
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# Para seleccionar el mejor k en kNN
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Rango de k para probar
k_values = range(1, 31)
accuracies = []
aucs = []

for k in k_values:
    pipe = Pipeline(
        [("preproc", preprocessor), ("clf", KNeighborsClassifier(n_neighbors=k))]
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    # ROC AUC
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = None
    aucs.append(auc)

# Graficamos accuracy vs k
plt.figure(figsize=(8, 4))
plt.plot(k_values, accuracies, marker="o", label="Accuracy")
plt.plot(k_values, aucs, marker="s", label="ROC AUC")
plt.xticks(k_values)
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Score")
plt.title("kNN Performance for Different k")
plt.legend()
plt.grid(True)
plt.show()

# Mejor k
best_k_acc = k_values[accuracies.index(max(accuracies))]
best_k_auc = k_values[aucs.index(max(aucs))]
print(f"Best k by accuracy: {best_k_acc} (score={max(accuracies):.3f})")
print(f"Best k by ROC AUC: {best_k_auc} (score={max(aucs):.3f})")

# Graficamos el mejor k por accuracy
final_k = best_k_acc
print(f"\n>>> Refitting final kNN with k={final_k}")

final_knn = Pipeline(
    [("preproc", preprocessor), ("clf", KNeighborsClassifier(n_neighbors=final_k))]
)
final_knn.fit(X_train, y_train)
y_pred_final = final_knn.predict(X_test)
y_proba_final = final_knn.predict_proba(X_test)[:, 1]

# Metrics
print("\nFinal kNN Classification Report:")
print(classification_report(y_test, y_pred_final))
print("ROC AUC:", roc_auc_score(y_test, y_proba_final))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(4, 3))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Pred: No", "Pred: Yes"],
    yticklabels=["True: No", "True: Yes"],
)
plt.title(f"Confusion Matrix - kNN (k={final_k})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Gráfica de Proyecciones 1D del Criterio de Fisher
fisher_pipe = Pipeline(
    [("preproc", preprocessor), ("fisher", LinearDiscriminantAnalysis(n_components=1))]
)

X_train_proj = fisher_pipe.fit_transform(X_train, y_train)
X_test_proj = fisher_pipe.transform(X_test)

# Graficamos las proyecciones para el conjunto de prueba
plt.figure(figsize=(8, 5))
plt.hist(X_test_proj[y_test == 0], bins=30, alpha=0.6, label="No (0)")
plt.hist(X_test_proj[y_test == 1], bins=30, alpha=0.6, label="Yes (1)")
plt.axvline(X_test_proj.mean(), color="k", linestyle="--", linewidth=1)
plt.title("Fisher's Linear Discriminant Projection (Test Set)")
plt.xlabel("Projected 1D Feature")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Para obtener las variables más significativas
from sklearn.ensemble import RandomForestClassifier

rf = Pipeline(
    [("preproc", preprocessor), ("randF", RandomForestClassifier(random_state=42))]
)
rf.fit(X_train, y_train)

feature_names = rf.named_steps["preproc"].get_feature_names_out()
importances = pd.Series(
    rf.named_steps["randF"].feature_importances_, index=feature_names
)
print(importances.sort_values(ascending=False))
