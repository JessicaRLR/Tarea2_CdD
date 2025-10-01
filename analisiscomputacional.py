import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


blue = "#4E84A5"
green = "#599C5F"
yellow = "#FEBD32"
red = "#D44127"
purple = "#D58BAA"
brown = "#B4917F"
orange = "#ED8038"
violet = "#DB547A"
gray = "#CCAA99"

# Pesos de la priori
pi0 = 0.5
pi1 = 1 - pi0

# Parámetros de las normales a considerar
mu0 = [1, 1]
mu1 = [-1, -1]

Sigma0 = [[1, 0], [0, 1]]
Sigma1 = [[2, -1], [-1, 4]]

n = 50

# Generamos los datos

X = []
y = []

for i in range(n):
    r = random.random()
    if r < pi0:
        newdata = np.random.multivariate_normal(mean=mu0, cov=Sigma0)
        y.append(0)
    else:
        newdata = np.random.multivariate_normal(mean=mu1, cov=Sigma1)
        y.append(1)
    X.append(newdata)


X = np.array(X)
y = np.array(y)

# print(X.shape)
# print(y.shape)

# Graficamos los puntos

df = pd.DataFrame(list(zip(X, y)), columns=["X", "y"])
df[["x1", "x2"]] = pd.DataFrame(df["X"].tolist(), index=df.index)
plt.figure(figsize=(6, 5))

for label in sorted(df["y"].unique()):
    subset = df[df["y"] == label]
    plt.scatter(
        subset["x1"],
        subset["x2"],
        label=f"Class {label}",
    )

plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Diagrama de dispersión de clases")
plt.show()

from sklearn.model_selection import train_test_split

# Separar train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("Tamaño entrenamiento:", X_train.shape)
print("Tamaño prueba:", X_test.shape)

# Función para graficar fronteras de decisión


def plot_decision_boundary(model, X, y, title):
    # Crear nueva figura cada vez
    plt.figure(figsize=(6, 5))

    # Crear grid en el espacio 2D
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Predicciones sobre el grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Graficar fronteras y puntos
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k", s=40)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.show()


# =====================================
# ======  Naive Bayes =================
# =====================================
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Entrenar
nb = GaussianNB()
nb.fit(X_train, y_train)

# Graficar frontera
plot_decision_boundary(nb, X_train, y_train, "Frontera de decisión - Naive Bayes")

# Evaluar
y_pred_nb = nb.predict(X_test)
cm = confusion_matrix(y_test, y_pred_nb)
print("Matriz de confusión (Naive Bayes):\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Precisión:", precision_score(y_test, y_pred_nb, average="weighted"))
print("Sensibilidad:", recall_score(y_test, y_pred_nb, average="weighted"))
print("F1-score:", f1_score(y_test, y_pred_nb, average="weighted"))


# =====================================
# =LDA (Linear Discriminant Analysis)=
# =====================================

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Entrenar
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Graficar frontera
plot_decision_boundary(lda, X_train, y_train, "Frontera de decisión - LDA")

# Evaluar
y_pred_lda = lda.predict(X_test)
cm = confusion_matrix(y_test, y_pred_lda)
print("Matriz de confusión (LDA):\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred_lda))
print("Precisión:", precision_score(y_test, y_pred_lda, average="weighted"))
print("Sensibilidad:", recall_score(y_test, y_pred_lda, average="weighted"))
print("F1-score:", f1_score(y_test, y_pred_lda, average="weighted"))

# =====================================
# QDA (Quadratic Discriminant Analysis) =
# =====================================
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Entrenar
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# Graficar frontera
plot_decision_boundary(qda, X_train, y_train, "Frontera de decisión - QDA")

# Evaluar
y_pred_qda = qda.predict(X_test)
cm = confusion_matrix(y_test, y_pred_qda)
print("Matriz de confusión (QDA):\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred_qda))
print("Precisión:", precision_score(y_test, y_pred_qda, average="weighted"))
print("Sensibilidad:", recall_score(y_test, y_pred_qda, average="weighted"))
print("F1-score:", f1_score(y_test, y_pred_qda, average="weighted"))


# =====================================
# k-NN (k-Nearest Neighbors) =
# =====================================
from sklearn.neighbors import KNeighborsClassifier

# Entrenar (con k=5 vecinos)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Graficar frontera
plot_decision_boundary(knn, X_train, y_train, "Frontera de decisión - k-NN (k=5)")

# Evaluar
y_pred_knn = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred_knn)
print("Matriz de confusión (k-NN):\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Precisión:", precision_score(y_test, y_pred_knn, average="weighted"))
print("Sensibilidad:", recall_score(y_test, y_pred_knn, average="weighted"))
print("F1-score:", f1_score(y_test, y_pred_knn, average="weighted"))


# =====================================
# Comparación de todos los modelos
# =====================================
import pandas as pd

# Guardar resultados en un diccionario
results = {
    "Naive Bayes": {
        "Acc": accuracy_score(y_test, y_pred_nb),
        "Precisión": precision_score(y_test, y_pred_nb, average="weighted"),
        "Recall": recall_score(y_test, y_pred_nb, average="weighted"),
        "F1": f1_score(y_test, y_pred_nb, average="weighted"),
    },
    "LDA": {
        "Acc": accuracy_score(y_test, y_pred_lda),
        "Precisión": precision_score(y_test, y_pred_lda, average="weighted"),
        "Recall": recall_score(y_test, y_pred_lda, average="weighted"),
        "F1": f1_score(y_test, y_pred_lda, average="weighted"),
    },
    "QDA": {
        "Acc": accuracy_score(y_test, y_pred_qda),
        "Precisión": precision_score(y_test, y_pred_qda, average="weighted"),
        "Recall": recall_score(y_test, y_pred_qda, average="weighted"),
        "F1": f1_score(y_test, y_pred_qda, average="weighted"),
    },
    "k-NN (k=5)": {
        "Acc": accuracy_score(y_test, y_pred_knn),
        "Precisión": precision_score(y_test, y_pred_knn, average="weighted"),
        "Recall": recall_score(y_test, y_pred_knn, average="weighted"),
        "F1": f1_score(y_test, y_pred_knn, average="weighted"),
    },
}

# Convertir a DataFrame para visualización
df_results = pd.DataFrame(results).T
print("\n=== Comparación Final de Modelos ===")
print(df_results.round(3))

# =====================================
# Validación cruzada comparativa
# =====================================

from sklearn.model_selection import StratifiedKFold, cross_val_score

models = {
    "Naive Bayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== Validación Cruzada (5-fold, accuracy) ===")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"{name:15s}: Acc = {scores.mean():.3f} ± {scores.std():.3f}")


# =====================================
# Matrices de confusión
# =====================================

import seaborn as sns

# Lista de modelos ya entrenados
trained_models = {"Naive Bayes": nb, "LDA": lda, "QDA": qda, "k-NN": knn}

# Graficar matrices de confusión
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for ax, (name, model) in zip(axes, trained_models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[0, 1],
        yticklabels=[0, 1],
        ax=ax,
    )
    ax.set_title(name)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")

plt.tight_layout()
plt.show()

# Clasificador óptimo de Bayes
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator, ClassifierMixin


def bayes_classifier(X, mu0, Sigma0, pi0, mu1, Sigma1, pi1):
    pdf0 = multivariate_normal.pdf(X, mean=mu0, cov=Sigma0)
    pdf1 = multivariate_normal.pdf(X, mean=mu1, cov=Sigma1)

    posterior0 = pdf0 * pi0
    posterior1 = pdf1 * pi1

    return (posterior1 > posterior0).astype(int)


# Predicción usando el clasificador de Bayes
y_pred_bayes = bayes_classifier(X_test, mu0, Sigma0, pi0, mu1, Sigma1, pi1)

# Evaluación del clasificador de Bayes
cm_bayes = confusion_matrix(y_test, y_pred_bayes)
print("Matriz de confusión (Optimal Bayes):\n", cm_bayes)
print("Accuracy (Optimal Bayes):", accuracy_score(y_test, y_pred_bayes))
print(
    "Precisión (Optimal Bayes):",
    precision_score(y_test, y_pred_bayes, average="weighted"),
)
print(
    "Sensibilidad (Optimal Bayes):",
    recall_score(y_test, y_pred_bayes, average="weighted"),
)
print("F1-score (Optimal Bayes):", f1_score(y_test, y_pred_bayes, average="weighted"))

# Graficar la frontera de decisión para clasificador de Bayes


# Creamos una clase para análisis posteriores
class BayesClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, mu0, Sigma0, pi0, mu1, Sigma1, pi1):
        self.mu0 = mu0
        self.Sigma0 = Sigma0
        self.pi0 = pi0
        self.mu1 = mu1
        self.Sigma1 = Sigma1
        self.pi1 = pi1

    def predict(self, X):
        return bayes_classifier(
            X, self.mu0, self.Sigma0, self.pi0, self.mu1, self.Sigma1, self.pi1
        )

    def fit(self, X, y):
        return self

    def get_params(self, deep=True):
        return {
            "mu0": self.mu0,
            "Sigma0": self.Sigma0,
            "pi0": self.pi0,
            "mu1": self.mu1,
            "Sigma1": self.Sigma1,
            "pi1": self.pi1,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


bayes_model = BayesClassifierWrapper(mu0, Sigma0, pi0, mu1, Sigma1, pi1)
plot_decision_boundary(
    bayes_model, X_train, y_train, "Frontera de decisión - Optimal Bayes"
)

y_pred = bayes_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=[0, 1],
    yticklabels=[0, 1],
    ax=ax,
)
ax.set_title("Matriz de confusión (Optimal Bayes)")
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
plt.show()

# Weighted kNN
from sklearn.neighbors import KNeighborsClassifier

knn_weighted = KNeighborsClassifier(n_neighbors=5, weights="distance")
knn_weighted.fit(X_train, y_train)

plot_decision_boundary(
    knn_weighted, X_train, y_train, "Frontera de decisión - Weighted k-NN (k=5)"
)

y_pred_knn_weighted = knn_weighted.predict(X_test)
cm_knn_weighted = confusion_matrix(y_test, y_pred_knn_weighted)
print("Matriz de confusión (Weighted k-NN):\n", cm_knn_weighted)
print("Accuracy (Weighted k-NN):", accuracy_score(y_test, y_pred_knn_weighted))
print(
    "Precisión (Weighted k-NN):",
    precision_score(y_test, y_pred_knn_weighted, average="weighted"),
)
print(
    "Sensibilidad (Weighted k-NN):",
    recall_score(y_test, y_pred_knn_weighted, average="weighted"),
)
print(
    "F1-score (Weighted k-NN):",
    f1_score(y_test, y_pred_knn_weighted, average="weighted"),
)

fig, ax = plt.subplots()
sns.heatmap(
    cm_knn_weighted,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=[0, 1],
    yticklabels=[0, 1],
    ax=ax,
)
ax.set_title("Weighted k-NN (k=5)")
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
plt.show()

# Comparación de todos los modelos

results["Optimal Bayes"] = {
    "Acc": accuracy_score(y_test, y_pred_bayes),
    "Precisión": precision_score(y_test, y_pred_bayes, average="weighted"),
    "Recall": recall_score(y_test, y_pred_bayes, average="weighted"),
    "F1": f1_score(y_test, y_pred_bayes, average="weighted"),
}

results["Weighted k-NN (k=5)"] = {
    "Acc": accuracy_score(y_test, y_pred_knn_weighted),
    "Precisión": precision_score(y_test, y_pred_knn_weighted, average="weighted"),
    "Recall": recall_score(y_test, y_pred_knn_weighted, average="weighted"),
    "F1": f1_score(y_test, y_pred_knn_weighted, average="weighted"),
}

df_results = pd.DataFrame(results).T
print("\n=== Comparación Final de Modelos (incl. Optimal Bayes and Weighted k-NN) ===")
print(df_results.round(3))

models["Optimal Bayes"] = BayesClassifierWrapper(mu0, Sigma0, pi0, mu1, Sigma1, pi1)
models["Weighted k-NN (k=5)"] = KNeighborsClassifier(n_neighbors=5, weights="distance")


print(
    "\n=== Validación Cruzada (5-fold, accuracy) (incl. Optimal Bayes and Weighted k-NN) ==="
)
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"{name:15s}: Acc = {scores.mean():.3f} ± {scores.std():.3f}")

trained_models["Optimal Bayes"] = bayes_model
trained_models["Weighted k-NN"] = knn_weighted

# Graficar matrices de confusión
fig, axes = plt.subplots(3, 2, figsize=(10, 12))  # Adjusted for 6 plots
axes = axes.ravel()

for ax, (name, model) in zip(axes, trained_models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[0, 1],
        yticklabels=[0, 1],
        ax=ax,
    )
    ax.set_title(name)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")

plt.tight_layout()
plt.show()

# Función para calcular el riesgo
from scipy.stats import multivariate_normal


def calculate_true_risk(g, mu0, Sigma0, pi0, mu1, Sigma1, pi1, num_samples=10000):
    num_samples0 = int(num_samples * pi0)
    num_samples1 = num_samples - num_samples0

    # Generar muestras de las distribuciones verdaderas
    X0 = np.random.multivariate_normal(mean=mu0, cov=Sigma0, size=num_samples0)
    X1 = np.random.multivariate_normal(mean=mu1, cov=Sigma1, size=num_samples1)

    # Combinar las muestras
    X_true = np.vstack((X0, X1))
    y_true = np.hstack((np.zeros(num_samples0), np.ones(num_samples1)))

    # Predecir usando el clasificador dado
    y_pred_true = g.predict(X_true)

    # Calcular el error de clasificación
    true_risk = 1 - accuracy_score(y_true, y_pred_true)

    return true_risk


# Calcular el riesgo usando distintos métodos
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import numpy as np


# Por validación cruzada
def estimate_risk_cv(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    return 1 - np.mean(accuracies)


# Por bootstrap
def estimate_risk_bootstrap(model, X, y, n_bootstraps=200):
    n_samples = X.shape[0]
    oob_errors = []
    no_information_errors = []

    for _ in range(n_bootstraps):
        # Hacemos una muestra de bootstrap
        bootstrap_indices = resample(
            np.arange(n_samples), n_samples=n_samples, replace=True, random_state=None
        )
        oob_indices = np.array(
            [i for i in np.arange(n_samples) if i not in bootstrap_indices]
        )

        if len(oob_indices) == 0:
            continue

        X_train_bs, y_train_bs = X[bootstrap_indices], y[bootstrap_indices]
        X_oob, y_oob = X[oob_indices], y[oob_indices]

        # Entrenamos la muestra bootstrap
        model.fit(X_train_bs, y_train_bs)

        # Calculamos el error (.632)
        y_pred_oob = model.predict(X_oob)
        oob_error = 1 - accuracy_score(y_oob, y_pred_oob)
        oob_errors.append(oob_error)

        # Calculamos el error (.632+)
        y_oob_shuffled = np.random.permutation(y_oob)
        y_pred_oob_shuffled = model.predict(X_oob)
        no_information_error = 1 - accuracy_score(y_oob_shuffled, y_pred_oob_shuffled)
        no_information_errors.append(no_information_error)

    avg_oob_error = np.mean(oob_errors)
    avg_no_information_error = np.mean(no_information_errors)

    # Calculamos los riesgos .632 y .632+
    risk_0632 = avg_oob_error
    risk_0632_plus = 0.632 * avg_oob_error + 0.368 * avg_no_information_error

    return risk_0632, risk_0632_plus


# Definimos los distintos parámetros para simulación
sample_sizes = [50, 100, 200, 500]
num_replications = 20
k_values = [1, 3, 5, 11, 21]

print("Simulation parameters defined:")
print(f"Sample sizes (n): {sample_sizes}")
print(f"Number of replications (R): {num_replications}")
print(f"k values for k-NN: {k_values}")

# Inicializamos las simulaciones
simulation_results = []


for n in sample_sizes:
    print(f"Simulating for n = {n}")

    for r in range(num_replications):

        X_sim = []
        y_sim = []
        for i in range(n):
            rand_num = random.random()
            if rand_num < pi0:
                newdata = np.random.multivariate_normal(mean=mu0, cov=Sigma0)
                y_sim.append(0)
            else:
                newdata = np.random.multivariate_normal(mean=mu1, cov=Sigma1)
                y_sim.append(1)
            X_sim.append(newdata)
        X_sim = np.array(X_sim)
        y_sim = np.array(y_sim)

        bayes_model = BayesClassifierWrapper(mu0, Sigma0, pi0, mu1, Sigma1, pi1)

        true_risk_bayes = calculate_true_risk(
            bayes_model, mu0, Sigma0, pi0, mu1, Sigma1, pi1
        )
        cv_risk_bayes = estimate_risk_cv(
            BayesClassifierWrapper(mu0, Sigma0, pi0, mu1, Sigma1, pi1), X_sim, y_sim
        )
        bootstrap_risk_bayes_0632, bootstrap_risk_bayes_0632_plus = (
            estimate_risk_bootstrap(
                BayesClassifierWrapper(mu0, Sigma0, pi0, mu1, Sigma1, pi1), X_sim, y_sim
            )
        )

        simulation_results.append(
            {
                "model": "Optimal Bayes",
                "n": n,
                "replication": r,
                "k": None,
                "true_risk": true_risk_bayes,
                "cv_risk": cv_risk_bayes,
                "bootstrap_0632_risk": bootstrap_risk_bayes_0632,
                "bootstrap_0632_plus_risk": bootstrap_risk_bayes_0632_plus,
            }
        )

        nb_model = GaussianNB()
        nb_model.fit(X_sim, y_sim)
        true_risk_nb = calculate_true_risk(nb_model, mu0, Sigma0, pi0, mu1, Sigma1, pi1)
        cv_risk_nb = estimate_risk_cv(GaussianNB(), X_sim, y_sim)
        bootstrap_risk_nb_0632, bootstrap_risk_nb_0632_plus = estimate_risk_bootstrap(
            GaussianNB(), X_sim, y_sim
        )
        simulation_results.append(
            {
                "model": "Naive Bayes",
                "n": n,
                "replication": r,
                "k": None,
                "true_risk": true_risk_nb,
                "cv_risk": cv_risk_nb,
                "bootstrap_0632_risk": bootstrap_risk_nb_0632,
                "bootstrap_0632_plus_risk": bootstrap_risk_nb_0632_plus,
            }
        )

        lda_model = LinearDiscriminantAnalysis()
        lda_model.fit(X_sim, y_sim)
        true_risk_lda = calculate_true_risk(
            lda_model, mu0, Sigma0, pi0, mu1, Sigma1, pi1
        )
        cv_risk_lda = estimate_risk_cv(LinearDiscriminantAnalysis(), X_sim, y_sim)
        bootstrap_risk_lda_0632, bootstrap_risk_lda_0632_plus = estimate_risk_bootstrap(
            LinearDiscriminantAnalysis(), X_sim, y_sim
        )
        simulation_results.append(
            {
                "model": "LDA",
                "n": n,
                "replication": r,
                "k": None,
                "true_risk": true_risk_lda,
                "cv_risk": cv_risk_lda,
                "bootstrap_0632_risk": bootstrap_risk_lda_0632,
                "bootstrap_0632_plus_risk": bootstrap_risk_lda_0632_plus,
            }
        )

        qda_model = QuadraticDiscriminantAnalysis()
        qda_model.fit(X_sim, y_sim)
        true_risk_qda = calculate_true_risk(
            qda_model, mu0, Sigma0, pi0, mu1, Sigma1, pi1
        )
        cv_risk_qda = estimate_risk_cv(QuadraticDiscriminantAnalysis(), X_sim, y_sim)
        bootstrap_risk_qda_0632, bootstrap_risk_qda_0632_plus = estimate_risk_bootstrap(
            QuadraticDiscriminantAnalysis(), X_sim, y_sim
        )
        simulation_results.append(
            {
                "model": "QDA",
                "n": n,
                "replication": r,
                "k": None,
                "true_risk": true_risk_qda,
                "cv_risk": cv_risk_qda,
                "bootstrap_0632_risk": bootstrap_risk_qda_0632,
                "bootstrap_0632_plus_risk": bootstrap_risk_qda_0632_plus,
            }
        )

        knn_weighted_model = KNeighborsClassifier(n_neighbors=5, weights="distance")
        knn_weighted_model.fit(X_sim, y_sim)
        true_risk_knn_weighted = calculate_true_risk(
            knn_weighted_model, mu0, Sigma0, pi0, mu1, Sigma1, pi1
        )
        cv_risk_knn_weighted = estimate_risk_cv(
            KNeighborsClassifier(n_neighbors=5, weights="distance"), X_sim, y_sim
        )
        bootstrap_risk_knn_weighted_0632, bootstrap_risk_knn_weighted_0632_plus = (
            estimate_risk_bootstrap(
                KNeighborsClassifier(n_neighbors=5, weights="distance"), X_sim, y_sim
            )
        )
        simulation_results.append(
            {
                "model": "Weighted k-NN (k=5)",
                "n": n,
                "replication": r,
                "k": 5,
                "true_risk": true_risk_knn_weighted,
                "cv_risk": cv_risk_knn_weighted,
                "bootstrap_0632_risk": bootstrap_risk_knn_weighted_0632,
                "bootstrap_0632_plus_risk": bootstrap_risk_knn_weighted_0632_plus,
            }
        )

        mu0_sim = np.mean(X_sim[y_sim == 0], axis=0)
        mu1_sim = np.mean(X_sim[y_sim == 1], axis=0)
        Sigma0_sim = np.cov(X_sim[y_sim == 0].T)
        Sigma1_sim = np.cov(X_sim[y_sim == 1].T)
        Sigma_comb_sim = (Sigma0_sim + Sigma1_sim) / 2

        from numpy.linalg import inv

        class FisherClassifierWrapper(BaseEstimator, ClassifierMixin):
            def __init__(self, mu0, mu1, Sigma):
                self.mu0 = mu0
                self.mu1 = mu1
                self.Sigma = Sigma

            def predict(self, X):
                Sigma_inv = inv(self.Sigma)
                w = Sigma_inv @ (self.mu1 - self.mu0)

                threshold = (w.T @ self.mu0 + w.T @ self.mu1) / 2
                projections = X @ w
                return (projections > threshold).astype(int)

            def fit(self, X, y):
                self.mu0 = np.mean(X[y == 0], axis=0)
                self.mu1 = np.mean(X[y == 1], axis=0)
                Sigma0 = np.cov(X[y == 0].T)
                Sigma1 = np.cov(X[y == 1].T)
                self.Sigma = (Sigma0 + Sigma1) / 2
                return self

            def get_params(self, deep=True):
                return {"mu0": self.mu0, "mu1": self.mu1, "Sigma": self.Sigma}

            def set_params(self, **parameters):
                for parameter, value in parameters.items():
                    setattr(self, parameter, value)
                return self

        fisher_model = FisherClassifierWrapper(mu0_sim, mu1_sim, Sigma_comb_sim)

        true_risk_fisher = calculate_true_risk(
            fisher_model, mu0, Sigma0, pi0, mu1, Sigma1, pi1
        )

        cv_risk_fisher = estimate_risk_cv(
            FisherClassifierWrapper(None, None, None), X_sim, y_sim
        )
        bootstrap_risk_fisher_0632, bootstrap_risk_fisher_0632_plus = (
            estimate_risk_bootstrap(
                FisherClassifierWrapper(None, None, None), X_sim, y_sim
            )
        )

        simulation_results.append(
            {
                "model": "Fisher",
                "n": n,
                "replication": r,
                "k": None,
                "true_risk": true_risk_fisher,
                "cv_risk": cv_risk_fisher,
                "bootstrap_0632_risk": bootstrap_risk_fisher_0632,
                "bootstrap_0632_plus_risk": bootstrap_risk_fisher_0632_plus,
            }
        )

        for k in k_values:
            # Train k-NN
            knn_model = KNeighborsClassifier(n_neighbors=k)
            knn_model.fit(X_sim, y_sim)

            true_risk_knn = calculate_true_risk(
                knn_model, mu0, Sigma0, pi0, mu1, Sigma1, pi1
            )
            cv_risk_knn = estimate_risk_cv(
                KNeighborsClassifier(n_neighbors=k), X_sim, y_sim
            )
            bootstrap_risk_knn_0632, bootstrap_risk_knn_0632_plus = (
                estimate_risk_bootstrap(
                    KNeighborsClassifier(n_neighbors=k), X_sim, y_sim
                )
            )

            simulation_results.append(
                {
                    "model": "k-NN",
                    "n": n,
                    "replication": r,
                    "k": k,
                    "true_risk": true_risk_knn,
                    "cv_risk": cv_risk_knn,
                    "bootstrap_0632_risk": bootstrap_risk_knn_0632,
                    "bootstrap_0632_plus_risk": bootstrap_risk_knn_0632_plus,
                }
            )

df_simulation_results = pd.DataFrame(simulation_results)

# Mostramos las primeras filas de los resultados de simulación
print(df_simulation_results.head())

# Presentamos los resultados en una tabla
aggregated_results = (
    df_simulation_results.groupby(["model", "n", "k"], dropna=False)[
        ["true_risk", "cv_risk", "bootstrap_0632_risk", "bootstrap_0632_plus_risk"]
    ]
    .agg(["mean", "std"])
    .reset_index()
)

aggregated_results.columns = [
    "_".join(col).strip("_") for col in aggregated_results.columns.values
]

# Display the aggregated results
print(aggregated_results)

# ================================
#  Creamos las gráficas
# ================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")


plt.figure(figsize=(10, 6))

for model_name in aggregated_results["model"].unique():
    if model_name == "k-NN":
        # Plot multiple k values for k-NN
        for k_val in aggregated_results[aggregated_results["model"] == "k-NN"][
            "k"
        ].unique():
            subset = aggregated_results[
                (aggregated_results["model"] == model_name)
                & (aggregated_results["k"] == k_val)
            ]
            plt.errorbar(
                subset["n"],
                subset["true_risk_mean"],
                yerr=subset["true_risk_std"],
                fmt="-o",
                label=f"{model_name} (k={k_val})",
            )
    else:
        # Plot Optimal Bayes, Naive Bayes, LDA, QDA, Fisher, Weighted k-NN
        subset = aggregated_results[aggregated_results["model"] == model_name]
        plt.errorbar(
            subset["n"],
            subset["true_risk_mean"],
            yerr=subset["true_risk_std"],
            fmt="-o",
            label=model_name,
        )


plt.xlabel("Sample Size (n)")
plt.ylabel("True Risk (L(g))")
plt.title("Evolution of True Risk with Sample Size")
plt.legend(title="Model")
plt.xscale("log")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# 2. (L(k-NN)) vs k para diferentes n
plt.figure(figsize=(10, 6))
knn_results = aggregated_results[aggregated_results["model"] == "k-NN"]
for n_val in knn_results["n"].unique():
    subset = knn_results[knn_results["n"] == n_val]
    plt.errorbar(
        subset["k"],
        subset["true_risk_mean"],
        yerr=subset["true_risk_std"],
        fmt="-o",
        label=f"n={n_val}",
    )

plt.xlabel("Number of Neighbors (k)")
plt.ylabel("True Risk (L(k-NN))")
plt.title("Evolution of True Risk with k for k-NN")
plt.legend(title="Sample Size (n)")
plt.show()


# 3. Brechas L(Bayes) - L(g) vs n para kNN
bayes_true_risk_means = aggregated_results[
    aggregated_results["model"] == "Optimal Bayes"
].set_index("n")["true_risk_mean"]

aggregated_results["true_risk_gap_mean"] = aggregated_results.apply(
    lambda row: bayes_true_risk_means[row["n"]] - row["true_risk_mean"], axis=1
)

plt.figure(figsize=(10, 6))
knn_results_with_gap = aggregated_results[aggregated_results["model"] == "k-NN"].copy()

for k_val in knn_results_with_gap["k"].unique():
    subset = knn_results_with_gap[knn_results_with_gap["k"] == k_val]
    plt.plot(subset["n"], subset["true_risk_gap_mean"], "-o", label=f"k={k_val}")

plt.xlabel("Sample Size (n)")
plt.ylabel("True Risk Gap (L(Optimal Bayes) - L(k-NN))")
plt.title("True Risk Gap between Optimal Bayes and k-NN vs Sample Size")
plt.legend(title="k for k-NN")
plt.xscale("log")
plt.show()

# 4. Brechas todos los modelos
plt.figure(figsize=(10, 6))
for model_name in aggregated_results["model"].unique():
    if model_name != "Optimal Bayes":
        if model_name == "k-NN":

            for k_val in aggregated_results[aggregated_results["model"] == "k-NN"][
                "k"
            ].unique():
                subset = aggregated_results[
                    (aggregated_results["model"] == model_name)
                    & (aggregated_results["k"] == k_val)
                ]
                plt.plot(
                    subset["n"],
                    subset["true_risk_gap_mean"],
                    "-o",
                    label=f"{model_name} (k={k_val})",
                )
        else:

            subset = aggregated_results[aggregated_results["model"] == model_name]
            plt.plot(subset["n"], subset["true_risk_gap_mean"], "-o", label=model_name)

plt.xlabel("Sample Size (n)")
plt.ylabel("True Risk Gap (L(Optimal Bayes) - L(g))")
plt.title("True Risk Gap between Optimal Bayes and Other Models vs Sample Size")
plt.legend(title="Model")
plt.xscale("log")
plt.show()

# Comparación de L_{CV} vs L_g
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

comparison_data = []
for index, row in aggregated_results.iterrows():
    model_name = row["model"]
    n_val = row["n"]
    k_val = row["k"] if pd.notna(row["k"]) else "N/A"

    true_risk_mean = row["true_risk_mean"]
    true_risk_std = row["true_risk_std"]

    comparison_data.append(
        {
            "model": model_name,
            "n": n_val,
            "k": k_val,
            "method": "True (Monte Carlo)",
            "risk_mean": true_risk_mean,
            "risk_std": true_risk_std,
        }
    )
    comparison_data.append(
        {
            "model": model_name,
            "n": n_val,
            "k": k_val,
            "method": "CV",
            "risk_mean": row["cv_risk_mean"],
            "risk_std": row["cv_risk_std"],
        }
    )
    comparison_data.append(
        {
            "model": model_name,
            "n": n_val,
            "k": k_val,
            "method": "Bootstrap .632",
            "risk_mean": row["bootstrap_0632_risk_mean"],
            "risk_std": row["bootstrap_0632_risk_std"],
        }
    )
    comparison_data.append(
        {
            "model": model_name,
            "n": n_val,
            "k": k_val,
            "method": "Bootstrap .632+",
            "risk_mean": row["bootstrap_0632_plus_risk_mean"],
            "risk_std": row["bootstrap_0632_plus_risk_std"],
        }
    )

df_comparison = pd.DataFrame(comparison_data)

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_comparison,
    x="risk_mean",
    y="risk_mean",
    hue="method",
    style="model",
    size="n",
    sizes=(50, 200),
    alpha=0.7,
)

max_risk = df_comparison["risk_mean"].max() * 1.1
plt.plot([0, max_risk], [0, max_risk], "k--", lw=1, label="Perfect Agreement")


plt.xlabel("True Risk (L(g)) - Mean over Replications")
plt.ylabel("Estimated Risk - Mean over Replications")
plt.title("Comparison of Estimated Risk vs True Risk")
plt.legend(title="Method / Model / n")
plt.xlim(0, max_risk)
plt.ylim(0, max_risk)
plt.show()
