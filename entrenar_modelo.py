"""
PASO 3: entrena un modelo que predice si un envío se va a atrasar.

Lee los datos desde SQLite (no desde el CSV) para que el flujo sea
BD -> modelo, igual que sería en un caso real.

Genera dos archivos:
  modelo_atrasos.joblib  -> el pipeline entrenado (lo carga la API)
  curva_umbral.json      -> precisión y recall para cada umbral posible
                            (lo usa el slider del dashboard). Se calcula con
                            validación cruzada SOLO en entrenamiento, para que
                            elegir el umbral no use los datos de prueba.

Además compara el modelo con dos líneas base en el mismo conjunto de prueba:
una regla de dos condiciones sacada del análisis SQL y una regresión logística.

Ejecutar:  python entrenar_modelo.py
"""
import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BD = "logistica.db"
MODELO = "modelo_atrasos.joblib"
CURVA = "curva_umbral.json"

CATEGORICAS = ["bodega", "modo_envio", "importancia"]
NUMERICAS = ["llamadas_cliente", "rating_cliente", "costo_producto",
             "compras_previas", "descuento", "peso_gr"]
# `genero` se deja fuera a propósito: no debería influir en un atraso logístico.

# --- [Paso 3.1] Leer los datos desde SQL --------------------------------------
with sqlite3.connect(BD) as con:
    df = pd.read_sql("SELECT * FROM envios", con)
print(f"[Paso 3.1] {len(df)} envíos leídos desde {BD}")

# --- [Paso 3.2] Separar variables (X) y objetivo (y); train/test 80/20 --------
X = df[CATEGORICAS + NUMERICAS]
y = df["atrasado"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[Paso 3.2] Train: {len(X_train)} | Test: {len(X_test)} (estratificado)")

# --- [Paso 3.3] Armar el pipeline: one-hot + Random Forest --------------------
# Pipeline = preprocesamiento + modelo en un solo objeto.
# Así la API le pasa los datos "crudos" y el pipeline hace el one-hot.
modelo = Pipeline([
    ("prep", ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)],
        remainder="passthrough",
    )),
    ("rf", RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42
    )),
])
print("[Paso 3.3] Pipeline armado: OneHotEncoder + RandomForest")

# --- [Paso 3.4] Entrenar ------------------------------------------------------
modelo.fit(X_train, y_train)
print("[Paso 3.4] Modelo entrenado")

# --- [Paso 3.5] Evaluar en test -----------------------------------------------
prob = modelo.predict_proba(X_test)[:, 1]
print("[Paso 3.5] Evaluación con umbral 0,5:")
print(classification_report(y_test, modelo.predict(X_test),
                            target_names=["a tiempo", "atrasado"]))
print(f"ROC AUC: {roc_auc_score(y_test, prob):.3f}")

# --- [Paso 3.5b] Comparar con líneas base (mismo conjunto de prueba) ----------
# Pregunta obligada: ¿el modelo aporta algo frente a algo simple?
# Regla: las dos condiciones que el análisis SQL mostró casi deterministas.
regla = ((X_test["descuento"] > 10)
         | ((X_test["peso_gr"] >= 2000) & (X_test["peso_gr"] < 4000))).astype(int)
logistica = Pipeline([
    ("prep", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ("num", StandardScaler(), NUMERICAS),
    ])),
    ("lr", LogisticRegression(max_iter=1000)),
]).fit(X_train, y_train)
prob_lr = logistica.predict_proba(X_test)[:, 1]
alerta_rf = (prob >= 0.5).astype(int)
print("\n[Paso 3.5b] Comparación en test (umbral 0,5):")
print(f"{'Método':<28}{'Accuracy':>9}{'Precisión':>11}{'Recall':>8}{'ROC AUC':>9}")
for nombre, pred, puntaje in [
    ("Regla (desc>10 o 2-4 kg)", regla, regla),
    ("Regresión logística", (prob_lr >= 0.5).astype(int), prob_lr),
    ("Random Forest", alerta_rf, prob),
]:
    print(f"{nombre:<28}{accuracy_score(y_test, pred):>9.3f}"
          f"{precision_score(y_test, pred):>11.3f}{recall_score(y_test, pred):>8.3f}"
          f"{roc_auc_score(y_test, puntaje):>9.3f}")
print(f"El Random Forest y la regla coinciden en el {100 * (alerta_rf == regla).mean():.1f}% "
      "de los envíos de prueba.")

# Estabilidad: ROC AUC con validación cruzada de 5 partes sobre entrenamiento
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_cv = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"ROC AUC con validación cruzada (train, 5 partes): {auc_cv.mean():.3f} ± {auc_cv.std():.3f}")

# --- [Paso 3.6] Importancia de variables --------------------------------------
nombres = modelo.named_steps["prep"].get_feature_names_out()
importancias = pd.Series(
    modelo.named_steps["rf"].feature_importances_, index=nombres
).sort_values(ascending=False)
print("\n[Paso 3.6] Variables más importantes:")
print(importancias.head(6).round(3).to_string())

# --- [Paso 3.7] Curva precisión / recall según el umbral ----------------------
# Para cada umbral u: se alerta si prob >= u.
#   precisión = de las alertas, cuántas eran atrasos reales
#   recall    = de los atrasos reales, cuántos alertamos
# La curva sirve para ELEGIR el umbral, así que no puede usar el test: se calcula
# con predicciones fuera de muestra (validación cruzada) del conjunto de entrenamiento.
prob_oof = cross_val_predict(modelo, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
curva = []
y_real = y_train.to_numpy()
for u in np.round(np.arange(0.30, 0.96, 0.01), 2):
    alerta = prob_oof >= u
    verdaderos = (alerta & (y_real == 1)).sum()
    curva.append({
        "umbral": float(u),
        "precision": round(verdaderos / max(alerta.sum(), 1), 3),
        "recall": round(verdaderos / (y_real == 1).sum(), 3),
        "pct_alertas": round(alerta.mean(), 3),
    })
with open(CURVA, "w") as f:
    json.dump(curva, f)
print(f"\n[Paso 3.7] Curva de {len(curva)} umbrales (validación cruzada en train) guardada en {CURVA}")
for punto in curva:
    if punto["umbral"] in (0.40, 0.45, 0.50, 0.75):
        print(f"  umbral {punto['umbral']:.2f}: precisión {punto['precision']:.2f} · "
              f"recall {punto['recall']:.2f} · {100 * punto['pct_alertas']:.0f}% alertados")

# --- [Paso 3.8] Guardar el modelo ---------------------------------------------
joblib.dump(modelo, MODELO)
print(f"[Paso 3.8] Modelo guardado en {MODELO}")
