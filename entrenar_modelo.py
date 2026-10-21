"""
PASO 3: entrena un modelo que predice si un envío se va a atrasar.

Lee los datos desde SQLite (no desde el CSV) para que el flujo sea
BD -> modelo, igual que sería en un caso real.

Genera dos archivos:
  modelo_atrasos.joblib  -> el pipeline entrenado (lo carga la API)
  curva_umbral.json      -> precisión y recall para cada umbral posible
                            (lo usa el slider del dashboard)

Ejecutar:  python entrenar_modelo.py
"""
import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
curva = []
y_real = y_test.to_numpy()
for u in np.round(np.arange(0.30, 0.96, 0.01), 2):
    alerta = prob >= u
    verdaderos = (alerta & (y_real == 1)).sum()
    curva.append({
        "umbral": float(u),
        "precision": round(verdaderos / max(alerta.sum(), 1), 3),
        "recall": round(verdaderos / (y_real == 1).sum(), 3),
        "pct_alertas": round(alerta.mean(), 3),
    })
with open(CURVA, "w") as f:
    json.dump(curva, f)
print(f"\n[Paso 3.7] Curva de {len(curva)} umbrales guardada en {CURVA}")

# --- [Paso 3.8] Guardar el modelo ---------------------------------------------
joblib.dump(modelo, MODELO)
print(f"[Paso 3.8] Modelo guardado en {MODELO}")
