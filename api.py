"""
PASO 4: API REST con FastAPI (+ mejoras del PASO 8).

Endpoints:
  GET  /kpis      -> indicadores históricos calculados con SQL
  POST /predict   -> probabilidad de atraso + explicación SHAP (no se guarda) [Paso 8.4]
  POST /envios    -> registra un envío "en curso", lo clasifica y lo guarda
  GET  /alertas   -> envíos en curso, con su nivel según el umbral actual
  DELETE /envios  -> reinicia la demo
  GET  /clima     -> clima actual en Quilicura (API externa Open-Meteo)  [Paso 8.1]
  GET  /umbral    -> umbral actual + curva precisión/recall              [Paso 8.2]
  PUT  /umbral    -> cambia el umbral de alerta                          [Paso 8.2]
  POST /simular   -> simula un lapso de tiempo (N horas de envíos)       [Paso 8.3]
  GET  /          -> dashboard (static/index.html)

Ejecutar:  uvicorn api:app --reload
Docs automáticas: http://127.0.0.1:8000/docs
"""
import json
import os
import random
import sqlite3
from datetime import datetime, timedelta
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from clima import clima_actual, clima_en

BD = os.environ.get("LOGISTICA_BD", "logistica.db")  # los tests usan una copia
COLUMNAS_MODELO = ["bodega", "modo_envio", "llamadas_cliente", "rating_cliente",
                   "costo_producto", "compras_previas", "importancia",
                   "descuento", "peso_gr"]
BANDA_MEDIO = 0.10  # "medio" = hasta 10 puntos bajo el umbral
APORTE_MINIMO = 0.02  # un motivo debe subir el riesgo al menos 2 puntos

# --- [Paso 4.1] Cargar el modelo y la curva una sola vez al arrancar ----------
modelo = joblib.load("modelo_atrasos.joblib")
with open("curva_umbral.json") as f:
    CURVA = json.load(f)

# --- [Paso 8.4] Explicador SHAP sobre el Random Forest ------------------------
# SHAP reparte la probabilidad de cada envío entre sus variables:
#   prob_atraso = BASE + suma de los aportes de cada variable
explicador = shap.TreeExplainer(modelo.named_steps["rf"])
BASE = float(np.ravel(explicador.expected_value)[-1])  # tasa media de atraso

# El one-hot convierte "bodega" en bodega_A, bodega_B... Para explicar, cada
# columna transformada se vuelve a asignar a la variable original.
VARIABLE_DE = []
for nombre in modelo.named_steps["prep"].get_feature_names_out():
    grupo, columna = nombre.split("__", 1)
    if grupo == "cat":
        columna = next(c for c in ["bodega", "modo_envio", "importancia"]
                       if columna.startswith(c + "_"))
    VARIABLE_DE.append(columna)

NOMBRE_VARIABLE = {
    "bodega": "Bodega", "modo_envio": "Modo", "importancia": "Importancia",
    "llamadas_cliente": "Llamadas", "rating_cliente": "Rating",
    "costo_producto": "Costo", "compras_previas": "Compras previas",
    "descuento": "Descuento", "peso_gr": "Peso",
}

app = FastAPI(title="Monitor de atrasos logísticos")
app.mount("/static", StaticFiles(directory="static"), name="static")


def conectar():
    con = sqlite3.connect(BD)
    con.row_factory = sqlite3.Row  # filas como diccionarios
    return con


# --- [Paso 4.2] Crear las tablas de trabajo si no existen ---------------------
with conectar() as con:
    con.executescript("""
        -- Envíos que van llegando (los alimentan simulador.py y /simular).
        -- No se guarda el nivel: se calcula al consultar, según el umbral vigente.
        CREATE TABLE IF NOT EXISTS envios_en_curso (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            recibido_en      TEXT,
            bodega           TEXT,
            modo_envio       TEXT,
            llamadas_cliente INTEGER,
            rating_cliente   INTEGER,
            costo_producto   REAL,
            compras_previas  INTEGER,
            importancia      TEXT,
            descuento        REAL,
            peso_gr          REAL,
            prob_atraso      REAL,
            motivos          TEXT,
            temperatura      REAL,   -- [Paso 8.1] clima de la hora de llegada
            lluvia_mm        REAL
        );

        -- [Paso 8.2] Parámetros configurables (por ahora solo el umbral)
        CREATE TABLE IF NOT EXISTS config (
            clave TEXT PRIMARY KEY,
            valor REAL
        );
        INSERT OR IGNORE INTO config VALUES ('umbral', 0.5);
    """)


# --- [Paso 4.3] Validación de entrada con Pydantic ----------------------------
class Envio(BaseModel):
    bodega: Literal["A", "B", "C", "D", "F"]
    modo_envio: Literal["Flight", "Ship", "Road"]
    llamadas_cliente: int = Field(ge=0)
    rating_cliente: int = Field(ge=1, le=5)
    costo_producto: float = Field(gt=0)
    compras_previas: int = Field(ge=0)
    importancia: Literal["low", "medium", "high"]
    descuento: float = Field(ge=0)
    peso_gr: float = Field(gt=0)


class NuevoUmbral(BaseModel):
    umbral: float = Field(ge=0.30, le=0.95)


# --- [Paso 4.4] Lógica de clasificación ---------------------------------------
def leer_umbral(con) -> float:
    return con.execute("SELECT valor FROM config WHERE clave = 'umbral'").fetchone()[0]


def nivel_de(prob: float, umbral: float) -> str:
    if prob >= umbral:
        return "alto"
    if prob >= umbral - BANDA_MEDIO:
        return "medio"
    return "bajo"


def predecir(envios: list) -> tuple:
    """[Paso 8.4.1] Probabilidad de atraso y explicación SHAP de cada envío.

    Una sola llamada al modelo y a SHAP para toda la lista.
    """
    X = pd.DataFrame(envios)[COLUMNAS_MODELO]
    probs = [round(float(p), 3) for p in modelo.predict_proba(X)[:, 1]]

    valores = explicador.shap_values(modelo.named_steps["prep"].transform(X))
    if isinstance(valores, list):  # versiones antiguas de shap: una lista por clase
        valores = valores[1]
    elif np.ndim(valores) == 3:    # versiones nuevas: (filas, columnas, clases)
        valores = valores[:, :, 1]

    explicaciones = []
    for envio, fila in zip(envios, valores):
        aportes = dict.fromkeys(COLUMNAS_MODELO, 0.0)
        for variable, aporte in zip(VARIABLE_DE, fila):
            aportes[variable] += float(aporte)
        explicaciones.append(sorted(
            ({"variable": v, "nombre": NOMBRE_VARIABLE[v], "valor": envio[v],
              "aporte": round(a, 3)}
             for v, a in aportes.items()),
            key=lambda e: abs(e["aporte"]), reverse=True,
        ))
    return probs, explicaciones


def motivos_de(explicacion: list) -> list:
    """[Paso 8.4.2] Hasta 3 variables que más suben el riesgo, en texto legible."""
    motivos = []
    for e in explicacion:
        if e["aporte"] < APORTE_MINIMO or len(motivos) == 3:
            continue
        valor = e["valor"]
        if e["variable"] == "peso_gr":
            valor = f"{valor / 1000:.1f} kg"
        elif isinstance(valor, float) and valor.is_integer():
            valor = int(valor)  # 25.0 -> 25
        motivos.append(f"{e['nombre']} {valor} "
                       f"(+{e['aporte'] * 100:.0f} pts)")
    return motivos


# El nivel se calcula en SQL con el umbral vigente, así al mover el slider
# se reclasifican todos los envíos sin reescribir la tabla.
NIVEL_SQL = """
    CASE WHEN prob_atraso >= :umbral THEN 'alto'
         WHEN prob_atraso >= :umbral - :banda THEN 'medio'
         ELSE 'bajo' END
"""


# --- [Paso 8.3] Reloj de la simulación ----------------------------------------
def reloj(con) -> datetime:
    """Hora actual de la simulación: la real, o la del último envío si /simular
    ya la adelantó."""
    ultimo = con.execute("SELECT MAX(recibido_en) FROM envios_en_curso").fetchone()[0]
    ahora = datetime.now().replace(microsecond=0)
    return max(ahora, datetime.fromisoformat(ultimo)) if ultimo else ahora


def guardar(con, envios: list, momentos: list):
    """Inserta envíos ya clasificados, con el clima de su hora de llegada."""
    probs, explicaciones = predecir(envios)
    filas = []
    for envio, momento, prob, expl in zip(envios, momentos, probs, explicaciones):
        temperatura, lluvia = clima_en(momento)
        filas.append((
            momento.isoformat(), *(envio[c] for c in COLUMNAS_MODELO),
            prob, ", ".join(motivos_de(expl)), temperatura, lluvia,
        ))
    con.executemany(f"""
        INSERT INTO envios_en_curso (
            recibido_en, {", ".join(COLUMNAS_MODELO)},
            prob_atraso, motivos, temperatura, lluvia_mm
        ) VALUES ({", ".join("?" * (len(COLUMNAS_MODELO) + 5))})
    """, filas)
    return probs, explicaciones


# --- [Paso 4.5] Endpoints -----------------------------------------------------
@app.get("/kpis")
def kpis():
    with conectar() as con:
        general = con.execute("""
            SELECT COUNT(*) AS total,
                   ROUND(AVG(atrasado) * 100, 1) AS pct_atraso
            FROM envios
        """).fetchone()

        def por(columna):
            filas = con.execute(f"""
                SELECT {columna} AS grupo,
                       COUNT(*) AS total,
                       ROUND(AVG(atrasado) * 100, 1) AS pct_atraso
                FROM envios
                GROUP BY {columna}
                ORDER BY {columna}
            """).fetchall()
            return [dict(f) for f in filas]

        return {
            **dict(general),
            "por_bodega": por("bodega"),
            "por_modo": por("modo_envio"),
            "por_importancia": por("importancia"),
        }


@app.post("/predict")
def predict(envio: Envio):
    probs, explicaciones = predecir([envio.model_dump()])
    with conectar() as con:
        umbral = leer_umbral(con)
    return {"prob_atraso": probs[0], "nivel": nivel_de(probs[0], umbral),
            "motivos": motivos_de(explicaciones[0]),
            "base": round(BASE, 3), "explicacion": explicaciones[0]}


@app.post("/envios")
def registrar_envio(envio: Envio):
    with conectar() as con:
        momento = reloj(con)
        probs, explicaciones = guardar(con, [envio.model_dump()], [momento])
        umbral = leer_umbral(con)
        nuevo_id = con.execute("SELECT MAX(id) FROM envios_en_curso").fetchone()[0]
    return {"id": nuevo_id, "recibido_en": momento.isoformat(), "prob_atraso": probs[0],
            "nivel": nivel_de(probs[0], umbral), "motivos": motivos_de(explicaciones[0])}


@app.get("/alertas")
def alertas(
    nivel: Literal["alto", "medio", "bajo", "todos"] = "todos",
    limite: int = Query(20, le=200),
):
    with conectar() as con:
        params = {"umbral": leer_umbral(con), "banda": BANDA_MEDIO,
                  "nivel": nivel, "limite": limite}
        filas = con.execute(f"""
            SELECT * FROM (
                SELECT *, {NIVEL_SQL} AS nivel FROM envios_en_curso
            )
            WHERE :nivel = 'todos' OR nivel = :nivel
            ORDER BY id DESC
            LIMIT :limite
        """, params).fetchall()
        resumen = con.execute(f"""
            SELECT {NIVEL_SQL} AS nivel, COUNT(*) AS n
            FROM envios_en_curso GROUP BY 1
        """, params).fetchall()
        hora = reloj(con)
    return {
        "reloj": hora.isoformat(),
        "umbral": params["umbral"],
        "resumen": {f["nivel"]: f["n"] for f in resumen},
        "envios": [dict(f) for f in filas],
    }


@app.delete("/envios")
def limpiar_envios():
    """Vacía la tabla de envíos en curso (reinicia la demo y el reloj)."""
    with conectar() as con:
        con.execute("DELETE FROM envios_en_curso")
    return {"ok": True}


# --- [Paso 8.1] Clima: consumo de API externa ---------------------------------
@app.get("/clima")
def clima():
    return clima_actual()


# --- [Paso 8.2] Umbral de alerta ajustable ------------------------------------
@app.get("/umbral")
def ver_umbral():
    with conectar() as con:
        return {"umbral": leer_umbral(con), "curva": CURVA}


@app.put("/umbral")
def cambiar_umbral(nuevo: NuevoUmbral):
    umbral = round(nuevo.umbral, 2)
    with conectar() as con:
        con.execute("UPDATE config SET valor = ? WHERE clave = 'umbral'", (umbral,))
    punto = min(CURVA, key=lambda p: abs(p["umbral"] - umbral))
    return {"umbral": umbral, **{k: v for k, v in punto.items() if k != "umbral"}}


# --- [Paso 8.3] Simular un lapso de tiempo ------------------------------------
@app.post("/simular")
def simular(horas: float = Query(1, gt=0, le=72)):
    """Genera los envíos que llegarían en las próximas `horas`:
    uno cada 2 a 8 minutos, tomados al azar del histórico."""
    with conectar() as con:
        inicio = reloj(con)
        fin = inicio + timedelta(hours=horas)

        # [Paso 8.3.1] Horas de llegada: cada 2 a 8 minutos hasta cubrir el lapso
        momentos, t = [], inicio
        while True:
            t += timedelta(minutes=random.uniform(2, 8))
            if t > fin:
                break
            momentos.append(t.replace(microsecond=0))
        if not momentos:
            return {"generados": 0, "desde": inicio.isoformat(), "hasta": fin.isoformat()}

        # [Paso 8.3.2] Envíos al azar desde el histórico, directo con SQL
        envios = [dict(f) for f in con.execute(f"""
            SELECT {", ".join(COLUMNAS_MODELO)} FROM envios
            ORDER BY RANDOM() LIMIT ?
        """, (len(momentos),))]

        # [Paso 8.3.3] Clasificar todos de una vez y guardar
        probs, _ = guardar(con, envios, momentos)
        umbral = leer_umbral(con)

    niveles = [nivel_de(p, umbral) for p in probs]
    return {
        "generados": len(momentos),
        "desde": inicio.isoformat(),
        "hasta": momentos[-1].isoformat(),
        "alto": niveles.count("alto"),
        "medio": niveles.count("medio"),
        "bajo": niveles.count("bajo"),
    }


@app.get("/")
def dashboard():
    return FileResponse("static/index.html")
