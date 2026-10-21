"""
PASO 1: carga Train.csv a una base SQLite (logistica.db).

Se renombran las columnas a español y se aclara la columna objetivo:
en el dataset original Reached.on.Time_Y.N = 1 significa que NO llegó a tiempo,
así que la guardamos como `atrasado` (1 = atrasado) para que sea más clara.

Ejecutar:  python cargar_bd.py
"""
import sqlite3
import pandas as pd

CSV = "Train.csv"
BD = "logistica.db"

COLUMNAS = {
    "ID": "id",
    "Warehouse_block": "bodega",
    "Mode_of_Shipment": "modo_envio",
    "Customer_care_calls": "llamadas_cliente",
    "Customer_rating": "rating_cliente",
    "Cost_of_the_Product": "costo_producto",
    "Prior_purchases": "compras_previas",
    "Product_importance": "importancia",
    "Gender": "genero",
    "Discount_offered": "descuento",
    "Weight_in_gms": "peso_gr",
    "Reached.on.Time_Y.N": "atrasado",
}

# --- [Paso 1.1] Leer el CSV ---------------------------------------------------
df = pd.read_csv(CSV)
print(f"[Paso 1.1] CSV leído: {len(df)} filas, {df.shape[1]} columnas, "
      f"{df.isna().sum().sum()} valores nulos")

# --- [Paso 1.2] Renombrar columnas a español ----------------------------------
df = df.rename(columns=COLUMNAS)
print("[Paso 1.2] Columnas renombradas (Reached.on.Time_Y.N -> atrasado)")

with sqlite3.connect(BD) as con:
    # --- [Paso 1.3] Crear la tabla con un esquema explícito -------------------
    con.executescript("""
        DROP TABLE IF EXISTS envios;
        CREATE TABLE envios (
            id               INTEGER PRIMARY KEY,
            bodega           TEXT,
            modo_envio       TEXT,
            llamadas_cliente INTEGER,
            rating_cliente   INTEGER,
            costo_producto   REAL,
            compras_previas  INTEGER,
            importancia      TEXT,
            genero           TEXT,
            descuento        REAL,
            peso_gr          REAL,
            atrasado         INTEGER   -- 1 = no llegó a tiempo
        );
    """)
    print("[Paso 1.3] Tabla `envios` creada en", BD)

    # --- [Paso 1.4] Insertar los datos ----------------------------------------
    df.to_sql("envios", con, if_exists="append", index=False)
    print("[Paso 1.4] Datos insertados")

    # --- [Paso 1.5] Verificar la carga con una consulta -----------------------
    total, atrasados = con.execute(
        "SELECT COUNT(*), SUM(atrasado) FROM envios"
    ).fetchone()

print(f"[Paso 1.5] Verificación: {total} envíos en {BD} "
      f"({atrasados / total:.1%} atrasados)")
