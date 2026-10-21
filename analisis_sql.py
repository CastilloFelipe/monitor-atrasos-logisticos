"""
PASO 2: análisis exploratorio con SQL.

Busca qué variables separan los envíos atrasados de los que llegan a tiempo.
La técnica es siempre la misma: GROUP BY + AVG(atrasado). Como `atrasado`
vale 0 o 1, el promedio es directamente la tasa de atraso.

Ejecutar:  python analisis_sql.py
"""
import sqlite3
import pandas as pd

BD = "logistica.db"

CONSULTAS = {
    # [Paso 2.1] Tasa general
    "2.1 Tasa general de atraso": """
        SELECT COUNT(*) AS envios,
               ROUND(AVG(atrasado) * 100, 1) AS pct_atraso
        FROM envios
    """,
    # [Paso 2.2] Bodega: ¿alguna bodega es peor?
    "2.2 Por bodega": """
        SELECT bodega, COUNT(*) AS envios,
               ROUND(AVG(atrasado) * 100, 1) AS pct_atraso
        FROM envios GROUP BY bodega ORDER BY bodega
    """,
    # [Paso 2.3] Modo de envío
    "2.3 Por modo de envío": """
        SELECT modo_envio, COUNT(*) AS envios,
               ROUND(AVG(atrasado) * 100, 1) AS pct_atraso
        FROM envios GROUP BY modo_envio
    """,
    # [Paso 2.4] Importancia del producto
    "2.4 Por importancia": """
        SELECT importancia, COUNT(*) AS envios,
               ROUND(AVG(atrasado) * 100, 1) AS pct_atraso
        FROM envios GROUP BY importancia
    """,
    # [Paso 2.5] Descuento: el patrón más fuerte
    "2.5 Descuento > 10%": """
        SELECT descuento > 10 AS descuento_alto, COUNT(*) AS envios,
               ROUND(AVG(atrasado) * 100, 1) AS pct_atraso
        FROM envios GROUP BY descuento_alto
    """,
    # [Paso 2.6] Peso por tramos con CASE WHEN
    "2.6 Por tramo de peso": """
        SELECT CASE
                 WHEN peso_gr < 2000 THEN '1) < 2 kg'
                 WHEN peso_gr < 4000 THEN '2) 2 a 4 kg'
                 ELSE '3) > 4 kg'
               END AS tramo_peso,
               COUNT(*) AS envios,
               ROUND(AVG(atrasado) * 100, 1) AS pct_atraso
        FROM envios GROUP BY tramo_peso ORDER BY tramo_peso
    """,
}

with sqlite3.connect(BD) as con:
    for titulo, sql in CONSULTAS.items():
        print(f"\n[Paso {titulo}]")
        print(pd.read_sql(sql, con).to_string(index=False))

print("\n[Paso 2.7] Conclusión: descuento y peso explican el atraso; "
      "bodega y modo de envío casi no influyen.")
