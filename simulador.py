"""
PASO 6: simula envíos llegando en tiempo real.

Toma envíos al azar de la tabla histórica, les agrega algo de ruido y los
manda a la API (POST /envios) cada 1-3 segundos, como lo haría otro sistema
(por ejemplo un TMS o un WMS) integrándose vía API.

Para simular horas completas de una vez, usar el botón del dashboard
o POST /simular (Paso 8.3).

Ejecutar (con la API corriendo):  python simulador.py
Si la API corre en otro puerto:   python simulador.py 8001
"""
import json
import random
import sqlite3
import sys
import time
import urllib.request

PUERTO = sys.argv[1] if len(sys.argv) > 1 else "8000"
API = f"http://127.0.0.1:{PUERTO}/envios"
BD = "logistica.db"

# --- [Paso 6.1] Leer el histórico SIN la columna `atrasado` -------------------
# (un envío en curso todavía no sabe si se va a atrasar)
with sqlite3.connect(BD) as con:
    con.row_factory = sqlite3.Row
    historicos = [dict(f) for f in con.execute("""
        SELECT bodega, modo_envio, llamadas_cliente, rating_cliente, costo_producto,
               compras_previas, importancia, descuento, peso_gr
        FROM envios
    """)]
print(f"[Paso 6.1] {len(historicos)} envíos históricos disponibles")

# --- [Paso 6.2] Ciclo: elegir, variar, enviar a la API, esperar ---------------
print(f"[Paso 6.2] Simulando envíos a {API} (Ctrl+C para detener)")
while True:
    envio = random.choice(historicos)
    envio["peso_gr"] = round(envio["peso_gr"] * random.uniform(0.95, 1.05))

    req = urllib.request.Request(
        API,
        data=json.dumps(envio).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        r = json.load(resp)

    print(f"#{r['id']:>4}  {envio['bodega']} {envio['modo_envio']:<6} "
          f"-> {r['prob_atraso']:.0%} [{r['nivel']}]  {', '.join(r['motivos'])}")
    time.sleep(random.uniform(1, 3))
