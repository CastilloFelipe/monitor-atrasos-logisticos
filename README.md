# Monitor de Atrasos Logísticos

Mini proyecto: **SQL + API REST + Machine Learning + dashboard JS + API externa**
para detectar envíos con riesgo de atraso en tiempo real.

Proyecto desarrollado en Python (FastAPI, scikit-learn, pandas, SQLite) y JavaScript (Chart.js), utilizando Claude (LLM) como asistente para la optimización de código y estructuración.

```
Train.csv ──► cargar_bd.py ──► SQLite (logistica.db) ──► entrenar_modelo.py ──► modelo + curva_umbral.json
                                    ▲                                                   │
                                    │                                                   ▼
simulador.py ──POST /envios──► api.py (FastAPI) ◄──────────────── usa el modelo ────────┘
dashboard ──POST /simular────►      │  ◄──── clima.py ◄──── Open-Meteo (API externa)
                                    │
               GET /kpis, /alertas, /clima, /umbral · POST /predict · PUT /umbral
                                    ▼
                         static/index.html (dashboard, fetch + Chart.js)
```

## Cómo correrlo

```bash
pip install -r requirements.txt
python cargar_bd.py          # Paso 1: CSV -> SQLite
python analisis_sql.py       # Paso 2: análisis exploratorio con SQL (opcional)
python entrenar_modelo.py    # Paso 3: entrena, guarda modelo y curva de umbral
uvicorn api:app --reload     # Paso 4: levanta la API -> http://127.0.0.1:8000
python simulador.py          # Paso 6: (otra terminal, opcional) envíos en vivo
```

Si el puerto 8000 está ocupado: `uvicorn api:app --port 8001` y `python simulador.py 8001`.

Documentación interactiva de la API: http://127.0.0.1:8000/docs

Cada script imprime sus pasos (`[Paso 1.1]`, `[Paso 1.2]`, ...) y el código tiene
las mismas marcas en comentarios, para seguirlo junto al manual.

## Archivos y pasos

| Paso | Archivo | Qué hace |
|---|---|---|
| 1 | `cargar_bd.py` | CSV → tabla `envios` en SQLite |
| 2 | `analisis_sql.py` | Consultas `GROUP BY` / `CASE WHEN` para encontrar patrones |
| 3 | `entrenar_modelo.py` | Random Forest + curva precisión/recall por umbral |
| 4 | `api.py` | Endpoints REST |
| 5 | `static/index.html` | Dashboard |
| 6 | `simulador.py` | Envíos en vivo vía `POST /envios` |
| 8.1 | `clima.py` | Consumo de API externa (Open-Meteo) |
| 8.2 | `api.py` + dashboard | Umbral de alerta ajustable |
| 8.3 | `api.py` + dashboard | Botones para simular un lapso de tiempo |

## Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/kpis` | % de atraso total, por bodega, modo e importancia |
| POST | `/predict` | Probabilidad, nivel (alto/medio/bajo) y motivos de un envío |
| POST | `/envios` | Igual que `/predict`, pero lo guarda con el clima de su hora |
| GET | `/alertas?nivel=alto&limite=20` | Envíos en curso; el nivel se calcula en SQL con el umbral vigente |
| DELETE | `/envios` | Reinicia la demo |
| GET | `/clima` | Clima actual en Quilicura (Open-Meteo, caché de 15 min) |
| GET / PUT | `/umbral` | Ver o cambiar el umbral de alerta (0,30 a 0,95) |
| POST | `/simular?horas=8` | Genera los envíos de N horas (uno cada 2 a 8 min) y adelanta el reloj |

## Datos

Dataset público ["E-Commerce Shipping Data" (Kaggle)](https://www.kaggle.com/datasets/prachi13/customer-analytics), 10.999 envíos.
**Ojo:** en el original `Reached.on.Time_Y.N = 1` significa que **NO** llegó a tiempo;
por eso se renombró a `atrasado`.

Hallazgos (`analisis_sql.py`):
- Descuento > 10 % → **100 %** atrasados (2.647 envíos).
- Peso entre 2 y 4 kg → **99,9 %** atrasados.
- Bodega y modo de envío casi no cambian la tasa (~58-60 %).
- Paradoja: los productos de importancia **alta** se atrasan más (65 % vs ~59 %).

## Modelo

- Random Forest (300 árboles, profundidad máx. 8) dentro de un `Pipeline` con One-Hot Encoding.
- Variables: bodega, modo, llamadas, rating, costo, compras previas, importancia, descuento, peso.
  Se excluyó `genero` a propósito.
- Resultado en test (20 %), umbral 0,5: **accuracy 0,68 · ROC AUC 0,75 · precisión 0,93 · recall 0,50**.
- Las variables más importantes son descuento (44 %) y peso (35 %).

### Umbral de alerta

| Umbral | Precisión | Recall | % envíos alertados |
|---|---|---|---|
| 0,40 | 0,60 | 0,98 | 97 % |
| 0,45 | 0,73 | 0,70 | 57 % |
| 0,50 | 0,93 | 0,50 | 32 % |
| 0,75 | 1,00 | 0,46 | 27 % |

El modelo separa con mucha seguridad ~la mitad de los atrasos (los de descuento alto
o peso 2-4 kg) y el resto queda cerca de 0,45. Por eso el cambio fuerte ocurre entre
0,40 y 0,50: el umbral lo debería decidir el negocio según el costo de un atraso
no detectado vs. una falsa alarma.

Niveles: **alto** ≥ umbral · **medio** hasta 10 puntos bajo el umbral · **bajo** el resto.

## Próximos pasos (para conversar)

- Usar valores SHAP para que los motivos salgan del modelo y no de reglas fijas.
- Reemplazar el polling cada 2 s por WebSockets.
- Conectar fuentes reales (GPS, TMS, WMS) en vez del simulador.
- Entrenar con el clima guardado en cada envío cuando haya resultados reales.
- Reentrenar periódicamente y monitorear el *drift* del modelo.
