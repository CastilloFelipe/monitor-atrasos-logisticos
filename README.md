# Monitor de Atrasos Logísticos

[![Tests](https://github.com/CastilloFelipe/monitor-atrasos-logisticos/actions/workflows/tests.yml/badge.svg)](https://github.com/CastilloFelipe/monitor-atrasos-logisticos/actions/workflows/tests.yml)

**App en línea:** https://monitor-atrasos-logisticos.onrender.com
(plan gratuito: si nadie la usó en los últimos 15 min, la primera carga tarda cerca de 1 min)

Mini proyecto: **SQL + API REST + Machine Learning explicable (SHAP) + dashboard JS + API externa**
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

Tests (requieren haber corrido los pasos 1 y 3):

```bash
pip install -r requirements-dev.txt
pytest
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
| 8.4 | `api.py` + dashboard | Motivos de cada alerta explicados con SHAP |
| 8.5 | `tests/` | 19 tests automáticos de la API, el clima y la seguridad (pytest) |
| 8.6 | `.github/workflows/tests.yml` | GitHub Actions corre los tests en cada push |
| 8.7 | `render.yaml` | Despliegue en Render (plan gratuito) |
| 8.8 | `api.py`, dashboard, `.github/` | Seguridad: clave para modificar datos, límites, versiones fijas, Dependabot |

## Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/kpis` | % de atraso total, por bodega, modo e importancia |
| POST | `/predict` | Probabilidad, nivel (alto/medio/bajo), motivos y aporte SHAP de cada variable |
| POST | `/envios` | Igual que `/predict`, pero lo guarda con el clima de su hora |
| GET | `/alertas?nivel=alto&limite=20` | Envíos en curso; el nivel se calcula en SQL con el umbral vigente |
| DELETE | `/envios` | Reinicia la demo |
| GET | `/clima` | Clima actual en Quilicura (Open-Meteo, caché de 15 min) |
| GET / PUT | `/umbral` | Ver o cambiar el umbral de alerta (0,30 a 0,95) |
| POST | `/simular?horas=8` | Genera los envíos de N horas (máx. 24; uno cada 2 a 8 min) y adelanta el reloj |
| GET | `/modo` | Indica si las rutas que modifican datos piden clave |

Con la variable `CLAVE_ADMIN` definida, `POST /envios`, `DELETE /envios`, `PUT /umbral`
y `POST /simular` exigen el header `X-Clave-Admin`. Consultar y predecir sigue siendo público.

## Datos

Dataset público ["E-Commerce Shipping Data" (Kaggle)](https://www.kaggle.com/datasets/prachi13/customer-analytics), 10.999 envíos.
**Ojo:** en el original `Reached.on.Time_Y.N = 1` significa que **NO** llegó a tiempo;
por eso se renombró a `atrasado`.

Hallazgos (`analisis_sql.py`):
- Descuento > 10 → **100 %** atrasados (2.647 envíos). El dataset no indica si el descuento es porcentaje o monto.
- Peso entre 2 y 4 kg → **99,9 %** atrasados.
- Bodega y modo de envío casi no cambian la tasa (~58-60 %).
- Paradoja: los productos de importancia **alta** se atrasan más (65 % vs ~59 %).

## Modelo

- Random Forest (300 árboles, profundidad máx. 8) dentro de un `Pipeline` con One-Hot Encoding.
- Variables: bodega, modo, llamadas, rating, costo, compras previas, importancia, descuento, peso.
  Se excluyó `genero` a propósito.
- Resultado en test (20 %), umbral 0,5: **accuracy 0,68 · ROC AUC 0,75 · precisión 0,93 · recall 0,50**.
  Con validación cruzada de 5 partes en entrenamiento: ROC AUC 0,736 ± 0,012 (estable).
- Las variables más importantes son descuento (44 %) y peso (35 %).

### ¿El modelo aporta algo frente a una regla simple?

El análisis SQL mostró dos patrones casi deterministas, así que la comparación obligada es
contra una regla con esas dos condiciones. Mismo conjunto de prueba, umbral 0,5:

| Método | Accuracy | Precisión | Recall | ROC AUC |
|---|---:|---:|---:|---:|
| Regla: descuento > 10 **o** peso entre 2 y 4 kg | 0,660 | 1,000 | 0,431 | 0,716 |
| Regresión logística | 0,639 | 0,709 | 0,670 | 0,717 |
| **Random Forest** | **0,679** | 0,928 | 0,501 | **0,748** |

El Random Forest y la regla coinciden en el **93,5 %** de los envíos. El modelo aporta, pero
poco: unos 7 puntos de recall y 3 de ROC AUC sobre la regla. En la práctica, la mayor parte
del valor está en descubrir esos dos patrones; el modelo los refina y, sobre todo, entrega una
probabilidad y una explicación (SHAP) para cada envío, que la regla no da. Este dataset es
conocido por tener patrones muy marcados, así que no conviene generalizar estas cifras a una
operación real.

### Umbral de alerta

La curva que usa el slider del dashboard se calcula con **validación cruzada sobre el
conjunto de entrenamiento**: elegir el umbral mirando el test sería usar el examen para
estudiar. El test queda solo para la evaluación final de arriba.

| Umbral | Precisión | Recall | % envíos alertados |
|---|---|---|---|
| 0,40 | 0,60 | 0,96 | 95 % |
| 0,45 | 0,70 | 0,71 | 60 % |
| 0,50 | 0,89 | 0,52 | 35 % |
| 0,75 | 0,99 | 0,46 | 28 % |

El modelo separa con mucha seguridad ~la mitad de los atrasos (los de descuento alto
o peso 2-4 kg) y el resto queda cerca de 0,45. Por eso el cambio fuerte ocurre entre
0,40 y 0,50: el umbral lo debería decidir el negocio según el costo de un atraso
no detectado vs. una falsa alarma.

Niveles: **alto** ≥ umbral · **medio** hasta 10 puntos bajo el umbral · **bajo** el resto.

### Explicaciones con SHAP

Cada predicción se descompone con `shap.TreeExplainer`:

```
probabilidad de atraso = promedio (59,7 %) + suma de los aportes de cada variable
```

Los **motivos** de una alerta son las hasta 3 variables que más suben el riesgo
(al menos 2 puntos), por ejemplo `Peso 2.5 kg (+20 pts) · Descuento 25 (+17 pts)`.
Antes eran reglas fijas sacadas del análisis SQL; ahora salen del propio modelo.
El dashboard muestra el aporte de las 9 variables como barras (rojo sube, verde baja).

## Seguridad

| Medida | Detalle |
|---|---|
| Clave de administrador | En Render, `CLAVE_ADMIN` la genera Render al azar (*Environment* del servicio). Sin la clave, el dashboard queda en solo lectura: se ven las alertas y se puede predecir, pero no simular, reiniciar ni cambiar el umbral. La clave se escribe en el campo "Clave admin" y solo se guarda mientras la pestaña está abierta. En local, sin la variable, todo queda abierto. |
| Límites | Máximo 24 h por simulación y 5.000 envíos en curso (los más antiguos se borran) |
| Inyección SQL | Todas las consultas usan parámetros; el texto SQL solo se arma con nombres de columnas fijos |
| Validación | Pydantic rechaza datos fuera de rango antes de llegar al modelo o a la base |
| Chart.js | Se carga con `integrity` (SRI): el navegador lo rechaza si el CDN lo modifica |
| Dependencias | Versiones exactas en `requirements.txt`; Dependabot propone las actualizaciones |
| GitHub Actions | Permisos de solo lectura |

Para usar el simulador contra una API con clave: definir `CLAVE_ADMIN` antes de `python simulador.py`.

## Despliegue en Render

`render.yaml` define el servicio: instala dependencias, genera la base y el modelo
durante el build y levanta `uvicorn`. Para publicarlo: en Render, *New → Blueprint*
y elegir este repositorio. En el plan gratuito el servicio se duerme tras 15 min sin
visitas (la primera carga tarda cerca de 1 min) y los envíos en curso se pierden en
cada despliegue. Con la cuenta de GitHub conectada en Render, cada push a `main`
se despliega solo (Auto-Deploy "On Commit").

Dos ajustes que salieron al publicarlo:
- El servidor de Render usa hora UTC; la API calcula la hora en `America/Santiago`
  para que el reloj y el clima por hora correspondan a la planta.
- Render comparte su IP con muchos servicios y Open-Meteo a veces responde 429
  (demasiadas consultas). `clima.py` recuerda el error 5 minutos, usa los últimos datos
  buenos si los tiene, y el resto de la app sigue funcionando sin clima.

## Próximos pasos (para conversar)

- Reemplazar el polling cada 2 s por WebSockets.
- Conectar fuentes reales (GPS, TMS, WMS) en vez del simulador.
- Entrenar con el clima guardado en cada envío cuando haya resultados reales.
- Reentrenar periódicamente y monitorear el *drift* del modelo.

## Licencia

Código bajo licencia [MIT](LICENSE). Autor: Felipe Castillo, estudiante de Ingeniería Civil
Industrial, Universidad del Desarrollo (Santiago, Chile). Los datos pertenecen a su autor en
Kaggle (ver [Datos](#datos)) y se incluyen solo con fines de demostración.
