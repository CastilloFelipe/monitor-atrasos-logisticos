"""
PASO 8.1: consumo de una API externa (Open-Meteo).

Obtiene el clima actual y el pronóstico por hora de la Planta Quilicura.
Open-Meteo es gratuita y no requiere clave.

Lo usa api.py para:
  - mostrar el clima actual en el dashboard (GET /clima)
  - guardar la temperatura y la lluvia de la hora en que llega cada envío,
    como dato extra para un modelo futuro

Probar solo:  python clima.py
"""
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime

# Coordenadas aproximadas de la Planta Quilicura
LAT, LON = -33.36, -70.73
URL = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
    "latitude": LAT,
    "longitude": LON,
    "current": "temperature_2m,precipitation,weather_code,wind_speed_10m",
    "hourly": "temperature_2m,precipitation",
    "forecast_days": 7,
    "timezone": "America/Santiago",
})
CACHE_SEGUNDOS = 15 * 60

# Códigos WMO más comunes -> descripción en español
DESCRIPCION = {
    0: "Despejado", 1: "Mayormente despejado", 2: "Parcialmente nublado",
    3: "Nublado", 45: "Niebla", 48: "Niebla", 51: "Llovizna", 53: "Llovizna",
    55: "Llovizna intensa", 61: "Lluvia débil", 63: "Lluvia", 65: "Lluvia intensa",
    80: "Chubascos", 81: "Chubascos", 82: "Chubascos fuertes",
    95: "Tormenta", 96: "Tormenta con granizo", 99: "Tormenta con granizo",
}

_cache = {"datos": None, "hora": 0.0}


def _descargar() -> dict:
    """[Paso 8.1.1] Llama a la API con caché de 15 minutos."""
    if _cache["datos"] and time.time() - _cache["hora"] < CACHE_SEGUNDOS:
        return _cache["datos"]
    with urllib.request.urlopen(URL, timeout=5) as resp:
        datos = json.load(resp)
    _cache.update(datos=datos, hora=time.time())
    return datos


def clima_actual() -> dict:
    """[Paso 8.1.2] Resume el clima actual. Si la API falla, no se cae el sistema."""
    try:
        c = _descargar()["current"]
    except Exception as e:
        return {"disponible": False, "error": str(e)}
    return {
        "disponible": True,
        "hora": c["time"],
        "temperatura": c["temperature_2m"],
        "lluvia_mm": c["precipitation"],
        "viento_kmh": c["wind_speed_10m"],
        "descripcion": DESCRIPCION.get(c["weather_code"], f"Código {c['weather_code']}"),
    }


def clima_en(momento: datetime) -> tuple:
    """[Paso 8.1.3] Temperatura y lluvia pronosticadas para la hora de `momento`.

    Devuelve (None, None) si la hora está fuera del pronóstico o la API falla.
    """
    try:
        horario = _descargar()["hourly"]
    except Exception:
        return None, None
    clave = momento.strftime("%Y-%m-%dT%H:00")
    if clave not in horario["time"]:
        return None, None
    i = horario["time"].index(clave)
    return horario["temperature_2m"][i], horario["precipitation"][i]


if __name__ == "__main__":
    print("[Paso 8.1] Clima actual:", clima_actual())
    print("[Paso 8.1] Pronóstico para esta hora:", clima_en(datetime.now()))
