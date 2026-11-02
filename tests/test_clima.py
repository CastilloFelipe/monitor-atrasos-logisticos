"""
PASO 8.5: tests de clima.py sin salir a internet.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import clima


@pytest.fixture(autouse=True)
def cache_vacio():
    clima._cache.update(datos=None, hora=0.0, error=None, error_hora=0.0)


def test_si_la_api_falla_no_insiste_en_cada_llamada(monkeypatch):
    llamadas = []

    def falla(*args, **kwargs):
        llamadas.append(1)
        raise OSError("HTTP Error 429: Too Many Requests")

    monkeypatch.setattr(clima.urllib.request, "urlopen", falla)
    assert clima.clima_actual()["disponible"] is False
    assert clima.clima_actual()["disponible"] is False
    assert clima.clima_en(clima.datetime.now()) == (None, None)
    assert len(llamadas) == 1  # el error se recuerda 5 minutos


def test_si_la_api_falla_usa_los_ultimos_datos_buenos(monkeypatch):
    clima._cache.update(datos={"current": {
        "time": "2026-11-02T12:00", "temperature_2m": 18.0, "precipitation": 0.0,
        "weather_code": 0, "wind_speed_10m": 4.0}}, hora=0.0)  # datos vencidos

    def falla(*args, **kwargs):
        raise OSError("HTTP Error 429: Too Many Requests")

    monkeypatch.setattr(clima.urllib.request, "urlopen", falla)
    c = clima.clima_actual()
    assert c["disponible"] is True
    assert c["temperatura"] == 18.0
