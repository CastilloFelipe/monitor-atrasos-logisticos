"""
PASO 8.5: tests automáticos de la API.

Requiere haber corrido antes cargar_bd.py y entrenar_modelo.py.
Los tests trabajan sobre una COPIA de logistica.db (no tocan tus datos)
y reemplazan la API del clima por valores fijos (no salen a internet).

Ejecutar:  pytest
"""
import importlib
import shutil
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

ENVIO_RIESGOSO = {
    "bodega": "A", "modo_envio": "Road", "llamadas_cliente": 3,
    "rating_cliente": 2, "costo_producto": 150, "compras_previas": 2,
    "importancia": "high", "descuento": 25, "peso_gr": 2500,
}
ENVIO_TRANQUILO = {**ENVIO_RIESGOSO, "descuento": 5, "peso_gr": 4500}
CLAVE = "clave-de-prueba"


@pytest.fixture(scope="module")
def cliente(tmp_path_factory, monkeypatch_module):
    # [Paso 8.5.1] Base de datos de prueba y clima falso
    bd = tmp_path_factory.mktemp("bd") / "logistica.db"
    shutil.copy(RAIZ / "logistica.db", bd)
    monkeypatch_module.setenv("LOGISTICA_BD", str(bd))
    monkeypatch_module.setenv("CLAVE_ADMIN", CLAVE)
    monkeypatch_module.chdir(RAIZ)  # el modelo y static/ se leen con rutas relativas

    import api
    api = importlib.reload(api)
    monkeypatch_module.setattr(api, "clima_en", lambda momento: (15.0, 0.0))
    monkeypatch_module.setattr(api, "clima_actual", lambda: {
        "disponible": True, "hora": "2026-11-02T12:00", "temperatura": 15.0,
        "lluvia_mm": 0.0, "viento_kmh": 5.0, "descripcion": "Despejado"})
    return TestClient(api.app, headers={"X-Clave-Admin": CLAVE})


@pytest.fixture(scope="module")
def anonimo(cliente):
    """Visitante sin clave."""
    return TestClient(cliente.app)


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(autouse=True)
def demo_limpia(cliente):
    cliente.delete("/envios")
    cliente.put("/umbral", json={"umbral": 0.5})


# --- [Paso 8.5.2] KPIs ----------------------------------------------------------
def test_kpis_cuadran_con_el_dataset(cliente):
    k = cliente.get("/kpis").json()
    assert k["total"] == 10999
    assert k["pct_atraso"] == 59.7
    assert sum(b["total"] for b in k["por_bodega"]) == 10999


# --- [Paso 8.5.3] Predicción y explicación SHAP ---------------------------------
def test_envio_riesgoso_queda_en_alto_y_explica_el_descuento(cliente):
    r = cliente.post("/predict", json=ENVIO_RIESGOSO).json()
    assert r["prob_atraso"] > 0.9
    assert r["nivel"] == "alto"
    assert any(m.startswith("Descuento") for m in r["motivos"])


def test_envio_tranquilo_no_queda_en_alto(cliente):
    r = cliente.post("/predict", json=ENVIO_TRANQUILO).json()
    assert r["prob_atraso"] < 0.5
    assert r["nivel"] != "alto"


def test_shap_suma_la_probabilidad(cliente):
    r = cliente.post("/predict", json=ENVIO_RIESGOSO).json()
    total = r["base"] + sum(e["aporte"] for e in r["explicacion"])
    assert total == pytest.approx(r["prob_atraso"], abs=0.02)
    assert len(r["explicacion"]) == 9  # una entrada por variable original


def test_datos_invalidos_responden_422(cliente):
    assert cliente.post("/predict", json={**ENVIO_RIESGOSO, "bodega": "Z"}).status_code == 422
    assert cliente.post("/predict", json={**ENVIO_RIESGOSO, "rating_cliente": 9}).status_code == 422
    assert cliente.post("/predict", json={"bodega": "A"}).status_code == 422


# --- [Paso 8.5.4] Registro de envíos y alertas ----------------------------------
def test_envio_registrado_aparece_en_alertas_con_clima(cliente):
    nuevo = cliente.post("/envios", json=ENVIO_RIESGOSO).json()
    a = cliente.get("/alertas").json()
    assert a["envios"][0]["id"] == nuevo["id"]
    assert a["envios"][0]["temperatura"] == 15.0
    assert a["resumen"] == {"alto": 1}


def test_reiniciar_vacia_los_envios(cliente):
    cliente.post("/envios", json=ENVIO_RIESGOSO)
    cliente.delete("/envios")
    assert cliente.get("/alertas").json()["envios"] == []


# --- [Paso 8.5.5] Umbral ---------------------------------------------------------
def test_cambiar_umbral_reclasifica_sin_reescribir(cliente):
    cliente.post("/envios", json=ENVIO_TRANQUILO)  # prob. cercana a 0,45
    assert cliente.get("/alertas").json()["resumen"] != {"alto": 1}
    cliente.put("/umbral", json={"umbral": 0.30})
    assert cliente.get("/alertas").json()["resumen"] == {"alto": 1}


def test_umbral_fuera_de_rango_responde_422(cliente):
    assert cliente.put("/umbral", json={"umbral": 0.1}).status_code == 422
    assert cliente.put("/umbral", json={"umbral": 0.99}).status_code == 422


def test_umbral_entrega_la_curva(cliente):
    r = cliente.get("/umbral").json()
    assert r["umbral"] == 0.5
    assert len(r["curva"]) == 66


# --- [Paso 8.5.6] Simulación de lapsos ------------------------------------------
def test_simular_una_hora_adelanta_el_reloj(cliente):
    r = cliente.post("/simular", params={"horas": 1}).json()
    assert 5 <= r["generados"] <= 30  # uno cada 2 a 8 minutos
    assert r["generados"] == r["alto"] + r["medio"] + r["bajo"]
    a = cliente.get("/alertas", params={"limite": 200}).json()
    assert a["reloj"] == r["hasta"]
    assert sum(a["resumen"].values()) == r["generados"]


def test_clima_usa_la_api_externa(cliente):
    assert cliente.get("/clima").json()["descripcion"] == "Despejado"


# --- [Paso 8.8] Seguridad --------------------------------------------------------
def test_sin_clave_no_se_puede_modificar_nada(anonimo):
    assert anonimo.post("/envios", json=ENVIO_RIESGOSO).status_code == 401
    assert anonimo.delete("/envios").status_code == 401
    assert anonimo.put("/umbral", json={"umbral": 0.4}).status_code == 401
    assert anonimo.post("/simular", params={"horas": 1}).status_code == 401


def test_clave_incorrecta_responde_401(anonimo):
    r = anonimo.delete("/envios", headers={"X-Clave-Admin": "otra-clave"})
    assert r.status_code == 401


def test_sin_clave_se_puede_consultar(anonimo):
    assert anonimo.get("/kpis").status_code == 200
    assert anonimo.get("/alertas").status_code == 200
    assert anonimo.get("/umbral").status_code == 200
    assert anonimo.post("/predict", json=ENVIO_RIESGOSO).status_code == 200
    assert anonimo.get("/modo").json() == {"requiere_clave": True}


def test_simular_mas_de_un_dia_responde_422(cliente):
    assert cliente.post("/simular", params={"horas": 48}).status_code == 422


def test_la_tabla_en_curso_tiene_tope(cliente, monkeypatch):
    import api
    monkeypatch.setattr(api, "MAX_ENVIOS_EN_CURSO", 20)
    cliente.post("/simular", params={"horas": 8})  # ~100 envíos
    resumen = cliente.get("/alertas").json()["resumen"]
    assert sum(resumen.values()) <= 20
