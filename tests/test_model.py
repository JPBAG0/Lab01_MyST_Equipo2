"""
Pruebas del modelo de Copeland-Galai.

Las tres primeras son las obligatorias del enunciado (seccion 3.6); el resto
cubre la simulacion y las propiedades economicas del optimo.
"""
import numpy as np
import pandas as pd
import pytest

from model import (Parametros, SEED, ganancia_liquidez, optimizar,
                   perdida_ask, perdida_bid, perdida_informados,
                   prob_ejecucion, sensibilidad, spread_monopolista,
                   utilidad_esperada)
from simulation import (REGIMENES_FIJOS, monte_carlo, resumen_monte_carlo,
                        simular_regimenes, simular_trades)


@pytest.fixture
def par():
    return Parametros()


# ---------------------------------------------------------------- obligatoria 1
def test_prob_ejecucion_nunca_es_negativa(par):
    """pi_LB(s) y pi_LS(s) se truncan en cero, nunca devuelven negativos."""
    s = np.linspace(0.0, 3.0 * par.s_max, 500)
    p = prob_ejecucion(s, par.alpha, par.beta)
    assert np.all(p >= 0.0)
    assert p[0] == pytest.approx(par.alpha)                  # s = 0 -> alpha
    assert prob_ejecucion(par.s_max, par.alpha, par.beta) == pytest.approx(0.0)
    assert prob_ejecucion(1e6, par.alpha, par.beta) == 0.0   # truncado, no negativo


# ---------------------------------------------------------------- obligatoria 2
@pytest.mark.parametrize("A", [20.0, 21.0, 22.0, 23.0, 24.0])
def test_perdida_ask_es_decreciente_en_A(par, A):
    """
    La perdida esperada frente a informados decrece al subir el ask.

    Es la razon economica del spread: cotizar mas arriba reduce el conjunto de
    valores P > A con los que el informado le gana al dealer.
    """
    assert perdida_ask(A, par) > perdida_ask(A + 0.5, par) >= 0.0


def test_perdida_informados_decrece_en_A_sobre_una_malla(par):
    """La monotonia se sostiene sobre toda la malla, no solo entre dos puntos."""
    As = np.linspace(par.S0, par.S0 + par.s_max, 12)
    perdidas = [perdida_ask(A, par) for A in As]
    assert np.all(np.diff(perdidas) < 0.0)


# ---------------------------------------------------------------- obligatoria 3
def test_sin_informados_el_optimo_es_el_del_monopolista():
    """
    Con pi_I = 0 el optimo coincide con el resultado analitico del monopolista.

    Sin seleccion adversa el dealer maximiza s*(alpha - beta*s) por lado, cuyo
    maximo esta en s* = alpha/(2*beta) = 3.125, o sea un spread total de
    alpha/beta = 0.50/0.08 = 6.25 (el numero que reporta el enunciado).
    """
    par = Parametros(pi_I=0.0)
    res = optimizar(par)

    assert res["semi_spread_ask"] == pytest.approx(spread_monopolista(par), abs=1e-4)
    assert res["semi_spread_bid"] == pytest.approx(spread_monopolista(par), abs=1e-4)
    assert res["semi_spread_ask"] == pytest.approx(0.50 / (2 * 0.08), abs=1e-4)
    assert res["spread"] == pytest.approx(0.50 / 0.08, abs=1e-4)
    assert res["perdida_informados"] == pytest.approx(0.0)


# ---------------------------------------------------------------- utilidad
def test_utilidad_es_ganancia_menos_perdida(par):
    """Pi(A,B) = G(A,B) - L(A,B), pieza por pieza."""
    A, B = 22.0, 18.0
    assert utilidad_esperada(A, B, par) == pytest.approx(
        ganancia_liquidez(A, B, par) - perdida_informados(A, B, par))


def test_sin_spread_no_hay_ganancia_y_si_perdida(par):
    """Cotizar A = B = S0 deja al dealer sin margen y expuesto: Pi < 0."""
    assert ganancia_liquidez(par.S0, par.S0, par) == pytest.approx(0.0)
    assert perdida_informados(par.S0, par.S0, par) > 0.0
    assert utilidad_esperada(par.S0, par.S0, par) < 0.0


def test_en_s_max_la_ganancia_se_anula(par):
    """En s = alpha/beta el no informado ya no llega: G = 0 y Pi <= 0."""
    A, B = par.S0 + par.s_max, par.S0 - par.s_max
    assert ganancia_liquidez(A, B, par) == pytest.approx(0.0)
    assert utilidad_esperada(A, B, par) <= 0.0


def test_las_integrales_son_consistentes_con_la_media(par):
    """
    Con A = B = S0 las dos colas reconstruyen E|P - S0|.

    Valida que quad esta integrando la densidad correcta: la suma de las dos
    perdidas debe dar la desviacion absoluta media respecto de S0.
    """
    P = par.distribucion()
    muestra = P.rvs(size=400_000, random_state=np.random.default_rng(SEED))
    esperado = np.mean(np.abs(muestra - par.S0))
    obtenido = perdida_ask(par.S0, par) + perdida_bid(par.S0, par)
    assert obtenido == pytest.approx(esperado, rel=0.01)


# ---------------------------------------------------------------- optimo
def test_el_optimo_le_gana_a_los_regimenes_fijos(par):
    """Ningun regimen del enunciado supera la utilidad esperada del optimo."""
    opt = optimizar(par)
    for q in REGIMENES_FIJOS.values():
        assert opt["utilidad"] >= utilidad_esperada(q["ask"], q["bid"], par)


def test_el_optimo_respeta_las_restricciones(par):
    """B en (0, S0] y A en [S0, inf)."""
    opt = optimizar(par)
    assert 0.0 < opt["bid"] <= par.S0 <= opt["ask"]


def test_mas_informados_ensancha_el_spread_y_encoge_la_utilidad(par):
    """
    Prediccion central de Copeland-Galai: el spread es el precio de la
    seleccion adversa, asi que crece de forma monotona en pi_I.
    """
    tabla = sensibilidad([0.1, 0.4, 0.7], par)
    assert tabla["spread"].is_monotonic_increasing
    assert tabla["utilidad"].is_monotonic_decreasing


# ---------------------------------------------------------------- simulacion
def test_la_simulacion_es_reproducible(par):
    """La misma semilla da exactamente la misma corrida."""
    kwargs = dict(bid=18.4, ask=21.4, n_trades=2_000, par=par)
    a = simular_trades(rng=np.random.default_rng(SEED), **kwargs)
    b = simular_trades(rng=np.random.default_rng(SEED), **kwargs)
    pd.testing.assert_frame_equal(a, b)


def test_la_simulacion_registra_lo_exigido(par):
    """Cada trade lleva P&L, cambio de inventario, tipo de trader y direccion."""
    df = simular_trades(18.4, 21.4, 1_000, par, np.random.default_rng(SEED))
    for col in ("pnl", "cambio_inventario", "informado", "compra", "ejecutado"):
        assert col in df.columns
    assert len(df) == 1_000
    assert set(np.unique(df["cambio_inventario"])) <= {-1, 0, 1}
    assert df["informado"].mean() == pytest.approx(par.pi_I, abs=0.05)


def test_el_dealer_siempre_pierde_contra_los_informados(par):
    """Por construccion, todo trade ejecutado contra un informado da P&L < 0."""
    df = simular_trades(18.4, 21.4, 5_000, par, np.random.default_rng(SEED))
    informados = df[df["informado"] & df["ejecutado"]]
    assert len(informados) > 0
    assert (informados["pnl"] < 0.0).all()


def test_el_regimen_estrecho_pierde_dinero(par):
    """
    El regimen tight no cubre el costo de seleccion adversa: P&L negativo.

    Es la evidencia numerica de por que el spread tiene que ser estrictamente
    positivo.
    """
    q = REGIMENES_FIJOS["estrecho"]
    df = simular_trades(q["bid"], q["ask"], 10_000, par, np.random.default_rng(SEED))
    assert df["pnl"].sum() < 0.0
    assert df.loc[df["informado"], "pnl"].sum() < df.loc[~df["informado"], "pnl"].sum()


def test_bid_mayor_que_ask_falla(par):
    with pytest.raises(ValueError):
        simular_trades(21.0, 20.0, 10, par, np.random.default_rng(SEED))


def test_la_simulacion_converge_a_la_utilidad_teorica(par):
    """
    Sin forzar ejecucion, el P&L medio por llegada converge a Pi(A,B).

    Es la validacion cruzada entre el modelo analitico (quad + minimize) y el
    simulador: dos caminos independientes al mismo numero.
    """
    opt = optimizar(par)
    df = simular_trades(opt["bid"], opt["ask"], 200_000, par,
                        np.random.default_rng(SEED), forzar_ejecucion=False)
    assert df["pnl"].mean() == pytest.approx(opt["utilidad"], abs=0.02)


# ---------------------------------------------------------------- monte carlo
def test_monte_carlo_reproducible_y_bien_formado(par):
    """Mismas semillas, mismos P&L finales; una entrada por corrida."""
    a = monte_carlo(18.4, 21.4, 50, 200, par, SEED)
    b = monte_carlo(18.4, 21.4, 50, 200, par, SEED)
    assert a.shape == (50,)
    np.testing.assert_allclose(a, b)


def test_resumen_monte_carlo_reporta_prob_de_perdida():
    """La probabilidad de perdida es la proporcion de corridas con P&L < 0."""
    res = resumen_monte_carlo(np.array([-2.0, -1.0, 1.0, 2.0]))
    assert res["prob_perdida"] == pytest.approx(0.5)
    assert res["pnl_promedio"] == pytest.approx(0.0)


def test_el_regimen_estrecho_pierde_casi_siempre(par):
    """Con el spread tight, casi toda corrida termina en perdida."""
    q = REGIMENES_FIJOS["estrecho"]
    pnl = monte_carlo(q["bid"], q["ask"], 100, 1_000, par, SEED)
    assert resumen_monte_carlo(pnl)["prob_perdida"] > 0.9


def test_los_regimenes_comparten_la_secuencia_aleatoria(par):
    """
    Correr los tres regimenes con la misma semilla les da los mismos traders.

    Asi las diferencias de P&L son atribuibles al spread y no al ruido.
    """
    sims = simular_regimenes(REGIMENES_FIJOS, 1_000, par, SEED)
    a, b = sims["estrecho"], sims["amplio"]
    np.testing.assert_allclose(a["valor_P"], b["valor_P"])
    np.testing.assert_array_equal(a["informado"], b["informado"])
