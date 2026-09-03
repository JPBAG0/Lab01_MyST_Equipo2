"""
Modelo de Copeland y Galai (1983): cotizaciones optimas de un formador de mercado.

Montaje
-------
El dealer tiene una estimacion a priori S0 del valor del activo y publica un bid
B <= S0 y un ask A >= S0. El valor verdadero P es una variable aleatoria
continua con densidad f(P) (aqui, Erlang(K, lambda)).

Llega un trader:
  - con probabilidad pi_I esta INFORMADO: observa P, compra al ask si P > A y
    vende al bid si P < B. El dealer siempre pierde contra el.
  - con probabilidad pi_L = 1 - pi_I es de LIQUIDEZ: no observa P y llega con
    probabilidad decreciente en lo caro que este el precio,
        pi_LB(s) = pi_LS(s) = max(alpha - beta * s, 0)
    evaluada en el semi-spread s de su lado. Contra el, el dealer gana s.

Utilidad esperada por trader que llega:

    Pi(A,B) = pi_L * [ pi_LB(A-S0)*(A-S0) + pi_LS(S0-B)*(S0-B) ]
            - pi_I * [ int_A^inf (P-A) f(P) dP + int_0^B (B-P) f(P) dP ]

El primer corchete es la ganancia esperada frente a traders de liquidez; el
segundo, la perdida esperada frente a informados (el costo de seleccion adversa).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import erlang, rv_continuous

# Semilla global del proyecto (ver README). Se pasa explicitamente, nunca
# via np.random.seed.
SEED = 42


@dataclass(frozen=True)
class Parametros:
    """Caso base del Lab 01. Todos los valores vienen fijos en el enunciado."""

    S0: float = 19.90        # precio de referencia
    K: int = 60              # forma de la Erlang
    lam: float = 3.0         # tasa de la Erlang
    pi_I: float = 0.40       # probabilidad de trader informado
    alpha: float = 0.50      # intercepto de la demanda no informada
    beta: float = 0.08       # pendiente de la demanda no informada

    @property
    def pi_L(self) -> float:
        """pi_L = 1 - pi_I: los dos tipos de trader agotan las llegadas."""
        return 1.0 - self.pi_I

    @property
    def s_max(self) -> float:
        """Semi-spread donde la demanda no informada se anula: alpha / beta."""
        return self.alpha / self.beta

    def distribucion(self) -> rv_continuous:
        """Distribucion congelada de P: Erlang(K, lambda) = gamma(K, 1/lambda)."""
        return erlang(self.K, scale=1.0 / self.lam)


# --------------------------------------------------------------------------
# Componentes de la utilidad
# --------------------------------------------------------------------------

def prob_ejecucion(s, alpha: float = 0.50, beta: float = 0.08):
    """
    Probabilidad de ejecucion del trader no informado: max(alpha - beta*s, 0).

    s es el semi-spread que ese lado le cobra. Regresa un escalar o un arreglo,
    segun la entrada, y nunca un valor negativo.
    """
    return np.maximum(alpha - beta * np.asarray(s, dtype=float), 0.0)


def ganancia_liquidez(A: float, B: float, par: Parametros) -> float:
    """
    Ganancia esperada del dealer frente a traders de liquidez.

    pi_L * [ pi_LB(A-S0)*(A-S0) + pi_LS(S0-B)*(S0-B) ].
    """
    s_ask, s_bid = A - par.S0, par.S0 - B
    return float(
        par.pi_L
        * (prob_ejecucion(s_ask, par.alpha, par.beta) * s_ask
           + prob_ejecucion(s_bid, par.alpha, par.beta) * s_bid)
    )


def perdida_ask(A: float, par: Parametros) -> float:
    """
    Perdida esperada del lado del ask: int_A^inf (P-A) f(P) dP.

    Se integra con scipy.integrate.quad (sin el factor pi_I).
    """
    f = par.distribucion().pdf
    valor, _ = quad(lambda P: (P - A) * f(P), A, np.inf, limit=200)
    return float(valor)


def perdida_bid(B: float, par: Parametros) -> float:
    """
    Perdida esperada del lado del bid: int_0^B (B-P) f(P) dP.

    Se integra con scipy.integrate.quad (sin el factor pi_I).
    """
    f = par.distribucion().pdf
    valor, _ = quad(lambda P: (B - P) * f(P), 0.0, B, limit=200)
    return float(valor)


def perdida_informados(A: float, B: float, par: Parametros) -> float:
    """Costo esperado de seleccion adversa: pi_I * (perdida_ask + perdida_bid)."""
    return float(par.pi_I * (perdida_ask(A, par) + perdida_bid(B, par)))


def utilidad_esperada(A: float, B: float, par: Parametros) -> float:
    """Pi(A,B) = ganancia frente a liquidez - perdida frente a informados."""
    return ganancia_liquidez(A, B, par) - perdida_informados(A, B, par)


def objetivo(x: Sequence[float], par: Parametros) -> float:
    """
    Funcion objetivo a minimizar: -Pi(A,B), con x = (A, B).

    Penaliza el orden invalido B > A para que el optimizador no lo explore.
    """
    A, B = float(x[0]), float(x[1])
    if B > A:
        return 1e6
    return -utilidad_esperada(A, B, par)


# --------------------------------------------------------------------------
# Optimizacion
# --------------------------------------------------------------------------

def optimizar(par: Parametros, n_arranques: int = 5) -> Dict[str, float]:
    """
    Bid y ask que maximizan Pi(A,B) con scipy.optimize.minimize.

    Restricciones del enunciado: B en (0, S0] y A en [S0, inf). La cota superior
    efectiva de A es S0 + alpha/beta, porque mas alla la demanda no informada ya
    es cero y la utilidad solo puede caer. Se usa multiarranque porque Pi no es
    concava en todo el dominio.

    Regresa un dict con bid, ask, spread y utilidad esperada por trade.
    """
    limites = [(par.S0, par.S0 + par.s_max),
               (max(1e-9, par.S0 - par.s_max), par.S0)]

    arranques = [
        (par.S0 + t * par.s_max, par.S0 - t * par.s_max)
        for t in np.linspace(0.05, 0.95, n_arranques)
    ]

    mejor = None
    for x0 in arranques:
        res = minimize(objetivo, np.array(x0, dtype=float), args=(par,),
                       method="L-BFGS-B", bounds=limites,
                       options={"ftol": 1e-14, "gtol": 1e-12})
        if mejor is None or res.fun < mejor.fun:
            mejor = res

    A, B = float(mejor.x[0]), float(mejor.x[1])
    return {
        "bid": B,
        "ask": A,
        "spread": A - B,
        "utilidad": -float(mejor.fun),
        "semi_spread_ask": A - par.S0,
        "semi_spread_bid": par.S0 - B,
        "ganancia_liquidez": ganancia_liquidez(A, B, par),
        "perdida_informados": perdida_informados(A, B, par),
    }


def sensibilidad(pi_is: Iterable[float], par: Parametros) -> pd.DataFrame:
    """
    Repite la optimizacion variando pi_I y regresa una tabla con el resultado.

    Una fila por pi_I, con bid, ask, spread y utilidad optimos.
    """
    filas = []
    for pi_I in pi_is:
        res = optimizar(Parametros(S0=par.S0, K=par.K, lam=par.lam, pi_I=pi_I,
                                   alpha=par.alpha, beta=par.beta))
        filas.append({"pi_I": pi_I, **res})
    return pd.DataFrame(filas).set_index("pi_I")


def spread_monopolista(par: Parametros) -> float:
    """
    Semi-spread optimo analitico cuando pi_I = 0 (monopolista sin informados).

    Sin seleccion adversa el dealer solo maximiza s*(alpha - beta*s) por lado,
    cuyo maximo esta en s* = alpha / (2*beta).

    NOTA sobre el enunciado: el Lab reporta "s* = 0.50/0.08 por lado" = 6.25.
    El numero 6.25 es correcto, pero corresponde al SPREAD TOTAL A* - B*, no al
    semi-spread de cada lado: alpha/beta = 2 * alpha/(2*beta). Por lado el optimo
    es alpha/(2*beta) = 3.125. La optimizacion numerica con pi_I = 0 confirma las
    dos lecturas: bid = 16.775, ask = 23.025, spread = 6.250. Ver README.
    """
    return par.alpha / (2.0 * par.beta)
