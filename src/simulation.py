"""
Simulador de trades contra el formador de mercado de Copeland-Galai.

Mecanica de una iteracion
-------------------------
1. Se sortea el tipo de trader: INFORMADO con probabilidad pi_I, de LIQUIDEZ con
   probabilidad pi_L = 1 - pi_I.
2. Se sortea el valor verdadero del activo, P ~ Erlang(K, lambda).
3. El informado observa P y opera solo en la direccion que le conviene: compra
   al ask si P > A, vende al bid si P < B. Si B <= P <= A no opera.
   El de liquidez no observa P: compra al ask con probabilidad pi_LB(A-S0),
   vende al bid con probabilidad pi_LS(S0-B), y con el resto de la masa,
   1 - pi_LB - pi_LS, no opera. Ojo: pi_LB y pi_LS son las probabilidades de
   compra y de venta, no probabilidades condicionadas a una direccion ya
   elegida; en s = 0 valen 0.5 cada una y suman exactamente 1.
4. P&L del dealer, marcado siempre contra el valor verdadero P:
     - el trader compra al ask  -> el dealer vende:  pnl = A - P, inventario -1
     - el trader vende al bid   -> el dealer compra: pnl = P - B, inventario +1

Marcar contra P (y no contra S0) es lo que hace que la simulacion sea consistente
con la utilidad teorica: como P es independiente del tipo de trader de liquidez y
E[P] = K/lambda, el P&L esperado frente a liquidez es A - E[P], que con los
parametros del Lab coincide practicamente con el semi-spread A - S0.

Sobre `forzar_ejecucion`
-----------------------
El enunciado pide que la simulacion fuerce un trade en cada iteracion. Con
forzar_ejecucion=True el trader de liquidez ejecuta siempre: se conserva el
sesgo relativo entre comprar y vender, pi_LB / (pi_LB + pi_LS), pero se elimina
la masa de "no opera", sin importar lo ancho que sea el spread. Eso es exactamente lo que la advertencia del enunciado senala: los
resultados son rentabilidad POR TRADE y no por unidad de tiempo, y un spread
muy amplio que en la realidad casi nunca se ejecutaria sale favorecido.
Con forzar_ejecucion=False se respeta pi_LB / pi_LS y el sesgo desaparece.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from model import Parametros, prob_ejecucion

# Regimenes de cotizacion del enunciado (el "optimo" se calcula en tiempo de
# ejecucion con model.optimizar).
REGIMENES_FIJOS: Dict[str, Dict[str, float]] = {
    "estrecho": {"bid": 19.75, "ask": 20.05},
    "amplio":   {"bid": 18.40, "ask": 21.40},
}


def simular_trades(bid: float, ask: float, n_trades: int, par: Parametros,
                   rng: np.random.Generator,
                   forzar_ejecucion: bool = True) -> pd.DataFrame:
    """
    Simula n_trades llegadas contra la cotizacion (bid, ask).

    Regresa un DataFrame con una fila por llegada y las columnas: informado,
    compra, ejecutado, valor_P, pnl, cambio_inventario, pnl_acumulado e
    inventario.
    """
    if bid > ask:
        raise ValueError("El bid no puede ser mayor que el ask.")

    informado = rng.random(n_trades) < par.pi_I
    valor_P = par.distribucion().rvs(size=n_trades, random_state=rng)

    # Demanda no informada: pi_LB es la probabilidad de COMPRA y pi_LS la de
    # VENTA; lo que sobra, 1 - pi_LB - pi_LS, es la masa de "no opera".
    p_compra = float(prob_ejecucion(ask - par.S0, par.alpha, par.beta))
    p_venta = float(prob_ejecucion(par.S0 - bid, par.alpha, par.beta))
    u = rng.random(n_trades)

    if forzar_ejecucion:
        # Se elimina la masa de "no opera" renormalizando; si el spread es tan
        # ancho que ninguno de los dos lados atrae flujo, se cae a 50/50.
        total = p_compra + p_venta
        umbral = p_compra / total if total > 0.0 else 0.5
        compra_liquidez = u < umbral
        ejec_liq = np.ones(n_trades, dtype=bool)
    else:
        compra_liquidez = u < p_compra
        ejec_liq = u < p_compra + p_venta

    # Direccion: el informado la elige mirando P; el de liquidez, segun pi_LB/pi_LS.
    compra = np.where(informado, valor_P > par.S0, compra_liquidez)

    # Ejecucion del informado: solo si el precio publicado le deja ganancia.
    ejec_inf = np.where(valor_P > par.S0, valor_P > ask, valor_P < bid)

    ejecutado = np.where(informado, ejec_inf, ejec_liq)

    # P&L del dealer, marcado contra el valor verdadero P.
    pnl = np.where(compra, ask - valor_P, valor_P - bid)
    pnl = np.where(ejecutado, pnl, 0.0)

    cambio_inventario = np.where(ejecutado, np.where(compra, -1, 1), 0)

    return pd.DataFrame({
        "informado":         informado,
        "compra":            compra,
        "ejecutado":         ejecutado,
        "valor_P":           valor_P,
        "pnl":               pnl,
        "cambio_inventario": cambio_inventario,
        "pnl_acumulado":     np.cumsum(pnl),
        "inventario":        np.cumsum(cambio_inventario),
    })


def simular_regimenes(regimenes: Dict[str, Dict[str, float]], n_trades: int,
                      par: Parametros, semilla: int,
                      forzar_ejecucion: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Corre simular_trades para cada regimen con la MISMA semilla.

    Usar la misma semilla en los tres regimenes hace que compartan la secuencia
    de tipos de trader y de valores P: las diferencias de P&L vienen del spread
    y no del ruido del muestreo.
    """
    return {
        nombre: simular_trades(q["bid"], q["ask"], n_trades, par,
                               np.random.default_rng(semilla), forzar_ejecucion)
        for nombre, q in regimenes.items()
    }


def monte_carlo(bid: float, ask: float, n_corridas: int, n_trades: int,
                par: Parametros, semilla: int,
                forzar_ejecucion: bool = True) -> np.ndarray:
    """
    P&L final de n_corridas independientes de n_trades cada una.

    Regresa un arreglo de longitud n_corridas. Cada corrida usa su propio
    Generator derivado de `semilla` via SeedSequence, asi que son independientes
    y el resultado completo es reproducible.
    """
    semillas = np.random.SeedSequence(semilla).spawn(n_corridas)
    return np.array([
        simular_trades(bid, ask, n_trades, par,
                       np.random.default_rng(s), forzar_ejecucion)["pnl"].sum()
        for s in semillas
    ])


def resumen_monte_carlo(pnl_finales: np.ndarray) -> Dict[str, float]:
    """P&L promedio, desviacion estandar y probabilidad de perdida."""
    return {
        "pnl_promedio":       float(np.mean(pnl_finales)),
        "pnl_desv_estandar":  float(np.std(pnl_finales, ddof=1)),
        "prob_perdida":       float(np.mean(pnl_finales < 0.0)),
    }


def tabla_monte_carlo(regimenes: Dict[str, Dict[str, float]], n_corridas: int,
                      n_trades: int, par: Parametros, semilla: int,
                      forzar_ejecucion: bool = True) -> pd.DataFrame:
    """Tabla con el resumen de Monte Carlo para cada regimen."""
    filas = {
        nombre: resumen_monte_carlo(
            monte_carlo(q["bid"], q["ask"], n_corridas, n_trades, par,
                        semilla, forzar_ejecucion))
        for nombre, q in regimenes.items()
    }
    return pd.DataFrame(filas).T


def tabla_regimenes(sims: Dict[str, pd.DataFrame],
                    regimenes: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Resumen descriptivo de la corrida larga: P&L, inventario y mezcla."""
    filas = {}
    for nombre, df in sims.items():
        ejec = df[df["ejecutado"]]
        filas[nombre] = {
            "bid":                 regimenes[nombre]["bid"],
            "ask":                 regimenes[nombre]["ask"],
            "spread":              regimenes[nombre]["ask"] - regimenes[nombre]["bid"],
            "pnl_total":           df["pnl"].sum(),
            "pnl_por_trade":       ejec["pnl"].mean(),
            "trades_ejecutados":   int(df["ejecutado"].sum()),
            "inventario_final":    int(df["cambio_inventario"].sum()),
            "inventario_max_abs":  int(df["inventario"].abs().max()),
            "pnl_vs_informados":   df.loc[df["informado"], "pnl"].sum(),
            "pnl_vs_liquidez":     df.loc[~df["informado"], "pnl"].sum(),
        }
    return pd.DataFrame(filas).T
