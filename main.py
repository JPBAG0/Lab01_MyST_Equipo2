"""
Lab 01 — Cotizaciones optimas de un formador de mercado (Copeland-Galai, 1983).

Punto de entrada unico del proyecto. Corre con:

    python main.py

Reproduce, en orden: la optimizacion del bid/ask, la simulacion larga de 10,000
trades en los tres regimenes, el analisis de Monte Carlo (1,000 corridas de
1,000 trades), el analisis de sensibilidad en pi_I y las cuatro figuras
obligatorias, que se guardan en docs/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

import plots                                                    # noqa: E402
from model import (Parametros, SEED, optimizar, sensibilidad,   # noqa: E402
                   spread_monopolista)
from simulation import (REGIMENES_FIJOS, monte_carlo,           # noqa: E402
                        resumen_monte_carlo, simular_regimenes,
                        tabla_regimenes)

N_TRADES_LARGA = 10_000
N_CORRIDAS_MC = 1_000
N_TRADES_MC = 1_000
PI_I_SENSIBILIDAD = (0.1, 0.4, 0.7)
FIGURAS = RAIZ / "docs" / "figures"


def _titulo(texto: str) -> None:
    """Imprime un encabezado de seccion en la consola."""
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}")


def main() -> None:
    """Corre el proyecto completo e imprime todos los resultados."""
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    par = Parametros()

    _titulo("1. PARAMETROS DEL CASO BASE")
    P = par.distribucion()
    print(f"  S0 (precio de referencia)     = {par.S0:.2f}")
    print(f"  P ~ Erlang(K={par.K}, lambda={par.lam:g})   E[P] = {P.mean():.2f}   "
          f"sd(P) = {P.std():.2f}")
    print(f"  pi_I = {par.pi_I:.2f}   pi_L = {par.pi_L:.2f}")
    print(f"  pi_LB(s) = pi_LS(s) = max({par.alpha:.2f} - {par.beta:.2f} s, 0)"
          f"   -> se anula en s = {par.s_max:.2f}")
    print(f"  Semilla = {SEED}")

    _titulo("2. COTIZACIONES OPTIMAS (scipy.optimize.minimize + quad)")
    opt = optimizar(par)
    print(f"  Bid optimo               = {opt['bid']:.2f}")
    print(f"  Ask optimo               = {opt['ask']:.2f}")
    print(f"  Spread optimo            = {opt['spread']:.2f}")
    print(f"  Utilidad esperada/trade  = {opt['utilidad']:.2f}")
    print(f"     ganancia vs liquidez  = {opt['ganancia_liquidez']:.2f}")
    print(f"     perdida vs informados = {opt['perdida_informados']:.2f}")

    regimenes = {"optimo": {"bid": opt["bid"], "ask": opt["ask"]},
                 **REGIMENES_FIJOS}

    _titulo(f"3. SIMULACION DE {N_TRADES_LARGA:,} TRADES POR REGIMEN")
    sims = simular_regimenes(regimenes, N_TRADES_LARGA, par, SEED)
    resumen = tabla_regimenes(sims, regimenes)
    print(resumen.to_string())

    _titulo(f"4. MONTE CARLO ({N_CORRIDAS_MC:,} corridas x {N_TRADES_MC:,} trades)")
    pnl_mc = {
        nombre: monte_carlo(q["bid"], q["ask"], N_CORRIDAS_MC, N_TRADES_MC,
                            par, SEED)
        for nombre, q in regimenes.items()
    }
    tabla_mc = pd.DataFrame({n: resumen_monte_carlo(p) for n, p in pnl_mc.items()}).T
    print(tabla_mc.to_string())

    _titulo("5. SENSIBILIDAD DEL SPREAD OPTIMO A pi_I")
    sens = sensibilidad(PI_I_SENSIBILIDAD, par)
    print(sens[["bid", "ask", "spread", "utilidad"]].to_string())
    s_teorico = spread_monopolista(par)
    print(f"\n  Referencia teorica con pi_I = 0: semi-spread alpha/(2 beta) = "
          f"{s_teorico:.3f}  ->  spread = {2 * s_teorico:.3f}")
    sin_informados = optimizar(Parametros(pi_I=0.0))
    print(f"  Optimizacion numerica con pi_I = 0: spread = "
          f"{sin_informados['spread']:.3f}")

    _titulo("6. FIGURAS")
    generadas = [
        plots.guardar(plots.figura_prob_ejecucion(par),
                      FIGURAS / "01_prob_ejecucion.png"),
        plots.guardar(plots.figura_pnl_acumulado(sims),
                      FIGURAS / "02_pnl_acumulado.png"),
        plots.guardar(plots.figura_inventario(sims),
                      FIGURAS / "03_inventario.png"),
        plots.guardar(plots.figura_histograma_mc(pnl_mc),
                      FIGURAS / "04_histograma_mc.png"),
        plots.guardar(plots.figura_sensibilidad(sens, par),
                      FIGURAS / "05_sensibilidad.png"),
    ]
    for ruta in generadas:
        print(f"  guardada: {ruta.relative_to(RAIZ)}")

    _titulo("LISTO")


if __name__ == "__main__":
    main()
