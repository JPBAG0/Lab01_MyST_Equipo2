"""
Generacion de las figuras del Lab 01.

Cada funcion arma una figura de matplotlib con titulo, ejes etiquetados y
leyenda, y regresa el objeto Figure. Guardar es responsabilidad de main.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")           # backend sin ventana: main.py solo guarda a disco
import matplotlib.pyplot as plt  # noqa: E402

from model import Parametros, prob_ejecucion  # noqa: E402

COLORES = {"optimo": "#1b6ca8", "estrecho": "#c2382c", "amplio": "#2e8b57"}
ETIQUETAS = {"optimo": "Optimo", "estrecho": "Estrecho (tight)", "amplio": "Amplio (wide)"}


def _color(nombre: str) -> str:
    """Color estable por regimen; gris si el regimen no esta en la paleta."""
    return COLORES.get(nombre, "#808080")


def figura_prob_ejecucion(par: Parametros) -> plt.Figure:
    """
    Probabilidad de ejecucion del trader no informado contra el semi-spread.

    Marca el punto donde la probabilidad llega a cero, s = alpha / beta.
    """
    s = np.linspace(0.0, par.s_max * 1.3, 400)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(s, prob_ejecucion(s, par.alpha, par.beta), color="#1b6ca8", lw=2,
            label=r"$\pi_{LB}(s) = \max(\alpha - \beta s,\ 0)$")
    ax.axvline(par.s_max, color="#c2382c", ls="--", lw=1.2,
               label=f"Demanda nula: $s = \\alpha/\\beta$ = {par.s_max:.2f}")
    ax.plot([par.s_max], [0.0], "o", color="#c2382c", ms=8, zorder=5)
    ax.annotate(f"({par.s_max:.2f}, 0.00)", xy=(par.s_max, 0.0),
                xytext=(par.s_max + 0.3, 0.06), color="#c2382c")
    ax.set_title("Probabilidad de ejecucion del trader no informado")
    ax.set_xlabel("Semi-spread $s$ (unidades de precio)")
    ax.set_ylabel(r"Probabilidad de ejecucion $\pi_{LB}(s)$")
    ax.set_ylim(-0.02, par.alpha * 1.15)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def figura_pnl_acumulado(sims: Dict[str, pd.DataFrame]) -> plt.Figure:
    """P&L acumulado a lo largo de la corrida larga, un trazo por regimen."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for nombre, df in sims.items():
        ax.plot(df.index + 1, df["pnl_acumulado"], color=_color(nombre), lw=1.6,
                label=ETIQUETAS.get(nombre, nombre))
    ax.axhline(0.0, color="black", lw=0.8, ls=":")
    ax.set_title(f"P&L acumulado del dealer ({len(next(iter(sims.values()))):,} trades)")
    ax.set_xlabel("Numero de trade")
    ax.set_ylabel("P&L acumulado (unidades de precio)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def figura_inventario(sims: Dict[str, pd.DataFrame]) -> plt.Figure:
    """Inventario acumulado del dealer a lo largo de la corrida larga."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for nombre, df in sims.items():
        ax.plot(df.index + 1, df["inventario"], color=_color(nombre), lw=1.6,
                label=ETIQUETAS.get(nombre, nombre))
    ax.axhline(0.0, color="black", lw=0.8, ls=":")
    ax.set_title("Inventario acumulado del dealer")
    ax.set_xlabel("Numero de trade")
    ax.set_ylabel("Inventario (unidades netas; + = largo)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def figura_histograma_mc(pnl_mc: Dict[str, np.ndarray]) -> plt.Figure:
    """Distribucion del P&L final de las corridas de Monte Carlo."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = 40
    for nombre, pnl in pnl_mc.items():
        ax.hist(pnl, bins=bins, alpha=0.55, color=_color(nombre),
                label=f"{ETIQUETAS.get(nombre, nombre)} "
                      f"($\\mu$={np.mean(pnl):,.1f})")
    ax.axvline(0.0, color="black", lw=1.0, ls="--", label="P&L = 0")
    ax.set_title("Distribucion del P&L final (Monte Carlo)")
    ax.set_xlabel("P&L final de la corrida (unidades de precio)")
    ax.set_ylabel("Frecuencia (numero de corridas)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def figura_sensibilidad(tabla: pd.DataFrame, par: Parametros) -> plt.Figure:
    """
    Spread optimo contra pi_I, con la referencia teorica de pi_I = 0.

    La linea punteada es el spread del monopolista sin seleccion adversa,
    2 * alpha / (2*beta): el piso al que tiende el spread cuando pi_I -> 0.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tabla.index, tabla["spread"], "o-", color="#1b6ca8", lw=2, ms=8,
            label="Spread optimo simulado")
    piso = 2.0 * par.alpha / (2.0 * par.beta)
    ax.axhline(piso, color="#c2382c", ls="--", lw=1.2,
               label=f"Teorico con $\\pi_I=0$: {piso:.2f}")
    for pi_I, fila in tabla.iterrows():
        ax.annotate(f"{fila['spread']:.2f}", xy=(pi_I, fila["spread"]),
                    xytext=(0, 9), textcoords="offset points", ha="center")
    ax.set_title("Sensibilidad del spread optimo a la probabilidad de informado")
    ax.set_xlabel(r"Probabilidad de trader informado $\pi_I$")
    ax.set_ylabel("Spread optimo $A^* - B^*$")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def guardar(fig: plt.Figure, ruta: Path, dpi: int = 150) -> Path:
    """Guarda la figura en `ruta` (creando la carpeta) y la cierra."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return ruta
