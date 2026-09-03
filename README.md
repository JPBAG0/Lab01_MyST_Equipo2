# Lab 01 — Cotizaciones Óptimas de un Formador de Mercado

Modelo de **Copeland y Galai (1983)**: cálculo del bid y el ask que maximizan la
utilidad esperada de un formador de mercado, y evaluación por simulación del
impacto de la selección adversa sobre su rentabilidad.

**Microestructura y Sistemas de Trading — ITESO, Otoño 2026. Equipo 2.**

## Integrantes

| Integrante |
|---|
| Juan Pablo Barba González |
| Gian Carlo Campos Sayavedra  |

## Descripción

Un formador de mercado publica un bid `B` y un ask `A` alrededor de su estimación
`S0 = 19.90` del valor de un activo cuyo valor verdadero `P` sigue una
`Erlang(K=60, λ=3)`. Frente a él llegan dos tipos de contraparte: con
probabilidad `π_I = 0.40` un trader **informado**, que observa `P` y solo opera
cuando el precio publicado le deja ganancia (y por lo tanto le quita dinero al
dealer siempre), y con probabilidad `π_L = 0.60` un trader **de liquidez**, cuya
probabilidad de operar decae linealmente con el semi-spread,
`π_LB(s) = π_LS(s) = max(0.50 − 0.08·s, 0)`. El proyecto optimiza numéricamente
la utilidad esperada `Π(A,B)` —integrando las pérdidas por selección adversa con
`scipy.integrate.quad` y maximizando con `scipy.optimize.minimize`—, simula
10,000 trades y 1,000 corridas de Monte Carlo bajo tres regímenes de cotización,
y mide cómo se ensancha el spread óptimo al aumentar la proporción de informados.

## Instalación

Requiere Python 3.9 o superior.

```bash
git clone https://github.com/JPBAG0/Lab01_FinanzasCuantitativas_Equipo2.git
cd Lab01_FinanzasCuantitativas_Equipo2
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducir todos los resultados

Un solo comando:

```bash
python main.py
```

Imprime en consola las cotizaciones óptimas, la tabla de los tres regímenes, el
resumen de Monte Carlo y el análisis de sensibilidad, y guarda las cinco figuras
en `docs/figures/`.

Las pruebas se corren con:

```bash
pytest tests/ -v
```

## Semilla aleatoria

`SEED = 42`, definida en `src/model.py` y pasada **explícitamente** a cada
generador con `np.random.default_rng(SEED)`; no se usa el estado global de
`np.random.seed`, que es global al proceso y hace que el resultado dependa del
orden en que se llamen las funciones. Las 1,000 corridas de Monte Carlo se
derivan de esa misma semilla con `np.random.SeedSequence(SEED).spawn(1000)`, así
que son independientes entre sí y el resultado completo es reproducible bit a
bit. Los tres regímenes se simulan con la misma semilla a propósito: comparten la
secuencia de tipos de trader y de valores `P`, de modo que las diferencias de P&L
son atribuibles al spread y no al ruido del muestreo.

## Estructura del proyecto

```
Lab01_FinanzasCuantitativas_Equipo2/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                  # ejecuta el proyecto completo
├── src/
│   ├── model.py             # utilidad esperada, integrales y optimización
│   ├── simulation.py        # simulador de trades y Monte Carlo
│   └── plots.py             # generación de las figuras
├── tests/
│   ├── conftest.py
│   └── test_model.py        # 25 pruebas
├── notebooks/
│   └── analysis.ipynb       # solo análisis y figuras, sin lógica
└── docs/
    ├── presentacion.pdf
    └── figures/
```

Toda la lógica del modelo y de la simulación vive en `src/`. El notebook importa
funciones de `src/` y genera figuras; no define ninguna fórmula.

---

## Resultados

### Cotizaciones óptimas (caso base, π_I = 0.40)

| | Valor |
|---|---|
| Bid óptimo | **16.45** |
| Ask óptimo | **23.43** |
| Spread óptimo | **6.98** |
| Utilidad esperada por trade | **0.84** |
| — ganancia frente a liquidez | 0.92 |
| — pérdida frente a informados | 0.08 |

El óptimo es casi simétrico (semi-spread de 3.53 del lado del ask contra 3.45 del
lado del bid). La ligera asimetría es real y tiene explicación: `S0 = 19.90` está
por debajo de `E[P] = K/λ = 20.00`, así que la cola de valores altos con la que el
informado ataca el ask pesa un poco más que la cola baja con la que ataca el bid,
y el dealer se protege marginalmente más arriba.

### Simulación de 10,000 trades

| Régimen | Bid | Ask | Spread | P&L total | P&L/trade | Trades ejec. | Invent. máx. | P&L vs informados | P&L vs liquidez |
|---|---|---|---|---|---|---|---|---|---|
| Óptimo | 16.45 | 23.43 | 6.98 | **+20,023.84** | +3.00 | 6,679 | 97 | −897.97 | +20,921.81 |
| Estrecho | 19.75 | 20.05 | 0.30 | **−6,737.32** | −0.69 | 9,809 | 117 | −7,782.68 | +1,045.36 |
| Amplio | 18.40 | 21.40 | 3.00 | **+5,396.85** | +0.66 | 8,215 | 88 | −3,702.61 | +9,099.46 |

### Monte Carlo (1,000 corridas × 1,000 trades)

| Régimen | P&L promedio | Desv. estándar | Probabilidad de pérdida |
|---|---|---|---|
| Óptimo | +2,012.13 | 81.61 | **0.000** |
| Estrecho | −672.41 | 74.66 | **1.000** |
| Amplio | +543.72 | 74.48 | **0.000** |

### Sensibilidad a π_I

| π_I | Bid | Ask | Spread óptimo | Utilidad |
|---|---|---|---|---|
| 0.00 | 16.775 | 23.025 | **6.250** | 1.5625 |
| 0.10 | 16.71 | 23.11 | **6.40** | 1.38 |
| 0.40 | 16.45 | 23.43 | **6.98** | 0.84 |
| 0.70 | 16.01 | 24.00 | **7.99** | 0.34 |

### Figuras

Las cinco figuras se generan en `docs/figures/`:

1. `01_prob_ejecucion.png` — probabilidad de ejecución contra el semi-spread, marcando `s = α/β = 6.25` donde llega a cero.
2. `02_pnl_acumulado.png` — P&L acumulado de los 10,000 trades, tres curvas.
3. `03_inventario.png` — inventario acumulado, tres regímenes.
4. `04_histograma_mc.png` — distribución del P&L final del Monte Carlo.
5. `05_sensibilidad.png` — spread óptimo contra π_I, con la referencia teórica.

---

## Preguntas de análisis

### 1. ¿Por qué los traders informados generan la necesidad de un spread?

Porque contra ellos el dealer **pierde siempre, por construcción**: el informado
solo cruza el spread cuando el precio publicado le deja ganancia, así que cada
trade que le ejecuta al dealer es negativo para el dealer. El spread es lo único
que puede financiar esa pérdida.

El régimen estrecho lo cuantifica. Con `(19.75, 20.05)` el dealer cobra 0.15 por
lado y su ganancia esperada frente a la liquidez es **0.088 por llegada**,
mientras que su pérdida esperada frente a los informados es **0.764** — casi
nueve veces mayor. La utilidad esperada es **−0.676 por llegada**, y en los
10,000 trades simulados el dealer terminó en **−6,737.32**: ganó +1,045.36 contra
traders de liquidez y perdió **−7,782.68** contra informados. En las 1,000
corridas de Monte Carlo, la probabilidad de pérdida fue **1.000**: ni una sola
corrida terminó en positivo. Un spread de 0.30 sobre un activo cuyo valor tiene
desviación estándar de 2.58 no es un margen, es una invitación.

El umbral es calculable: con estos parámetros el dealer no alcanza utilidad
esperada positiva hasta un spread de aproximadamente **1.95** (en `spread = 2.00`,
`Π = +0.021`). Debajo de eso, cualquier cotización pierde dinero en el largo plazo.

### 2. ¿Cómo cambia el costo de selección adversa conforme se amplía el spread?

Cae de forma **monótona y muy convexa**. Al alejar el ask, la cola `P > A` de la
Erlang que el informado puede explotar se adelgaza rápido, y con ella la integral
`∫_A^∞ (P−A)·f(P)dP`:

| Spread | Pérdida vs informados `L` | Ganancia vs liquidez `G` | Utilidad `Π` |
|---|---|---|---|
| 0.30 | 0.7635 | 0.0878 | −0.6757 |
| 1.00 | 0.6376 | 0.2760 | −0.3616 |
| 2.00 | 0.4834 | 0.5040 | +0.0206 |
| 3.00 | 0.3579 | 0.6840 | +0.3261 |
| 4.00 | 0.2585 | 0.8160 | +0.5575 |
| 6.00 | 0.1252 | 0.9360 | +0.8109 |
| **6.98** | **0.0846** | **0.9247** | **+0.8403** |
| 8.00 | 0.0550 | 0.8640 | +0.8090 |
| 10.00 | 0.0223 | 0.6000 | +0.5778 |
| 12.50 | 0.0066 | 0.0000 | −0.0066 |

De 0.30 a 6.98 el costo de selección adversa cae **89%** (0.7635 → 0.0846). Pero
el óptimo **no** está donde `L` es mínima: ampliar el spread también espanta al
flujo del que el dealer vive. La ganancia `G` es cuadrática cóncava, sube hasta
`spread = 6.25` y luego se derrumba, y en `spread = 12.50` (`s = α/β = 6.25`) la
probabilidad de ejecución es exactamente cero y `G = 0`. El óptimo en 6.98 es el
punto donde el ahorro marginal en `L` deja de compensar la pérdida marginal de
flujo: es un compromiso, no una esquina.

### 3. ¿Cuál régimen acumula el mayor desbalance de inventario y por qué?

El **estrecho**, con un inventario máximo en valor absoluto de **117 unidades**,
contra 97 del óptimo y 88 del amplio. La razón es la mezcla de contrapartes: al
cotizar pegado a `S0` el dealer ejecuta **9,809 de 10,000** llegadas, la
proporción más alta de los tres, y una fracción desproporcionada de esas
ejecuciones son informadas —justamente las que llegan todas del mismo lado a la
vez, porque el informado compra cuando `P > A` y vende cuando `P < B`. Los trades
de liquidez son aproximadamente simétricos y se cancelan entre sí; los informados
no, porque su dirección está correlacionada con el valor verdadero. Ese es el
mecanismo: **el desbalance de inventario es el rastro de la selección adversa.**

El riesgo real al que eso lo expone y que **el modelo no captura** es el *riesgo
de inventario*. Copeland-Galai valúa cada posición al valor verdadero `P` del
momento y liquida instantáneamente: en el modelo, quedarse largo 117 unidades no
cuesta nada. En un mercado real ese inventario (i) inmoviliza capital y consume
margen, (ii) queda expuesto al movimiento del precio entre el momento de
adquirirlo y el de deshacerse de él, y (iii) tiene que liquidarse contra el mismo
libro, con impacto de mercado. Los modelos de inventario (Ho-Stoll, Amihud-Mendelson,
Avellaneda-Stoikov) existen precisamente para incorporar lo que aquí se omite: un
dealer real desplaza sus cotizaciones para sesgar el flujo hacia el lado que le
reduce la posición, algo que en este modelo estático no tiene sentido hacer.

### 4. ¿Cómo se comporta el spread óptimo al variar π_I? ¿Coincide con la teoría?

Crece de forma monótona: **6.25 → 6.40 → 6.98 → 7.99** para
`π_I ∈ {0.0, 0.1, 0.4, 0.7}`. Sí coincide con la teoría, en dos sentidos.

**Cualitativo:** es la predicción central de Copeland-Galai — el spread es el
precio de la selección adversa. Cuanta más masa de flujo es informada, más caro
es el seguro que el dealer tiene que cobrarle al flujo no informado, y más ancho
cotiza. La utilidad, en paralelo, se desploma de 1.5625 a 0.34: el dealer puede
defenderse ampliando el spread, pero no puede evitar que la selección adversa le
cueste.

**Cuantitativo:** el caso `π_I = 0` tiene solución cerrada y sirve de anclaje.
Sin informados el dealer solo maximiza `s·(α − β·s)` por lado, cuyo máximo está en
`s* = α/(2β) = 0.50/0.16 = 3.125`, es decir un spread total de
`α/β = 0.50/0.08 = 6.25`. La optimización numérica devuelve exactamente
`bid = 16.775`, `ask = 23.025`, `spread = 6.250`. Esa es la prueba obligatoria
`test_sin_informados_el_optimo_es_el_del_monopolista`, y confirma que `quad` y
`minimize` están resolviendo el problema correcto. El spread de 6.25 es el
**piso**: la parte del spread que existe por poder de mercado y no por
información. Todo lo que excede ese piso —0.15 con `π_I = 0.1`, 0.73 con
`π_I = 0.4`, 1.74 con `π_I = 0.7`— es la componente de selección adversa,
y crece de forma convexa en `π_I`.

> **Nota sobre el enunciado.** La sección 3.6 pide verificar que con `π_I = 0` el
> spread óptimo sea `s* = 0.50/0.08 = 6.25` "por lado". El número 6.25 es correcto,
> pero corresponde al **spread total** `A* − B*`, no al semi-spread de cada lado:
> por lado el óptimo es `α/(2β) = 3.125`, y `α/β` es su doble. La prueba verifica
> las dos lecturas explícitamente para que no quede ambigüedad.

### 5. Tres limitaciones del modelo para un formador de mercado real

**1. No hay inventario ni horizonte temporal.** El modelo es estático y de un solo
período: valúa cada trade contra `P` y supone liquidación instantánea sin costo.
Un dealer real acumula posición (117 unidades en nuestro régimen estrecho), la
carga a través del tiempo, paga capital por ella y la deshace con impacto de
mercado. Nada de eso entra en `Π(A,B)`, y por eso el modelo nunca recomienda
sesgar las cotizaciones para reducir inventario, que es la decisión más frecuente
de un market maker real.

**2. La demanda no informada es lineal, exógena y no compite.** `π_LB(s) = 0.50 − 0.08·s`
supone que el flujo de liquidez reacciona solo al spread *de este dealer*, de forma
lineal, y que se anula abruptamente en `s = 6.25`. En un mercado real el flujo se
va al **mejor precio disponible entre todos los dealers**: la variable relevante no
es el spread propio sino el spread relativo, y basta que un competidor cotice un
tick mejor para llevarse todo el flujo. El modelo describe un monopolista, y ese
supuesto es justo el que explica por qué el spread óptimo que obtenemos (6.98
sobre un precio de 19.90, o sea **35% del valor del activo**) es órdenes de
magnitud mayor que cualquier spread observable en un mercado competitivo.

**3. La información es binaria, perfecta y gratuita.** El informado observa `P`
exactamente y sin costo; el dealer nunca aprende nada del flujo que recibe. En la
realidad la información es ruidosa y graduada, y sobre todo el **order flow es
informativo**: una racha de compras es evidencia de que el valor está arriba, y un
dealer real actualiza su `S0` con cada trade. Ese es el mecanismo central de
Glosten-Milgrom y Kyle, y aquí está ausente por completo: nuestro dealer se deja
pegar 10,000 veces seguidas sin mover su cotización ni una vez.

### Advertencia de interpretación (obligatoria)

**La simulación fuerza la ejecución de un trade de liquidez en cada iteración**
(`forzar_ejecucion=True`, el modo que pide el enunciado). Eso significa que las
cifras de P&L reportadas son **rentabilidad por trade, no por unidad de tiempo**:
al forzar la ejecución se elimina la masa de probabilidad "el trader no opera",
que es precisamente el castigo que un spread amplio debería recibir.

El sesgo es cuantificable con nuestros propios números. En el régimen óptimo la
probabilidad real de ejecución del lado del ask es **0.218**, contra 0.488 en el
estrecho: el régimen óptimo debería ejecutar menos de la mitad de las veces que el
estrecho, pero la simulación forzada le regala ese flujo. Un dealer que cotizara
`(16.45, 23.43)` sobre un activo de 19.90 no vería casi ningún trade en un mercado
real, y su P&L **por hora** sería muy inferior al que sugiere la tabla, aunque su
P&L **por trade** fuera excelente. Por eso `src/simulation.py` implementa también
`forzar_ejecucion=False`, que respeta `π_LB` y `π_LS`; en ese modo el P&L medio
por llegada converge a la utilidad teórica `Π(A,B) = 0.84`, y es la validación
cruzada entre el modelo analítico y el simulador
(`test_la_simulacion_converge_a_la_utilidad_teorica`).

---

## Pruebas

`tests/test_model.py` contiene **25 pruebas**, entre ellas las tres obligatorias
de la sección 3.6 del enunciado:

| Prueba | Qué verifica |
|---|---|
| `test_prob_ejecucion_nunca_es_negativa` | **(Obligatoria 1)** `π_LB(s)` y `π_LS(s)` se truncan en cero sobre toda la malla. |
| `test_perdida_ask_es_decreciente_en_A` | **(Obligatoria 2)** La pérdida esperada frente a informados decrece en `A`, verificado en cinco valores de `A` más una malla de 12 puntos. |
| `test_sin_informados_el_optimo_es_el_del_monopolista` | **(Obligatoria 3)** Con `π_I = 0` el óptimo coincide con `α/(2β)` por lado y `α/β` de spread total. |
| `test_las_integrales_son_consistentes_con_la_media` | Las integrales de `quad` reconstruyen `E\|P − S0\|` medido por Monte Carlo. |
| `test_la_simulacion_converge_a_la_utilidad_teorica` | El P&L medio del simulador converge a `Π(A,B)` calculada analíticamente. |
| `test_el_dealer_siempre_pierde_contra_los_informados` | Todo trade ejecutado contra un informado tiene P&L negativo. |
| `test_el_regimen_estrecho_pierde_dinero` | El régimen tight termina en P&L negativo y pierde más contra informados. |
| `test_mas_informados_ensancha_el_spread_y_encoge_la_utilidad` | El spread óptimo es monótono creciente en `π_I`. |

```bash
pytest tests/ -v
```

## Uso de asistencia de IA

Se usó **Claude (Claude Code)** en las siguientes partes del proyecto:

- **Estructura del repositorio y configuración de git.** Creación del árbol de
  carpetas exigido, `.gitignore`, `requirements.txt` y el flujo de ramas.
- **Redacción de `src/model.py`, `src/simulation.py`, `src/plots.py` y `main.py`.**
  El diseño del modelo y la interpretación económica son del equipo; la asistencia
  se usó para la traducción a código, la parametrización de `quad` y `minimize`, y
  el formato de las figuras.
- **Redacción de `tests/test_model.py`.** Las tres pruebas obligatorias las define
  el enunciado; las 22 restantes se diseñaron con asistencia.
- **Depuración.** La prueba `test_la_simulacion_converge_a_la_utilidad_teorica`
  detectó un error de modelado en el simulador: se había implementado la dirección
  del trader de liquidez como un volado 50/50 con `π_LB` aplicada después, cuando
  en el modelo `π_LB` y `π_LS` **son** las probabilidades de compra y de venta
  (suman 1 en `s = 0`). La corrección hizo que el simulador convergiera a la
  utilidad teórica.
- **Redacción de este README.** Las respuestas a las cinco preguntas de análisis
  se apoyan en las cifras generadas por `main.py`.

Todo el código entregado fue revisado por ambos integrantes, que pueden explicar
cualquier línea.

## Referencia

Copeland, T. E., & Galai, D. (1983). *Information Effects on the Bid-Ask Spread*.
The Journal of Finance, 38(5), 1457–1469.
