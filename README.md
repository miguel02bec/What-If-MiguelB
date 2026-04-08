# SoccerSolver — Simulador What-If de Valor de Mercado

> Herramienta interactiva para ojeadores, directores deportivos y agentes que permite explorar cómo cambia el precio de un jugador al variar sus atributos en tiempo real.

---

## Índice

1. [Arquitectura](#arquitectura)
2. [Instalación y uso](#instalación-y-uso)
3. [Pipeline de datos](#pipeline-de-datos)
4. [Modelo predictivo](#modelo-predictivo)
5. [Evaluación del modelo](#evaluación-del-modelo)
6. [Motor What-If](#motor-what-if)
7. [Interfaz de usuario](#interfaz-de-usuario)
8. [Limitaciones conocidas](#limitaciones-conocidas)
9. [Con 10x más datos haría...](#con-10x-más-datos-haría)

---

## Arquitectura

```
male_players.csv (SoFIFA)
player_valuations.csv (Transfermarkt)   ──→  Fuzzy matching  ──→  players_unified.csv
players.csv (Transfermarkt)

players_unified.csv  ──→  Feature engineering  ──→  GradientBoosting × 4 grupos
                                                      (GK / DEF / MID / FWD)

Modelos .pkl  ──→  Motor What-If  ──→  soccersolver_ui.html
```

**Stack tecnológico:**
- Datos: `pandas`, `numpy`, `sqlite3`
- Matching: `difflib.SequenceMatcher` (fuzzy matching sin dependencias externas)
- Modelo: `sklearn.ensemble.GradientBoostingRegressor`
- Importancia de features: `sklearn.inspection.permutation_importance`
- UI: HTML + CSS + JavaScript vanilla (sin frameworks, sin servidor)

---

## Instalación y uso

```bash
# 1. Instalar dependencias
pip install pandas numpy scikit-learn

# 2. Colocar los CSVs en el directorio raíz
#    male_players.csv
#    player_valuations.csv
#    players.csv

# 3. Ejecutar el pipeline completo
python soccersolver_pipeline.py

# 4. Abrir el simulador
#    Abrir soccersolver_ui.html en cualquier navegador
#    No requiere servidor — todo está embebido en el HTML
```

**Estructura de archivos generados:**
```
data/
  players_clean.csv          # Dataset FIFA 24 Big 5 limpio (3.467 jugadores)
  players_unified.csv        # Fusión SoFIFA + Transfermarkt
  tm_valuations_history.csv  # Historial de valuaciones TM (55.109 registros)
  metadata.json              # Features y configuración del modelo
models/
  model_GK.pkl               # Modelo porteros
  model_DEF.pkl              # Modelo defensas
  model_MID.pkl              # Modelo centrocampistas
  model_FWD.pkl              # Modelo delanteros
  feature_importances.json   # Importancia por grupo de posición
  evaluation_report.json     # Métricas de evaluación
```

---

## Pipeline de datos

### Fuente 1 — SoFIFA (male_players.csv)

| Parámetro | Valor |
|---|---|
| Versión FIFA utilizada | FIFA 24 (temporada 2023-24) |
| Ligas | Premier League, La Liga, Bundesliga, Serie A, Ligue 1 |
| Jugadores tras filtrado | 3.467 |
| Features extraídas | 45 |
| Nulls en features clave | 0% (imputados por mediana de grupo de posición) |

**Decisiones de limpieza:**
- Porteros (402 jugadores): `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic` → imputados a 0. Se usan sus `goalkeeping_*` como features específicas.
- `contract_years_left` = `club_contract_valid_until_year` − 2023
- `growth_potential` = `potential` − `overall`
- `age_squared` añadido para capturar la no-linealidad de la curva de valor por edad
- Target: `log1p(value_eur)` — los valores de mercado siguen distribución log-normal

### Fuente 2 — Transfermarkt (player_valuations.csv + players.csv)

| Parámetro | Valor |
|---|---|
| Jugadores en TM Big 5 con valor | 10.683 |
| Registros históricos de valuación | 616.377 |
| Rango temporal | 2000-01-20 → 2026-03-30 |

### Matching SoFIFA ↔ Transfermarkt

**Algoritmo (3 capas de matching):**

1. **Normalización**: eliminar acentos, minúsculas, solo letras — `"Kylian Mbappé Lottin"` → `"kylian mbappe lottin"`
2. **Filtro por año de nacimiento ±1** para reducir espacio de búsqueda
3. **Token overlap score**: fracción de tokens compartidos entre nombres — resuelve nombres legales completos vs nombres comerciales (`"kylian mbappe lottin"` ↔ `"kylian mbappe"` = 0.77)
4. **Bonus por apellido exacto** compartido (+0.10 al score)
5. **Umbral de aceptación**: score ≥ 0.50

**Resultados:**

| Métrica | Valor |
|---|---|
| Jugadores emparejados | 2.742 / 3.467 (79.1%) |
| Matches con score = 1.0 (exactos) | 1.616 (58.9%) |
| Matches con score ≥ 0.8 | 1.630 (59.4%) |
| Sin match | 725 (20.9%) |

**Problemas de calidad de datos identificados:**
- Jugadores con nombre artístico de una sola palabra (Vinicius → "Vinícius José Paixão de Oliveira Júnior" en SoFIFA) tienen scores bajos (0.3–0.5) y quedan sin match
- Casemiro, Rodri y similares: nombre corto en TM vs nombre legal completo en SoFIFA
- Jugadores transferidos entre la fecha del CSV de SoFIFA y la de TM pueden no coincidir en club
- El historial de valuaciones TM está disponible para 2.538 de los 2.742 jugadores emparejados

---

## Modelo predictivo

### Decisiones de diseño

**¿Por qué GradientBoostingRegressor y no XGBoost/LightGBM?**
Se usó `sklearn.ensemble.GradientBoostingRegressor` por disponibilidad en el entorno sin conexión a internet. En producción, LightGBM daría resultados equivalentes con entrenamiento ~10x más rápido.

**¿Por qué target en escala logarítmica?**
Los valores de mercado siguen una distribución log-normal (Mbappé vale 180M, un jugador medio 3M, un reserva 200K). Entrenar en escala lineal daría un MAE dominado por los outliers de alta valoración, ignorando la precisión en el rango donde vive el 95% de los jugadores.

**¿Por qué estratificación por posición?**
Un portero tiene atributos completamente distintos a un delantero. Un modelo único aprendería reglas promedio que serían incorrectas para todos los grupos. Los 4 modelos especializados (GK / DEF / MID / FWD) tienen mejor rendimiento y las importancias de features son interpretables por posición.

**Hiperparámetros:**
```python
GradientBoostingRegressor(
    n_estimators  = 300,
    max_depth     = 5,
    learning_rate = 0.05,
    subsample     = 0.8,
    min_samples_leaf = 3,
    random_state  = 42,
)
```

---

## Evaluación del modelo

### Métricas por grupo de posición

| Grupo | N jugadores | R² (test set) | R² (CV 5-fold) | MAE € | RMSE € |
|---|---|---|---|---|---|
| GK  | 402  | 0.984 | 0.454 ± 0.526 | €1.008.386 | €3.131.515 |
| DEF | 1181 | 0.996 | −0.032 ± 0.733 | €302.311 | €694.321 |
| MID | 1234 | 0.994 | −0.137 ± 0.945 | €573.294 | €3.005.078 |
| FWD | 650  | 0.997 | −0.330 ± 1.005 | €884.836 | €3.594.009 |

### Error por rango de valor de mercado

| Rango | N | MAE € | Error medio % | R² |
|---|---|---|---|---|
| < €1M    | 708  | €11.781     | 2.3% | 0.983 |
| €1–5M    | 1477 | €46.026     | 1.8% | 0.983 |
| €5–20M   | 835  | €165.120    | 1.6% | 0.992 |
| €20–50M  | 364  | €485.880    | 1.5% | 0.984 |
| €50–100M | 66   | €1.655.598  | 2.5% | 0.834 |
| > €100M  | 17   | €4.369.565  | 3.4% | 0.805 |

### Sanity check — jugadores conocidos

| Jugador | Overall | Valor real | Predicho | Error % |
|---|---|---|---|---|
| K. Mbappé       | 91 | €181.500.000 | €181.202.043 | −0.2% ✓ |
| E. Haaland      | 91 | €185.000.000 | €184.535.299 | −0.3% ✓ |
| K. De Bruyne    | 91 | €103.000.000 | €103.045.591 | +0.0% ✓ |
| J. Bellingham   | 86 | €100.500.000 | €100.115.706 | −0.4% ✓ |
| Vini Jr.        | 89 | €158.500.000 | €158.526.030 | +0.0% ✓ |
| H. Kane         | 90 | €119.500.000 | €119.501.908 | +0.0% ✓ |
| M. Salah        | 89 | €85.500.000  | €85.417.393  | −0.1% ✓ |
| T. Courtois     | 90 | €63.000.000  | €58.591.550  | −7.0% ✓ |
| R. Lewandowski  | 90 | €58.000.000  | €92.583.617  | +59.6% ⚠️ |

### Top 5 features por grupo de posición

**GK:** potential (0.510) · overall (0.312) · age² (0.068) · age (0.063) · goalkeeping_reflexes (0.026)

**DEF:** overall (1.156) · potential (0.197) · age (0.033) · age² (0.026) · growth_potential (0.001)

**MID:** overall (1.355) · potential (0.115) · age² (0.014) · age (0.013) · dribbling (0.003)

**FWD:** overall (1.468) · potential (0.105) · age (0.007) · age² (0.007) · growth_potential (0.001)

---

## Motor What-If

El motor calcula en tiempo real el impacto de modificar cualquier atributo:

```python
def whatif_delta(base_features, modified_features, position_group):
    base = predict_value(base_features, position_group)
    new  = predict_value(modified_features, position_group)
    return {
        "base_eur":  base["predicted_eur"],
        "new_eur":   new["predicted_eur"],
        "delta_eur": new["predicted_eur"] - base["predicted_eur"],
        "delta_pct": (delta / base) * 100,
        "conf_low":  new["conf_low"],
        "conf_high": new["conf_high"],
    }
```

**Intervalo de confianza** basado en RMSE relativo por grupo:
- GK: ±18% · DEF: ±8% · MID: ±15% · FWD: ±17%

---

## Interfaz de usuario

El simulador (`soccersolver_ui.html`) incluye:

- **Búsqueda de jugadores** — filtrado en tiempo real sobre 150 jugadores top de las Big 5
- **Panel hero** — valor predicho con intervalo de confianza + comparativa con Transfermarkt
- **Key Levers** — top 5 atributos con mayor impacto en el precio para ese jugador específico, con delta en EUR al subir +5 puntos
- **Sliders What-If** — actualización de valor instantánea al mover cualquier atributo
- **Comparador de escenarios** — guarda hasta 2 escenarios y los compara side-by-side
- **Sin servidor** — todo el modelo está embebido en el HTML, funciona offline

---

## Limitaciones conocidas

### 1. El modelo aprende la función de EA Sports, no el mercado real

**Qué pasa:** El R² en test set es 0.994–0.997, pero el R² en cross-validation estricta cae a valores negativos para DEF/MID/FWD. Esto indica que el modelo memorizó la relación `overall+age → value_eur` tal como la codifica FIFA, no como la evalúa el mercado real.

**Por qué:** EA Sports calcula `value_eur` internamente usando casi los mismos atributos que usamos como features. El modelo aprende la función inversa de EA, no el mercado de transferencias.

**Impacto práctico:** Las predicciones son muy precisas dentro del dataset FIFA 24, pero no deben usarse para valorar jugadores fuera de ese universo (ligas menores, jugadores de academias, versiones futuras de FIFA).

**Para el simulador What-If:** El impacto es limitado — el uso principal es mostrar sensibilidad relativa ("si sube X, el valor sube Y"), no predicción absoluta de precio de traspaso.

### 2. Jugadores veteranos top infravalorados por el mercado

**Casos identificados:** Lewandowski (+59.6%), Lloris (+88.5%), Mandanda (+72.6%), Ochoa (+63.1%)

**Por qué:** Estos jugadores tienen atributos FIFA todavía altos (overall 85–90), pero el mercado ya descuenta su declive físico inminente y la reducción de años de contrato. El modelo no captura este "descuento de carrera tardía".

**Solución con más datos:** Añadir `years_since_peak` (años desde los 27) y el historial de lesiones como features.

### 3. Jugadores únicos con nombre artístico no tienen datos TM

**Afectados:** Vinicius Jr., Casemiro, Rodri, y ~725 jugadores (20.9% del dataset)

**Por qué:** SoFIFA usa nombres legales completos; Transfermarkt usa nombres comerciales. El fuzzy matching falla cuando el solapamiento de tokens es < 0.5.

**Solución:** Mantener una tabla manual de equivalencias para los 50 jugadores más valiosos.

### 4. Una sola instantánea temporal (FIFA 24)

El modelo solo conoce el estado de octubre 2023. No sabe si un jugador mejoró en los últimos 18 meses, si se lesionó, ni si cambió de club.

### 5. Cobertura limitada de la UI

El simulador embebido incluye los 150 jugadores con mayor overall. Los 3.317 restantes requieren el pipeline Python completo.

### 6. Intervalos de confianza aproximados

Los intervalos ±8–18% son estimaciones basadas en el RMSE relativo del grupo, no intervalos estadísticos formales. Un modelo con incertidumbre calibrada (quantile regression o conformal prediction) daría intervalos más precisos.

---

## Con 10x más datos haría...

### Datos adicionales (10x)
Con 35.000+ jugadores históricos de todas las ligas y temporadas de FIFA 15–24 tendríamos suficiente masa para:

**1. Series temporales reales**
Entrenar un modelo de evolución de valor: dado el perfil de un jugador a los 22 años, predecir su trayectoria de valor hasta los 32. El dataset de TM ya tiene 616.377 registros históricos — el cuello de botella es cruzarlos con los atributos FIFA por temporada.

**2. Separar FIFA rating del valor de mercado real**
Con suficiente volumen, entrenar dos modelos: uno que predice `overall` → `value_FIFA` y otro que predice `value_FIFA` → `value_TM`. El gap entre ambos es el "descuento de mercado" — útil para detectar jugadores sobrevalorados o infravalorados por FIFA.

**3. Historial de lesiones como feature**
Cruzar con bases de datos de lesiones (Transfermarkt tiene el historial). Un jugador con 3 lesiones musculares en 2 años vale sistemáticamente menos aunque sus atributos FIFA sean iguales.

**4. Contexto de ventana de transferencias**
Con datos de traspasos reales, modelar el efecto del tiempo hasta la expiración del contrato. Un jugador a 6 meses de quedar libre vale ~30–40% menos en negociación real que su `value_eur` de FIFA.

**5. Modelos por liga**
El mercado de la Premier League tiene una prima del 20–30% sobre La Liga para el mismo perfil de jugador. Con más datos, modelos estratificados por liga darían predicciones más precisas.

**6. Quantile regression para intervalos reales**
Reemplazar los intervalos ±RMSE% por un modelo de quantile regression (percentil 10 y 90), dando intervalos estadísticamente calibrados en lugar de estimaciones.

---

## Decisiones de arquitectura

| Decisión | Alternativa considerada | Razón de la elección |
|---|---|---|
| GradientBoosting de sklearn | XGBoost / LightGBM | Disponibilidad sin red en entorno de desarrollo |
| Permutation importance | SHAP values | Sin dependencia de `shap`, misma interpretabilidad para el simulador |
| HTML vanilla | React / Streamlit | Sin servidor, funciona offline, cero dependencias en producción |
| Log-transform del target | Target lineal | Distribución log-normal de valores; mejora MAE en rango bajo |
| 4 modelos por posición | 1 modelo global | Features y patrones completamente distintos por posición |
| SQLite | PostgreSQL | Suficiente para 3.467 jugadores; sin infraestructura |

---

*SoccerSolver — Reto Técnico SoccerSolver*
*Dataset: FIFA 24 Big 5 + Transfermarkt · Modelo: GradientBoostingRegressor × 4*
