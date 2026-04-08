"""
model_training.py
=================
Entrena dos modelos para predecir el valor de mercado de jugadores:
  1. Regresión lineal  → baseline
  2. Gradient Boosting → modelo final

Entrada:  data/players_clean.csv
Salida:   models/model_{GK|DEF|MID|FWD}.pkl
          models/baseline_{GK|DEF|MID|FWD}.pkl
          models/evaluation_report.json
          models/feature_importances.json

Uso:
    python model_training.py
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Rutas ─────────────────────────────────────────────────────────────────────
DATA_PATH   = Path("data/players_clean.csv")
META_PATH   = Path("data/metadata.json")
MODELS_DIR  = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ── Cargar datos ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

with open(META_PATH) as f:
    meta = json.load(f)

FEATURES         = meta["features"]
TARGET           = "log_value_eur"          # entrenamos en log-scale
POSITION_GROUPS  = ["GK", "DEF", "MID", "FWD"]

print(f"Dataset: {len(df)} jugadores | {len(FEATURES)} features")
print(f"Target:  log1p(value_eur)  →  rango [{df[TARGET].min():.2f}, {df[TARGET].max():.2f}]")


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def metrics(y_true_log, y_pred_log, label=""):
    """Calcula MAE, RMSE y R² tanto en log-scale como en euros."""
    y_true_eur = np.expm1(y_true_log)
    y_pred_eur = np.expm1(y_pred_log)

    r2   = r2_score(y_true_log, y_pred_log)       # R² en log (estándar)
    mae  = mean_absolute_error(y_true_eur, y_pred_eur)
    rmse = np.sqrt(mean_squared_error(y_true_eur, y_pred_eur))
    mape = np.mean(np.abs(y_true_eur - y_pred_eur) / y_true_eur) * 100

    if label:
        print(f"    R²={r2:.4f}  |  MAE=€{mae:>12,.0f}  |  RMSE=€{rmse:>12,.0f}  |  MAPE={mape:.1f}%")

    return {"r2": round(r2, 4), "mae_eur": round(mae), "rmse_eur": round(rmse), "mape": round(mape, 2)}


def cross_validate(model, X, y, label=""):
    """Cross-validation 5-fold. Devuelve media ± std del R²."""
    kf     = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    print(f"    CV R²: {[round(s, 3) for s in scores]}  →  {scores.mean():.4f} ± {scores.std():.4f}")
    return round(scores.mean(), 4), round(scores.std(), 4)


# ══════════════════════════════════════════════════════════════════════════════
# MODELO 1 — REGRESIÓN LINEAL (BASELINE)
# ══════════════════════════════════════════════════════════════════════════════
#
# Usamos Ridge (regresión lineal con regularización L2) en lugar de OLS puro
# porque con 45 features y multicolinealidad (overall correlaciona con muchos
# atributos individuales), OLS tiende a overfitting. Ridge lo estabiliza.
#
# Pipeline: StandardScaler → Ridge
#   - StandardScaler necesario porque Ridge es sensible a la escala
#   - overall está en [40–99], contract_years_left en [-2, 8]: sin escalar,
#     el coeficiente de overall dominaría artificialmente

def build_baseline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=10.0)),   # alpha=10 → regularización moderada
    ])


# ══════════════════════════════════════════════════════════════════════════════
# MODELO 2 — GRADIENT BOOSTING (MODELO FINAL)
# ══════════════════════════════════════════════════════════════════════════════
#
# GradientBoostingRegressor de sklearn. En producción se reemplazaría por
# LightGBM (pip install lightgbm) con los mismos hiperparámetros y mejor
# velocidad de entrenamiento.
#
# Hiperparámetros elegidos:
#   n_estimators=300   → suficientes árboles para convergencia sin overfitting
#   max_depth=5        → árboles moderadamente profundos; >6 overfittea con 3.467 filas
#   learning_rate=0.05 → tasa baja + más árboles = mejor generalización
#   subsample=0.8      → stochastic GB: usa 80% de filas por árbol (reduce varianza)
#   min_samples_leaf=3 → hoja mínima de 3 muestras: evita memorizar outliers

def build_gbm():
    return GradientBoostingRegressor(
        n_estimators     = 300,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        min_samples_leaf = 3,
        random_state     = 42,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO ESTRATIFICADO POR POSICIÓN
# ══════════════════════════════════════════════════════════════════════════════

baseline_models = {}
gbm_models      = {}
all_reports     = {}
all_importances = {}

for group in POSITION_GROUPS:
    print(f"\n{'═'*55}")
    print(f"  GRUPO: {group}  ({(df['position_group']==group).sum()} jugadores)")
    print(f"{'═'*55}")

    # Filtrar por grupo de posición
    df_g = df[df["position_group"] == group].copy()
    X    = df_g[FEATURES].values
    y    = df_g[TARGET].values

    # Train / test split estratificado (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # ── BASELINE: Ridge ───────────────────────────────────────────────────────
    print(f"\n  [1] BASELINE — Ridge Regression")
    baseline = build_baseline()
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)

    print(f"  Test set:")
    base_metrics = metrics(y_test, y_pred_base, label="baseline")

    print(f"  Cross-validation:")
    base_cv_mean, base_cv_std = cross_validate(baseline, X, y)
    baseline_models[group] = baseline

    # ── MODELO FINAL: GBM ─────────────────────────────────────────────────────
    print(f"\n  [2] GRADIENT BOOSTING")
    gbm = build_gbm()
    gbm.fit(X_train, y_train)
    y_pred_gbm = gbm.predict(X_test)

    print(f"  Test set:")
    gbm_metrics = metrics(y_test, y_pred_gbm, label="gbm")

    print(f"  Cross-validation:")
    gbm_cv_mean, gbm_cv_std = cross_validate(gbm, X, y)
    gbm_models[group] = gbm

    # ── Comparación baseline vs GBM ───────────────────────────────────────────
    r2_gain  = gbm_metrics["r2"] - base_metrics["r2"]
    mae_gain = base_metrics["mae_eur"] - gbm_metrics["mae_eur"]
    print(f"\n  Ganancia GBM sobre baseline:")
    print(f"    R²:  {base_metrics['r2']:.4f} → {gbm_metrics['r2']:.4f}  (+{r2_gain:.4f})")
    print(f"    MAE: €{base_metrics['mae_eur']:>10,.0f} → €{gbm_metrics['mae_eur']:>10,.0f}  (−€{mae_gain:,.0f})")

    # ── Importancia de features (permutation importance) ──────────────────────
    # Permutation importance: mide cuánto sube el error al aleatorizar cada feature.
    # Más robusto que feature_importances_ de sklearn (que sobrevalora features con
    # alta cardinalidad). Usamos n_repeats=15 para estabilizar el resultado.
    perm = permutation_importance(
        gbm, X_test, y_test,
        n_repeats=15,
        random_state=42,
        n_jobs=-1,
    )
    feat_imp = {
        feat: round(float(imp), 6)
        for feat, imp in sorted(
            zip(FEATURES, perm.importances_mean),
            key=lambda x: x[1],
            reverse=True
        )
    }

    print(f"\n  Top 5 features por importancia:")
    for feat, imp in list(feat_imp.items())[:5]:
        bar = "█" * max(1, int(imp * 20))
        print(f"    {feat:<35s} {bar} {imp:.4f}")

    all_importances[group] = feat_imp

    # ── Guardar reporte ───────────────────────────────────────────────────────
    all_reports[group] = {
        "n_total":  len(df_g),
        "n_train":  len(X_train),
        "n_test":   len(X_test),
        "baseline": {**base_metrics, "cv_r2": base_cv_mean, "cv_r2_std": base_cv_std},
        "gbm":      {**gbm_metrics,  "cv_r2": gbm_cv_mean,  "cv_r2_std": gbm_cv_std},
    }


# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR MODELOS Y REPORTES
# ══════════════════════════════════════════════════════════════════════════════

for group in POSITION_GROUPS:
    with open(MODELS_DIR / f"model_{group}.pkl",    "wb") as f: pickle.dump(gbm_models[group],      f)
    with open(MODELS_DIR / f"baseline_{group}.pkl", "wb") as f: pickle.dump(baseline_models[group], f)

with open(MODELS_DIR / "evaluation_report.json",   "w") as f: json.dump(all_reports,     f, indent=2)
with open(MODELS_DIR / "feature_importances.json", "w") as f: json.dump(all_importances, f, indent=2)

print(f"\n\n{'═'*55}")
print(f"  RESUMEN FINAL")
print(f"{'═'*55}")
print(f"  {'Grupo':<6} {'Baseline R²':>12} {'GBM R²':>10} {'GBM MAE €':>14} {'GBM MAPE':>10}")
print(f"  {'─'*55}")
for group in POSITION_GROUPS:
    rep = all_reports[group]
    b   = rep["baseline"]
    g   = rep["gbm"]
    ok  = "✓" if g["r2"] >= 0.80 else "✗"
    print(f"  {group:<6} {b['r2']:>12.4f} {g['r2']:>10.4f} {g['mae_eur']:>14,.0f} {g['mape']:>9.1f}%  {ok}")

print(f"\n  Modelos guardados en models/")
print(f"  Reporte en models/evaluation_report.json")
