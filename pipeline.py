"""
SoccerSolver — Pipeline completo
=================================
Ejecutar en orden. Requiere:
    pip install scikit-learn pandas numpy

El CSV male_players.csv debe estar en el mismo directorio.
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELDA 1 — LIMPIEZA Y FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

Path("data").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

# ── Cargar ────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv("male_players.csv", low_memory=False)

# ── Filtrar FIFA 24 + Big 5 ───────────────────────────────────────────────────
BIG5 = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
df = df_raw[(df_raw["fifa_version"] == 24.0) & (df_raw["league_name"].isin(BIG5))].copy()

# ── Posición principal y grupo ────────────────────────────────────────────────
df["position_primary"] = df["player_positions"].str.split(",").str[0].str.strip()

def position_group(pos):
    if pos == "GK":                                                          return "GK"
    if pos in ["CB","LB","RB","LWB","RWB"]:                                 return "DEF"
    if pos in ["CDM","CM","CAM","LM","RM","LCM","RCM","LDM","RDM"]:        return "MID"
    if pos in ["ST","CF","LW","RW","LF","RF","LS","RS"]:                    return "FWD"
    return "MID"

df["position_group"] = df["position_primary"].apply(position_group)

# ── Porteros: rellenar stats de campo con 0 ───────────────────────────────────
gk_mask     = df["position_group"] == "GK"
field_stats = ["pace","shooting","passing","dribbling","defending","physic"]
df.loc[gk_mask, field_stats] = df.loc[gk_mask, field_stats].fillna(0)

# ── Features derivadas ────────────────────────────────────────────────────────
df["contract_years_left"] = df["club_contract_valid_until_year"].astype(float) - 2023
df["age_squared"]         = df["age"] ** 2
df["is_peak_age"]         = ((df["age"] >= 24) & (df["age"] <= 28)).astype(int)
df["growth_potential"]    = df["potential"] - df["overall"]

def encode_work_rate(wr):
    if not isinstance(wr, str): return 0.5
    mapping = {"Low": 0, "Medium": 1, "High": 2}
    parts   = wr.split("/")
    return (mapping.get(parts[0],1) + mapping.get(parts[1],1)) / 4.0 if len(parts)==2 else 0.5

df["work_rate_enc"] = df["work_rate"].apply(encode_work_rate)
df["is_right_foot"] = (df["preferred_foot"] == "Right").astype(int)

# ── Target en log scale ───────────────────────────────────────────────────────
df = df[df["value_eur"].notna() & (df["value_eur"] > 0)].copy()
df["log_value_eur"] = np.log1p(df["value_eur"])

# ── Selección de features ─────────────────────────────────────────────────────
FEATURES = [
    "overall","potential","age","age_squared","is_peak_age","growth_potential",
    "pace","shooting","passing","dribbling","defending","physic",
    "attacking_crossing","attacking_finishing","attacking_heading_accuracy",
    "attacking_short_passing","attacking_volleys",
    "skill_dribbling","skill_curve","skill_ball_control",
    "movement_acceleration","movement_sprint_speed","movement_agility",
    "movement_reactions","movement_balance",
    "power_shot_power","power_stamina","power_strength","power_long_shots",
    "mentality_positioning","mentality_vision","mentality_composure",
    "defending_marking_awareness","defending_standing_tackle",
    "goalkeeping_diving","goalkeeping_handling","goalkeeping_reflexes",
    "weak_foot","skill_moves","international_reputation",
    "height_cm","weight_kg",
    "contract_years_left","work_rate_enc","is_right_foot",
]

# Convertir a numérico e imputar por mediana de grupo
for col in FEATURES:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            group_med = df.groupby("position_group")[col].transform("median")
            df[col]   = df[col].fillna(group_med).fillna(df[col].median())

df.to_csv("data/players_clean.csv", index=False)
print(f"✓ Dataset limpio: {len(df)} jugadores | {len(FEATURES)} features")


# ══════════════════════════════════════════════════════════════════════════════
# CELDA 2 — ENTRENAMIENTO DE MODELOS (estratificado por posición)
# ══════════════════════════════════════════════════════════════════════════════
models      = {}
importances = {}
reports     = {}

for group in ["GK","DEF","MID","FWD"]:
    df_g = df[df["position_group"] == group].copy()
    X    = df_g[FEATURES].values
    y    = df_g["log_value_eur"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=3, random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred     = model.predict(X_test)
    y_pred_eur = np.expm1(y_pred)
    y_test_eur = np.expm1(y_test)

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test_eur, y_pred_eur)
    rmse = np.sqrt(mean_squared_error(y_test_eur, y_pred_eur))

    # Permutation importance (equivalente a SHAP en GBM)
    perm      = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    feat_imp  = dict(sorted(zip(FEATURES, perm.importances_mean.tolist()), key=lambda x: x[1], reverse=True))

    models[group]      = model
    importances[group] = feat_imp
    reports[group]     = {"n_train": len(X_train), "n_test": len(X_test),
                          "r2": round(r2,4), "mae_eur": round(mae), "rmse_eur": round(rmse)}

    print(f"{group:4s} | R²={r2:.4f} | MAE=€{mae:>10,.0f} | RMSE=€{rmse:>10,.0f}")
    with open(f"models/model_{group}.pkl","wb") as f:
        pickle.dump(model, f)

with open("models/feature_importances.json","w") as f: json.dump(importances, f, indent=2)
with open("models/evaluation_report.json","w")  as f: json.dump(reports,     f, indent=2)

meta = {"features": FEATURES, "position_groups": list(models.keys()),
        "n_players": len(df), "leagues": BIG5}
with open("data/metadata.json","w") as f: json.dump(meta, f, indent=2)

print("✓ Modelos guardados en models/")


# ══════════════════════════════════════════════════════════════════════════════
# CELDA 3 — MOTOR WHAT-IF
# ══════════════════════════════════════════════════════════════════════════════

# Intervalo de confianza por grupo (basado en RMSE relativo)
RMSE_PCT = {"GK": 0.18, "DEF": 0.08, "MID": 0.15, "FWD": 0.17}


def predict_value(player_features: dict, position_group: str) -> dict:
    """
    Predice el valor de mercado de un jugador.

    Args:
        player_features: dict {feature_name: value}
        position_group:  "GK" | "DEF" | "MID" | "FWD"

    Returns:
        {predicted_eur, conf_low, conf_high, top_levers}
    """
    model    = models[position_group]
    feat_imp = importances[position_group]

    x        = np.array([[player_features.get(f, 0) for f in FEATURES]])
    log_pred = model.predict(x)[0]
    pred_eur = np.expm1(log_pred)

    pct       = RMSE_PCT.get(position_group, 0.15)
    conf_low  = pred_eur * (1 - pct)
    conf_high = pred_eur * (1 + pct)

    levers = [
        {"feature": f, "importance": imp, "current_value": player_features.get(f, 0)}
        for f, imp in list(feat_imp.items())[:5]
    ]

    return {
        "predicted_eur": round(pred_eur),
        "conf_low":      round(conf_low),
        "conf_high":     round(conf_high),
        "top_levers":    levers,
    }


def whatif_delta(base_features: dict, modified_features: dict, position_group: str) -> dict:
    """
    Calcula el impacto en valor de mercado al modificar atributos.

    Returns:
        {base_eur, new_eur, delta_eur, delta_pct, conf_low, conf_high, top_levers}
    """
    base  = predict_value(base_features, position_group)
    new   = predict_value(modified_features, position_group)
    delta = new["predicted_eur"] - base["predicted_eur"]

    return {
        "base_eur":   base["predicted_eur"],
        "new_eur":    new["predicted_eur"],
        "delta_eur":  round(delta),
        "delta_pct":  round(delta / base["predicted_eur"] * 100, 2),
        "conf_low":   new["conf_low"],
        "conf_high":  new["conf_high"],
        "top_levers": new["top_levers"],
    }


def sensitivity_report(player_row: pd.Series, test_features=None, increment=5) -> pd.DataFrame:
    """
    Calcula el impacto de subir +increment cada feature para un jugador.
    Útil para el panel 'Key Levers' de la UI.
    """
    if test_features is None:
        test_features = ["overall","potential","dribbling","pace","passing",
                         "shooting","movement_reactions","skill_ball_control",
                         "contract_years_left","international_reputation","age"]

    base_feats = {f: player_row[f] for f in FEATURES if f in player_row.index}
    group      = player_row["position_group"]
    base_val   = predict_value(base_feats, group)["predicted_eur"]

    rows = []
    for feat in test_features:
        if feat not in base_feats:
            continue
        inc    = 1 if feat in ["contract_years_left","international_reputation"] else increment
        mod    = base_feats.copy()
        mod[feat] += inc
        new_val    = predict_value(mod, group)["predicted_eur"]
        delta_eur  = new_val - base_val
        rows.append({
            "feature":   feat,
            "increment": inc,
            "delta_eur": round(delta_eur),
            "delta_pct": round(delta_eur / base_val * 100, 2),
        })

    return pd.DataFrame(rows).sort_values("delta_eur", ascending=False)


# ── Test rápido ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Jugador medio para ver sensibilidad real
    player = df[(df["overall"].between(72,78)) & (df["position_group"]=="MID")].iloc[5]
    print(f"\nTest: {player['short_name']} | overall={player['overall']} | €{player['value_eur']:,.0f}")

    report = sensitivity_report(player)
    print(report.to_string(index=False))

    # What-If: ¿cuánto vale si sube 2 puntos de overall?
    base  = {f: player[f] for f in FEATURES if f in player.index}
    mod   = base.copy()
    mod["overall"] += 2
    delta = whatif_delta(base, mod, player["position_group"])
    print(f"\n+2 overall → €{delta['delta_eur']:+,.0f} ({delta['delta_pct']:+.1f}%)")
