"""
app.py — SoccerSolver Streamlit
================================
Ejecutar local:  streamlit run app.py
Streamlit Cloud: conecta el repo en share.streamlit.io
"""

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ── Rutas absolutas ───────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="SoccerSolver — What-If Miguel Becerra",
    page_icon="⚽",
    layout="wide",
)

# ── Cargar modelos y datos (cacheados) ────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for g in ["GK", "DEF", "MID", "FWD"]:
        with open(MODELS_DIR / f"model_{g}.pkl", "rb") as f:
            models[g] = pickle.load(f)
    return models

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / "players_clean.csv")
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    with open(MODELS_DIR / "feature_importances.json") as f:
        imp = json.load(f)
    return df, meta["features"], imp

models                    = load_models()
df, FEATURES, importances = load_data()

# ── Config sliders ────────────────────────────────────────────────────────────
SLIDER_CONFIG = {
    "overall":                  ("Overall",            40, 99),
    "potential":                ("Potential",           40, 99),
    "pace":                     ("Pace",                 1, 99),
    "shooting":                 ("Shooting",             1, 99),
    "passing":                  ("Passing",              1, 99),
    "dribbling":                ("Dribbling",            1, 99),
    "defending":                ("Defending",            1, 99),
    "physic":                   ("Physic",               1, 99),
    "movement_reactions":       ("Reactions",            1, 99),
    "skill_ball_control":       ("Ball Control",         1, 99),
    "mentality_vision":         ("Vision",               1, 99),
    "mentality_composure":      ("Composure",            1, 99),
    "weak_foot":                ("Weak Foot ★",          1,  5),
    "skill_moves":              ("Skill Moves ★",        1,  5),
    "international_reputation": ("Intl. Reputation ★",   1,  5),
    "contract_years_left":      ("Contrato (años)",     -2,  8),
    "age":                      ("Edad",                16, 40),
}

STAR_FEATURES = {"contract_years_left", "international_reputation",
                 "weak_foot", "skill_moves"}
RMSE_PCT = {"GK": 0.18, "DEF": 0.08, "MID": 0.15, "FWD": 0.17}

# ── Funciones ─────────────────────────────────────────────────────────────────
def predict(features: dict, group: str) -> dict:
    x        = np.array([[features.get(f, 0) for f in FEATURES]])
    log_pred = models[group].predict(x)[0]
    eur      = float(np.expm1(log_pred))
    pct      = RMSE_PCT[group]
    return {"eur": round(eur), "low": round(eur*(1-pct)), "high": round(eur*(1+pct))}

def fmt(v: float) -> str:
    if v >= 1e8: return f"€{v/1e6:.0f}M"
    if v >= 1e6: return f"€{v/1e6:.1f}M"
    if v >= 1e3: return f"€{v/1e3:.0f}K"
    return f"€{v:.0f}"

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚽ SoccerSolver")
st.sidebar.caption("Simulador What-If · FIFA 24 · Big Five")

search = st.sidebar.text_input("Buscar jugador", placeholder="Mbappé, Haaland...")

if search:
    mask     = (df["short_name"].str.contains(search, case=False, na=False) |
                df["long_name"].str.contains(search, case=False, na=False))
    filtered = df[mask].sort_values("overall", ascending=False)
else:
    filtered = df.sort_values("overall", ascending=False)

if filtered.empty:
    st.sidebar.warning("Sin resultados.")
    st.stop()

selected_name = st.sidebar.selectbox("Jugador", filtered["short_name"].tolist())
player        = df[df["short_name"] == selected_name].iloc[0]
player_id     = int(player["player_id"])  # ← clave única por jugador

# ── FIX: resetear sliders al cambiar de jugador ───────────────────────────────
# Si el jugador seleccionado cambió, limpiamos las claves de los sliders
# del session_state para que se inicialicen con los valores del nuevo jugador.
if st.session_state.get("active_player_id") != player_id:
    for feat in SLIDER_CONFIG:
        slider_key = f"{feat}_{player_id}"
        # Borrar cualquier valor previo de este slider con otro jugador
        old_key = f"{feat}_{st.session_state.get('active_player_id', '')}"
        if old_key in st.session_state:
            del st.session_state[old_key]
    st.session_state["active_player_id"] = player_id

# ── Features base del jugador ─────────────────────────────────────────────────
base_feats = {
    f: float(player[f])
    for f in FEATURES
    if f in player.index and pd.notna(player[f])
}
base_pred = predict(base_feats, player["position_group"])

# ── Header ────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    st.title(player["short_name"])
    st.caption(f"{player['long_name']}  ·  {player['club_name']}  ·  {player['league_name']}")
    st.markdown(
        f"`{player['position_primary']}` "
        f"`{player['position_group']}` "
        f"`{player['nationality_name']}` "
        f"`Age {int(player['age'])}`"
    )
with c2:
    st.metric("Overall",   int(player["overall"]))
with c3:
    st.metric("Potential", int(player["potential"]))

st.divider()

left, right = st.columns([1, 1])

# ── Sliders ───────────────────────────────────────────────────────────────────
# La clave incluye player_id → cuando cambia el jugador, Streamlit crea
# sliders nuevos inicializados con los valores del jugador actual.
modified_feats = base_feats.copy()
with left:
    st.subheader("⚡ Estadísticas:")
    for feat, (label, vmin, vmax) in SLIDER_CONFIG.items():
        if feat not in base_feats:
            continue
        slider_key           = f"{feat}_{player_id}"   # ← clave única por jugador
        new_val              = st.slider(
            label,
            min_value=vmin,
            max_value=vmax,
            value=int(base_feats[feat]),   # siempre arranca en el valor real del jugador
            key=slider_key,
        )
        modified_feats[feat] = float(new_val)

# ── Resultados ────────────────────────────────────────────────────────────────
new_pred   = predict(modified_feats, player["position_group"])
delta_eur  = new_pred["eur"] - base_pred["eur"]
delta_pct  = delta_eur / base_pred["eur"] * 100
has_change = any(
    modified_feats.get(f) != base_feats.get(f)
    for f in SLIDER_CONFIG if f in base_feats
)

with right:
    st.subheader("Valor de mercado predicho")
    st.metric(
        value=fmt(new_pred["eur"]),
        delta=(
            f"{'+' if delta_eur >= 0 else ''}"
            f"{fmt(abs(delta_eur))} ({delta_pct:+.1f}%)"
        ) if has_change else None,
    )
    st.caption(f"Intervalo de confianza: {fmt(new_pred['low'])} – {fmt(new_pred['high'])}")

    # Comparativa Transfermarkt
    if "tm_market_value" in player.index and pd.notna(player.get("tm_market_value")):
        tm   = float(player["tm_market_value"])
        diff = new_pred["eur"] - tm
        st.metric(
            "Valor Transfermarkt",
            fmt(tm),
            delta=f"{'+' if diff >= 0 else ''}{fmt(abs(diff))} vs TM",
        )

    st.divider()

    # Key levers
    st.subheader("🔑 Key levers — mayor impacto en precio")
    feat_imp = importances[player["position_group"]]
    top5     = list(feat_imp.items())[:5]
    max_imp  = max(v for _, v in top5) or 1

    for feat, imp in top5:
        label  = SLIDER_CONFIG.get(feat, (feat,))[0]
        inc    = 1 if feat in STAR_FEATURES else 5
        tf     = base_feats.copy()
        tf[feat] = min(base_feats.get(feat, 0) + inc, 99)
        deur   = predict(tf, player["position_group"])["eur"] - base_pred["eur"]

        ca, cb = st.columns([3, 1])
        ca.progress(imp / max_imp, text=f"**{label}**")
        cb.markdown(f"`{'+' if deur >= 0 else ''}{fmt(abs(deur))}`")

    st.caption("Impacto de subir +5 puntos cada atributo (+1 para estrellas/contrato)")

    st.divider()

    # Comparador de escenarios
    st.subheader("📊 Comparador de escenarios")

    if "scenarios" not in st.session_state:
        st.session_state.scenarios = []

    if st.button("💾 Guardar escenario actual", disabled=not has_change):
        changes = [
            f"{SLIDER_CONFIG.get(f,(f,))[0]}: "
            f"{int(base_feats[f])}→{int(modified_feats[f])}"
            for f in SLIDER_CONFIG
            if f in base_feats and modified_feats.get(f) != base_feats[f]
        ]
        st.session_state.scenarios.append({
            "Jugador": player["short_name"],
            "Valor":   fmt(new_pred["eur"]),
            "Delta":   f"{delta_pct:+.1f}%",
            "Cambios": ", ".join(changes[:3]),
        })
        st.success("Escenario guardado.")

    if st.session_state.scenarios:
        st.dataframe(
            pd.DataFrame(st.session_state.scenarios),
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Borrar todos los escenarios"):
            st.session_state.scenarios = []
            st.rerun()
    else:
        st.caption("Modifica atributos y pulsa 'Guardar' para comparar escenarios.")
