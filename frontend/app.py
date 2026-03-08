from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


default_backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_URL = st.sidebar.text_input("Backend URL", value=default_backend_url)

st.set_page_config(page_title="LaLiga Predictor", layout="wide")
st.title("LaLiga Predictor - TFM")
st.caption("Prediccion 1X2, evaluacion de modelos y simulacion de Value Betting")


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def _get(path: str) -> Dict[str, Any]:
    response = requests.get(f"{API_URL}{path}", timeout=60)
    response.raise_for_status()
    return response.json()


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Prediccion partido",
        "Metricas modelo",
        "Explorador historico",
        "Simulacion Value Betting",
    ]
)

with tab1:
    st.subheader("Prediccion de partido")

    c1, c2, c3, c4 = st.columns(4)
    season = c1.text_input("Temporada", value="2025/2026")
    round_number = c2.number_input("Jornada", min_value=1, max_value=50, value=38)

    fixtures_data = {"fixtures": []}
    if c3.button("Cargar fixtures"):
        try:
            fixtures_data = _get(f"/fixtures/{season}/{round_number}")
            st.session_state["fixtures_data"] = fixtures_data
        except Exception as exc:
            st.error(f"Error cargando fixtures: {exc}")

    fixtures_data = st.session_state.get("fixtures_data", fixtures_data)
    fixture_labels = [f"{f['home_team']} vs {f['away_team']}" for f in fixtures_data.get("fixtures", [])]

    selected_fixture = c4.selectbox("Partido", options=fixture_labels) if fixture_labels else ""
    home_team = st.text_input("Equipo local", value=selected_fixture.split(" vs ")[0] if selected_fixture else "Athletic")
    away_team = st.text_input("Equipo visitante", value=selected_fixture.split(" vs ")[1] if selected_fixture else "Barcelona")

    if st.button("Predecir"):
        payload = {
            "season": season,
            "round": int(round_number),
            "home_team": home_team,
            "away_team": away_team,
        }
        try:
            pred = _post("/predict", payload)

            probs = pd.DataFrame(
                {
                    "outcome": ["home_win", "draw", "away_win"],
                    "probability": [pred["home_win"], pred["draw"], pred["away_win"]],
                }
            )
            fig = px.bar(probs, x="outcome", y="probability", title="Probabilidades 1X2", text_auto=".2f")
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Prediccion final: {pred['prediction']}")
            st.write("Top features:", pred.get("top_features", []))
        except Exception as exc:
            st.error(f"Error en prediccion: {exc}")

with tab2:
    st.subheader("Metricas del modelo")
    col_train, col_metrics = st.columns(2)

    if col_train.button("Entrenar modelo"):
        try:
            result = _post("/train", {})
            st.session_state["metrics_data"] = result
            st.success("Entrenamiento completado")
        except Exception as exc:
            st.error(f"Error entrenando: {exc}")

    if col_metrics.button("Actualizar metricas"):
        try:
            metrics = _get("/metrics")
            fi = _get("/feature-importance")
            st.session_state["metrics_data"] = metrics
            st.session_state["feature_importance"] = fi
        except Exception as exc:
            st.error(f"Error leyendo metricas: {exc}")

    metrics_data = st.session_state.get("metrics_data", {})
    table = metrics_data.get("metrics", []) if isinstance(metrics_data, dict) else []
    if table:
        df_metrics = pd.DataFrame(table)
        st.dataframe(df_metrics, use_container_width=True)

        plot_cols = [c for c in ["accuracy", "f1_macro", "log_loss", "brier_score", "ece"] if c in df_metrics.columns]
        for metric_name in plot_cols:
            fig = px.bar(df_metrics, x="model", y=metric_name, title=f"Comparativa {metric_name}")
            st.plotly_chart(fig, use_container_width=True)

    fi_data = st.session_state.get("feature_importance", {})
    fi_rows = fi_data.get("top_features", []) if isinstance(fi_data, dict) else []
    if fi_rows:
        fi_df = pd.DataFrame(fi_rows)
        fig = px.bar(fi_df, x="importance", y="feature", orientation="h", title="Feature importance")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Explorador historico")
    st.info("Version base: conecta aqui tus agregaciones historicas (tabla liga, ELO y comparador).")
    st.write(
        "Siguiente iteracion recomendada: endpoint /history para devolver clasificacion, forma y evolucion ELO por jornada."
    )

with tab4:
    st.subheader("Simulacion Value Betting")

    c1, c2, c3, c4 = st.columns(4)
    s_season = c1.text_input("Temporada simulacion", value="2025/2026")
    s_round = c2.number_input("Jornada simulacion", min_value=1, max_value=50, value=38)
    budget = c3.number_input("Presupuesto", min_value=1.0, value=100.0, step=10.0)
    odds_source = c4.selectbox("Fuente de cuotas", options=["api", "simulated"], index=0)

    if st.button("Ejecutar simulacion"):
        payload = {
            "season": s_season,
            "round": int(s_round),
            "budget": float(budget),
            "odds_source": odds_source,
        }
        try:
            sim = _post("/simulate-value-betting", payload)
            bets = pd.DataFrame(sim.get("bets", []))
            st.metric("ROI", f"{sim.get('roi', 0.0):.2%}")
            st.metric("Profit", f"{sim.get('profit', 0.0):.2f}")

            if not bets.empty:
                st.dataframe(bets, use_container_width=True)
                bets["cum_profit"] = bets["profit"].cumsum()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=bets.index, y=bets["cum_profit"], mode="lines+markers", name="Capital"))
                fig.update_layout(title="Evolucion del beneficio acumulado", xaxis_title="Apuesta", yaxis_title="Profit")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No se detectaron apuestas de valor en esta simulacion")
        except Exception as exc:
            st.error(f"Error en simulacion: {exc}")
