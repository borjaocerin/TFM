from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import settings
from app.models.model_store import ModelStore
from app.schemas.predict import OddsCompareRequest, PredictRequest
from app.services.evaluation import compare_market_vs_model


def _resolve_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (settings.data_dir.parent / path).resolve()


def _prepare_prediction_input(request: PredictRequest) -> pd.DataFrame:
    if request.fixtures is not None:
        return pd.DataFrame(request.fixtures)

    path = _resolve_path(request.fixtures_enriched_path)
    if path is None or not path.exists():
        raise FileNotFoundError("No existe fixtures_enriched.csv")
    return pd.read_csv(path)


def predict_matches(request: PredictRequest) -> dict[str, Any]:
    features_df = _prepare_prediction_input(request)

    store = ModelStore()
    payload, _ = store.load()
    model = payload["model"]
    feature_columns: list[str] = payload["feature_columns"]

    for column in feature_columns:
        if column not in features_df.columns:
            features_df[column] = pd.NA

    proba = model.predict_proba(features_df[feature_columns])

    output = features_df.copy()
    output["p_H"] = proba[:, 0]
    output["p_D"] = proba[:, 1]
    output["p_A"] = proba[:, 2]

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = settings.output_dir / "predictions.csv"
    output.to_csv(output_csv, index=False)

    keep = [
        column
        for column in [
            "date",
            "home_team",
            "away_team",
            "elo_diff",
            "xg_last5_diff",
            "xg_last10_diff",
            "odds_avg_h",
            "odds_avg_d",
            "odds_avg_a",
            "odds_close_h",
            "odds_close_d",
            "odds_close_a",
            "p_H",
            "p_D",
            "p_A",
            "target",
            "result",
        ]
        if column in output.columns
    ]
    preview = output[keep] if keep else output

    return {
        "rows": int(len(output)),
        "output_csv": str(output_csv),
        "predictions": preview.to_dict(orient="records"),
    }


def _prepare_compare_input(request: OddsCompareRequest) -> pd.DataFrame:
    if request.predictions is not None:
        return pd.DataFrame(request.predictions)

    path = _resolve_path(request.predictions_csv)
    if path is None or not path.exists():
        raise FileNotFoundError("No existe CSV para comparar cuotas")
    return pd.read_csv(path)


def compare_odds(request: OddsCompareRequest) -> dict[str, Any]:
    df = _prepare_compare_input(request)
    required = [
        f"{request.odds_kind}_h",
        f"{request.odds_kind}_d",
        f"{request.odds_kind}_a",
        "p_H",
        "p_D",
        "p_A",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para comparar cuotas: {missing}")

    metrics, compared = compare_market_vs_model(df, request.odds_kind, request.value_threshold)
    output_csv = settings.output_dir / "predictions_with_odds.csv"
    compared.to_csv(output_csv, index=False)

    value_bets = compared[compared["value_bet"]].copy()
    keep = [
        column
        for column in [
            "date",
            "home_team",
            "away_team",
            "best_ev_pick",
            "best_ev",
            "ev_H",
            "ev_D",
            "ev_A",
        ]
        if column in value_bets.columns
    ]
    return {
        "rows": int(len(compared)),
        "metrics": metrics,
        "value_bets": value_bets[keep].to_dict(orient="records") if keep else value_bets.to_dict(orient="records"),
        "output_csv": str(output_csv),
    }
