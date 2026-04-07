from __future__ import annotations

import pandas as pd

from app.schemas.predict import PredictUpcomingRequest
from modelos.services.predict import predict_selected_upcoming_match


def test_predict_upcoming_fallback_without_enrichment(monkeypatch) -> None:
    monkeypatch.setattr(
        "modelos.services.predict._ensure_default_enriched_historical",
        lambda: "ignored.csv",
    )
    monkeypatch.setattr(
        "modelos.services.predict._load_team_map",
        lambda: {},
    )
    monkeypatch.setattr(
        "modelos.services.predict.pd.read_csv",
        lambda _path: pd.DataFrame(),
    )

    def _boom(*_args, **_kwargs):
        raise ValueError("no features")

    monkeypatch.setattr("modelos.services.predict.enrich_fixtures", _boom)

    def _predict_stub(request):
        fixture = request.fixtures[0]
        return {
            "rows": 1,
            "output_csv": "data/out/predictions.csv",
            "predictions": [
                {
                    "date": fixture.get("date"),
                    "home_team": fixture.get("home_team"),
                    "away_team": fixture.get("away_team"),
                    "p_H": 0.5,
                    "p_D": 0.3,
                    "p_A": 0.2,
                }
            ],
        }

    monkeypatch.setattr("modelos.services.predict.predict_matches", _predict_stub)
    monkeypatch.setattr("modelos.services.predict.find_fixture_odds", lambda **_kwargs: None)

    result = predict_selected_upcoming_match(
        PredictUpcomingRequest(date="2026-04-15", home_team="Real Madrid", away_team="Barcelona")
    )

    assert result["market_odds"] is None
    assert result["prediction"]["p_H"] == 0.5
    assert "prediction_note" in result["prediction"]
