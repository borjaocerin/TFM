from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from app.services.datasets import _augment_historical_with_manual_results


def test_augment_historical_adds_played_manual_2026_rows(tmp_path) -> None:
    historical = pd.DataFrame(
        [
            {
                "date": "2025-10-27",
                "season": 2025.0,
                "home_team": "Barcelona",
                "away_team": "Valencia",
                "home_goals": 2.0,
                "away_goals": 1.0,
                "result": "H",
            }
        ]
    )

    manual_payload = {
        "matches": [
            {
                "date": "2026-01-10",
                "team1": "FC Barcelona",
                "team2": "Rayo Vallecano de Madrid",
                "score": {"ht": [1, 0], "ft": [2, 1]},
            },
            {
                "date": "2026-03-18",
                "team1": "Valencia CF",
                "team2": "Getafe CF",
                "score": {"ht": [0, 0], "ft": [None, None]},
            },
            {
                "date": "2026-04-01",
                "team1": "Valencia CF",
                "team2": "Getafe CF",
                "score": {"ht": [0, 0], "ft": [1, 1]},
            },
        ]
    }

    manual_path = tmp_path / "proximosPartidos.json"
    manual_path.write_text(json.dumps(manual_payload), encoding="utf-8")

    out = _augment_historical_with_manual_results(
        historical,
        team_map={},
        manual_results_path=manual_path,
        cutoff_date=date(2026, 3, 20),
    )

    assert len(out) == 2

    added = out[
        (out["date"] == "2026-01-10")
        & (out["home_team"] == "Barcelona")
        & (out["away_team"] == "Rayo Vallecano")
    ]
    assert len(added) == 1

    row = added.iloc[0]
    assert float(row["season"]) == 2025.0
    assert float(row["home_goals"]) == 2.0
    assert float(row["away_goals"]) == 1.0
    assert row["result"] == "H"
    assert float(row["ht_home_goals"]) == 1.0
    assert float(row["ht_away_goals"]) == 0.0
    assert row["ht_result"] == "H"


def test_augment_historical_fills_missing_existing_match_values(tmp_path) -> None:
    historical = pd.DataFrame(
        [
            {
                "date": "2026-01-10",
                "season": np.nan,
                "home_team": "Barcelona",
                "away_team": "Rayo Vallecano",
                "home_goals": np.nan,
                "away_goals": np.nan,
                "result": np.nan,
                "ht_home_goals": np.nan,
                "ht_away_goals": np.nan,
                "ht_result": np.nan,
            }
        ]
    )

    manual_payload = {
        "matches": [
            {
                "date": "2026-01-10",
                "team1": "FC Barcelona",
                "team2": "Rayo Vallecano de Madrid",
                "score": {"ht": [1, 0], "ft": [2, 1]},
            }
        ]
    }

    manual_path = tmp_path / "proximosPartidos.json"
    manual_path.write_text(json.dumps(manual_payload), encoding="utf-8")

    out = _augment_historical_with_manual_results(
        historical,
        team_map={},
        manual_results_path=manual_path,
        cutoff_date=date(2026, 3, 20),
    )

    assert len(out) == 1
    row = out.iloc[0]
    assert float(row["season"]) == 2025.0
    assert float(row["home_goals"]) == 2.0
    assert float(row["away_goals"]) == 1.0
    assert row["result"] == "H"
    assert float(row["ht_home_goals"]) == 1.0
    assert float(row["ht_away_goals"]) == 0.0
    assert row["ht_result"] == "H"
