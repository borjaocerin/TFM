from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests

try:
    from .preprocessing import infer_column_map, load_raw_dataset
except ImportError:
    from preprocessing import infer_column_map, load_raw_dataset


API_BASE_URL = "https://v3.football.api-sports.io"


def _headers() -> Dict[str, str]:
    key = os.getenv("API_FOOTBALL_KEY", "")
    return {"x-apisports-key": key} if key else {}


def get_fixtures(season: str, round_number: int, dataset_path: Path | str) -> List[Dict[str, Any]]:
    headers = _headers()
    if headers:
        try:
            response = requests.get(
                f"{API_BASE_URL}/fixtures",
                headers=headers,
                params={"league": 140, "season": season.split("/")[0], "round": f"Regular Season - {round_number}"},
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json().get("response", [])
            fixtures = []
            for row in payload:
                fixtures.append(
                    {
                        "match_id": row.get("fixture", {}).get("id"),
                        "date": row.get("fixture", {}).get("date"),
                        "home_team": row.get("teams", {}).get("home", {}).get("name"),
                        "away_team": row.get("teams", {}).get("away", {}).get("name"),
                    }
                )
            if fixtures:
                return fixtures
        except Exception:
            pass

    # Local fallback for offline work.
    raw = load_raw_dataset(dataset_path)
    cmap = infer_column_map(raw)

    if cmap.season and cmap.round:
        filt = raw[(raw[cmap.season].astype(str) == season) & (pd.to_numeric(raw[cmap.round], errors="coerce") == round_number)]
    else:
        filt = raw.tail(10)

    fixtures = []
    for _, row in filt.iterrows():
        fixtures.append(
            {
                "match_id": None,
                "date": str(row[cmap.date]),
                "home_team": row[cmap.home_team],
                "away_team": row[cmap.away_team],
            }
        )
    return fixtures


def get_team_stats(team_name: str, dataset_path: Path | str, last_n_matches: int = 5) -> Dict[str, Any]:
    raw = load_raw_dataset(dataset_path)
    cmap = infer_column_map(raw)

    team_mask = (raw[cmap.home_team] == team_name) | (raw[cmap.away_team] == team_name)
    df = raw[team_mask].sort_values(cmap.date).tail(last_n_matches)

    if df.empty:
        return {"team": team_name, "matches": 0}

    goals_for = []
    goals_against = []
    for _, row in df.iterrows():
        if row[cmap.home_team] == team_name:
            goals_for.append(float(row[cmap.home_goals]))
            goals_against.append(float(row[cmap.away_goals]))
        else:
            goals_for.append(float(row[cmap.away_goals]))
            goals_against.append(float(row[cmap.home_goals]))

    return {
        "team": team_name,
        "matches": len(df),
        "avg_goals_for": float(pd.Series(goals_for).mean()),
        "avg_goals_against": float(pd.Series(goals_against).mean()),
    }


def get_lineup(match_id: int) -> Dict[str, Any]:
    headers = _headers()
    if not headers:
        return {"match_id": match_id, "lineups": [], "source": "no_api_key"}

    try:
        response = requests.get(
            f"{API_BASE_URL}/fixtures/lineups",
            headers=headers,
            params={"fixture": match_id},
            timeout=20,
        )
        response.raise_for_status()
        return {"match_id": match_id, "lineups": response.json().get("response", [])}
    except Exception:
        return {"match_id": match_id, "lineups": [], "source": "error"}
