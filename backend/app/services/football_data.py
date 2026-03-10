from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _to_iso_date(series: pd.Series) -> pd.Series:
    date_value = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return date_value.dt.strftime("%Y-%m-%d")


def _std_team(team_name: Any, team_map: dict[str, str]) -> Any:
    if pd.isna(team_name):
        return team_name
    return team_map.get(str(team_name), str(team_name))


def _first_non_null(row: pd.Series, candidates: list[str]) -> float:
    for column in candidates:
        if column in row.index and pd.notna(row[column]):
            return float(row[column])
    return np.nan


def _collect_csvs(directory: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    if not directory.exists():
        return frames
    for csv_path in sorted(directory.glob("*.csv")):
        frame = pd.read_csv(csv_path)
        frame["source_file"] = csv_path.name
        frames.append(frame)
    return frames


def load_football_data(directory: Path, team_map: dict[str, str]) -> pd.DataFrame:
    frames = _collect_csvs(directory)
    if not frames:
        return pd.DataFrame(columns=["date", "home_team", "away_team"])

    df = pd.concat(frames, ignore_index=True, sort=False)
    column_lookup = {column.lower(): column for column in df.columns}

    date_column = column_lookup.get("date")
    home_column = column_lookup.get("hometeam") or column_lookup.get("home_team")
    away_column = column_lookup.get("awayteam") or column_lookup.get("away_team")
    if not date_column or not home_column or not away_column:
        raise ValueError("Los CSV de football-data deben incluir Date/HomeTeam/AwayTeam")

    out = pd.DataFrame()
    out["date"] = _to_iso_date(df[date_column])
    out["home_team"] = df[home_column].map(lambda value: _std_team(value, team_map))
    out["away_team"] = df[away_column].map(lambda value: _std_team(value, team_map))

    stats_map = {
        "referee": "referee",
        "attendance": "attendance",
        "hs": "sh_home",
        "as": "sh_away",
        "hst": "sot_home",
        "ast": "sot_away",
        "hc": "corners_home",
        "ac": "corners_away",
        "hy": "yellow_home",
        "ay": "yellow_away",
        "hr": "red_home",
        "ar": "red_away",
    }
    for source_lower, target in stats_map.items():
        source_column = column_lookup.get(source_lower)
        if source_column:
            out[target] = df[source_column]

    odds_open_candidates = {
        "odds_avg_h": ["AvgH", "B365H", "PSH", "MaxH"],
        "odds_avg_d": ["AvgD", "B365D", "PSD", "MaxD"],
        "odds_avg_a": ["AvgA", "B365A", "PSA", "MaxA"],
    }
    odds_close_candidates = {
        "odds_close_h": ["AvgCH", "B365CH", "PSCH", "MaxCH"],
        "odds_close_d": ["AvgCD", "B365CD", "PSCD", "MaxCD"],
        "odds_close_a": ["AvgCA", "B365CA", "PSCA", "MaxCA"],
    }

    for target_column, candidates in odds_open_candidates.items():
        out[target_column] = df.apply(lambda row: _first_non_null(row, candidates), axis=1)

    for target_column, candidates in odds_close_candidates.items():
        out[target_column] = df.apply(lambda row: _first_non_null(row, candidates), axis=1)

    # Goles y resultado final por si hacen falta para chequeos de calidad.
    for source_column, target_column in {
        "fthg": "fd_home_goals",
        "ftag": "fd_away_goals",
        "ftr": "fd_result",
    }.items():
        source = column_lookup.get(source_column)
        if source:
            out[target_column] = df[source]

    out = out.dropna(subset=["date", "home_team", "away_team"])
    out = out.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    return out
