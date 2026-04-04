from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _infer_home_score(result: object, home_goals: object, away_goals: object) -> float | None:
    result_text = str(result).strip().upper()
    if result_text == "H":
        return 1.0
    if result_text in {"D", "X"}:
        return 0.5
    if result_text == "A":
        return 0.0

    if pd.notna(home_goals) and pd.notna(away_goals):
        home_value = float(home_goals)
        away_value = float(away_goals)
        if home_value > away_value:
            return 1.0
        if home_value < away_value:
            return 0.0
        return 0.5

    return None


def add_internal_elo_features(
    matches: pd.DataFrame,
    initial_rating: float = 1500.0,
    k_factor: float = 24.0,
    home_advantage: float = 60.0,
) -> pd.DataFrame:
    if not {"date", "home_team", "away_team"}.issubset(matches.columns):
        return matches

    out = matches.copy()
    had_date_dt = "date_dt" in out.columns
    out["__elo_row_id"] = range(len(out))
    out["date_dt"] = pd.to_datetime(out["date"], errors="coerce")
    out["elo_home"] = np.nan
    out["elo_away"] = np.nan
    out["elo_diff"] = np.nan

    ratings: dict[str, float] = {}
    ordered = out.sort_values(["date_dt", "__elo_row_id"])

    for row_index in ordered.index:
        row = out.loc[row_index]
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])

        home_rating = ratings.get(home_team, initial_rating)
        away_rating = ratings.get(away_team, initial_rating)

        out.at[row_index, "elo_home"] = home_rating
        out.at[row_index, "elo_away"] = away_rating
        out.at[row_index, "elo_diff"] = home_rating - away_rating

        home_score = _infer_home_score(
            row.get("result"),
            row.get("home_goals"),
            row.get("away_goals"),
        )
        if home_score is None:
            continue

        expected_home = 1.0 / (1.0 + 10.0 ** (((away_rating) - (home_rating + home_advantage)) / 400.0))
        goal_margin_multiplier = 1.0

        if pd.notna(row.get("home_goals")) and pd.notna(row.get("away_goals")):
            goal_margin = abs(float(row["home_goals"]) - float(row["away_goals"]))
            if goal_margin > 1.0:
                rating_gap = abs(home_rating - away_rating)
                goal_margin_multiplier = np.log(goal_margin + 1.0) * (2.2 / (rating_gap * 0.001 + 2.2))

        delta = k_factor * goal_margin_multiplier * (home_score - expected_home)
        ratings[home_team] = home_rating + delta
        ratings[away_team] = away_rating - delta

    drop_columns = ["__elo_row_id"]
    if not had_date_dt:
        drop_columns.append("date_dt")
    return out.drop(columns=drop_columns)


def load_elo(elo_path: Path, team_map: dict[str, str]) -> pd.DataFrame:
    elo = pd.read_csv(elo_path)
    elo.columns = [column.strip().lower() for column in elo.columns]
    if "date" not in elo.columns or "club" not in elo.columns or "elo" not in elo.columns:
        raise ValueError("ELO_RATINGS.csv debe contener columnas Date, Club, Elo")
    elo["club"] = elo["club"].map(lambda value: team_map.get(str(value), str(value)))
    elo["date"] = pd.to_datetime(elo["date"], errors="coerce")
    elo = elo.dropna(subset=["date", "club", "elo"]).sort_values("date")
    return elo[["date", "club", "elo"]]


def _merge_side_elo(
    matches: pd.DataFrame,
    elo: pd.DataFrame,
    team_column: str,
    output_column: str,
) -> pd.DataFrame:
    left = matches[["match_id", "date_dt", team_column]].rename(columns={team_column: "club"})
    merged = pd.merge_asof(
        left.sort_values("date_dt"),
        elo.sort_values("date"),
        left_on="date_dt",
        right_on="date",
        by="club",
        direction="backward",
    )
    return merged[["match_id", "elo"]].rename(columns={"elo": output_column})


def enrich_with_elo(
    matches: pd.DataFrame,
    elo_path: Path | None,
    team_map: dict[str, str],
) -> pd.DataFrame:
    if elo_path is None or not elo_path.exists():
        return add_internal_elo_features(matches)

    elo = load_elo(elo_path, team_map)
    out = matches.copy()
    out["match_id"] = range(len(out))
    out["date_dt"] = pd.to_datetime(out["date"], errors="coerce")

    home = _merge_side_elo(out, elo, "home_team", "elo_home")
    away = _merge_side_elo(out, elo, "away_team", "elo_away")

    out = out.merge(home, on="match_id", how="left")
    out = out.merge(away, on="match_id", how="left")
    out["elo_diff"] = out["elo_home"] - out["elo_away"]

    return out.drop(columns=["match_id", "date_dt"])
