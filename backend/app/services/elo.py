from __future__ import annotations

from pathlib import Path

import pandas as pd


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
        return matches

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
