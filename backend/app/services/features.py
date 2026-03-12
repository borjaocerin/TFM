from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.services.elo import add_internal_elo_features


ENRICHMENT_BASE_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "xg_home",
    "xg_away",
    "xga_home",
    "xga_away",
    "poss_home",
    "poss_away",
    "sh_home",
    "sh_away",
    "sot_home",
    "sot_away",
]


def _safe_points(goals_for: Any, goals_against: Any) -> float:
    if pd.isna(goals_for) or pd.isna(goals_against):
        return np.nan
    if goals_for > goals_against:
        return 3.0
    if goals_for < goals_against:
        return 0.0
    return 1.0


def add_basic_differentials(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pairs = [
        ("xg_home", "xg_away", "xg_diff"),
        ("xga_home", "xga_away", "xga_diff"),
        ("poss_home", "poss_away", "poss_diff"),
        ("sh_home", "sh_away", "sh_diff"),
        ("sot_home", "sot_away", "sot_diff"),
        ("home_goals", "away_goals", "goal_diff"),
    ]
    for home_col, away_col, output_col in pairs:
        if home_col in out.columns and away_col in out.columns:
            out[output_col] = pd.to_numeric(out[home_col], errors="coerce") - pd.to_numeric(
                out[away_col], errors="coerce"
            )
    return out


def _build_team_tall(matches: pd.DataFrame) -> pd.DataFrame:
    base = matches.copy()
    required = [
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "xg_home",
        "xg_away",
        "xga_home",
        "xga_away",
        "poss_home",
        "poss_away",
        "sh_home",
        "sh_away",
        "sot_home",
        "sot_away",
    ]
    for column in required:
        if column not in base.columns:
            base[column] = np.nan

    base["date_dt"] = pd.to_datetime(base["date"], errors="coerce")

    home = pd.DataFrame(
        {
            "team": base["home_team"],
            "date": base["date_dt"],
            "gf": pd.to_numeric(base["home_goals"], errors="coerce"),
            "ga": pd.to_numeric(base["away_goals"], errors="coerce"),
            "xg": pd.to_numeric(base["xg_home"], errors="coerce"),
            "xga": pd.to_numeric(base["xga_home"], errors="coerce"),
            "poss": pd.to_numeric(base["poss_home"], errors="coerce"),
            "sh": pd.to_numeric(base["sh_home"], errors="coerce"),
            "sot": pd.to_numeric(base["sot_home"], errors="coerce"),
        }
    )
    away = pd.DataFrame(
        {
            "team": base["away_team"],
            "date": base["date_dt"],
            "gf": pd.to_numeric(base["away_goals"], errors="coerce"),
            "ga": pd.to_numeric(base["home_goals"], errors="coerce"),
            "xg": pd.to_numeric(base["xg_away"], errors="coerce"),
            "xga": pd.to_numeric(base["xga_away"], errors="coerce"),
            "poss": pd.to_numeric(base["poss_away"], errors="coerce"),
            "sh": pd.to_numeric(base["sh_away"], errors="coerce"),
            "sot": pd.to_numeric(base["sot_away"], errors="coerce"),
        }
    )
    tall = pd.concat([home, away], ignore_index=True)
    tall["points"] = [_safe_points(row.gf, row.ga) for row in tall.itertuples()]
    return tall.dropna(subset=["team", "date"]).sort_values(["team", "date"])


def compute_team_rolling_features(matches: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    tall = _build_team_tall(matches)

    metrics = ["xg", "xga", "poss", "sh", "sot", "gf", "ga"]
    for window in windows:
        grouped = tall.groupby("team", group_keys=False)
        for metric in metrics:
            tall[f"{metric}_last{window}"] = grouped[metric].apply(
                lambda series: series.rolling(window, min_periods=1).mean().shift(1)
            )
        tall[f"points_last{window}"] = grouped["points"].apply(
            lambda series: series.rolling(window, min_periods=1).sum().shift(1)
        )

    keep = ["team", "date"] + [
        column
        for column in tall.columns
        if any(column.endswith(f"last{window}") for window in windows)
    ]
    return tall[keep].drop_duplicates(subset=["team", "date"], keep="last")


def _join_side_features(
    matches: pd.DataFrame,
    team_features: pd.DataFrame,
    team_column: str,
    suffix: str,
) -> pd.DataFrame:
    left = matches[["match_id", "date_dt", team_column]].rename(columns={team_column: "team"})
    right = team_features.rename(columns={"date": "feature_date"})
    joined = pd.merge_asof(
        left.sort_values("date_dt"),
        right.sort_values("feature_date"),
        left_on="date_dt",
        right_on="feature_date",
        by="team",
        direction="backward",
    )
    keep_columns = [column for column in joined.columns if column not in {"date_dt", "team", "feature_date"}]
    rename_map = {
        column: f"{column}_{suffix}" for column in keep_columns if column != "match_id"
    }
    return joined[keep_columns].rename(columns=rename_map)


def add_pre_match_rolling_features(matches: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    out = matches.copy()
    out["match_id"] = range(len(out))
    out["date_dt"] = pd.to_datetime(out["date"], errors="coerce")

    team_features = compute_team_rolling_features(out, windows)

    home = _join_side_features(out, team_features, "home_team", "home")
    away = _join_side_features(out, team_features, "away_team", "away")

    out = out.merge(home, on="match_id", how="left")
    out = out.merge(away, on="match_id", how="left")

    diff_bases = [
        f"{metric}_last{window}"
        for metric in ["xg", "xga", "poss", "sh", "sot", "gf", "ga", "points"]
        for window in windows
    ]
    for base in diff_bases:
        home_col = f"{base}_home"
        away_col = f"{base}_away"
        if home_col in out.columns and away_col in out.columns:
            out[f"{base}_diff"] = out[home_col] - out[away_col]

    out = out.drop(columns=["match_id", "date_dt"])
    return out


def _normalize_fixture_columns(fixtures: pd.DataFrame) -> pd.DataFrame:
    out = fixtures.copy()
    lowercase_map = {column.lower(): column for column in out.columns}

    if "date" not in out.columns and "date" in lowercase_map:
        out = out.rename(columns={lowercase_map["date"]: "date"})
    if "Date" in out.columns:
        out = out.rename(columns={"Date": "date"})
    if "HomeTeam" in out.columns:
        out = out.rename(columns={"HomeTeam": "home_team"})
    if "AwayTeam" in out.columns:
        out = out.rename(columns={"AwayTeam": "away_team"})

    raw_dates = out["date"].astype(str).str.strip()
    parsed_iso = pd.to_datetime(raw_dates, errors="coerce", format="%Y-%m-%d")
    parsed_dayfirst = pd.to_datetime(raw_dates, errors="coerce", dayfirst=True)
    parsed_standard = pd.to_datetime(raw_dates, errors="coerce", dayfirst=False)
    preferred_fallback = (
        parsed_dayfirst
        if int(parsed_dayfirst.notna().sum()) > int(parsed_standard.notna().sum())
        else parsed_standard
    )
    out["date"] = parsed_iso.fillna(preferred_fallback).dt.strftime("%Y-%m-%d")
    return out


def _normalize_historical_for_enrichment(
    historical: pd.DataFrame,
    team_map: dict[str, str],
) -> pd.DataFrame:
    out = historical.copy()
    rename_map = {
        "fthg": "home_goals",
        "ftag": "away_goals",
        "home_goals": "home_goals",
        "away_goals": "away_goals",
        "ftr": "result",
        "result": "result",
        "xg_home": "xg_home",
        "xg_away": "xg_away",
        "xga_home": "xga_home",
        "xga_away": "xga_away",
        "poss_home": "poss_home",
        "poss_away": "poss_away",
        "sh_home": "sh_home",
        "sh_away": "sh_away",
        "sot_home": "sot_home",
        "sot_away": "sot_away",
    }
    normalized_map: dict[str, str] = {}
    for column in out.columns:
        key = column.strip().lower()
        if key in rename_map:
            normalized_map[column] = rename_map[key]

    out = out.rename(columns=normalized_map)
    out = _normalize_fixture_columns(out)

    for column in ENRICHMENT_BASE_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan

    out["home_team"] = out["home_team"].map(lambda value: team_map.get(str(value), str(value)))
    out["away_team"] = out["away_team"].map(lambda value: team_map.get(str(value), str(value)))

    home_goals = pd.to_numeric(out["home_goals"], errors="coerce")
    away_goals = pd.to_numeric(out["away_goals"], errors="coerce")
    result_text = out["result"].astype(str).str.strip().str.upper()
    has_result = ~result_text.isin(["", "NAN", "NONE"])
    has_score = home_goals.notna() & away_goals.notna()

    out = out[has_result | has_score].copy()
    out = out.dropna(subset=["date", "home_team", "away_team"])
    out = out.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return out[ENRICHMENT_BASE_COLUMNS]


def enrich_fixtures(
    fixtures: pd.DataFrame,
    historical: pd.DataFrame,
    windows: tuple[int, ...],
    team_map: dict[str, str],
) -> pd.DataFrame:
    base = _normalize_fixture_columns(fixtures)
    if "home_team" not in base.columns or "away_team" not in base.columns:
        raise ValueError("fixtures.csv debe contener HomeTeam/AwayTeam o home_team/away_team")

    base["home_team"] = base["home_team"].map(lambda value: team_map.get(str(value), str(value)))
    base["away_team"] = base["away_team"].map(lambda value: team_map.get(str(value), str(value)))
    base = base.reset_index(drop=True).copy()
    base["__fixture_row_id"] = range(len(base))
    base["__is_prediction_fixture"] = True

    historical_base = _normalize_historical_for_enrichment(historical, team_map)
    historical_base["__fixture_row_id"] = np.nan
    historical_base["__is_prediction_fixture"] = False

    combined = pd.concat([historical_base, base], ignore_index=True, sort=False)

    out = add_pre_match_rolling_features(combined, windows)
    out = add_internal_elo_features(out)
    out = add_basic_differentials(out)
    out = out[out["__is_prediction_fixture"]].copy()
    out = out.sort_values("__fixture_row_id").reset_index(drop=True)
    return out.drop(columns=["__fixture_row_id", "__is_prediction_fixture"])


def add_target_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "result" not in out.columns:
        out["result"] = np.nan

    result_upper = out["result"].astype(str).str.upper().str.strip()
    missing_result = result_upper.isin(["NAN", "NONE", ""])

    home_goals = pd.to_numeric(out.get("home_goals"), errors="coerce")
    away_goals = pd.to_numeric(out.get("away_goals"), errors="coerce")
    inferred = np.where(home_goals > away_goals, "H", np.where(home_goals < away_goals, "A", "D"))
    result_upper = np.where(missing_result, inferred, result_upper)

    mapping = {"H": 0, "D": 1, "A": 2}
    result_series = pd.Series(result_upper, index=out.index)
    out["target"] = result_series.map(mapping)
    out["result"] = result_series
    return out
