from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "dataset_clean.csv"

RESULT_TO_INT = {"H": 0, "D": 1, "A": 2}
INT_TO_RESULT = {0: "home_win", 1: "draw", 2: "away_win"}

TEAM_ALIASES = {
    "athletic club": "Athletic",
    "athletic bilbao": "Athletic",
    "fc barcelona": "Barcelona",
    "real madrid cf": "Real Madrid",
    "atletico madrid": "Atletico Madrid",
}


@dataclass
class ColumnMap:
    date: str
    season: Optional[str]
    round: Optional[str]
    home_team: str
    away_team: str
    home_goals: str
    away_goals: str
    ftr: Optional[str]
    home_shots: Optional[str]
    away_shots: Optional[str]
    home_shots_on_target: Optional[str]
    away_shots_on_target: Optional[str]
    home_possession: Optional[str]
    away_possession: Optional[str]


def normalize_team_name(name: str) -> str:
    clean = str(name).strip()
    return TEAM_ALIASES.get(clean.lower(), clean)


def _choose_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"No se encontro ninguna columna entre: {candidates}")
    return None


def infer_column_map(df: pd.DataFrame) -> ColumnMap:
    return ColumnMap(
        date=_choose_column(df, ["date", "Date", "match_date"]),
        season=_choose_column(df, ["season", "Season", "temporada"], required=False),
        round=_choose_column(df, ["round", "Round", "jornada", "matchday"], required=False),
        home_team=_choose_column(df, ["home_team", "HomeTeam", "local", "team_home"]),
        away_team=_choose_column(df, ["away_team", "AwayTeam", "visitante", "team_away"]),
        home_goals=_choose_column(df, ["home_goals", "FTHG", "goals_home"]),
        away_goals=_choose_column(df, ["away_goals", "FTAG", "goals_away"]),
        ftr=_choose_column(df, ["FTR", "result", "full_time_result"], required=False),
        home_shots=_choose_column(df, ["HS", "home_shots", "shots_home"], required=False),
        away_shots=_choose_column(df, ["AS", "away_shots", "shots_away"], required=False),
        home_shots_on_target=_choose_column(df, ["HST", "home_shots_on_target", "sot_home"], required=False),
        away_shots_on_target=_choose_column(df, ["AST", "away_shots_on_target", "sot_away"], required=False),
        home_possession=_choose_column(df, ["home_possession", "possession_home"], required=False),
        away_possession=_choose_column(df, ["away_possession", "possession_away"], required=False),
    )


def load_raw_dataset(dataset_path: Path | str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro dataset en {path}. Copia el CSV historico en data/dataset_clean.csv"
        )
    df = pd.read_csv(path)
    cmap = infer_column_map(df)

    df[cmap.date] = pd.to_datetime(df[cmap.date], errors="coerce")
    df = df.dropna(subset=[cmap.date])

    df[cmap.home_team] = df[cmap.home_team].map(normalize_team_name)
    df[cmap.away_team] = df[cmap.away_team].map(normalize_team_name)

    return df.sort_values(cmap.date).reset_index(drop=True)


def _compute_result_label(row: pd.Series, cmap: ColumnMap) -> str:
    if cmap.ftr and pd.notna(row[cmap.ftr]):
        val = str(row[cmap.ftr]).strip().upper()
        if val in RESULT_TO_INT:
            return val

    hg = row[cmap.home_goals]
    ag = row[cmap.away_goals]
    if hg > ag:
        return "H"
    if hg < ag:
        return "A"
    return "D"


def _build_matches(df: pd.DataFrame, cmap: ColumnMap) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "date": df[cmap.date],
            "season": df[cmap.season] if cmap.season else "unknown",
            "round": df[cmap.round] if cmap.round else np.nan,
            "home_team": df[cmap.home_team],
            "away_team": df[cmap.away_team],
            "home_goals": pd.to_numeric(df[cmap.home_goals], errors="coerce"),
            "away_goals": pd.to_numeric(df[cmap.away_goals], errors="coerce"),
            "home_shots": pd.to_numeric(df[cmap.home_shots], errors="coerce") if cmap.home_shots else np.nan,
            "away_shots": pd.to_numeric(df[cmap.away_shots], errors="coerce") if cmap.away_shots else np.nan,
            "home_sot": pd.to_numeric(df[cmap.home_shots_on_target], errors="coerce")
            if cmap.home_shots_on_target
            else np.nan,
            "away_sot": pd.to_numeric(df[cmap.away_shots_on_target], errors="coerce")
            if cmap.away_shots_on_target
            else np.nan,
            "home_possession": pd.to_numeric(df[cmap.home_possession], errors="coerce")
            if cmap.home_possession
            else np.nan,
            "away_possession": pd.to_numeric(df[cmap.away_possession], errors="coerce")
            if cmap.away_possession
            else np.nan,
        }
    )

    base["result"] = df.apply(lambda row: _compute_result_label(row, cmap), axis=1)
    base["result_int"] = base["result"].map(RESULT_TO_INT).astype(int)
    base = base.dropna(subset=["home_goals", "away_goals"])
    base = base.reset_index(drop=True)
    base["match_id"] = np.arange(len(base))
    return base


def _build_team_form_table(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "date": matches["date"],
            "team": matches["home_team"],
            "goals_for": matches["home_goals"],
            "goals_against": matches["away_goals"],
            "shots_for": matches["home_shots"],
            "shots_against": matches["away_shots"],
            "sot_for": matches["home_sot"],
            "sot_against": matches["away_sot"],
            "possession": matches["home_possession"],
            "is_home": 1,
        }
    )

    away = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "date": matches["date"],
            "team": matches["away_team"],
            "goals_for": matches["away_goals"],
            "goals_against": matches["home_goals"],
            "shots_for": matches["away_shots"],
            "shots_against": matches["home_shots"],
            "sot_for": matches["away_sot"],
            "sot_against": matches["home_sot"],
            "possession": matches["away_possession"],
            "is_home": 0,
        }
    )

    long_df = pd.concat([home, away], ignore_index=True).sort_values(["team", "date", "match_id"])

    points = []
    win = []
    for _, row in long_df.iterrows():
        gf = row["goals_for"]
        ga = row["goals_against"]
        if gf > ga:
            points.append(3)
            win.append(1)
        elif gf == ga:
            points.append(1)
            win.append(0)
        else:
            points.append(0)
            win.append(0)

    long_df["points"] = points
    long_df["win"] = win

    rolling_cols = ["points", "win", "goals_for", "goals_against", "shots_for", "shots_against", "sot_for", "sot_against", "possession"]
    for col in rolling_cols:
        long_df[f"last{window}_{col}"] = (
            long_df.groupby("team")[col]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
            .fillna(0.0)
        )

    return long_df


def _compute_elo_features(matches: pd.DataFrame, k_factor: float = 20.0) -> pd.DataFrame:
    elo: Dict[str, float] = {}
    rows: List[Tuple[float, float, float]] = []

    for _, row in matches.sort_values(["date", "match_id"]).iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_elo = elo.get(home, 1500.0)
        away_elo = elo.get(away, 1500.0)

        exp_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400.0))
        exp_away = 1.0 - exp_home

        result = row["result"]
        if result == "H":
            score_home, score_away = 1.0, 0.0
        elif result == "A":
            score_home, score_away = 0.0, 1.0
        else:
            score_home, score_away = 0.5, 0.5

        rows.append((home_elo, away_elo, home_elo - away_elo))

        elo[home] = home_elo + k_factor * (score_home - exp_home)
        elo[away] = away_elo + k_factor * (score_away - exp_away)

    elo_df = pd.DataFrame(rows, columns=["home_elo", "away_elo", "elo_difference"])
    return elo_df


def build_feature_dataset(raw_df: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:
    cmap = infer_column_map(raw_df)
    matches = _build_matches(raw_df, cmap)
    team_form = _build_team_form_table(matches, window=rolling_window)

    home_form = team_form[team_form["is_home"] == 1].copy()
    away_form = team_form[team_form["is_home"] == 0].copy()

    home_form = home_form[["match_id"] + [c for c in home_form.columns if c.startswith(f"last{rolling_window}_")]]
    away_form = away_form[["match_id"] + [c for c in away_form.columns if c.startswith(f"last{rolling_window}_")]]

    home_form = home_form.rename(columns={c: f"home_{c}" for c in home_form.columns if c != "match_id"})
    away_form = away_form.rename(columns={c: f"away_{c}" for c in away_form.columns if c != "match_id"})

    features = matches.merge(home_form, on="match_id", how="left").merge(away_form, on="match_id", how="left")

    features["goals_diff_recent"] = features[f"home_last{rolling_window}_goals_for"] - features[f"away_last{rolling_window}_goals_for"]
    features["defense_diff_recent"] = features[f"away_last{rolling_window}_goals_against"] - features[f"home_last{rolling_window}_goals_against"]
    features["points_diff_recent"] = features[f"home_last{rolling_window}_points"] - features[f"away_last{rolling_window}_points"]
    features["shots_diff_recent"] = features[f"home_last{rolling_window}_shots_for"] - features[f"away_last{rolling_window}_shots_for"]
    features["sot_diff_recent"] = features[f"home_last{rolling_window}_sot_for"] - features[f"away_last{rolling_window}_sot_for"]
    features["possession_diff_recent"] = (
        features[f"home_last{rolling_window}_possession"] - features[f"away_last{rolling_window}_possession"]
    )

    elo_df = _compute_elo_features(features)
    features = pd.concat([features.reset_index(drop=True), elo_df], axis=1)

    numeric_cols = features.select_dtypes(include=["number"]).columns
    features[numeric_cols] = features[numeric_cols].fillna(0.0)

    return features.sort_values(["date", "match_id"]).reset_index(drop=True)


def get_model_feature_columns(features_df: pd.DataFrame) -> List[str]:
    drop_cols = {
        "match_id",
        "date",
        "season",
        "round",
        "home_team",
        "away_team",
        "result",
        "result_int",
        "home_goals",
        "away_goals",
    }
    return [c for c in features_df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(features_df[c])]


def split_xy(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    feature_columns = get_model_feature_columns(features_df)
    x = features_df[feature_columns].copy()
    y = features_df["result_int"].copy()
    return x, y, feature_columns


def build_prediction_row(features_df: pd.DataFrame, home_team: str, away_team: str) -> pd.DataFrame:
    home_team = normalize_team_name(home_team)
    away_team = normalize_team_name(away_team)
    feature_columns = get_model_feature_columns(features_df)

    home_hist = features_df[(features_df["home_team"] == home_team) | (features_df["away_team"] == home_team)].tail(1)
    away_hist = features_df[(features_df["home_team"] == away_team) | (features_df["away_team"] == away_team)].tail(1)

    if home_hist.empty or away_hist.empty:
        league_avg = features_df[feature_columns].mean(numeric_only=True)
        return pd.DataFrame([league_avg], columns=feature_columns).fillna(0.0)

    latest_home = home_hist.iloc[-1]
    latest_away = away_hist.iloc[-1]

    row = {}
    for col in feature_columns:
        if col.startswith("home_"):
            row[col] = float(latest_home.get(col, 0.0))
        elif col.startswith("away_"):
            row[col] = float(latest_away.get(col, 0.0))
        elif col == "elo_difference":
            row[col] = float(latest_home.get("home_elo", 1500.0) - latest_away.get("away_elo", 1500.0))
        elif col == "home_elo":
            row[col] = float(latest_home.get("home_elo", 1500.0))
        elif col == "away_elo":
            row[col] = float(latest_away.get("away_elo", 1500.0))
        elif col.endswith("_diff_recent"):
            base = col.replace("_diff_recent", "")
            h_col = f"home_last5_{base}"
            a_col = f"away_last5_{base}"
            row[col] = float(latest_home.get(h_col, 0.0) - latest_away.get(a_col, 0.0))
        else:
            row[col] = float(features_df[col].mean()) if col in features_df else 0.0

    return pd.DataFrame([row], columns=feature_columns).fillna(0.0)
