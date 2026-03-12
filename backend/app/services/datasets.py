from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.schemas.datasets import DatasetIngestRequest, FixturesFeatureRequest
from app.services.elo import add_internal_elo_features, enrich_with_elo
from app.services.features import add_basic_differentials, add_pre_match_rolling_features, add_target_label, enrich_fixtures
from app.services.football_data import load_football_data


def _resolve_path(path_value: str | None) -> Path | None:
    if path_value is None or path_value == "":
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (settings.data_dir.parent / path).resolve()


def _load_team_map(path_value: str | None) -> dict[str, str]:
    candidates: list[Path] = []
    if path_value:
        resolved = _resolve_path(path_value)
        if resolved is not None:
            candidates.append(resolved)
    candidates.append(settings.data_dir.parent / "etl" / "team_name_map_es.json")
    candidates.append(settings.data_dir.parent / "backend" / "team_name_map_es.json")

    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _normalize_historical_columns(raw: pd.DataFrame, team_map: dict[str, str]) -> pd.DataFrame:
    df = raw.copy()
    rename_map: dict[str, str] = {
        "date": "date",
        "hometeam": "home_team",
        "awayteam": "away_team",
        "home_team": "home_team",
        "away_team": "away_team",
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
        "season": "season",
    }
    normalized_map: dict[str, str] = {}
    for column in df.columns:
        key = column.strip().lower()
        if key in rename_map:
            normalized_map[column] = rename_map[key]

    df = df.rename(columns=normalized_map)
    expected = [
        "date",
        "season",
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
    for column in expected:
        if column not in df.columns:
            df[column] = np.nan

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d")
    df["home_team"] = df["home_team"].map(lambda value: team_map.get(str(value), str(value)))
    df["away_team"] = df["away_team"].map(lambda value: team_map.get(str(value), str(value)))

    df = df.dropna(subset=["date", "home_team", "away_team"])
    df = df.sort_values("date")
    return df


def _build_model_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Evita fuga: para modelado solo conservar señales pre-partido.
    # Se permite rolling ("_last"), ELO y cuotas, mas claves/target para trazabilidad.
    fixed_columns = {
        "date",
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "target",
    }

    blocked_columns = {
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
        "xg_diff",
        "xga_diff",
        "poss_diff",
        "sh_diff",
        "sot_diff",
        "goal_diff",
    }

    selected = [
        column
        for column in out.columns
        if (
            column in fixed_columns
            or "_last" in column
            or column.startswith("elo_")
            or column.startswith("odds_")
        )
    ]
    selected = [column for column in selected if column not in blocked_columns]
    selected = sorted(set(selected), key=selected.index)
    return out[selected]


def _summary(df: pd.DataFrame, output_all: Path, output_model: Path) -> dict[str, Any]:
    rows_by_season = (
        df["season"].astype(str).fillna("unknown").value_counts(dropna=False).sort_index().to_dict()
        if "season" in df.columns
        else {}
    )
    missing_pct = (df.isna().mean() * 100.0).round(2).to_dict()

    return {
        "rows_total": int(len(df)),
        "rows_by_season": {str(key): int(value) for key, value in rows_by_season.items()},
        "missing_pct_by_column": {str(key): float(value) for key, value in missing_pct.items()},
        "columns": list(df.columns),
        "output_all": str(output_all),
        "output_model": str(output_model),
    }


def ingest_datasets(request: DatasetIngestRequest) -> dict[str, Any]:
    historical_path = _resolve_path(request.historical)
    football_data_dir = _resolve_path(request.football_data_dir)
    elo_path = _resolve_path(request.elo_csv)

    if historical_path is None or not historical_path.exists():
        raise FileNotFoundError("No existe historical CSV")
    if football_data_dir is None or not football_data_dir.exists():
        raise FileNotFoundError("No existe football_data_dir")

    team_map = _load_team_map(request.team_map)
    historical_raw = pd.read_csv(historical_path)
    historical = _normalize_historical_columns(historical_raw, team_map)

    fdata = load_football_data(football_data_dir, team_map)
    merged = historical.merge(fdata, on=["date", "home_team", "away_team"], how="left")
    merged = add_basic_differentials(merged)
    merged = add_pre_match_rolling_features(merged, tuple(request.windows))
    merged = add_internal_elo_features(merged)
    merged = enrich_with_elo(merged, elo_path, team_map)
    merged = add_target_label(merged)

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_all = settings.output_dir / "laliga_enriched_all.csv"
    output_model = settings.output_dir / "laliga_enriched_model.csv"

    merged.to_csv(output_all, index=False)
    model_df = _build_model_dataset(merged)
    model_df.to_csv(output_model, index=False)

    return _summary(merged, output_all, output_model)


def build_fixtures_features(request: FixturesFeatureRequest) -> dict[str, Any]:
    fixtures_path = _resolve_path(request.fixtures_csv)
    historical_path = _resolve_path(request.historical_csv) or (settings.output_dir / "laliga_enriched_all.csv")
    elo_path = _resolve_path(request.elo_csv)
    if fixtures_path is None or not fixtures_path.exists():
        raise FileNotFoundError("No existe fixtures CSV")
    if not historical_path.exists():
        raise FileNotFoundError(
            "No existe historico enriquecido. Ejecuta /api/v1/datasets/ingest primero."
        )

    team_map = _load_team_map(request.team_map)
    fixtures_df = pd.read_csv(fixtures_path)
    historical_df = pd.read_csv(historical_path)

    enriched = enrich_fixtures(fixtures_df, historical_df, tuple(request.windows), team_map)
    enriched = enrich_with_elo(enriched, elo_path, team_map)

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "fixtures_enriched.csv"
    enriched.to_csv(output_path, index=False)

    return {
        "rows_total": int(len(enriched)),
        "generated_columns": list(enriched.columns),
        "output_path": str(output_path),
    }
