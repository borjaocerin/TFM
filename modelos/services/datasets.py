from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
from typing import Any
import unicodedata

import numpy as np
import pandas as pd

from app.core.config import settings
from app.schemas.datasets import DatasetIngestRequest, FixturesFeatureRequest
from modelos.services.elo import add_internal_elo_features, enrich_with_elo
from modelos.services.features import add_basic_differentials, add_pre_match_rolling_features, add_target_label, enrich_fixtures
from modelos.services.football_data import load_football_data


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


TEAM_MANUAL_ALIASES: dict[str, str] = {
    "deportivo alaves": "Alaves",
    "alaves": "Alaves",
    "girona fc": "Girona",
    "villarreal cf": "Villarreal",
    "rcd mallorca": "Mallorca",
    "fc barcelona": "Barcelona",
    "valencia cf": "Valencia",
    "rc celta de vigo": "Celta",
    "club atletico de madrid": "Atletico Madrid",
    "atletico de madrid": "Atletico Madrid",
    "real oviedo": "Oviedo",
    "ca osasuna": "Osasuna",
    "real madrid cf": "Real Madrid",
    "real betis balompie": "Betis",
    "real sociedad de futbol": "Real Sociedad",
    "rayo vallecano de madrid": "Rayo Vallecano",
    "levante ud": "Levante",
    "getafe cf": "Getafe",
    "elche cf": "Elche",
    "sevilla fc": "Sevilla",
    "rcd espanyol de barcelona": "Espanyol",
}


def _normalize_text_basic(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "":
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    for token in [".", ",", ";", ":", "'", '"', "-", "_"]:
        text = text.replace(token, " ")
    return " ".join(text.split())


def _canonical_team_name(team_name: Any, team_map: dict[str, str]) -> str:
    raw = str(team_name or "").strip()
    if raw == "":
        return ""

    direct = team_map.get(raw)
    if direct:
        return str(direct)

    normalized_input = _normalize_text_basic(raw)
    normalized_map: dict[str, str] = {}
    for source_name, canonical_name in team_map.items():
        canonical = str(canonical_name)
        normalized_map[_normalize_text_basic(source_name)] = canonical
        normalized_map[_normalize_text_basic(canonical)] = canonical

    for alias, canonical in TEAM_MANUAL_ALIASES.items():
        normalized_map[_normalize_text_basic(alias)] = canonical

    mapped = normalized_map.get(normalized_input)
    return mapped if mapped else raw


def _season_start_from_match_date(match_date: date) -> float:
    start_year = match_date.year if match_date.month >= 7 else match_date.year - 1
    return float(start_year)


def _result_from_score(home_goals: float, away_goals: float) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def _parse_score_pair(value: Any) -> tuple[float | None, float | None]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None, None

    try:
        home_goals = float(value[0])
        away_goals = float(value[1])
    except Exception:
        return None, None

    if not np.isfinite(home_goals) or not np.isfinite(away_goals):
        return None, None
    return home_goals, away_goals


def _empty_manual_results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "season",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
            "ht_home_goals",
            "ht_away_goals",
            "ht_result",
        ]
    )


def _extract_completed_manual_results(
    manual_results_path: Path,
    team_map: dict[str, str],
    cutoff_date: date | None = None,
) -> pd.DataFrame:
    if not manual_results_path.exists():
        return _empty_manual_results_frame()

    payload: Any
    try:
        payload = json.loads(manual_results_path.read_text(encoding="utf-8"))
    except Exception:
        try:
            payload = json.loads(manual_results_path.read_text(encoding="latin-1"))
        except Exception:
            return _empty_manual_results_frame()

    matches = payload.get("matches") if isinstance(payload, dict) else None
    if not isinstance(matches, list):
        return _empty_manual_results_frame()

    cutoff = cutoff_date or datetime.now().date()
    rows: list[dict[str, Any]] = []

    for item in matches:
        if not isinstance(item, dict):
            continue

        raw_date = str(item.get("date", "")).strip()
        if raw_date == "":
            continue

        parsed_date = pd.to_datetime(raw_date, errors="coerce", format="%Y-%m-%d")
        if pd.isna(parsed_date):
            parsed_date = pd.to_datetime(raw_date, errors="coerce", dayfirst=True)
        if pd.isna(parsed_date):
            continue

        match_date = parsed_date.date()
        if match_date > cutoff:
            continue

        score_obj = item.get("score")
        score = score_obj if isinstance(score_obj, dict) else {}

        home_goals, away_goals = _parse_score_pair(score.get("ft"))
        if home_goals is None or away_goals is None:
            continue

        ht_home_goals, ht_away_goals = _parse_score_pair(score.get("ht"))
        ht_result = (
            _result_from_score(ht_home_goals, ht_away_goals)
            if ht_home_goals is not None and ht_away_goals is not None
            else np.nan
        )

        home_raw = item.get("team1") or item.get("home_team") or item.get("home")
        away_raw = item.get("team2") or item.get("away_team") or item.get("away")
        home_team = _canonical_team_name(home_raw, team_map)
        away_team = _canonical_team_name(away_raw, team_map)
        if home_team == "" or away_team == "":
            continue

        rows.append(
            {
                "date": match_date.isoformat(),
                "season": _season_start_from_match_date(match_date),
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": _result_from_score(home_goals, away_goals),
                "ht_home_goals": ht_home_goals,
                "ht_away_goals": ht_away_goals,
                "ht_result": ht_result,
            }
        )

    if not rows:
        return _empty_manual_results_frame()

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    out = out.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return out


def _augment_historical_with_manual_results(
    historical: pd.DataFrame,
    team_map: dict[str, str],
    manual_results_path: Path | None = None,
    cutoff_date: date | None = None,
) -> pd.DataFrame:
    manual_path = manual_results_path or (settings.data_dir / "fixtures" / "proximosPartidos.json")
    manual = _extract_completed_manual_results(manual_path, team_map, cutoff_date=cutoff_date)
    if manual.empty:
        return historical

    out = historical.copy()
    for column in [
        "date",
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "ht_home_goals",
        "ht_away_goals",
        "ht_result",
    ]:
        if column not in out.columns:
            out[column] = np.nan

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["home_team"] = out["home_team"].map(lambda value: _canonical_team_name(value, team_map))
    out["away_team"] = out["away_team"].map(lambda value: _canonical_team_name(value, team_map))

    out["__key"] = out["date"].astype(str) + "|" + out["home_team"].astype(str) + "|" + out["away_team"].astype(str)
    manual_with_key = manual.copy()
    manual_with_key["__key"] = (
        manual_with_key["date"].astype(str)
        + "|"
        + manual_with_key["home_team"].astype(str)
        + "|"
        + manual_with_key["away_team"].astype(str)
    )
    manual_by_key = manual_with_key.set_index("__key")

    numeric_fill_columns = ["season", "home_goals", "away_goals", "ht_home_goals", "ht_away_goals"]
    for column in numeric_fill_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        mapped = out["__key"].map(manual_by_key[column])
        out[column] = out[column].where(out[column].notna(), mapped)

    for column in ["result", "ht_result"]:
        current = out[column].astype(str).str.strip()
        missing = current.isin(["", "nan", "none", "null", "NaN", "None", "NULL"])
        mapped = out["__key"].map(manual_by_key[column])
        out.loc[missing, column] = mapped[missing]

    existing_keys = set(out["__key"])
    new_rows = manual_with_key[~manual_with_key["__key"].isin(existing_keys)].copy()

    base_columns = [column for column in out.columns if column != "__key"]
    out = out.drop(columns=["__key"])

    if not new_rows.empty:
        new_rows = new_rows.drop(columns=["__key"])
        for column in base_columns:
            if column not in new_rows.columns:
                new_rows[column] = np.nan
        new_rows = new_rows[base_columns]
        out = pd.concat([out, new_rows], ignore_index=True, sort=False)

    out = out.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return out


def _to_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _normalize_team_level_historical(raw: pd.DataFrame, team_map: dict[str, str]) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(column).strip().lower() for column in df.columns]

    required = {"date", "team", "opponent", "venue", "gf", "ga"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d")
    df["team"] = df["team"].map(lambda value: _canonical_team_name(value, team_map))
    df["opponent"] = df["opponent"].map(lambda value: _canonical_team_name(value, team_map))
    df["venue"] = df["venue"].astype(str).str.strip().str.lower()

    df["gf"] = _to_numeric_column(df, "gf")
    df["ga"] = _to_numeric_column(df, "ga")
    df["xg"] = _to_numeric_column(df, "xg")
    df["xga"] = _to_numeric_column(df, "xga")
    df["poss"] = _to_numeric_column(df, "poss")
    df["sh"] = _to_numeric_column(df, "sh")
    df["sot"] = _to_numeric_column(df, "sot")
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
    else:
        df["season"] = np.nan

    home_mask = df["venue"].isin(["home", "casa", "local"])
    away_mask = df["venue"].isin(["away", "fuera", "visitante"])

    df["home_team"] = np.where(home_mask, df["team"], np.where(away_mask, df["opponent"], np.nan))
    df["away_team"] = np.where(home_mask, df["opponent"], np.where(away_mask, df["team"], np.nan))
    df["home_goals"] = np.where(home_mask, df["gf"], np.where(away_mask, df["ga"], np.nan))
    df["away_goals"] = np.where(home_mask, df["ga"], np.where(away_mask, df["gf"], np.nan))

    df = df.dropna(subset=["date", "home_team", "away_team"])

    rows: list[dict[str, Any]] = []
    grouped = df.groupby(["date", "home_team", "away_team"], dropna=False)
    for (match_date, home_team, away_team), group in grouped:
        home_row = group[group["team"] == home_team].head(1)
        away_row = group[group["team"] == away_team].head(1)

        home_row_series = home_row.iloc[0] if len(home_row) > 0 else None
        away_row_series = away_row.iloc[0] if len(away_row) > 0 else None

        home_goals = (
            float(home_row_series["gf"]) if home_row_series is not None and pd.notna(home_row_series["gf"]) else np.nan
        )
        away_goals = (
            float(away_row_series["gf"]) if away_row_series is not None and pd.notna(away_row_series["gf"]) else np.nan
        )
        if pd.isna(home_goals) or pd.isna(away_goals):
            fallback = group.head(1).iloc[0]
            home_goals = fallback["home_goals"]
            away_goals = fallback["away_goals"]

        xg_home = home_row_series["xg"] if home_row_series is not None else np.nan
        xg_away = away_row_series["xg"] if away_row_series is not None else np.nan
        xga_home = home_row_series["xga"] if home_row_series is not None else np.nan
        xga_away = away_row_series["xga"] if away_row_series is not None else np.nan

        if pd.isna(xg_home) and pd.notna(xga_away):
            xg_home = xga_away
        if pd.isna(xg_away) and pd.notna(xga_home):
            xg_away = xga_home
        if pd.isna(xga_home) and pd.notna(xg_away):
            xga_home = xg_away
        if pd.isna(xga_away) and pd.notna(xg_home):
            xga_away = xg_home

        poss_home = home_row_series["poss"] if home_row_series is not None else np.nan
        poss_away = away_row_series["poss"] if away_row_series is not None else np.nan
        if pd.isna(poss_home) and pd.notna(poss_away):
            poss_home = 100.0 - float(poss_away)
        if pd.isna(poss_away) and pd.notna(poss_home):
            poss_away = 100.0 - float(poss_home)

        sh_home = home_row_series["sh"] if home_row_series is not None else np.nan
        sh_away = away_row_series["sh"] if away_row_series is not None else np.nan
        sot_home = home_row_series["sot"] if home_row_series is not None else np.nan
        sot_away = away_row_series["sot"] if away_row_series is not None else np.nan

        season = np.nan
        if home_row_series is not None and pd.notna(home_row_series.get("season")):
            season = float(home_row_series["season"])
        elif away_row_series is not None and pd.notna(away_row_series.get("season")):
            season = float(away_row_series["season"])
        else:
            parsed = pd.to_datetime(match_date, errors="coerce")
            if pd.notna(parsed):
                season = float(parsed.year if parsed.month >= 7 else parsed.year - 1)

        result = _result_from_score(float(home_goals), float(away_goals)) if pd.notna(home_goals) and pd.notna(away_goals) else np.nan

        rows.append(
            {
                "date": match_date,
                "season": season,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result,
                "xg_home": xg_home,
                "xg_away": xg_away,
                "xga_home": xga_home,
                "xga_away": xga_away,
                "poss_home": poss_home,
                "poss_away": poss_away,
                "sh_home": sh_home,
                "sh_away": sh_away,
                "sot_home": sot_home,
                "sot_away": sot_away,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    out = out.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return out


def _normalize_historical_columns(raw: pd.DataFrame, team_map: dict[str, str]) -> pd.DataFrame:
    normalized_team_level = _normalize_team_level_historical(raw, team_map)
    if not normalized_team_level.empty:
        return normalized_team_level

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
    df["home_team"] = df["home_team"].map(lambda value: _canonical_team_name(value, team_map))
    df["away_team"] = df["away_team"].map(lambda value: _canonical_team_name(value, team_map))

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


def _summary(
    df: pd.DataFrame,
    output_all: Path,
    output_model: Path,
    historical_augmented_output: Path | None = None,
) -> dict[str, Any]:
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
        "historical_augmented_output": str(historical_augmented_output) if historical_augmented_output else None,
    }


def ingest_datasets(request: DatasetIngestRequest) -> dict[str, Any]:
    historical_path = _resolve_path(request.historical)
    football_data_dir = _resolve_path(request.football_data_dir)
    elo_path = _resolve_path(request.elo_csv)
    manual_results_path = _resolve_path(request.manual_results_json)

    if historical_path is None or not historical_path.exists():
        raise FileNotFoundError("No existe historical CSV")
    if football_data_dir is None or not football_data_dir.exists():
        raise FileNotFoundError("No existe football_data_dir")

    team_map = _load_team_map(request.team_map)
    historical_raw = pd.read_csv(historical_path)
    historical = _normalize_historical_columns(historical_raw, team_map)
    if request.include_manual_results:
        historical = _augment_historical_with_manual_results(
            historical,
            team_map,
            manual_results_path=manual_results_path,
        )

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    historical_augmented_output = settings.output_dir / "laliga_historical_augmented.csv"
    historical.to_csv(historical_augmented_output, index=False)

    fdata = load_football_data(football_data_dir, team_map)
    merged = historical.merge(fdata, on=["date", "home_team", "away_team"], how="left")
    merged = add_basic_differentials(merged)
    merged = add_pre_match_rolling_features(merged, tuple(request.windows))
    merged = add_internal_elo_features(merged)
    merged = enrich_with_elo(merged, elo_path, team_map)
    merged = add_target_label(merged)

    output_all = settings.output_dir / "laliga_enriched_all.csv"
    output_model = settings.output_dir / "laliga_enriched_model.csv"

    merged.to_csv(output_all, index=False)
    model_df = _build_model_dataset(merged)
    model_df.to_csv(output_model, index=False)

    return _summary(
        merged,
        output_all,
        output_model,
        historical_augmented_output=historical_augmented_output,
    )


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
