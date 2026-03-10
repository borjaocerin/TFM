from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from app.core.config import settings
from app.models.model_store import ModelStore
from app.schemas.predict import OddsCompareRequest, PredictRequest, PredictUpcomingRequest
from app.services.elo import enrich_with_elo
from app.services.evaluation import compare_market_vs_model
from app.services.features import enrich_fixtures

DEFAULT_WINDOWS: tuple[int, ...] = (5, 10)


def _resolve_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (settings.data_dir.parent / path).resolve()


def _default_source_paths() -> dict[str, Path]:
    return {
        "historical_csv": settings.data_dir / "historical" / "laliga_merged_matches.csv",
        "football_data_dir": settings.data_dir / "football-data",
        "elo_csv": settings.data_dir / "elo" / "ELO_RATINGS.csv",
        "team_map": settings.data_dir.parent / "etl" / "team_name_map_es.json",
        "fixtures_csv": settings.data_dir / "fixtures" / "fixtures.csv",
        "fallback_fixtures": settings.data_dir.parent / "matches_laliga.csv",
        "fallback_fixtures_alt": settings.data_dir.parent / "LaLiga_Matches.csv",
        "historical_fallback": settings.data_dir / "historical" / "laliga_merged_matches.csv",
    }


def _current_season_start_year(reference: date | None = None) -> int:
    today = reference or datetime.now().date()
    return today.year if today.month >= 7 else today.year - 1


def _current_season_label(reference: date | None = None) -> str:
    start_year = _current_season_start_year(reference)
    return f"{start_year}/{start_year + 1}"


def _load_team_map() -> dict[str, str]:
    paths = _default_source_paths()
    team_map_path = paths["team_map"]
    if team_map_path.exists():
        return json.loads(team_map_path.read_text(encoding="utf-8"))
    return {}


def _parse_dates_best_effort(series: pd.Series) -> pd.Series:
    as_dayfirst = pd.to_datetime(series, errors="coerce", dayfirst=True)
    as_standard = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return as_dayfirst if int(as_dayfirst.notna().sum()) > int(as_standard.notna().sum()) else as_standard


def _season_start_from_value(value: Any) -> int | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if text == "":
        return None

    try:
        as_float = float(text)
        as_int = int(as_float)
        if 1900 <= as_int <= 2100:
            return as_int
    except Exception:
        pass

    for separator in ["/", "-"]:
        if separator in text:
            first = text.split(separator)[0].strip()
            try:
                as_int = int(float(first))
                if 1900 <= as_int <= 2100:
                    return as_int
            except Exception:
                continue
    return None


def _canonicalize_fixture_source(raw: pd.DataFrame, team_map: dict[str, str]) -> pd.DataFrame:
    columns_lookup = {column.strip().lower(): column for column in raw.columns}

    def pick(candidates: list[str]) -> str | None:
        for candidate in candidates:
            if candidate.lower() in columns_lookup:
                return columns_lookup[candidate.lower()]
        return None

    date_col = pick(["date"])
    season_col = pick(["season"])

    home_col = pick(["hometeam", "home_team"])
    away_col = pick(["awayteam", "away_team"])

    out = pd.DataFrame()

    if date_col and home_col and away_col:
        out["date"] = _parse_dates_best_effort(raw[date_col]).dt.strftime("%Y-%m-%d")
        out["home_team"] = raw[home_col].astype(str)
        out["away_team"] = raw[away_col].astype(str)
        if season_col:
            out["season_raw"] = raw[season_col]

        result_col = pick(["result", "ftr"])
        home_goals_col = pick(["home_goals", "fthg"])
        away_goals_col = pick(["away_goals", "ftag"])

        played_result = (
            raw[result_col].astype(str).str.strip().ne("")
            if result_col
            else pd.Series(False, index=raw.index)
        )
        played_goals = (
            raw[home_goals_col].notna() & raw[away_goals_col].notna()
            if home_goals_col and away_goals_col
            else pd.Series(False, index=raw.index)
        )
        out["played"] = played_result | played_goals

    else:
        team_col = pick(["team"])
        opponent_col = pick(["opponent"])
        venue_col = pick(["venue"])
        result_col = pick(["result"])
        gf_col = pick(["gf"])
        ga_col = pick(["ga"])

        if not (date_col and team_col and opponent_col and venue_col):
            raise ValueError("No se pudo identificar estructura de fixtures en el CSV")

        home_rows = raw[raw[venue_col].astype(str).str.lower() == "home"].copy()
        out["date"] = _parse_dates_best_effort(home_rows[date_col]).dt.strftime("%Y-%m-%d")
        out["home_team"] = home_rows[team_col].astype(str)
        out["away_team"] = home_rows[opponent_col].astype(str)
        if season_col:
            out["season_raw"] = home_rows[season_col]

        played_result = (
            home_rows[result_col].astype(str).str.strip().ne("")
            if result_col
            else pd.Series(False, index=home_rows.index)
        )
        played_goals = (
            home_rows[gf_col].notna() & home_rows[ga_col].notna()
            if gf_col and ga_col
            else pd.Series(False, index=home_rows.index)
        )
        out["played"] = played_result | played_goals

    out["home_team"] = out["home_team"].map(lambda value: team_map.get(str(value), str(value)))
    out["away_team"] = out["away_team"].map(lambda value: team_map.get(str(value), str(value)))
    out["date_dt"] = pd.to_datetime(out["date"], errors="coerce")

    out = out.dropna(subset=["date_dt", "home_team", "away_team"])
    out = out.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    return out


def _filter_current_season_upcoming(fixtures: pd.DataFrame) -> pd.DataFrame:
    start_year = _current_season_start_year()
    today = datetime.now().date()
    season_start = date(start_year, 7, 1)
    season_end = date(start_year + 1, 6, 30)

    out = fixtures.copy()
    if "season_raw" in out.columns:
        out["season_start_year"] = out["season_raw"].map(_season_start_from_value)
    else:
        out["season_start_year"] = out["date_dt"].apply(
            lambda value: value.year if value.month >= 7 else value.year - 1
        )

    out = out[out["season_start_year"] == start_year]
    out = out[
        (out["date_dt"].dt.date >= season_start)
        & (out["date_dt"].dt.date <= season_end)
    ]

    if "played" in out.columns:
        out = out[(~out["played"]) | (out["date_dt"].dt.date >= today)]
    else:
        out = out[out["date_dt"].dt.date >= today]

    out = out.sort_values(["date_dt", "home_team", "away_team"]).reset_index(drop=True)
    return out


def _candidate_fixtures_sources() -> list[Path]:
    paths = _default_source_paths()
    return [
        paths["fixtures_csv"],
        paths["fallback_fixtures"],
        paths["historical_fallback"],
        paths["fallback_fixtures_alt"],
    ]


def _fixture_id(date_value: str, home_team: str, away_team: str) -> str:
    return f"{date_value}|{home_team}|{away_team}"


def _build_fixtures_response(
    upcoming: pd.DataFrame,
    season_label: str,
    source_path: str,
) -> dict[str, Any]:
    fixtures: list[dict[str, str]] = []
    for row in upcoming.itertuples(index=False):
        date_iso = str(row.date)
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        fixtures.append(
            {
                "fixture_id": _fixture_id(date_iso, home_team, away_team),
                "date": date_iso,
                "home_team": home_team,
                "away_team": away_team,
                "label": f"{date_iso} | {home_team} vs {away_team}",
            }
        )

    return {
        "season_label": season_label,
        "source_path": source_path,
        "rows": int(len(fixtures)),
        "fixtures": fixtures,
    }


def _effective_fixtures_api_url() -> str | None:
    configured = (settings.fixtures_api_url or "").strip()
    if configured == "":
        return None

    try:
        return configured.format(
            season_start_year=_current_season_start_year(),
            season_label=_current_season_label(),
        )
    except Exception:
        return configured


def _local_api_key_fallback() -> str | None:
    # Keep demo setup simple: if no env key is configured, reuse oddapikey.txt when present.
    fallback_path = settings.data_dir.parent / "oddapikey.txt"
    if not fallback_path.exists():
        return None

    try:
        key = fallback_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return key or None


def _effective_fixtures_api_key() -> str | None:
    configured = (settings.fixtures_api_key or "").strip()
    if configured:
        return configured
    return _local_api_key_fallback()


def _extract_rows_from_api_payload(payload: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add_row(date_value: Any, home_team: Any, away_team: Any) -> None:
        if date_value is None or home_team is None or away_team is None:
            return
        date_text = str(date_value).strip()
        home_text = str(home_team).strip()
        away_text = str(away_team).strip()
        if date_text == "" or home_text == "" or away_text == "":
            return
        rows.append(
            {
                "date": date_text,
                "home_team": home_text,
                "away_team": away_text,
            }
        )

    if isinstance(payload, dict):
        matches = payload.get("matches")
        if isinstance(matches, list):
            blocked_status = {
                "FINISHED",
                "IN_PLAY",
                "PAUSED",
                "AFTER_EXTRA_TIME",
                "PENALTY_SHOOTOUT",
                "AWARDED",
                "CANCELLED",
                "ABANDONED",
                "SUSPENDED",
            }
            for match in matches:
                if not isinstance(match, dict):
                    continue
                status = str(match.get("status", "")).upper().strip()
                if status in blocked_status:
                    continue
                home_team = match.get("home_team")
                if home_team is None and isinstance(match.get("homeTeam"), dict):
                    home_team = match["homeTeam"].get("name")
                away_team = match.get("away_team")
                if away_team is None and isinstance(match.get("awayTeam"), dict):
                    away_team = match["awayTeam"].get("name")
                date_value = match.get("date") or match.get("utcDate")
                add_row(date_value, home_team, away_team)

        response_rows = payload.get("response")
        if isinstance(response_rows, list):
            allowed_short_status = {"NS", "TBD", "PST"}
            for item in response_rows:
                if not isinstance(item, dict):
                    continue
                fixture = item.get("fixture") if isinstance(item.get("fixture"), dict) else {}
                status_obj = fixture.get("status") if isinstance(fixture.get("status"), dict) else {}
                status_short = str(status_obj.get("short", "")).upper().strip()
                if status_short and status_short not in allowed_short_status:
                    continue
                teams = item.get("teams") if isinstance(item.get("teams"), dict) else {}
                home_team = (
                    teams.get("home", {}).get("name")
                    if isinstance(teams.get("home"), dict)
                    else item.get("home_team")
                )
                away_team = (
                    teams.get("away", {}).get("name")
                    if isinstance(teams.get("away"), dict)
                    else item.get("away_team")
                )
                date_value = fixture.get("date") or item.get("date")
                add_row(date_value, home_team, away_team)

        events = payload.get("events")
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                status_value = str(event.get("strStatus", "")).strip().lower()
                if status_value not in {"", "not started", "ns", "tbd"}:
                    continue
                date_value = event.get("strTimestamp") or event.get("dateEvent") or event.get("date")
                home_team = event.get("strHomeTeam") or event.get("home_team")
                away_team = event.get("strAwayTeam") or event.get("away_team")
                add_row(date_value, home_team, away_team)

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            add_row(item.get("date"), item.get("home_team"), item.get("away_team"))

    return rows


def _canonicalize_api_rows(rows: list[dict[str, str]], team_map: dict[str, str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "played", "date_dt"])

    out = pd.DataFrame(rows)
    out["date_dt"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_localize(None)
    out = out.dropna(subset=["date_dt"])
    out["date"] = out["date_dt"].dt.strftime("%Y-%m-%d")
    out["home_team"] = out["home_team"].map(lambda value: team_map.get(str(value), str(value)))
    out["away_team"] = out["away_team"].map(lambda value: team_map.get(str(value), str(value)))
    out["played"] = False
    out = out.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    return out


def _load_upcoming_from_api(team_map: dict[str, str], season_label: str) -> dict[str, Any] | None:
    api_url = _effective_fixtures_api_url()
    if not api_url:
        return None

    headers = {"Accept": "application/json"}
    api_key = _effective_fixtures_api_key()
    if api_key:
        headers["X-Auth-Token"] = api_key
        headers["x-apisports-key"] = api_key
        headers["x-rapidapi-key"] = api_key
        headers["apikey"] = api_key

    api_host = (settings.fixtures_api_host or "").strip()
    if api_host:
        headers["x-rapidapi-host"] = api_host

    try:
        request = Request(api_url, headers=headers, method="GET")
        with urlopen(request, timeout=settings.fixtures_api_timeout_sec) as response:
            payload_raw = response.read().decode("utf-8")
        payload = json.loads(payload_raw)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None

    api_rows = _extract_rows_from_api_payload(payload)
    canonical = _canonicalize_api_rows(api_rows, team_map)
    if canonical.empty:
        return None

    upcoming = _filter_current_season_upcoming(canonical)
    if upcoming.empty:
        return None

    return _build_fixtures_response(upcoming, season_label, f"api:{api_url}")


def list_upcoming_fixture_options() -> dict[str, Any]:
    team_map = _load_team_map()
    season_label = _current_season_label()

    source_seen: Path | None = None
    for candidate_path in _candidate_fixtures_sources():
        if not candidate_path.exists():
            continue

        source_seen = candidate_path
        try:
            raw = pd.read_csv(candidate_path)
            canonical = _canonicalize_fixture_source(raw, team_map)
            upcoming = _filter_current_season_upcoming(canonical)
        except Exception:
            continue

        if upcoming.empty:
            continue

        return _build_fixtures_response(upcoming, season_label, str(candidate_path))

    from_api = _load_upcoming_from_api(team_map, season_label)
    if from_api is not None:
        return from_api

    return {
        "season_label": season_label,
        "source_path": str(source_seen) if source_seen else "",
        "rows": 0,
        "fixtures": [],
    }


def _ensure_default_enriched_historical() -> Path:
    output_all = settings.output_dir / "laliga_enriched_all.csv"
    output_model = settings.output_dir / "laliga_enriched_model.csv"
    if output_all.exists() and output_model.exists():
        return output_all

    defaults = _default_source_paths()
    historical_csv = defaults["historical_csv"]
    football_data_dir = defaults["football_data_dir"]
    elo_csv = defaults["elo_csv"] if defaults["elo_csv"].exists() else None
    team_map = defaults["team_map"] if defaults["team_map"].exists() else None

    if not historical_csv.exists():
        raise FileNotFoundError(
            f"No existe historico base en {historical_csv}. Coloca el CSV y vuelve a intentar."
        )
    if not football_data_dir.exists():
        raise FileNotFoundError(
            f"No existe directorio football-data en {football_data_dir}."
        )

    from app.schemas.datasets import DatasetIngestRequest
    from app.services.datasets import ingest_datasets

    ingest_datasets(
        DatasetIngestRequest(
            historical=str(historical_csv),
            football_data_dir=str(football_data_dir),
            elo_csv=str(elo_csv) if elo_csv else None,
            team_map=str(team_map) if team_map else None,
            windows=list(DEFAULT_WINDOWS),
        )
    )

    if not output_all.exists():
        raise FileNotFoundError("No se pudo generar laliga_enriched_all.csv automaticamente")
    return output_all


def _prepare_prediction_input(request: PredictRequest) -> pd.DataFrame:
    if request.fixtures is not None:
        return pd.DataFrame(request.fixtures)

    path = _resolve_path(request.fixtures_enriched_path)
    if path is None or not path.exists():
        raise FileNotFoundError("No existe fixtures_enriched.csv")
    return pd.read_csv(path)


def predict_matches(request: PredictRequest) -> dict[str, Any]:
    features_df = _prepare_prediction_input(request)

    store = ModelStore()
    try:
        payload, _ = store.load()
    except FileNotFoundError:
        from app.schemas.model import TrainRequest
        from app.services.train import train_and_calibrate

        train_and_calibrate(TrainRequest())
        payload, _ = store.load()

    model = payload["model"]
    feature_columns: list[str] = payload["feature_columns"]

    for column in feature_columns:
        if column not in features_df.columns:
            features_df[column] = np.nan

    model_input = features_df[feature_columns].copy()
    model_input = model_input.apply(pd.to_numeric, errors="coerce").astype(float)

    proba = model.predict_proba(model_input)

    output = features_df.copy()
    output["p_H"] = proba[:, 0]
    output["p_D"] = proba[:, 1]
    output["p_A"] = proba[:, 2]

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = settings.output_dir / "predictions.csv"
    output.to_csv(output_csv, index=False)

    keep = [
        column
        for column in [
            "date",
            "home_team",
            "away_team",
            "elo_diff",
            "xg_last5_diff",
            "xg_last10_diff",
            "odds_avg_h",
            "odds_avg_d",
            "odds_avg_a",
            "odds_close_h",
            "odds_close_d",
            "odds_close_a",
            "p_H",
            "p_D",
            "p_A",
            "target",
            "result",
        ]
        if column in output.columns
    ]
    preview = output[keep] if keep else output

    return {
        "rows": int(len(output)),
        "output_csv": str(output_csv),
        "predictions": preview.to_dict(orient="records"),
    }


def predict_selected_upcoming_match(request: PredictUpcomingRequest) -> dict[str, Any]:
    historical_path = _ensure_default_enriched_historical()
    defaults = _default_source_paths()
    team_map = _load_team_map()

    parsed_date = pd.to_datetime(request.date, errors="coerce", format="%Y-%m-%d")
    if pd.isna(parsed_date):
        parsed_date = pd.to_datetime(request.date, errors="coerce", dayfirst=True)
    if pd.isna(parsed_date):
        raise ValueError("La fecha enviada no es valida")

    fixture_df = pd.DataFrame(
        [
            {
                "date": parsed_date.strftime("%Y-%m-%d"),
                "home_team": team_map.get(request.home_team, request.home_team),
                "away_team": team_map.get(request.away_team, request.away_team),
            }
        ]
    )

    historical_df = pd.read_csv(historical_path)
    enriched = enrich_fixtures(fixture_df, historical_df, DEFAULT_WINDOWS, team_map)

    elo_path = defaults["elo_csv"]
    if elo_path.exists():
        enriched = enrich_with_elo(enriched, elo_path, team_map)

    prediction = predict_matches(PredictRequest(fixtures=enriched.to_dict(orient="records")))
    prediction_row = prediction["predictions"][0] if prediction["predictions"] else {}

    return {
        "season_label": _current_season_label(),
        "selected_fixture": {
            "date": str(fixture_df.iloc[0]["date"]),
            "home_team": str(fixture_df.iloc[0]["home_team"]),
            "away_team": str(fixture_df.iloc[0]["away_team"]),
        },
        "prediction": prediction_row,
        "output_csv": str(prediction["output_csv"]),
    }


def _prepare_compare_input(request: OddsCompareRequest) -> pd.DataFrame:
    if request.predictions is not None:
        return pd.DataFrame(request.predictions)

    path = _resolve_path(request.predictions_csv)
    if path is None or not path.exists():
        raise FileNotFoundError("No existe CSV para comparar cuotas")
    return pd.read_csv(path)


def compare_odds(request: OddsCompareRequest) -> dict[str, Any]:
    df = _prepare_compare_input(request)
    required = [
        f"{request.odds_kind}_h",
        f"{request.odds_kind}_d",
        f"{request.odds_kind}_a",
        "p_H",
        "p_D",
        "p_A",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para comparar cuotas: {missing}")

    metrics, compared = compare_market_vs_model(df, request.odds_kind, request.value_threshold)
    output_csv = settings.output_dir / "predictions_with_odds.csv"
    compared.to_csv(output_csv, index=False)

    value_bets = compared[compared["value_bet"]].copy()
    keep = [
        column
        for column in [
            "date",
            "home_team",
            "away_team",
            "best_ev_pick",
            "best_ev",
            "ev_H",
            "ev_D",
            "ev_A",
        ]
        if column in value_bets.columns
    ]
    return {
        "rows": int(len(compared)),
        "metrics": metrics,
        "value_bets": value_bets[keep].to_dict(orient="records") if keep else value_bets.to_dict(orient="records"),
        "output_csv": str(output_csv),
    }
