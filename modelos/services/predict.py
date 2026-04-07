from __future__ import annotations

import json
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.core.config import settings
from modelos.models.model_store import ModelStore
from app.schemas.predict import OddsCompareRequest, PredictRequest, PredictUpcomingRequest
from modelos.services.odds_api import fetch_upcoming_laliga_odds, find_fixture_odds
from modelos.services.elo import enrich_with_elo
from modelos.services.evaluation import compare_market_vs_model
from modelos.services.features import enrich_fixtures

DEFAULT_WINDOWS: tuple[int, ...] = (5, 10)

ROUND_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "real betis": ("betis",),
    "betis": ("real betis",),
    "celta vigo": ("celta",),
    "celta": ("celta vigo",),
    "real oviedo": ("oviedo",),
    "oviedo": ("real oviedo",),
    "ca osasuna": ("osasuna",),
    "osasuna": ("ca osasuna",),
}


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
        "manual_fixtures_json": settings.data_dir / "fixtures" / "proximosPartidos.json",
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

    out["home_team"] = out["home_team"].map(lambda value: _canonical_team_name(value, team_map))
    out["away_team"] = out["away_team"].map(lambda value: _canonical_team_name(value, team_map))
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
        fixture = {
            "fixture_id": _fixture_id(date_iso, home_team, away_team),
            "date": date_iso,
            "home_team": home_team,
            "away_team": away_team,
            "label": f"{date_iso} | {home_team} vs {away_team}",
            "round": ""  # valor por defecto
        }
        # Si el DataFrame tiene columna 'round', usar su valor aunque sea vacía
        if hasattr(row, "round"):
            round_val = getattr(row, "round")
            if round_val is not None and str(round_val).strip().lower() not in ("nan", "none"):
                fixture["round"] = str(round_val).strip()
        fixtures.append(fixture)

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

    manual_aliases = {
        "deportivo alaves": "Alaves",
        "girona fc": "Girona",
        "villarreal cf": "Villarreal",
        "rcd mallorca": "Mallorca",
        "fc barcelona": "Barcelona",
        "valencia cf": "Valencia",
        "rc celta de vigo": "Celta",
        "club atletico de madrid": "Atletico Madrid",
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
    for alias, canonical in manual_aliases.items():
        normalized_map[_normalize_text_basic(alias)] = canonical

    mapped = normalized_map.get(normalized_input)
    return mapped if mapped else raw


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "si", "s"}


def _normalize_competition_text(value: Any) -> str:
    return _normalize_text_basic(value)


def _is_laliga_competition(name: Any, country: Any, code: Any = None) -> bool:
    name_text = _normalize_competition_text(name)
    country_text = _normalize_competition_text(country)
    code_text = _normalize_competition_text(code)

    if code_text == "pd":
        return True

    if name_text and not any(
        token in name_text
        for token in ["laliga", "la liga", "primera division", "spanish la liga"]
    ):
        return False

    if country_text and country_text not in {"spain", "espana", "españa"}:
        return False

    if not name_text and not country_text and code_text == "":
        return True

    return True


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
            payload_competition = (
                payload.get("competition") if isinstance(payload.get("competition"), dict) else {}
            )
            for match in matches:
                if not isinstance(match, dict):
                    continue

                match_competition = (
                    match.get("competition") if isinstance(match.get("competition"), dict) else {}
                )
                competition_info = match_competition or payload_competition
                if competition_info:
                    if not _is_laliga_competition(
                        competition_info.get("name"),
                        competition_info.get("area", {}).get("name")
                        if isinstance(competition_info.get("area"), dict)
                        else competition_info.get("country"),
                        competition_info.get("code"),
                    ):
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

                league = item.get("league") if isinstance(item.get("league"), dict) else {}
                if league:
                    if not _is_laliga_competition(
                        league.get("name"),
                        league.get("country"),
                        league.get("code") or league.get("id"),
                    ):
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

                if not _is_laliga_competition(
                    event.get("strLeague") or event.get("league"),
                    event.get("strCountry") or event.get("country"),
                    event.get("idLeague"),
                ):
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
            date_value = item.get("date") or item.get("utcDate") or item.get("commence_time")
            home_team = item.get("home_team") or item.get("strHomeTeam")
            away_team = item.get("away_team") or item.get("strAwayTeam")
            if home_team is None and isinstance(item.get("homeTeam"), dict):
                home_team = item["homeTeam"].get("name")
            if away_team is None and isinstance(item.get("awayTeam"), dict):
                away_team = item["awayTeam"].get("name")
            add_row(date_value, home_team, away_team)

    return rows


def _extract_rows_from_manual_json(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    matches: Any = None
    if isinstance(payload, dict):
        matches = payload.get("matches")
    elif isinstance(payload, list):
        matches = payload

    if not isinstance(matches, list):
        return rows

    for item in matches:
        if not isinstance(item, dict):
            continue

        date_value = item.get("date")
        home_team = item.get("team1") or item.get("home_team") or item.get("homeTeam")
        away_team = item.get("team2") or item.get("away_team") or item.get("awayTeam")

        if date_value is None or home_team is None or away_team is None:
            continue

        date_text = str(date_value).strip()
        home_text = str(home_team).strip()
        away_text = str(away_team).strip()
        if date_text == "" or home_text == "" or away_text == "":
            continue

        played = False
        score = item.get("score") if isinstance(item.get("score"), dict) else {}
        ft = score.get("ft") if isinstance(score, dict) else None
        if isinstance(ft, list) and len(ft) >= 2:
            home_ft = ft[0]
            away_ft = ft[1]
            if home_ft is not None and away_ft is not None:
                if str(home_ft).strip() != "" and str(away_ft).strip() != "":
                    played = True

        row = {
            "date": date_text,
            "home_team": home_text,
            "away_team": away_text,
            "played": played,
        }
        round_value = item.get("round") or item.get("jornada") or item.get("matchday")
        if round_value is not None:
            row["round"] = str(round_value).strip()
        rows.append(row)

    return rows


def _canonicalize_api_rows(rows: list[dict[str, str]], team_map: dict[str, str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "played", "date_dt", "round"])

    out = pd.DataFrame(rows)
    # Forzar columna 'round' aunque falte en algunos registros
    if "round" not in out.columns:
        out["round"] = None
    out["date_dt"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_localize(None)
    out = out.dropna(subset=["date_dt"])
    out["date"] = out["date_dt"].dt.strftime("%Y-%m-%d")
    out["home_team"] = out["home_team"].map(lambda value: _canonical_team_name(value, team_map))
    out["away_team"] = out["away_team"].map(lambda value: _canonical_team_name(value, team_map))
    if "played" in out.columns:
        out["played"] = out["played"].map(_as_bool)
    else:
        out["played"] = False
    out = out.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    # Mantener columna 'round' como string
    out["round"] = out["round"].astype(str)
    return out


def _load_laliga_team_pool(team_map: dict[str, str]) -> set[str]:
    paths = _default_source_paths()
    candidates = [
        paths["historical_csv"],
        paths["historical_fallback"],
        settings.data_dir / "laliga_merged_matches.csv",
        settings.data_dir.parent / "backend" / "laliga_merged_matches.csv",
    ]
    start_year = _current_season_start_year()

    for candidate_path in candidates:
        if not candidate_path.exists():
            continue

        try:
            raw = pd.read_csv(candidate_path)
            canonical = _canonicalize_fixture_source(raw, team_map)
        except Exception:
            continue

        if canonical.empty:
            continue

        season_filtered = canonical.copy()
        if "season_raw" in season_filtered.columns:
            season_filtered["season_start_year"] = season_filtered["season_raw"].map(_season_start_from_value)
        else:
            season_filtered["season_start_year"] = season_filtered["date_dt"].apply(
                lambda value: value.year if value.month >= 7 else value.year - 1
            )

        season_filtered = season_filtered[season_filtered["season_start_year"] == start_year]
        source = season_filtered if not season_filtered.empty else canonical

        teams = set(source["home_team"].astype(str)) | set(source["away_team"].astype(str))
        clean_teams = {team for team in teams if team and team.lower() != "nan"}
        if clean_teams:
            return clean_teams

    return set()


def _round_match_keys(team_name: Any) -> set[str]:
    normalized = _normalize_text_basic(team_name)
    if normalized == "":
        return set()

    keys = {normalized}
    for alias in ROUND_NAME_ALIASES.get(normalized, ()):  # pragma: no branch
        alias_normalized = _normalize_text_basic(alias)
        if alias_normalized:
            keys.add(alias_normalized)
    return keys


def _teams_match_for_round(left_team: Any, right_team: Any) -> bool:
    left_keys = _round_match_keys(left_team)
    right_keys = _round_match_keys(right_team)
    if not left_keys or not right_keys:
        return False

    if left_keys.intersection(right_keys):
        return True

    for left_key in left_keys:
        for right_key in right_keys:
            if min(len(left_key), len(right_key)) < 5:
                continue
            if left_key in right_key or right_key in left_key:
                return True

    return False


def _build_manual_round_index(
    team_map: dict[str, str],
) -> tuple[dict[str, str], dict[str, list[tuple[str, str, str]]]]:
    json_path = _default_source_paths()["manual_fixtures_json"]
    if not json_path.exists():
        return {}, {}

    payload: Any
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        try:
            payload = json.loads(json_path.read_text(encoding="latin-1"))
        except Exception:
            return {}, {}

    manual_rows = _extract_rows_from_manual_json(payload)
    canonical = _canonicalize_api_rows(manual_rows, team_map)
    if canonical.empty:
        return {}, {}

    exact_round_by_fixture: dict[str, str] = {}
    candidate_rounds_by_date: dict[str, list[tuple[str, str, str]]] = {}

    for row in canonical.itertuples(index=False):
        round_value = str(getattr(row, "round", "")).strip()
        if round_value == "" or round_value.lower() in {"none", "nan", "null", "undefined"}:
            continue

        date_value = str(row.date)
        home_team = str(row.home_team)
        away_team = str(row.away_team)

        fixture_key = _fixture_id(date_value, home_team, away_team)
        exact_round_by_fixture[fixture_key] = round_value
        candidate_rounds_by_date.setdefault(date_value, []).append((home_team, away_team, round_value))

    return exact_round_by_fixture, candidate_rounds_by_date


def _attach_rounds_from_manual_source(upcoming: pd.DataFrame, team_map: dict[str, str]) -> pd.DataFrame:
    if upcoming.empty:
        return upcoming

    exact_round_by_fixture, candidate_rounds_by_date = _build_manual_round_index(team_map)
    if not exact_round_by_fixture and not candidate_rounds_by_date:
        return upcoming

    enriched = upcoming.copy()
    rounds: list[str] = []

    for row in enriched.itertuples(index=False):
        date_value = str(row.date)
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        fixture_key = _fixture_id(date_value, home_team, away_team)

        round_value = exact_round_by_fixture.get(fixture_key, "")
        if round_value == "":
            for candidate_home, candidate_away, candidate_round in candidate_rounds_by_date.get(date_value, []):
                if _teams_match_for_round(home_team, candidate_home) and _teams_match_for_round(
                    away_team, candidate_away
                ):
                    round_value = candidate_round
                    break

        rounds.append(round_value)

    enriched["round"] = rounds
    return enriched


def _filter_to_laliga_teams(fixtures: pd.DataFrame, team_map: dict[str, str]) -> pd.DataFrame:
    team_pool = _load_laliga_team_pool(team_map)
    if not team_pool:
        return fixtures

    filtered = fixtures[
        fixtures["home_team"].astype(str).isin(team_pool)
        & fixtures["away_team"].astype(str).isin(team_pool)
    ].copy()
    return filtered.reset_index(drop=True)


def _load_upcoming_from_odds_api(team_map: dict[str, str], season_label: str) -> dict[str, Any] | None:
    try:
        odds_payload = fetch_upcoming_laliga_odds(team_map=team_map, limit=200)
    except Exception as exc:
        raise ValueError(f"The Odds API: {exc}") from exc

    odds_rows = odds_payload.get("odds")
    if not isinstance(odds_rows, list) or not odds_rows:
        return None

    raw_rows: list[dict[str, str]] = []
    for row in odds_rows:
        if not isinstance(row, dict):
            continue
        raw_rows.append(
            {
                "date": str(row.get("date", "")),
                "home_team": str(row.get("home_team", "")),
                "away_team": str(row.get("away_team", "")),
            }
        )

    canonical = _canonicalize_api_rows(raw_rows, team_map)
    if canonical.empty:
        return None

    # The Odds API sport_key is already scoped to soccer_spain_la_liga — skip
    # the team-pool filter to avoid false negatives from name-format differences.
    upcoming = _filter_current_season_upcoming(canonical)
    if upcoming.empty:
        return None

    upcoming = _attach_rounds_from_manual_source(upcoming, team_map)

    source_path = str(odds_payload.get("source_path", "api:the-odds-api"))
    return _build_fixtures_response(upcoming, season_label, source_path or "api:the-odds-api")


def _load_upcoming_from_manual_json(team_map: dict[str, str], season_label: str) -> dict[str, Any] | None:
    json_path = _default_source_paths()["manual_fixtures_json"]
    if not json_path.exists():
        return None

    payload: Any
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        try:
            payload = json.loads(json_path.read_text(encoding="latin-1"))
        except Exception:
            return None

    manual_rows = _extract_rows_from_manual_json(payload)
    canonical = _canonicalize_api_rows(manual_rows, team_map)
    if canonical.empty:
        return None

    # The manual JSON file is expected to be LaLiga-only, so we keep this path permissive.
    upcoming = _filter_current_season_upcoming(canonical)
    if upcoming.empty:
        return None

    return _build_fixtures_response(upcoming, season_label, str(json_path))


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

    canonical = _filter_to_laliga_teams(canonical, team_map)
    if canonical.empty:
        return None

    upcoming = _filter_current_season_upcoming(canonical)
    if upcoming.empty:
        return None

    upcoming = _attach_rounds_from_manual_source(upcoming, team_map)

    return _build_fixtures_response(upcoming, season_label, f"api:{api_url}")


def _load_upcoming_from_csv(team_map: dict[str, str], season_label: str) -> dict[str, Any] | None:
    for candidate_path in _candidate_fixtures_sources():
        if not candidate_path.exists():
            continue

        try:
            raw = pd.read_csv(candidate_path)
            canonical = _canonicalize_fixture_source(raw, team_map)
            upcoming = _filter_current_season_upcoming(canonical)
        except Exception:
            continue

        if upcoming.empty:
            continue

        upcoming = _attach_rounds_from_manual_source(upcoming, team_map)

        return _build_fixtures_response(upcoming, season_label, str(candidate_path))

    return None


def _load_demo_from_csv(team_map: dict[str, str], season_label: str) -> dict[str, Any] | None:
    """Last-resort: return the most recent historical LaLiga matches as demo fixtures
    so the UI is never completely empty when no live data source is available."""
    defaults = _default_source_paths()
    candidates = [
        defaults["historical_csv"],
        defaults["historical_fallback"],
        settings.data_dir / "laliga_merged_matches.csv",
    ]
    for candidate_path in candidates:
        if not candidate_path.exists():
            continue
        try:
            raw = pd.read_csv(candidate_path)
            canonical = _canonicalize_fixture_source(raw, team_map)
        except Exception:
            continue
        if canonical.empty:
            continue
        # Return the 10 most recent rows as demo options regardless of date
        recent = canonical.sort_values("date_dt").tail(10).copy()
        recent = recent.reset_index(drop=True)
        return _build_fixtures_response(recent, season_label, f"demo:{candidate_path}")
    return None


def list_upcoming_fixture_options() -> dict[str, Any]:
    team_map = _load_team_map()
    season_label = _current_season_label()
    api_url = _effective_fixtures_api_url()

    from_manual_json = _load_upcoming_from_manual_json(team_map, season_label)
    if from_manual_json is not None:
        return from_manual_json

    from_api = _load_upcoming_from_api(team_map, season_label)
    if from_api is not None:
        return from_api

    if settings.fixtures_allow_csv_fallback:
        from_csv = _load_upcoming_from_csv(team_map, season_label)
        if from_csv is not None:
            return from_csv

    odds_api_error: str | None = None
    try:
        from_odds_api = _load_upcoming_from_odds_api(team_map, season_label)
        if from_odds_api is not None:
            from_odds_api["error"] = (
                "No se encontraron jornadas completas en fixtures; mostrando solo partidos con cuota."
            )
            return from_odds_api
    except ValueError as exc:
        odds_api_error = str(exc)

    # Demo fallback: last historical matches so the UI is never completely blank
    from_demo = _load_demo_from_csv(team_map, season_label)
    if from_demo is not None:
        from_demo["error"] = (
            odds_api_error
            or "No hay partidos futuros en fuentes de fixtures. Mostrando datos historicos de demo."
        )
        return from_demo

    error_msg = (
        odds_api_error
        or "No se encontraron partidos proximos. Comprueba las fuentes de fixtures y ODDS_API_KEY."
    )
    return {
        "season_label": season_label,
        "source_path": f"api:{api_url}" if api_url else "",
        "rows": 0,
        "fixtures": [],
        "error": error_msg,
    }


def _find_fixture_odds_in_rows(
    date_iso: str,
    home_team: str,
    away_team: str,
    odds_rows: list[dict[str, Any]],
    team_map: dict[str, str],
) -> dict[str, Any] | None:
    if not odds_rows:
        return None

    normalized_home = _normalize_text_basic(_canonical_team_name(home_team, team_map))
    normalized_away = _normalize_text_basic(_canonical_team_name(away_team, team_map))
    date_key = str(date_iso).strip()

    best_row: dict[str, Any] | None = None
    best_score = -1

    for row in odds_rows:
        row_home_raw = str(row.get("home_team", "")).strip()
        row_away_raw = str(row.get("away_team", "")).strip()
        row_date = str(row.get("date", "")).strip()

        row_home = _normalize_text_basic(_canonical_team_name(row_home_raw, team_map))
        row_away = _normalize_text_basic(_canonical_team_name(row_away_raw, team_map))

        if row_home == "" or row_away == "":
            continue

        exact_match = row_home == normalized_home and row_away == normalized_away
        fuzzy_match = _teams_match_for_round(row_home_raw, home_team) and _teams_match_for_round(
            row_away_raw, away_team
        )

        if exact_match and row_date == date_key:
            return row

        if exact_match:
            score = 3
        elif fuzzy_match and row_date == date_key:
            score = 2
        elif fuzzy_match:
            score = 1
        else:
            continue

        if score > best_score:
            best_score = score
            best_row = row

    return best_row


def _to_float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def list_upcoming_fixture_options_with_value(value_threshold: float = 0.02) -> dict[str, Any]:
    response = list_upcoming_fixture_options()

    raw_fixtures = response.get("fixtures")
    if not isinstance(raw_fixtures, list) or not raw_fixtures:
        return response

    fixtures = [dict(item) for item in raw_fixtures if isinstance(item, dict)]
    if not fixtures:
        response["fixtures"] = []
        response["rows"] = 0
        return response

    team_map = _load_team_map()

    fixture_rows: list[dict[str, str]] = []
    for fixture in fixtures:
        date_value = str(fixture.get("date", "")).strip()
        home_team = _canonical_team_name(fixture.get("home_team", ""), team_map)
        away_team = _canonical_team_name(fixture.get("away_team", ""), team_map)
        round_value = str(fixture.get("round", "") or "").strip()

        if date_value == "" or home_team == "" or away_team == "":
            continue

        fixture["date"] = date_value
        fixture["home_team"] = home_team
        fixture["away_team"] = away_team
        fixture["fixture_id"] = str(
            fixture.get("fixture_id") or _fixture_id(date_value, home_team, away_team)
        )
        fixture["round"] = round_value

        fixture_rows.append(
            {
                "fixture_id": fixture["fixture_id"],
                "date": date_value,
                "home_team": home_team,
                "away_team": away_team,
                "round": round_value,
            }
        )

    if not fixture_rows:
        for fixture in fixtures:
            fixture["p_H"] = None
            fixture["p_D"] = None
            fixture["p_A"] = None
            fixture["odds_avg_h"] = None
            fixture["odds_avg_d"] = None
            fixture["odds_avg_a"] = None
            fixture["best_ev"] = None
            fixture["best_ev_pick"] = None
            fixture["value_bet"] = None
        response["fixtures"] = fixtures
        response["rows"] = int(len(fixtures))
        return response

    try:
        historical_path = _ensure_default_enriched_historical()
        historical_df = pd.read_csv(historical_path)
        fixture_df = pd.DataFrame(fixture_rows)

        enriched = enrich_fixtures(fixture_df, historical_df, DEFAULT_WINDOWS, team_map)
        elo_path = _default_source_paths()["elo_csv"]
        if elo_path.exists():
            enriched = enrich_with_elo(enriched, elo_path, team_map)

        prediction_payload = predict_matches(PredictRequest(fixtures=enriched.to_dict(orient="records")))
        predictions_df = pd.DataFrame(prediction_payload.get("predictions", []))

        odds_rows: list[dict[str, Any]] = []
        try:
            odds_payload = fetch_upcoming_laliga_odds(team_map=team_map, limit=400)
            payload_rows = odds_payload.get("odds")
            if isinstance(payload_rows, list):
                odds_rows = [row for row in payload_rows if isinstance(row, dict)]
        except Exception:
            odds_rows = []

        prediction_by_key: dict[str, dict[str, Any]] = {}
        if not predictions_df.empty:
            for row in predictions_df.to_dict(orient="records"):
                key = _fixture_id(
                    str(row.get("date", "")),
                    str(row.get("home_team", "")),
                    str(row.get("away_team", "")),
                )
                prediction_by_key[key] = row

        compare_rows: list[dict[str, Any]] = []
        for fixture in fixture_rows:
            key = _fixture_id(fixture["date"], fixture["home_team"], fixture["away_team"])
            prediction_row = prediction_by_key.get(key)
            if not prediction_row:
                continue

            odds_row = _find_fixture_odds_in_rows(
                date_iso=fixture["date"],
                home_team=fixture["home_team"],
                away_team=fixture["away_team"],
                odds_rows=odds_rows,
                team_map=team_map,
            )

            compare_rows.append(
                {
                    "fixture_id": fixture["fixture_id"],
                    "p_H": _to_float_or_none(prediction_row.get("p_H")),
                    "p_D": _to_float_or_none(prediction_row.get("p_D")),
                    "p_A": _to_float_or_none(prediction_row.get("p_A")),
                    "odds_avg_h": _to_float_or_none((odds_row or {}).get("odds_avg_h")),
                    "odds_avg_d": _to_float_or_none((odds_row or {}).get("odds_avg_d")),
                    "odds_avg_a": _to_float_or_none((odds_row or {}).get("odds_avg_a")),
                }
            )

        compared_by_fixture_id: dict[str, dict[str, Any]] = {}
        if compare_rows:
            _, compared = compare_market_vs_model(pd.DataFrame(compare_rows), "odds_avg", value_threshold)
            for row in compared.to_dict(orient="records"):
                fixture_id = str(row.get("fixture_id", "")).strip()
                if fixture_id:
                    compared_by_fixture_id[fixture_id] = row

        for fixture in fixtures:
            fixture_id = str(fixture.get("fixture_id", "")).strip()
            compared_row = compared_by_fixture_id.get(fixture_id)

            if not compared_row:
                fixture["p_H"] = None
                fixture["p_D"] = None
                fixture["p_A"] = None
                fixture["odds_avg_h"] = None
                fixture["odds_avg_d"] = None
                fixture["odds_avg_a"] = None
                fixture["best_ev"] = None
                fixture["best_ev_pick"] = None
                fixture["value_bet"] = None
                continue

            p_h = _to_float_or_none(compared_row.get("p_H"))
            p_d = _to_float_or_none(compared_row.get("p_D"))
            p_a = _to_float_or_none(compared_row.get("p_A"))
            odds_avg_h = _to_float_or_none(compared_row.get("odds_avg_h"))
            odds_avg_d = _to_float_or_none(compared_row.get("odds_avg_d"))
            odds_avg_a = _to_float_or_none(compared_row.get("odds_avg_a"))

            fixture["p_H"] = p_h
            fixture["p_D"] = p_d
            fixture["p_A"] = p_a
            fixture["odds_avg_h"] = odds_avg_h
            fixture["odds_avg_d"] = odds_avg_d
            fixture["odds_avg_a"] = odds_avg_a

            best_ev = _to_float_or_none(compared_row.get("best_ev"))
            if best_ev is None:
                fixture["best_ev"] = None
                fixture["best_ev_pick"] = None
                fixture["value_bet"] = None
                continue

            best_pick = str(compared_row.get("best_ev_pick", "")).strip()
            value_bet_value = compared_row.get("value_bet")

            fixture["best_ev"] = best_ev
            fixture["best_ev_pick"] = best_pick if best_pick else None
            fixture["value_bet"] = bool(value_bet_value) if pd.notna(value_bet_value) else None

    except Exception as exc:
        for fixture in fixtures:
            fixture["p_H"] = None
            fixture["p_D"] = None
            fixture["p_A"] = None
            fixture["odds_avg_h"] = None
            fixture["odds_avg_d"] = None
            fixture["odds_avg_a"] = None
            fixture["best_ev"] = None
            fixture["best_ev_pick"] = None
            fixture["value_bet"] = None

        previous_error = str(response.get("error") or "").strip()
        value_error = f"No se pudo calcular ranking de value bet: {exc}"
        response["error"] = f"{previous_error} | {value_error}" if previous_error else value_error

    response["fixtures"] = fixtures
    response["rows"] = int(len(fixtures))
    return response


def _ensure_default_enriched_historical() -> Path:
    output_all = settings.output_dir / "laliga_enriched_all.csv"
    # For prediction flows we only need the enriched historical frame.
    # Do not force a full re-ingest if laliga_enriched_all.csv is already present.
    if output_all.exists():
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
    from modelos.services.datasets import ingest_datasets

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


def _ensure_logreg_compat(model: Any) -> None:
    """Patch legacy sklearn payloads that miss LogisticRegression.multi_class."""

    visited: set[int] = set()

    def walk(node: Any) -> None:
        if node is None:
            return

        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        if isinstance(node, LogisticRegression) and not hasattr(node, "multi_class"):
            node.multi_class = "auto"

        if isinstance(node, dict):
            for value in node.values():
                walk(value)
            return

        if isinstance(node, (list, tuple, set)):
            for value in node:
                walk(value)
            return

        for attr in ["steps", "named_steps", "estimator", "estimator_", "calibrated_classifiers_"]:
            if hasattr(node, attr):
                walk(getattr(node, attr))

    walk(model)


def predict_matches(request: PredictRequest) -> dict[str, Any]:
    features_df = _prepare_prediction_input(request)

    store = ModelStore()
    try:
        payload, _ = store.load()
    except FileNotFoundError:
        from app.schemas.model import TrainRequest
        from modelos.services.train import train_and_calibrate

        # Auto-training fallback: prefer robust baseline models for cold start.
        train_and_calibrate(TrainRequest(use_xgb=False, use_catboost=False))
        payload, _ = store.load()

    model = payload["model"]
    _ensure_logreg_compat(model)
    feature_columns: list[str] = payload["feature_columns"]

    for column in feature_columns:
        if column not in features_df.columns:
            features_df[column] = np.nan

    model_input = features_df[feature_columns].copy()
    model_input = model_input.apply(pd.to_numeric, errors="coerce").astype(float)

    proba = model.predict_proba(model_input)
    classes = getattr(model, "classes_", np.array([0, 1, 2]))

    proba_h = np.zeros(len(model_input), dtype=float)
    proba_d = np.zeros(len(model_input), dtype=float)
    proba_a = np.zeros(len(model_input), dtype=float)

    class_to_col = {int(cls): idx for idx, cls in enumerate(np.asarray(classes).tolist())}
    if 0 in class_to_col:
        proba_h = proba[:, class_to_col[0]]
    if 1 in class_to_col:
        proba_d = proba[:, class_to_col[1]]
    if 2 in class_to_col:
        proba_a = proba[:, class_to_col[2]]

    output = features_df.copy()
    output["p_H"] = proba_h
    output["p_D"] = proba_d
    output["p_A"] = proba_a

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

    requested_round = str(request.round or "").strip()
    fixture_payload: dict[str, Any] = {
        "date": parsed_date.strftime("%Y-%m-%d"),
        "home_team": _canonical_team_name(request.home_team, team_map),
        "away_team": _canonical_team_name(request.away_team, team_map),
    }
    if requested_round != "":
        fixture_payload["round"] = requested_round

    fixture_df = pd.DataFrame([fixture_payload])

    prediction_note: str | None = None
    try:
        historical_df = pd.read_csv(historical_path)
        enriched = enrich_fixtures(fixture_df, historical_df, DEFAULT_WINDOWS, team_map)

        elo_path = defaults["elo_csv"]
        if elo_path.exists():
            enriched = enrich_with_elo(enriched, elo_path, team_map)

        prediction = predict_matches(PredictRequest(fixtures=enriched.to_dict(orient="records")))
    except Exception as exc:
        # Keep /predict/upcoming available even when feature enrichment fails for a fixture.
        prediction = predict_matches(PredictRequest(fixtures=fixture_df.to_dict(orient="records")))
        prediction_note = (
            "Prediccion generada con fallback sin enriquecimiento de features"
            f" ({type(exc).__name__})"
        )
    prediction_row = prediction["predictions"][0] if prediction["predictions"] else {}

    market_odds: dict[str, Any] | None = None
    try:
        market_odds = find_fixture_odds(
            date_iso=str(fixture_df.iloc[0]["date"]),
            home_team=str(fixture_df.iloc[0]["home_team"]),
            away_team=str(fixture_df.iloc[0]["away_team"]),
            team_map=team_map,
        )
    except Exception:
        market_odds = None

    if market_odds is not None:
        for key in ["odds_avg_h", "odds_avg_d", "odds_avg_a", "odds_best_h", "odds_best_d", "odds_best_a"]:
            if key in market_odds:
                prediction_row[key] = market_odds[key]

    if prediction_note is not None:
        prediction_row["prediction_note"] = prediction_note

    round_value = ""
    if "round" in fixture_df.columns:
        raw_round = fixture_df.iloc[0]["round"]
        if raw_round is not None and str(raw_round).strip().lower() not in ("", "nan", "none", "null"):
            round_value = str(raw_round).strip()

    return {
        "season_label": _current_season_label(),
        "selected_fixture": {
            "date": str(fixture_df.iloc[0]["date"]),
            "home_team": str(fixture_df.iloc[0]["home_team"]),
            "away_team": str(fixture_df.iloc[0]["away_team"]),
            "round": round_value,
        },
        "prediction": prediction_row,
        "market_odds": market_odds,
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
    settings.output_dir.mkdir(parents=True, exist_ok=True)
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
            "best_ev_odds",
            "ev_H",
            "ev_D",
            "ev_A",
            "target",
            "target_label",
            "bet_won",
            "bet_profit",
        ]
        if column in value_bets.columns
    ]
    return {
        "rows": int(len(compared)),
        "metrics": metrics,
        "value_bets": value_bets[keep].to_dict(orient="records") if keep else value_bets.to_dict(orient="records"),
        "output_csv": str(output_csv),
    }
