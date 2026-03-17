from __future__ import annotations

import csv
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.core.config import settings

DRAW_LABELS = {"draw", "tie", "empate", "x"}
ODDS_EXPORT_COLUMNS = [
    "fetched_at_utc",
    "fixture_id",
    "event_id",
    "date",
    "home_team",
    "away_team",
    "source",
    "bookmakers",
    "odds_avg_h",
    "odds_avg_d",
    "odds_avg_a",
    "odds_best_h",
    "odds_best_d",
    "odds_best_a",
]
TEAM_MANUAL_ALIASES = {
    "deportivo alaves": "Alaves",
    "girona fc": "Girona",
    "villarreal cf": "Villarreal",
    "rcd mallorca": "Mallorca",
    "fc barcelona": "Barcelona",
    "barcelona fc": "Barcelona",
    "valencia cf": "Valencia",
    "rc celta de vigo": "Celta",
    "club atletico de madrid": "Atletico Madrid",
    "atletico de madrid": "Atletico Madrid",
    "ca osasuna": "Osasuna",
    "real madrid cf": "Real Madrid",
    "real betis balompie": "Real Betis",
    "betis": "Real Betis",
    "real sociedad de futbol": "Real Sociedad",
    "rayo vallecano de madrid": "Rayo Vallecano",
    "levante ud": "Levante",
    "getafe cf": "Getafe",
    "elche cf": "Elche",
    "sevilla fc": "Sevilla",
    "rcd espanyol de barcelona": "Espanyol",
    "athletic bilbao": "Athletic Club",
}


def load_team_map() -> dict[str, str]:
    team_map_path = settings.data_dir.parent / "etl" / "team_name_map_es.json"
    if not team_map_path.exists():
        return {}

    try:
        return json.loads(team_map_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.strip().lower()
    for token in [".", ",", ";", ":", "'", '"', "-", "_"]:
        text = text.replace(token, " ")
    text = " ".join(text.split())
    return text


def _team_alias_map(team_map: dict[str, str]) -> dict[str, str]:
    normalized_map: dict[str, str] = {}
    for source_name, canonical_name in team_map.items():
        canonical = str(canonical_name)
        normalized_map[_normalize_text(str(source_name))] = canonical
        normalized_map[_normalize_text(canonical)] = canonical

    for alias, canonical in TEAM_MANUAL_ALIASES.items():
        normalized_map[_normalize_text(alias)] = canonical
        normalized_map[_normalize_text(canonical)] = canonical

    return normalized_map


def _canonical_team(team_name: str, team_map: dict[str, str]) -> str:
    mapped = team_map.get(team_name)
    if mapped:
        return mapped

    normalized_team = _normalize_text(team_name)
    normalized_map = _team_alias_map(team_map)
    mapped = normalized_map.get(normalized_team)
    if mapped:
        return mapped
    return team_name


def _to_iso_date(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if text == "":
        return ""

    fixed = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(fixed)
        return dt.date().isoformat()
    except ValueError:
        return text[:10]


def _local_odds_key_fallback() -> str | None:
    fallback_path = settings.data_dir.parent / "oddapikey.txt"
    if not fallback_path.exists():
        return None

    try:
        key = fallback_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return key or None


def _effective_odds_api_key() -> str | None:
    configured = (settings.odds_api_key or "").strip()
    if configured:
        return configured
    return _local_odds_key_fallback()


def _effective_odds_api_url() -> str:
    configured = (settings.odds_api_url or "").strip()
    if configured:
        return configured

    sport_key = (settings.odds_api_sport_key or "soccer_spain_la_liga").strip()
    return f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"


def _write_odds_csv(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    write_header = (not append) or (not path.exists()) or path.stat().st_size == 0
    mode = "a" if append else "w"
    with path.open(mode=mode, newline="", encoding="utf-8") as handler:
        writer = csv.DictWriter(handler, fieldnames=ODDS_EXPORT_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in ODDS_EXPORT_COLUMNS})


def persist_laliga_odds_snapshot(odds_rows: list[dict[str, Any]]) -> dict[str, str]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    latest_csv = settings.output_dir / "laliga_upcoming_odds.csv"
    history_csv = settings.output_dir / "laliga_odds_history.csv"
    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    rows_for_export: list[dict[str, Any]] = []
    for row in odds_rows:
        if not isinstance(row, dict):
            continue
        rows_for_export.append({"fetched_at_utc": fetched_at, **row})

    _write_odds_csv(latest_csv, rows_for_export, append=False)
    _write_odds_csv(history_csv, rows_for_export, append=True)

    return {
        "output_csv": str(latest_csv),
        "history_csv": str(history_csv),
        "fetched_at_utc": fetched_at,
    }


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _best(values: list[float]) -> float | None:
    if not values:
        return None
    return float(max(values))


def _parse_h2h_outcomes(
    event: dict[str, Any],
    home_team: str,
    away_team: str,
    team_map: dict[str, str],
) -> dict[str, Any]:
    home_key = _normalize_text(home_team)
    away_key = _normalize_text(away_team)

    prices_h: list[float] = []
    prices_d: list[float] = []
    prices_a: list[float] = []

    bookmakers = event.get("bookmakers")
    if not isinstance(bookmakers, list):
        bookmakers = []

    for bookmaker in bookmakers:
        if not isinstance(bookmaker, dict):
            continue
        markets = bookmaker.get("markets")
        if not isinstance(markets, list):
            continue

        for market in markets:
            if not isinstance(market, dict):
                continue
            if str(market.get("key", "")).strip().lower() != "h2h":
                continue

            outcomes = market.get("outcomes")
            if not isinstance(outcomes, list):
                continue

            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    continue

                outcome_name = str(outcome.get("name", "")).strip()
                if outcome_name == "":
                    continue
                canonical_outcome = _canonical_team(outcome_name, team_map)
                normalized_outcome = _normalize_text(canonical_outcome)
                price = _safe_float(outcome.get("price"))
                if price is None:
                    continue

                if normalized_outcome == home_key:
                    prices_h.append(price)
                    continue
                if normalized_outcome == away_key:
                    prices_a.append(price)
                    continue
                if normalized_outcome in DRAW_LABELS:
                    prices_d.append(price)

    return {
        "bookmakers": len(bookmakers),
        "odds_avg_h": _avg(prices_h),
        "odds_avg_d": _avg(prices_d),
        "odds_avg_a": _avg(prices_a),
        "odds_best_h": _best(prices_h),
        "odds_best_d": _best(prices_d),
        "odds_best_a": _best(prices_a),
    }


def _event_to_row(event: dict[str, Any], team_map: dict[str, str]) -> dict[str, Any] | None:
    home_raw = str(event.get("home_team", "")).strip()
    away_raw = str(event.get("away_team", "")).strip()
    if home_raw == "" or away_raw == "":
        return None

    home_team = _canonical_team(home_raw, team_map)
    away_team = _canonical_team(away_raw, team_map)
    date_iso = _to_iso_date(event.get("commence_time"))

    odds = _parse_h2h_outcomes(event, home_team, away_team, team_map)
    row = {
        "fixture_id": f"{date_iso}|{home_team}|{away_team}",
        "event_id": str(event.get("id", "")),
        "date": date_iso,
        "home_team": home_team,
        "away_team": away_team,
        "source": "api:the-odds-api",
        **odds,
    }
    return row


def fetch_upcoming_laliga_odds(team_map: dict[str, str], limit: int = 100) -> dict[str, Any]:
    api_key = _effective_odds_api_key()
    if not api_key:
        raise ValueError("No hay ODDS_API_KEY configurada (ni oddapikey.txt)")

    params: dict[str, str] = {
        "apiKey": api_key,
        "regions": settings.odds_api_regions,
        "markets": settings.odds_api_markets,
        "oddsFormat": settings.odds_api_odds_format,
        "dateFormat": settings.odds_api_date_format,
    }
    if settings.odds_api_bookmakers:
        params["bookmakers"] = settings.odds_api_bookmakers

    api_url = _effective_odds_api_url()
    request_url = f"{api_url}?{urlencode(params)}"
    safe_params = {key: value for key, value in params.items() if key.lower() != "apikey"}
    safe_source_path = f"{api_url}?{urlencode(safe_params)}" if safe_params else api_url
    headers = {"Accept": "application/json"}

    try:
        request = Request(request_url, headers=headers, method="GET")
        with urlopen(request, timeout=settings.odds_api_timeout_sec) as response:
            payload_raw = response.read().decode("utf-8")
            quota_remaining = response.headers.get("x-requests-remaining", "")
            quota_used = response.headers.get("x-requests-used", "")
    except HTTPError as exc:
        raise ValueError(f"The Odds API devolvio HTTP {exc.code}") from exc
    except (URLError, TimeoutError) as exc:
        raise ValueError("No se pudo conectar con The Odds API") from exc

    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Respuesta invalida de The Odds API") from exc

    if not isinstance(payload, list):
        raise ValueError("Formato inesperado en The Odds API")

    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        row = _event_to_row(item, team_map)
        if row is None:
            continue
        rows.append(row)

    rows = sorted(rows, key=lambda item: (str(item.get("date", "")), str(item.get("home_team", ""))))
    if limit > 0:
        rows = rows[:limit]

    snapshot_info = persist_laliga_odds_snapshot(rows)

    return {
        "sport_key": settings.odds_api_sport_key,
        "source_path": safe_source_path,
        "rows": len(rows),
        "odds": rows,
        "requests_remaining": quota_remaining,
        "requests_used": quota_used,
        "output_csv": snapshot_info["output_csv"],
        "history_csv": snapshot_info["history_csv"],
        "fetched_at_utc": snapshot_info["fetched_at_utc"],
    }


def find_fixture_odds(
    date_iso: str,
    home_team: str,
    away_team: str,
    team_map: dict[str, str],
) -> dict[str, Any] | None:
    odds_payload = fetch_upcoming_laliga_odds(team_map=team_map, limit=200)

    home_key = _normalize_text(_canonical_team(home_team, team_map))
    away_key = _normalize_text(_canonical_team(away_team, team_map))
    date_key = date_iso.strip()

    for row in odds_payload["odds"]:
        row_home = _normalize_text(_canonical_team(str(row.get("home_team", "")), team_map))
        row_away = _normalize_text(_canonical_team(str(row.get("away_team", "")), team_map))
        row_date = str(row.get("date", "")).strip()
        if row_home == home_key and row_away == away_key and row_date == date_key:
            return row

    for row in odds_payload["odds"]:
        row_home = _normalize_text(_canonical_team(str(row.get("home_team", "")), team_map))
        row_away = _normalize_text(_canonical_team(str(row.get("away_team", "")), team_map))
        if row_home == home_key and row_away == away_key:
            return row

    return None
