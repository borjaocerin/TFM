from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import requests


ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_SPORT = "soccer_spain_la_liga"

TEAM_ALIASES = {
    "atletico": "atletico madrid",
    "ath bilbao": "athletic",
    "athletic club": "athletic",
    "barca": "barcelona",
    "real madrid cf": "real madrid",
}


def _normalize_team(name: str) -> str:
    raw = re.sub(r"[^a-zA-Z0-9 ]+", "", str(name).strip().lower())
    raw = re.sub(r"\s+", " ", raw)
    return TEAM_ALIASES.get(raw, raw)


def _parse_h2h_market(event: Dict[str, Any], preferred_bookmaker: Optional[str] = None) -> Optional[Dict[str, float]]:
    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return None

    ordered = bookmakers
    if preferred_bookmaker:
        ordered = sorted(
            bookmakers,
            key=lambda b: 0 if str(b.get("key", "")).lower() == preferred_bookmaker.lower() else 1,
        )

    for book in ordered:
        for market in book.get("markets", []):
            if market.get("key") != "h2h":
                continue

            out = {str(x.get("name")): x.get("price") for x in market.get("outcomes", [])}
            home = event.get("home_team")
            away = event.get("away_team")

            home_odd = out.get(home)
            away_odd = out.get(away)
            draw_odd = out.get("Draw")

            if home_odd and away_odd and draw_odd:
                return {
                    "home_win": float(home_odd),
                    "draw": float(draw_odd),
                    "away_win": float(away_odd),
                    "bookmaker": str(book.get("key", "unknown")),
                }
    return None


def fetch_odds_events(region: str = "eu", markets: str = "h2h") -> List[Dict[str, Any]]:
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Falta ODDS_API_KEY en el entorno para usar cuotas reales")

    response = requests.get(
        f"{ODDS_BASE_URL}/sports/{DEFAULT_SPORT}/odds",
        params={
            "apiKey": api_key,
            "regions": region,
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        },
        timeout=25,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        return []
    return data


def find_match_odds(
    home_team: str,
    away_team: str,
    events: List[Dict[str, Any]],
    preferred_bookmaker: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    target_home = _normalize_team(home_team)
    target_away = _normalize_team(away_team)

    for event in events:
        ev_home = _normalize_team(event.get("home_team", ""))
        ev_away = _normalize_team(event.get("away_team", ""))

        if ev_home == target_home and ev_away == target_away:
            return _parse_h2h_market(event, preferred_bookmaker=preferred_bookmaker)

    for event in events:
        ev_home = _normalize_team(event.get("home_team", ""))
        ev_away = _normalize_team(event.get("away_team", ""))
        if target_home in ev_home and target_away in ev_away:
            return _parse_h2h_market(event, preferred_bookmaker=preferred_bookmaker)

    return None
