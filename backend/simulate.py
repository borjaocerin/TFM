from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from .api_data import get_fixtures
    from .odds_data import fetch_odds_events, find_match_odds
    from .predict import predict_match
except ImportError:
    from api_data import get_fixtures
    from odds_data import fetch_odds_events, find_match_odds
    from predict import predict_match


@dataclass
class ValueBet:
    fixture: str
    market: str
    model_probability: float
    bookmaker_odds: float
    implied_probability: float
    expected_value: float
    stake: float
    won: bool
    profit: float


def _simulated_odds(home_win: float, draw: float, away_win: float, margin: float = 0.06) -> Dict[str, float]:
    probs = np.array([home_win, draw, away_win], dtype=float)
    probs = probs / probs.sum()

    # Add realistic bookmaker margin and some noise.
    noisy = np.clip(probs + np.random.normal(0.0, 0.02, size=3), 0.03, 0.9)
    noisy = noisy / noisy.sum()
    implied = noisy * (1.0 + margin)

    return {
        "home_win": float(1.0 / implied[0]),
        "draw": float(1.0 / implied[1]),
        "away_win": float(1.0 / implied[2]),
    }


def _value_bets_for_match(
    pred: Dict[str, Any],
    stake: float,
    fixture_name: str,
    odds: Dict[str, float],
) -> List[ValueBet]:
    outcomes = ["home_win", "draw", "away_win"]

    bets: List[ValueBet] = []
    for market in outcomes:
        p_model = float(pred[market])
        odd = float(odds[market])
        implied_p = 1.0 / odd
        ev = p_model * odd - 1.0

        if p_model > implied_p and ev > 0:
            won = market == pred["prediction"]
            profit = stake * (odd - 1.0) if won else -stake
            bets.append(
                ValueBet(
                    fixture=fixture_name,
                    market=market,
                    model_probability=p_model,
                    bookmaker_odds=odd,
                    implied_probability=implied_p,
                    expected_value=ev,
                    stake=stake,
                    won=won,
                    profit=profit,
                )
            )
    return bets


def simulate_value_betting_round(
    season: str,
    round_number: int,
    budget: float,
    dataset_path: Path | str,
    model_path: Path | str,
    odds_source: str = "api",
    preferred_bookmaker: Optional[str] = None,
) -> Dict[str, Any]:
    fixtures = get_fixtures(season=season, round_number=round_number, dataset_path=dataset_path)
    if not fixtures:
        return {"bets": [], "roi": 0.0, "profit": 0.0, "message": "No hay fixtures para simular"}

    odds_events: List[Dict[str, Any]] = []
    if not preferred_bookmaker:
        preferred_bookmaker = os.getenv("PREFERRED_BOOKMAKER", "").strip() or None
    if odds_source == "api":
        odds_events = fetch_odds_events()

    stake_per_bet = max(1.0, budget / max(len(fixtures), 1) / 3.0)

    all_bets: List[ValueBet] = []
    skipped_fixtures: List[str] = []
    for fx in fixtures:
        pred = predict_match(
            home_team=fx["home_team"],
            away_team=fx["away_team"],
            dataset_path=dataset_path,
            model_path=model_path,
        )
        fixture_name = f"{fx['home_team']} vs {fx['away_team']}"

        if odds_source == "api":
            odds = find_match_odds(
                home_team=fx["home_team"],
                away_team=fx["away_team"],
                events=odds_events,
                preferred_bookmaker=preferred_bookmaker,
            )
            if not odds:
                skipped_fixtures.append(fixture_name)
                continue
        else:
            odds = _simulated_odds(pred["home_win"], pred["draw"], pred["away_win"])

        all_bets.extend(
            _value_bets_for_match(
                pred=pred,
                stake=stake_per_bet,
                fixture_name=fixture_name,
                odds=odds,
            )
        )

    total_staked = float(sum(b.stake for b in all_bets))
    total_profit = float(sum(b.profit for b in all_bets))
    roi = (total_profit / total_staked) if total_staked > 0 else 0.0

    if odds_source == "api" and len(skipped_fixtures) == len(fixtures):
        raise ValueError(
            "No se encontraron cuotas reales para los partidos solicitados. Revisa ODDS_API_KEY, cobertura de mercado o nombres de equipos."
        )

    return {
        "season": season,
        "round": round_number,
        "budget": budget,
        "odds_source": odds_source,
        "bets": [b.__dict__ for b in all_bets],
        "total_bets": len(all_bets),
        "staked": total_staked,
        "profit": total_profit,
        "roi": roi,
        "skipped_fixtures_no_real_odds": skipped_fixtures,
    }
