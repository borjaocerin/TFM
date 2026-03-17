import csv

from app.core.config import settings
from app.services.odds_api import _parse_h2h_outcomes, find_fixture_odds
from app.services.odds_api import persist_laliga_odds_snapshot


def test_parse_h2h_outcomes_matches_common_team_aliases() -> None:
    event = {
        "bookmakers": [
            {
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Barcelona", "price": 1.9},
                            {"name": "Draw", "price": 3.6},
                            {"name": "Club Atletico de Madrid", "price": 4.1},
                        ],
                    }
                ]
            }
        ]
    }

    odds = _parse_h2h_outcomes(
        event=event,
        home_team="FC Barcelona",
        away_team="Atletico Madrid",
        team_map={},
    )

    assert odds["odds_avg_h"] == 1.9
    assert odds["odds_avg_d"] == 3.6
    assert odds["odds_avg_a"] == 4.1


def test_find_fixture_odds_matches_aliases_in_request_and_rows(monkeypatch) -> None:
    def fake_fetch_upcoming_laliga_odds(*_args, **_kwargs) -> dict:
        return {
            "odds": [
                {
                    "date": "2026-03-15",
                    "home_team": "Barcelona",
                    "away_team": "Atletico Madrid",
                    "odds_avg_h": 1.95,
                    "odds_avg_d": 3.7,
                    "odds_avg_a": 4.0,
                }
            ]
        }

    monkeypatch.setattr(
        "app.services.odds_api.fetch_upcoming_laliga_odds",
        fake_fetch_upcoming_laliga_odds,
    )

    matched = find_fixture_odds(
        date_iso="2026-03-15",
        home_team="FC Barcelona",
        away_team="Club Atletico de Madrid",
        team_map={},
    )

    assert matched is not None
    assert matched["odds_avg_h"] == 1.95


def test_persist_laliga_odds_snapshot_writes_latest_and_history(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "output_dir", tmp_path)

    odds_rows = [
        {
            "fixture_id": "2026-03-15|Barcelona|Atletico Madrid",
            "event_id": "evt-1",
            "date": "2026-03-15",
            "home_team": "Barcelona",
            "away_team": "Atletico Madrid",
            "source": "api:the-odds-api",
            "bookmakers": 4,
            "odds_avg_h": 1.95,
            "odds_avg_d": 3.60,
            "odds_avg_a": 4.10,
            "odds_best_h": 2.00,
            "odds_best_d": 3.70,
            "odds_best_a": 4.20,
        }
    ]

    first = persist_laliga_odds_snapshot(odds_rows)
    second = persist_laliga_odds_snapshot(odds_rows)

    latest_path = tmp_path / "laliga_upcoming_odds.csv"
    history_path = tmp_path / "laliga_odds_history.csv"

    assert first["output_csv"] == str(latest_path)
    assert second["history_csv"] == str(history_path)
    assert latest_path.exists()
    assert history_path.exists()

    with latest_path.open("r", encoding="utf-8", newline="") as latest_file:
        latest_rows = list(csv.DictReader(latest_file))

    with history_path.open("r", encoding="utf-8", newline="") as history_file:
        history_rows = list(csv.DictReader(history_file))

    assert len(latest_rows) == 1
    assert len(history_rows) == 2
    assert latest_rows[0]["home_team"] == "Barcelona"
    assert history_rows[0]["fetched_at_utc"] != ""
