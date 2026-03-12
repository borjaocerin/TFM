from app.services.odds_api import _parse_h2h_outcomes, find_fixture_odds


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
