from fastapi.testclient import TestClient

from app.main import app


def test_upcoming_options_include_round(monkeypatch) -> None:
    def fake_list_upcoming_fixture_options() -> dict:
        return {
            "season_label": "2025-2026",
            "source_path": "demo:test",
            "rows": 1,
            "fixtures": [
                {
                    "fixture_id": "2026-03-12|Real Madrid|Barcelona",
                    "date": "2026-03-12",
                    "home_team": "Real Madrid",
                    "away_team": "Barcelona",
                    "label": "2026-03-12 | Real Madrid vs Barcelona",
                    "round": "28",
                }
            ],
        }

    monkeypatch.setattr(
        "app.api.v1.routes_predict.list_upcoming_fixture_options",
        fake_list_upcoming_fixture_options,
    )

    client = TestClient(app)
    response = client.get("/api/v1/predict/options/upcoming")

    assert response.status_code == 200
    assert response.json()["fixtures"][0]["round"] == "28"


def test_predict_upcoming_includes_round(monkeypatch) -> None:
    def fake_predict_selected_upcoming_match(_request) -> dict:
        return {
            "season_label": "2025-2026",
            "selected_fixture": {
                "date": "2026-03-12",
                "home_team": "Real Madrid",
                "away_team": "Barcelona",
                "round": "28",
            },
            "prediction": {"p_H": 0.4, "p_D": 0.3, "p_A": 0.3},
            "market_odds": None,
            "output_csv": "data/out.csv",
        }

    monkeypatch.setattr(
        "app.api.v1.routes_predict.predict_selected_upcoming_match",
        fake_predict_selected_upcoming_match,
    )

    client = TestClient(app)
    response = client.post(
        "/api/v1/predict/upcoming",
        json={
            "date": "2026-03-12",
            "home_team": "Real Madrid",
            "away_team": "Barcelona",
        },
    )

    assert response.status_code == 200
    assert response.json()["selected_fixture"]["round"] == "28"


def test_upcoming_options_include_value_ranking(monkeypatch) -> None:
    def fake_list_upcoming_fixture_options() -> dict:
        return {
            "season_label": "2025-2026",
            "source_path": "demo:test",
            "rows": 1,
            "fixtures": [
                {
                    "fixture_id": "2026-03-12|Real Madrid|Barcelona",
                    "date": "2026-03-12",
                    "home_team": "Real Madrid",
                    "away_team": "Barcelona",
                    "label": "2026-03-12 | Real Madrid vs Barcelona",
                    "round": "28",
                }
            ],
        }

    def fake_list_upcoming_fixture_options_with_value(value_threshold: float = 0.02) -> dict:
        assert value_threshold == 0.05
        return {
            "season_label": "2025-2026",
            "source_path": "demo:test",
            "rows": 1,
            "fixtures": [
                {
                    "fixture_id": "2026-03-12|Real Madrid|Barcelona",
                    "date": "2026-03-12",
                    "home_team": "Real Madrid",
                    "away_team": "Barcelona",
                    "label": "2026-03-12 | Real Madrid vs Barcelona",
                    "round": "28",
                    "p_H": 0.228,
                    "p_D": 0.228,
                    "p_A": 0.544,
                    "odds_avg_h": 3.8,
                    "odds_avg_d": 7.0,
                    "odds_avg_a": 1.95,
                    "best_ev": 0.1234,
                    "best_ev_pick": "H",
                    "value_bet": True,
                }
            ],
        }

    monkeypatch.setattr(
        "app.api.v1.routes_predict.list_upcoming_fixture_options",
        fake_list_upcoming_fixture_options,
    )
    monkeypatch.setattr(
        "app.api.v1.routes_predict.list_upcoming_fixture_options_with_value",
        fake_list_upcoming_fixture_options_with_value,
    )

    client = TestClient(app)
    response = client.get("/api/v1/predict/options/upcoming?include_value=true&value_threshold=0.05")

    assert response.status_code == 200
    fixture = response.json()["fixtures"][0]
    assert fixture["p_H"] == 0.228
    assert fixture["odds_avg_d"] == 7.0
    assert fixture["best_ev"] == 0.1234
    assert fixture["best_ev_pick"] == "H"
    assert fixture["value_bet"] is True