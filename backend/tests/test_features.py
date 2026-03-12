import pandas as pd

from app.services.features import enrich_fixtures


def test_enrich_fixtures_uses_historical_context_for_upcoming_matches() -> None:
    historical = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "home_team": "Team A",
                "away_team": "Team C",
                "home_goals": 2,
                "away_goals": 1,
                "result": "H",
                "xg_home": 2.0,
                "xg_away": 0.7,
                "xga_home": 0.8,
                "xga_away": 1.6,
                "poss_home": 58,
                "poss_away": 42,
                "sh_home": 15,
                "sh_away": 9,
                "sot_home": 6,
                "sot_away": 3,
            },
            {
                "date": "2026-01-02",
                "home_team": "Team B",
                "away_team": "Team D",
                "home_goals": 0,
                "away_goals": 1,
                "result": "A",
                "xg_home": 0.5,
                "xg_away": 1.3,
                "xga_home": 1.2,
                "xga_away": 0.6,
                "poss_home": 47,
                "poss_away": 53,
                "sh_home": 8,
                "sh_away": 11,
                "sot_home": 2,
                "sot_away": 5,
            },
        ]
    )
    fixtures = pd.DataFrame(
        [
            {"date": "2026-01-10", "home_team": "Team A", "away_team": "Team B"},
            {"date": "2026-01-10", "home_team": "Team C", "away_team": "Team D"},
        ]
    )

    enriched = enrich_fixtures(fixtures, historical, (1,), {})

    assert len(enriched) == 2
    assert pd.notna(enriched.loc[0, "xg_last1_home"])
    assert pd.notna(enriched.loc[0, "xg_last1_away"])
    assert pd.notna(enriched.loc[1, "xg_last1_home"])
    assert pd.notna(enriched.loc[1, "xg_last1_away"])
    assert pd.notna(enriched.loc[0, "elo_diff"])
    assert pd.notna(enriched.loc[1, "elo_diff"])
    assert enriched.loc[0, "xg_last1_diff"] != enriched.loc[1, "xg_last1_diff"]
    assert enriched.loc[0, "points_last1_diff"] != enriched.loc[1, "points_last1_diff"]
    assert enriched.loc[0, "elo_diff"] != enriched.loc[1, "elo_diff"]