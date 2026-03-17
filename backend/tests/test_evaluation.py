import numpy as np
import pandas as pd
import pytest

from app.services.evaluation import (
    compare_market_vs_model,
    expected_calibration_error,
    multiclass_brier_score,
)


def test_brier_score_in_range() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_prob = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    score = multiclass_brier_score(y_true, y_prob)
    assert 0.0 <= score <= 2.0


def test_ece_non_negative() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_prob = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6],
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    score = expected_calibration_error(y_true, y_prob)
    assert score >= 0.0


def test_compare_market_vs_model_computes_profitability_metrics() -> None:
    frame = pd.DataFrame(
        [
            {
                "target": "H",
                "p_H": 0.60,
                "p_D": 0.20,
                "p_A": 0.20,
                "odds_avg_h": 2.10,
                "odds_avg_d": 3.40,
                "odds_avg_a": 3.80,
            },
            {
                "target": "A",
                "p_H": 0.50,
                "p_D": 0.20,
                "p_A": 0.30,
                "odds_avg_h": 2.30,
                "odds_avg_d": 3.20,
                "odds_avg_a": 3.20,
            },
        ]
    )

    metrics, compared = compare_market_vs_model(frame, odds_kind="odds_avg", value_threshold=0.10)

    assert metrics["model_log_loss"] is not None
    assert metrics["value_bets_total"] == 2.0
    assert metrics["value_bets_settled"] == 2.0
    assert metrics["value_bets_won"] == 1.0
    assert metrics["value_bets_profit"] == pytest.approx(0.1)
    assert metrics["value_bets_roi"] == pytest.approx(0.05)
    assert metrics["value_bets_hit_rate"] == pytest.approx(0.5)

    profits = compared["bet_profit"].dropna().tolist()
    assert sorted(profits) == [-1.0, 1.1]
