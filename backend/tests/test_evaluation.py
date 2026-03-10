import numpy as np

from app.services.evaluation import expected_calibration_error, multiclass_brier_score


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
