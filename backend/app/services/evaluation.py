from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, f1_score, log_loss


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int = 3) -> float:
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece_values: list[float] = []

    for class_index in range(y_prob.shape[1]):
        class_true = (y_true == class_index).astype(float)
        class_prob = y_prob[:, class_index]
        class_ece = 0.0
        for lower, upper in zip(bins[:-1], bins[1:]):
            mask = (class_prob > lower) & (class_prob <= upper)
            if not np.any(mask):
                continue
            acc = float(np.mean(class_true[mask]))
            conf = float(np.mean(class_prob[mask]))
            class_ece += (np.sum(mask) / total) * abs(acc - conf)
        ece_values.append(class_ece)
    return float(np.mean(ece_values))


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = np.argmax(y_prob, axis=1)
    return {
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1, 2])),
        "brier": multiclass_brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def reliability_points(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for class_index, label in enumerate(["H", "D", "A"]):
        class_true = (y_true == class_index).astype(int)
        frac_pos, mean_pred = calibration_curve(
            class_true,
            y_prob[:, class_index],
            n_bins=n_bins,
            strategy="uniform",
        )
        output[label] = {
            "fraction_positives": frac_pos.tolist(),
            "mean_predicted_value": mean_pred.tolist(),
        }
    return output


def market_implied_probabilities(df: pd.DataFrame, odds_kind: str) -> pd.DataFrame:
    output = df.copy()
    home_col = f"{odds_kind}_h"
    draw_col = f"{odds_kind}_d"
    away_col = f"{odds_kind}_a"

    inv_h = 1.0 / pd.to_numeric(output[home_col], errors="coerce")
    inv_d = 1.0 / pd.to_numeric(output[draw_col], errors="coerce")
    inv_a = 1.0 / pd.to_numeric(output[away_col], errors="coerce")
    total = inv_h + inv_d + inv_a

    output["mkt_p_H"] = inv_h / total
    output["mkt_p_D"] = inv_d / total
    output["mkt_p_A"] = inv_a / total
    return output


def compare_market_vs_model(
    frame: pd.DataFrame,
    odds_kind: str,
    value_threshold: float,
) -> tuple[dict[str, float | None], pd.DataFrame]:
    df = market_implied_probabilities(frame, odds_kind)

    for outcome in ["H", "D", "A"]:
        df[f"ev_{outcome}"] = pd.to_numeric(df[f"p_{outcome}"], errors="coerce") * pd.to_numeric(
            df[f"{odds_kind}_{outcome.lower()}"], errors="coerce"
        ) - 1.0

    ev_cols = ["ev_H", "ev_D", "ev_A"]
    ev_matrix = df[ev_cols].to_numpy(dtype=float)
    safe_ev = np.where(np.isnan(ev_matrix), -np.inf, ev_matrix)
    best_idx = np.argmax(safe_ev, axis=1)
    mapping = np.array(["H", "D", "A"])
    df["best_ev_pick"] = mapping[best_idx]
    best_ev = np.take_along_axis(safe_ev, best_idx[:, None], axis=1).reshape(-1)
    best_ev = np.where(np.isneginf(best_ev), np.nan, best_ev)
    df["best_ev"] = best_ev
    df["value_bet"] = df["best_ev"].fillna(-999.0) > value_threshold

    metrics: dict[str, float | None] = {
        "model_log_loss": None,
        "market_log_loss": None,
        "model_brier": None,
        "market_brier": None,
    }
    if "target" in df.columns:
        valid = df.dropna(subset=["target", "p_H", "p_D", "p_A", "mkt_p_H", "mkt_p_D", "mkt_p_A"])
        if not valid.empty:
            y_true = valid["target"].astype(int).to_numpy()
            model_prob = valid[["p_H", "p_D", "p_A"]].astype(float).to_numpy()
            market_prob = valid[["mkt_p_H", "mkt_p_D", "mkt_p_A"]].astype(float).to_numpy()
            metrics["model_log_loss"] = float(log_loss(y_true, model_prob, labels=[0, 1, 2]))
            metrics["market_log_loss"] = float(log_loss(y_true, market_prob, labels=[0, 1, 2]))
            metrics["model_brier"] = multiclass_brier_score(y_true, model_prob)
            metrics["market_brier"] = multiclass_brier_score(y_true, market_prob)
    return metrics, df
