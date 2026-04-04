from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data" / "out" / "laliga_enriched_model.csv"
OUT_DIR = ROOT / "out" / "model_improvement"


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece_values: list[float] = []

    for class_idx in range(y_prob.shape[1]):
        class_true = (y_true == class_idx).astype(float)
        class_prob = y_prob[:, class_idx]
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


def _build_models() -> dict[str, Pipeline]:
    return {
        "logreg": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=160,
                        random_state=42,
                        max_depth=8,
                        min_samples_leaf=2,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=220,
                        random_state=42,
                        min_samples_leaf=2,
                        n_jobs=1,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        ),
    }


def _load_data(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError("El dataset debe contener columna target")

    df["date_dt"] = pd.to_datetime(df.get("date"), errors="coerce")
    df = df.sort_values("date_dt").dropna(subset=["target"]).copy()
    df["target"] = pd.to_numeric(df["target"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    excluded = {
        "date",
        "date_dt",
        "season",
        "home_team",
        "away_team",
        "result",
        "home_goals",
        "away_goals",
        "target",
    }

    feature_columns = [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any()
    ]

    if not feature_columns:
        raise ValueError("No hay columnas numéricas para entrenar")

    return df.reset_index(drop=True), feature_columns


def _make_time_splits(n_rows: int, n_splits: int = 4, min_train_size: int = 1200) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_rows <= min_train_size + 50:
        raise ValueError("Dataset demasiado pequeño para walk-forward")

    test_size = max(80, (n_rows - min_train_size) // n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    train_end = min_train_size
    while train_end + test_size <= n_rows:
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, train_end + test_size)
        splits.append((train_idx, test_idx))
        train_end += test_size

    return splits


def _settle_strategy(
    frame: pd.DataFrame,
    threshold: float,
    min_prob: float,
) -> dict[str, float | int | None]:
    work = frame.copy()

    prob_cols = ["p_H", "p_D", "p_A"]
    probs = work[prob_cols].to_numpy(dtype=float)
    best_idx = np.argmax(probs, axis=1)
    labels = np.array(["H", "D", "A"])

    work["pick"] = labels[best_idx]
    work["pick_prob"] = probs[np.arange(len(work)), best_idx]

    has_odds = all(column in work.columns for column in ["odds_avg_h", "odds_avg_d", "odds_avg_a"])

    if has_odds:
        work["ev_H"] = pd.to_numeric(work["p_H"], errors="coerce") * pd.to_numeric(work["odds_avg_h"], errors="coerce") - 1.0
        work["ev_D"] = pd.to_numeric(work["p_D"], errors="coerce") * pd.to_numeric(work["odds_avg_d"], errors="coerce") - 1.0
        work["ev_A"] = pd.to_numeric(work["p_A"], errors="coerce") * pd.to_numeric(work["odds_avg_a"], errors="coerce") - 1.0
        ev_matrix = work[["ev_H", "ev_D", "ev_A"]].to_numpy(dtype=float)
        pick_ev = ev_matrix[np.arange(len(work)), best_idx]
        work["pick_ev"] = pick_ev
        selected = (work["pick_ev"] > threshold) & (work["pick_prob"] >= min_prob)
    else:
        work["pick_ev"] = np.nan
        selected = work["pick_prob"] >= min_prob

    work = work[selected].copy()
    if work.empty:
        return {
            "bets": 0,
            "wins": 0,
            "hit_rate": None,
            "profit": 0.0,
            "roi": None,
        }

    work["won"] = work["pick"] == work["target_label"]

    if has_odds:
        odds_pick = np.select(
            [work["pick"] == "H", work["pick"] == "D", work["pick"] == "A"],
            [
                pd.to_numeric(work["odds_avg_h"], errors="coerce"),
                pd.to_numeric(work["odds_avg_d"], errors="coerce"),
                pd.to_numeric(work["odds_avg_a"], errors="coerce"),
            ],
            default=np.nan,
        )
        work["profit"] = np.where(work["won"], odds_pick - 1.0, -1.0)
    else:
        work["profit"] = np.where(work["won"], 1.0, -1.0)

    bets = int(len(work))
    wins = int(work["won"].sum())
    total_profit = float(pd.to_numeric(work["profit"], errors="coerce").sum())

    return {
        "bets": bets,
        "wins": wins,
        "hit_rate": float(wins / bets) if bets > 0 else None,
        "profit": total_profit,
        "roi": float(total_profit / bets) if bets > 0 else None,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, feature_columns = _load_data(DATASET_PATH)
    splits = _make_time_splits(len(df), n_splits=4, min_train_size=1200)

    models = _build_models()
    thresholds_ev = [0.00, 0.02, 0.04, 0.06]
    thresholds_prob = [0.34, 0.40, 0.46, 0.50]

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model_name, model in models.items():
        split_predictions: list[pd.DataFrame] = []

        fold_metrics: list[dict[str, float]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            train = df.iloc[train_idx].copy()
            test = df.iloc[test_idx].copy()

            x_train = train[feature_columns].apply(pd.to_numeric, errors="coerce").astype(float)
            y_train = train["target"].astype(int).to_numpy()

            x_test = test[feature_columns].apply(pd.to_numeric, errors="coerce").astype(float)
            y_test = test["target"].astype(int).to_numpy()

            model.fit(x_train, y_train)
            prob = model.predict_proba(x_test)
            pred = np.argmax(prob, axis=1)

            fold_metrics.append(
                {
                    "model": model_name,
                    "fold": fold_idx,
                    "rows": int(len(test)),
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "log_loss": float(log_loss(y_test, prob, labels=[0, 1, 2])),
                    "ece": float(_expected_calibration_error(y_test, prob)),
                }
            )

            scored = pd.DataFrame(
                {
                    "date": test["date"].values if "date" in test.columns else None,
                    "home_team": test["home_team"].values if "home_team" in test.columns else None,
                    "away_team": test["away_team"].values if "away_team" in test.columns else None,
                    "target": y_test,
                    "target_label": np.array(["H", "D", "A"])[y_test],
                    "p_H": prob[:, 0],
                    "p_D": prob[:, 1],
                    "p_A": prob[:, 2],
                }
            )

            for odds_col in ["odds_avg_h", "odds_avg_d", "odds_avg_a"]:
                if odds_col in test.columns:
                    scored[odds_col] = test[odds_col].values

            split_predictions.append(scored)

        pred_frame = pd.concat(split_predictions, ignore_index=True)

        for ev_t in thresholds_ev:
            for prob_t in thresholds_prob:
                strategy = _settle_strategy(pred_frame, threshold=ev_t, min_prob=prob_t)
                all_rows.append(
                    {
                        "model": model_name,
                        "ev_threshold": ev_t,
                        "min_prob_threshold": prob_t,
                        **strategy,
                    }
                )

        fold_df = pd.DataFrame(fold_metrics)
        summary_rows.append(
            {
                "model": model_name,
                "cv_accuracy_mean": float(fold_df["accuracy"].mean()),
                "cv_log_loss_mean": float(fold_df["log_loss"].mean()),
                "cv_ece_mean": float(fold_df["ece"].mean()),
            }
        )

    strategy_df = pd.DataFrame(all_rows)
    strategy_df = strategy_df.sort_values(["roi", "profit", "hit_rate"], ascending=False, na_position="last")

    model_df = pd.DataFrame(summary_rows).sort_values("cv_log_loss_mean")

    strategy_df.to_csv(OUT_DIR / "walkforward_strategy_ranking.csv", index=False)
    model_df.to_csv(OUT_DIR / "walkforward_model_metrics.csv", index=False)

    best_row = strategy_df.iloc[0].to_dict() if not strategy_df.empty else {}

    report = {
        "dataset_path": str(DATASET_PATH).replace("\\", "/"),
        "rows": int(len(df)),
        "n_splits": int(len(splits)),
        "feature_count": int(len(feature_columns)),
        "best_strategy": best_row,
    }

    (OUT_DIR / "walkforward_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
