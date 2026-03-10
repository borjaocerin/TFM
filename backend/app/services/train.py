from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import settings
from app.models.model_store import ModelStore
from app.schemas.model import TrainRequest
from app.services.evaluation import compute_classification_metrics, reliability_points

matplotlib.use("Agg")

try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    XGB_AVAILABLE = False


def _resolve_dataset_path(dataset_path: str | None) -> Path:
    if dataset_path:
        path = Path(dataset_path)
        if path.is_absolute():
            return path
        return (settings.data_dir.parent / dataset_path).resolve()
    return settings.output_dir / "laliga_enriched_model.csv"


def _load_training_data(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe dataset de entrenamiento en {dataset_path}")

    df = pd.read_csv(dataset_path)
    if "target" not in df.columns:
        from app.services.features import add_target_label

        df = add_target_label(df)

    df["date_dt"] = pd.to_datetime(df.get("date"), errors="coerce")
    df = df.sort_values("date_dt")
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
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not feature_columns:
        raise ValueError("No hay columnas numericas para entrenar")

    X = df[feature_columns].copy()
    y = df["target"].copy()
    return X, y, feature_columns


def _candidate_estimators(use_xgb: bool) -> dict[str, Any]:
    estimators: dict[str, Any] = {
        "logreg": LogisticRegression(max_iter=2000, multi_class="multinomial"),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            max_depth=10,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
    }
    if use_xgb and XGB_AVAILABLE:
        estimators["xgb"] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=4,
        )
    return estimators


def _build_pipeline(name: str, estimator: Any) -> Pipeline:
    if name == "logreg":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def _cross_val_metrics(X: pd.DataFrame, y: pd.Series, model: Pipeline) -> dict[str, float]:
    if len(X) < 60:
        splits = 2
    elif len(X) < 150:
        splits = 3
    else:
        splits = 5

    tss = TimeSeriesSplit(n_splits=splits)
    y_true_parts: list[np.ndarray] = []
    y_prob_parts: list[np.ndarray] = []
    for train_idx, test_idx in tss.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)
        y_true_parts.append(y_test.to_numpy())
        y_prob_parts.append(y_prob)

    y_true = np.concatenate(y_true_parts)
    y_prob = np.concatenate(y_prob_parts)
    return compute_classification_metrics(y_true, y_prob)


def _save_reliability_plot(y_true: np.ndarray, y_prob: np.ndarray) -> str:
    curves = reliability_points(y_true, y_prob, n_bins=10)
    labels = ["H", "D", "A"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    for label, color in zip(labels, colors):
        mean_pred = curves[label]["mean_predicted_value"]
        frac_pos = curves[label]["fraction_positives"]
        if mean_pred and frac_pos:
            plt.plot(mean_pred, frac_pos, marker="o", color=color, label=f"Clase {label}")

    plt.title("Reliability Diagram 1X2")
    plt.xlabel("Probabilidad predicha")
    plt.ylabel("Frecuencia observada")
    plt.legend(loc="best")
    plt.tight_layout()

    static_dir = settings.backend_app_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    output_path = static_dir / "reliability_latest.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return f"/static/{output_path.name}"


def train_and_calibrate(request: TrainRequest) -> dict[str, Any]:
    dataset_path = _resolve_dataset_path(request.dataset_path)
    X, y, feature_columns = _load_training_data(dataset_path)

    estimators = _candidate_estimators(request.use_xgb)
    leaderboard: list[dict[str, Any]] = []

    for model_name, estimator in estimators.items():
        pipeline = _build_pipeline(model_name, estimator)
        metrics = _cross_val_metrics(X, y, pipeline)
        leaderboard.append({"model": model_name, **metrics})

    leaderboard = sorted(leaderboard, key=lambda item: (item["log_loss"], -item["accuracy"]))
    best_model_name = str(leaderboard[0]["model"])

    best_pipeline = _build_pipeline(best_model_name, estimators[best_model_name])
    method = "sigmoid" if request.calibration == "platt" else "isotonic"
    calibrated = CalibratedClassifierCV(estimator=best_pipeline, method=method, cv=3)
    calibrated.fit(X, y)

    train_prob = calibrated.predict_proba(X)
    train_metrics = compute_classification_metrics(y.to_numpy(), train_prob)
    reliability_plot = _save_reliability_plot(y.to_numpy(), train_prob)

    trained_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "trained_at": trained_at,
        "dataset_path": str(dataset_path),
        "feature_columns": feature_columns,
        "rows_trained": int(len(X)),
        "best_model": best_model_name,
        "calibration": request.calibration,
        "metrics": train_metrics,
        "leaderboard": leaderboard,
        "reliability_plot": reliability_plot,
    }
    payload = {
        "model": calibrated,
        "feature_columns": feature_columns,
    }

    store = ModelStore()
    model_path, metadata_path = store.save(payload, metadata)

    return {
        "best_model": best_model_name,
        "metrics": train_metrics,
        "leaderboard": leaderboard,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "reliability_plot": reliability_plot,
    }


def get_active_model_status() -> dict[str, Any]:
    store = ModelStore()
    if not store.model_path.exists() or not store.metadata_path.exists():
        return {"model_available": False, "metadata": None}
    _, metadata = store.load()
    return {"model_available": True, "metadata": metadata}
