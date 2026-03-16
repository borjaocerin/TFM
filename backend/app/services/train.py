from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import settings
from app.models.model_store import ModelStore
from app.schemas.datasets import DatasetIngestRequest
from app.schemas.model import TrainRequest
from app.services.datasets import ingest_datasets
from app.services.evaluation import compute_classification_metrics, reliability_points

matplotlib.use("Agg")

try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    CATBOOST_AVAILABLE = False


LEAKAGE_COLUMNS = {
    "xg_home",
    "xg_away",
    "xga_home",
    "xga_away",
    "poss_home",
    "poss_away",
    "sh_home",
    "sh_away",
    "sot_home",
    "sot_away",
    "xg_diff",
    "xga_diff",
    "poss_diff",
    "sh_diff",
    "sot_diff",
    "goal_diff",
}


def _resolve_dataset_path(dataset_path: str | None) -> Path:
    if dataset_path:
        path = Path(dataset_path)
        if path.is_absolute():
            return path
        return (settings.data_dir.parent / dataset_path).resolve()

    default_model_path = settings.output_dir / "laliga_enriched_model.csv"
    if default_model_path.exists():
        return default_model_path

    historical_csv = settings.data_dir / "historical" / "laliga_merged_matches.csv"
    football_data_dir = settings.data_dir / "football-data"
    elo_csv = settings.data_dir / "elo" / "ELO_RATINGS.csv"
    team_map = settings.data_dir.parent / "etl" / "team_name_map_es.json"

    if not historical_csv.exists():
        raise FileNotFoundError(
            f"No existe historico base en {historical_csv}. Colocalo para entrenar."
        )
    if not football_data_dir.exists():
        raise FileNotFoundError(
            f"No existe directorio football-data en {football_data_dir}."
        )

    ingest_datasets(
        DatasetIngestRequest(
            historical=str(historical_csv),
            football_data_dir=str(football_data_dir),
            elo_csv=str(elo_csv) if elo_csv.exists() else None,
            team_map=str(team_map) if team_map.exists() else None,
            windows=[5, 10],
        )
    )
    return default_model_path


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
    excluded = excluded | LEAKAGE_COLUMNS

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(df[column])
        and df[column].notna().any()
    ]
    if not feature_columns:
        raise ValueError("No hay columnas numericas para entrenar")

    X = df[feature_columns].copy()
    y = df["target"].copy()
    return X, y, feature_columns


def _candidate_estimators(use_xgb: bool, use_catboost: bool) -> dict[str, Any]:
    estimators: dict[str, Any] = {
        "logreg": LogisticRegression(max_iter=2000, multi_class="multinomial"),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            max_depth=10,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=600,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=400,
            min_samples_leaf=20,
            random_state=42,
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
    if use_catboost and CATBOOST_AVAILABLE:
        estimators["catboost"] = CatBoostClassifier(
            iterations=700,
            learning_rate=0.03,
            depth=6,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
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


def _write_metrics_report(
    trained_at: str,
    best_model_name: str,
    cv_metrics: dict[str, float],
    leaderboard: list[dict[str, Any]],
    dataset_path: Path,
    fit_metrics: dict[str, float] | None = None,
) -> Path:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = settings.output_dir / "model_metrics.txt"

    lines: list[str] = [
        "MODEL METRICS REPORT",
        "====================",
        f"trained_at_utc: {trained_at}",
        f"dataset_path: {dataset_path}",
        f"best_model: {best_model_name}",
        "",
        "GLOBAL METRICS (TIME-SPLIT CV)",
        f"log_loss: {cv_metrics.get('log_loss', 0.0):.6f}",
        f"brier: {cv_metrics.get('brier', 0.0):.6f}",
        f"ece: {cv_metrics.get('ece', 0.0):.6f}",
        f"accuracy: {cv_metrics.get('accuracy', 0.0):.6f}",
        f"f1_macro: {cv_metrics.get('f1_macro', 0.0):.6f}",
        "",
        "LEADERBOARD",
    ]

    if fit_metrics is not None:
        lines.extend(
            [
                "",
                "FIT METRICS (IN-SAMPLE, SOLO REFERENCIA)",
                f"fit_log_loss: {fit_metrics.get('log_loss', 0.0):.6f}",
                f"fit_brier: {fit_metrics.get('brier', 0.0):.6f}",
                f"fit_ece: {fit_metrics.get('ece', 0.0):.6f}",
                f"fit_accuracy: {fit_metrics.get('accuracy', 0.0):.6f}",
                f"fit_f1_macro: {fit_metrics.get('f1_macro', 0.0):.6f}",
            ]
        )

    for item in leaderboard:
        lines.append(
            " - "
            + f"{item['model']}: "
            + f"log_loss={float(item['log_loss']):.6f}, "
            + f"accuracy={float(item['accuracy']):.6f}, "
            + f"brier={float(item['brier']):.6f}, "
            + f"ece={float(item['ece']):.6f}, "
            + f"f1_macro={float(item['f1_macro']):.6f}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def train_and_calibrate(request: TrainRequest) -> dict[str, Any]:
    dataset_path = _resolve_dataset_path(request.dataset_path)
    X, y, feature_columns = _load_training_data(dataset_path)

    estimators = _candidate_estimators(request.use_xgb, request.use_catboost)
    leaderboard: list[dict[str, Any]] = []

    for model_name, estimator in estimators.items():
        pipeline = _build_pipeline(model_name, estimator)
        metrics = _cross_val_metrics(X, y, pipeline)
        leaderboard.append({"model": model_name, **metrics})

    leaderboard = sorted(
        leaderboard,
        key=lambda item: (-item["accuracy"], item["log_loss"], -item["f1_macro"], item["brier"]),
    )
    best_model_name = str(leaderboard[0]["model"])
    best_cv_metrics = {
        key: float(value)
        for key, value in leaderboard[0].items()
        if key != "model"
    }

    best_pipeline = _build_pipeline(best_model_name, estimators[best_model_name])
    method = "sigmoid" if request.calibration == "platt" else "isotonic"
    calibrated = CalibratedClassifierCV(estimator=best_pipeline, method=method, cv=3)
    calibrated.fit(X, y)

    fit_prob = calibrated.predict_proba(X)
    fit_metrics = compute_classification_metrics(y.to_numpy(), fit_prob)
    reliability_plot = _save_reliability_plot(y.to_numpy(), fit_prob)

    trained_at = datetime.now(timezone.utc).isoformat()
    metrics_report_path = _write_metrics_report(
        trained_at=trained_at,
        best_model_name=best_model_name,
        cv_metrics=best_cv_metrics,
        leaderboard=leaderboard,
        dataset_path=dataset_path,
        fit_metrics=fit_metrics,
    )

    metadata = {
        "trained_at": trained_at,
        "dataset_path": str(dataset_path),
        "feature_columns": feature_columns,
        "rows_trained": int(len(X)),
        "best_model": best_model_name,
        "calibration": request.calibration,
        "metrics": best_cv_metrics,
        "fit_metrics": fit_metrics,
        "leaderboard": leaderboard,
        "reliability_plot": reliability_plot,
        "metrics_report_path": str(metrics_report_path),
    }
    payload = {
        "model": calibrated,
        "feature_columns": feature_columns,
    }

    store = ModelStore()
    model_path, metadata_path = store.save(payload, metadata)

    return {
        "best_model": best_model_name,
        "metrics": best_cv_metrics,
        "leaderboard": leaderboard,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "reliability_plot": reliability_plot,
        "metrics_report_path": str(metrics_report_path),
    }


def get_active_model_status() -> dict[str, Any]:
    store = ModelStore()
    if not store.model_path.exists() or not store.metadata_path.exists():
        return {"model_available": False, "metadata": None}
    _, metadata = store.load()
    return {"model_available": True, "metadata": metadata}
