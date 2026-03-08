from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .preprocessing import INT_TO_RESULT, build_feature_dataset, load_raw_dataset, split_xy
except ImportError:
    from preprocessing import INT_TO_RESULT, build_feature_dataset, load_raw_dataset, split_xy

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False


MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


def _ece_score(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    confidence = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)

    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        lo, hi = boundaries[i], boundaries[i + 1]
        mask = (confidence >= lo) & (confidence < hi)
        if not np.any(mask):
            continue
        acc = correctness[mask].mean()
        conf = confidence[mask].mean()
        ece += np.abs(acc - conf) * (mask.sum() / len(y_true))
    return float(ece)


def _multiclass_brier_score(y_true: np.ndarray, proba: np.ndarray, n_classes: int = 3) -> float:
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def _get_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        multi_class="multinomial",
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
        )
    return models


def _evaluate_model_time_series(model: Any, x: np.ndarray, y: np.ndarray, splits: int = 5) -> Dict[str, float]:
    tscv = TimeSeriesSplit(n_splits=splits)
    acc_list: List[float] = []
    f1_list: List[float] = []
    ll_list: List[float] = []
    brier_list: List[float] = []
    ece_list: List[float] = []

    for train_idx, test_idx in tscv.split(x):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(x_train, y_train)
        proba = model.predict_proba(x_test)
        pred = np.argmax(proba, axis=1)

        acc_list.append(accuracy_score(y_test, pred))
        f1_list.append(f1_score(y_test, pred, average="macro"))
        ll_list.append(log_loss(y_test, proba, labels=[0, 1, 2]))
        brier_list.append(_multiclass_brier_score(y_test.to_numpy(), proba, n_classes=3))
        ece_list.append(_ece_score(y_test.to_numpy(), proba))

    return {
        "accuracy": float(np.mean(acc_list)),
        "f1_macro": float(np.mean(f1_list)),
        "log_loss": float(np.mean(ll_list)),
        "brier_score": float(np.mean(brier_list)),
        "ece": float(np.mean(ece_list)),
    }


def train_and_save_model(
    dataset_path: Path | str,
    model_path: Path | str = MODEL_PATH,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    raw = load_raw_dataset(dataset_path)
    features_df = build_feature_dataset(raw)
    x, y, feature_columns = split_xy(features_df)

    models = _get_models()
    metrics_table: List[Dict[str, Any]] = []

    for name, model in models.items():
        scores = _evaluate_model_time_series(model, x, y)
        row = {"model": name, **scores}
        metrics_table.append(row)

    metrics_table = sorted(metrics_table, key=lambda m: (m["log_loss"], -m["accuracy"]))
    best_name = metrics_table[0]["model"]
    best_model = models[best_name]

    calibrated = CalibratedClassifierCV(best_model, method="sigmoid", cv=3)
    calibrated.fit(x, y)

    payload = {
        "model": calibrated,
        "best_model_name": best_name,
        "metrics": metrics_table,
        "feature_columns": feature_columns,
        "class_mapping": INT_TO_RESULT,
        "training_rows": int(len(x)),
    }

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)

    return payload, metrics_table
