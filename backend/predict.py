from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

try:
    from .preprocessing import INT_TO_RESULT, build_feature_dataset, build_prediction_row, load_raw_dataset
except ImportError:
    from preprocessing import INT_TO_RESULT, build_feature_dataset, build_prediction_row, load_raw_dataset


MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


def load_model_artifact(model_path: Path | str = MODEL_PATH) -> Dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError("No hay modelo entrenado. Ejecuta /train primero.")
    return joblib.load(path)


def _top_feature_importance(model: Any, feature_columns: List[str], top_k: int = 5) -> List[str]:
    # Fallback interpretable ranking when SHAP is unavailable.
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        imp = np.abs(coef).mean(axis=0)
    elif hasattr(model, "estimator") and hasattr(model.estimator, "feature_importances_"):
        imp = np.asarray(model.estimator.feature_importances_)
    elif hasattr(model, "estimator") and hasattr(model.estimator, "coef_"):
        coef = np.asarray(model.estimator.coef_)
        imp = np.abs(coef).mean(axis=0)
    else:
        return feature_columns[:top_k]

    idx = np.argsort(imp)[::-1][:top_k]
    return [feature_columns[i] for i in idx]


def predict_match(
    home_team: str,
    away_team: str,
    dataset_path: Path | str,
    model_path: Path | str = MODEL_PATH,
) -> Dict[str, Any]:
    artifact = load_model_artifact(model_path)
    model = artifact["model"]
    feature_columns: List[str] = artifact["feature_columns"]

    raw = load_raw_dataset(dataset_path)
    feature_dataset = build_feature_dataset(raw)
    x_one = build_prediction_row(feature_dataset, home_team=home_team, away_team=away_team)
    x_one = x_one[feature_columns]

    proba = model.predict_proba(x_one)[0]
    pred_idx = int(np.argmax(proba))

    response = {
        "home_win": float(proba[0]),
        "draw": float(proba[1]),
        "away_win": float(proba[2]),
        "prediction": INT_TO_RESULT[pred_idx],
        "top_features": _top_feature_importance(model, feature_columns, top_k=8),
    }
    return response
