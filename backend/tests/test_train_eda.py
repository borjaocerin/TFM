from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app.core.config import settings
from app.schemas.model import TrainRequest
from app.services.train import _apply_training_filters, _write_missing_eda_report


def _sample_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2017-09-01",
                "season": 2017,
                "target": 0,
                "xg_last5_home": np.nan,
                "xg_last5_away": np.nan,
                "poss_last5_home": np.nan,
                "poss_last5_away": np.nan,
                "points_last5_home": 7,
            },
            {
                "date": "2019-09-01",
                "season": 2019,
                "target": 1,
                "xg_last5_home": 1.2,
                "xg_last5_away": 0.8,
                "poss_last5_home": 55.0,
                "poss_last5_away": 45.0,
                "points_last5_home": 10,
            },
            {
                "date": "2020-09-01",
                "season": 2020,
                "target": 2,
                "xg_last5_home": 1.1,
                "xg_last5_away": 0.9,
                "poss_last5_home": 52.0,
                "poss_last5_away": 48.0,
                "points_last5_home": 9,
            },
        ]
    )


def test_apply_training_filters_by_season_and_coverage() -> None:
    df = _sample_training_frame()
    request = TrainRequest(
        min_season=2017,
        xg_poss_min_coverage_pct=80.0,
        use_xgb=False,
        use_catboost=False,
    )

    filtered, info = _apply_training_filters(df, request)

    assert len(filtered) == 2
    assert sorted(filtered["season"].unique().tolist()) == [2019.0, 2020.0]
    assert info["rows_before"] == 3
    assert info["rows_after"] == 2
    assert info["seasons_selected"] == [2019.0, 2020.0]


def test_write_missing_eda_report_outputs_json(tmp_path) -> None:
    raw = _sample_training_frame()
    request = TrainRequest(
        min_season=2019,
        use_xgb=False,
        use_catboost=False,
    )
    filtered, info = _apply_training_filters(raw, request)

    old_output = settings.output_dir
    settings.output_dir = tmp_path
    try:
        report_path = _write_missing_eda_report(raw, filtered, tmp_path / "dataset.csv", info)
    finally:
        settings.output_dir = old_output

    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["rows_raw"] == 3
    assert payload["rows_filtered"] == 2
    assert "top_missing_pct_raw" in payload
    assert "xg_poss_coverage_by_season_filtered" in payload
