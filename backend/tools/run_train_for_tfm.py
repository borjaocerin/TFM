from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas.model import TrainRequest
from app.services.train import train_and_calibrate


def main() -> None:
    request = TrainRequest(
        dataset_path="data/out/laliga_enriched_model.csv",
        use_xgb=False,
        use_catboost=False,
        calibration="platt",
        min_season=2019,
        xg_poss_min_coverage_pct=80.0,
        selection_metric="log_loss",
    )
    result = train_and_calibrate(request)

    print("best_model:", result["best_model"])
    print("selection_metric:", result["selection_metric"])
    print("leaderboard:", [item["model"] for item in result["leaderboard"]])
    print("metrics_report_path:", result["metrics_report_path"])
    print("eda_missing_report_path:", result["eda_missing_report_path"])


if __name__ == "__main__":
    main()
