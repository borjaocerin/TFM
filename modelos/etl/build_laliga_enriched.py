#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.datasets import DatasetIngestRequest, FixturesFeatureRequest
from modelos.services.datasets import build_fixtures_features, ingest_datasets


def _parse_windows(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    values = sorted(set(value for value in values if value > 0))
    if not values:
        raise ValueError("--windows debe tener al menos un entero positivo")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye datasets enriquecidos de LaLiga")
    parser.add_argument(
        "--hist",
        required=True,
        help="Ruta al CSV historico base (ej. data/historical/laliga_merged_matches.csv)",
    )
    parser.add_argument(
        "--fdata_dir",
        required=True,
        help="Directorio con CSV por temporada de football-data",
    )
    parser.add_argument("--elo", default="", help="Ruta opcional a data/elo/ELO_RATINGS.csv")
    parser.add_argument("--fixtures", default="", help="Ruta opcional a data/fixtures/fixtures.csv")
    parser.add_argument(
        "--team_map",
        default="etl/team_name_map_es.json",
        help="Ruta al JSON de normalizacion de nombres de equipo",
    )
    parser.add_argument("--windows", default="5,10", help="Ventanas rolling. Ejemplo: 5,10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = _parse_windows(args.windows)

    ingest_request = DatasetIngestRequest(
        historical=args.hist,
        football_data_dir=args.fdata_dir,
        elo_csv=args.elo or None,
        team_map=args.team_map,
        windows=windows,
    )

    ingest_summary = ingest_datasets(ingest_request)
    response: dict[str, object] = {"ingest": ingest_summary}

    if args.fixtures:
        fixtures_request = FixturesFeatureRequest(
            fixtures_csv=args.fixtures,
            historical_csv=ingest_summary["output_all"],
            elo_csv=args.elo or None,
            team_map=args.team_map,
            windows=windows,
        )
        response["fixtures"] = build_fixtures_features(fixtures_request)

    print(json.dumps(response, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
