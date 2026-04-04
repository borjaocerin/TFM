from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append("backend")

from app.services.odds_api import fetch_upcoming_laliga_odds, load_team_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga cuotas upcoming de LaLiga y guarda snapshot + historico CSV."
    )
    parser.add_argument("--limit", type=int, default=200, help="Numero maximo de partidos a guardar")
    args = parser.parse_args()

    team_map = load_team_map()
    result = fetch_upcoming_laliga_odds(team_map=team_map, limit=max(1, int(args.limit)))

    summary = {
        "rows": int(result.get("rows", 0)),
        "output_csv": result.get("output_csv"),
        "history_csv": result.get("history_csv"),
        "fetched_at_utc": result.get("fetched_at_utc"),
        "requests_remaining": result.get("requests_remaining"),
        "requests_used": result.get("requests_used"),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
