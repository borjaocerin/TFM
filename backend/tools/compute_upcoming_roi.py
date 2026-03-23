import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append("backend")
from app.services.features import enrich_fixtures


def norm_team(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("fc ", "")
        .replace(" cf", "")
        .replace(" de madrid", "")
        .replace(" vigo", "")
    )


def parse_score_ft(value: Any) -> tuple[float, float] | tuple[None, None]:
    if isinstance(value, list) and len(value) >= 2:
        try:
            home = float(value[0]) if value[0] is not None else None
            away = float(value[1]) if value[1] is not None else None
            return home, away
        except Exception:
            return None, None

    if isinstance(value, str) and "-" in value:
        left, right = value.split("-", 1)
        try:
            return float(left.strip()), float(right.strip())
        except Exception:
            return None, None

    return None, None


def result_from_goals(home_goals: Any, away_goals: Any) -> str | None:
    try:
        home = float(home_goals)
        away = float(away_goals)
    except Exception:
        return None
    if home > away:
        return "H"
    if home < away:
        return "A"
    return "D"


def _season_label_for_results(reference: pd.Timestamp | None = None) -> str:
    today = reference or pd.Timestamp.now("UTC")
    year = int(today.year)
    month = int(today.month)
    start_year = year if month >= 7 else year - 1
    return f"{start_year}-{start_year + 1}"


def _extract_results_from_sportsdb_payload(payload: Any) -> pd.DataFrame:
    events = []
    if isinstance(payload, dict):
        possible = payload.get("events")
        if isinstance(possible, list):
            events = possible
    if not events:
        return pd.DataFrame(columns=["_key", "actual_result", "source"])

    rows: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue

        date_value = event.get("dateEvent") or event.get("strTimestamp") or event.get("date")
        home_team = event.get("strHomeTeam") or event.get("home_team")
        away_team = event.get("strAwayTeam") or event.get("away_team")
        home_score = event.get("intHomeScore")
        away_score = event.get("intAwayScore")

        if date_value is None or home_team is None or away_team is None:
            continue

        date_dt = pd.to_datetime(date_value, errors="coerce", utc=True)
        if pd.isna(date_dt):
            continue

        result = result_from_goals(home_score, away_score)
        if result is None:
            continue

        key = (
            date_dt.tz_localize(None).strftime("%Y-%m-%d")
            + "|"
            + norm_team(home_team)
            + "|"
            + norm_team(away_team)
        )
        rows.append({"_key": key, "actual_result": result, "source": "api:sportsdb"})

    if not rows:
        return pd.DataFrame(columns=["_key", "actual_result", "source"])

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["_key"], keep="last")
    return out


def _fetch_external_results() -> pd.DataFrame:
    season = _season_label_for_results()
    urls = [
        f"https://www.thesportsdb.com/api/v1/json/3/eventsseason.php?id=4335&s={season}",
        "https://www.thesportsdb.com/api/v1/json/3/eventspastleague.php?id=4335",
    ]

    frames: list[pd.DataFrame] = []
    for url in urls:
        try:
            req = Request(url, headers={"Accept": "application/json"}, method="GET")
            with urlopen(req, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
            parsed = _extract_results_from_sportsdb_payload(payload)
            if not parsed.empty:
                frames.append(parsed)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["_key", "actual_result", "source"])

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["_key"], keep="last")
    return merged


def _extract_results_from_odds_scores_payload(payload: Any) -> pd.DataFrame:
    if not isinstance(payload, list):
        return pd.DataFrame(columns=["_key", "actual_result", "source"])

    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue

        completed = bool(item.get("completed", False))
        if not completed:
            continue

        home_team = item.get("home_team")
        away_team = item.get("away_team")
        commence_time = item.get("commence_time")
        scores = item.get("scores")
        if home_team is None or away_team is None or commence_time is None or not isinstance(scores, list):
            continue

        home_score = None
        away_score = None
        for score_item in scores:
            if not isinstance(score_item, dict):
                continue
            name = norm_team(score_item.get("name", ""))
            score_value = score_item.get("score")
            if name == norm_team(home_team):
                home_score = score_value
            elif name == norm_team(away_team):
                away_score = score_value

        result = result_from_goals(home_score, away_score)
        if result is None:
            continue

        date_dt = pd.to_datetime(commence_time, errors="coerce", utc=True)
        if pd.isna(date_dt):
            continue

        key = (
            date_dt.tz_localize(None).strftime("%Y-%m-%d")
            + "|"
            + norm_team(home_team)
            + "|"
            + norm_team(away_team)
        )
        rows.append({"_key": key, "actual_result": result, "source": "api:odds-scores"})

    if not rows:
        return pd.DataFrame(columns=["_key", "actual_result", "source"])
    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["_key"], keep="last")
    return out


def _fetch_odds_api_scores_results() -> pd.DataFrame:
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        fallback = Path("oddapikey.txt")
        if fallback.exists():
            api_key = fallback.read_text(encoding="utf-8").strip()
    if not api_key:
        return pd.DataFrame(columns=["_key", "actual_result", "source"])

    sport_key = os.getenv("ODDS_API_SPORT_KEY", "soccer_spain_la_liga").strip()
    base_url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores"
    params = {
        "apiKey": api_key,
        "daysFrom": "14",
        "dateFormat": "iso",
    }
    url = f"{base_url}?{urlencode(params)}"

    try:
        req = Request(url, headers={"Accept": "application/json"}, method="GET")
        timeout_sec = int(os.getenv("ODDS_API_TIMEOUT_SEC", "15") or "15")
        with urlopen(req, timeout=max(15, timeout_sec)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return pd.DataFrame(columns=["_key", "actual_result", "source"])

    return _extract_results_from_odds_scores_payload(payload)


def write_roi_readme(out_dir: Path) -> None:
    readme = f"""# ROI tracking (apuestas jornada)

Este directorio guarda:

- `roi_upcoming_summary.json`: resumen de ROI esperado y (si hay resultados) ROI real.
- `roi_upcoming_detail.csv`: detalle por partido (EV, pick, stake, bankroll esperado).
- `roi_expected_profit_by_match.png`: gráfico beneficio esperado por partido.
- `roi_cumulative_bankroll.png`: gráfico bankroll acumulado esperado.
- `settlement_realized.csv`: liquidación real (ganado/perdido) cuando existan marcadores finales.
- `odds_snapshots/`: snapshots históricos de cuotas descargadas.
- `update_roi_jornada.bat`: ejecución rápida (doble clic) para recalcular todo.
- Integración automática de resultados reales desde API (TheSportsDB) para no cargar marcadores a mano.

## Qué decirle al asistente para actualizar todo

Usa este mensaje:

"Actualiza ROI de próximas jornadas: vuelve a leer cuotas, recalcula predicciones con el modelo, regenera CSV y gráficos, y liquida si ya hay resultados."

## Comando directo (terminal)

```bash
python backend/tools/compute_upcoming_roi.py
```

## Automatizar para cada jornada (Windows)

1. Ejecuta manualmente con doble clic:

```bat
backend\\tools\\update_roi_jornada.bat
```

2. Para programarlo automáticamente cada día a las 09:00:

```powershell
powershell -ExecutionPolicy Bypass -File backend/tools/register_weekly_roi_task.ps1
```

Esto crea la tarea `TFM_ROI_Update_Jornada` en el Programador de tareas.

## Cómo ver si habríamos ganado o perdido esta jornada

1. Revisa `settlement_realized.csv`.
2. Columnas clave:
   - `bet_won` (`True`/`False`)
   - `realized_profit_eur` (beneficio por apuesta: gana `odds-1`, pierde `-1`)
   - `realized_bankroll_cumulative_eur`
3. En `roi_upcoming_summary.json` verás:
    - `next_jornada_expected_bankroll_eur`
    - `next_jornada_realized_bankroll_eur`
   - `settled_bets`
   - `realized_total_profit_eur`
   - `realized_roi`

Si todavía no hay resultados finales en `data/fixtures/proximosPartidos.json`, esas métricas saldrán como `null`.

## Fuente de resultados reales (automática)

1. Primero intenta leer resultados reales desde API externa.
2. Después aplica cualquier resultado manual de `data/fixtures/proximosPartidos.json` (si existe) como prioridad final.

Si no hay Internet o la API no responde, la liquidación seguirá funcionando con resultados manuales.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    root = Path(".")

    meta = json.loads((root / "backend/app/models/store/metadata.json").read_text(encoding="utf-8"))
    features = meta["feature_columns"]

    train_df = pd.read_csv(root / "data/out/laliga_enriched_model.csv")
    for column in features:
        if column not in train_df.columns:
            train_df[column] = np.nan

    x_train = train_df[features].apply(pd.to_numeric, errors="coerce").astype(float)
    y_train = pd.to_numeric(train_df["target"], errors="coerce").astype("Int64")
    valid_mask = y_train.notna()
    x_train = x_train[valid_mask]
    y_train = y_train[valid_mask].astype(int)

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    model.fit(x_train, y_train)

    odds = pd.read_csv(root / "data/out/laliga_upcoming_odds.csv")
    fixtures = odds[["date", "home_team", "away_team"]].drop_duplicates().copy()

    team_map_path = root / "etl/team_name_map_es.json"
    team_map = json.loads(team_map_path.read_text(encoding="utf-8")) if team_map_path.exists() else {}

    historical = pd.read_csv(root / "data/historical/laliga_merged_matches.csv")
    enriched = enrich_fixtures(fixtures, historical, (5, 10), team_map)

    for column in features:
        if column not in enriched.columns:
            enriched[column] = np.nan

    x_pred = enriched[features].apply(pd.to_numeric, errors="coerce").astype(float)
    probabilities = model.predict_proba(x_pred)

    predictions = enriched[["date", "home_team", "away_team"]].copy()
    predictions["p_H"] = probabilities[:, 0]
    predictions["p_D"] = probabilities[:, 1]
    predictions["p_A"] = probabilities[:, 2]

    for frame in (odds, predictions):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame["_key"] = frame["date"].astype(str) + "|" + frame["home_team"].map(norm_team) + "|" + frame["away_team"].map(norm_team)

    merged = odds.merge(predictions[["_key", "p_H", "p_D", "p_A"]], on="_key", how="left")

    for side, label in [("h", "H"), ("d", "D"), ("a", "A")]:
        merged[f"ev_{label}"] = pd.to_numeric(merged[f"odds_avg_{side}"], errors="coerce") * pd.to_numeric(
            merged[f"p_{label}"], errors="coerce"
        ) - 1.0

    ev_matrix = merged[["ev_H", "ev_D", "ev_A"]].to_numpy(dtype=float)
    ev_safe = np.where(np.isnan(ev_matrix), -np.inf, ev_matrix)
    best_index = np.argmax(ev_safe, axis=1)
    labels = np.array(["H", "D", "A"])

    merged["best_pick"] = labels[best_index]
    best_ev = np.take_along_axis(ev_safe, best_index[:, None], axis=1).reshape(-1)
    best_ev = np.where(np.isneginf(best_ev), np.nan, best_ev)
    merged["best_ev"] = best_ev

    merged = merged.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

    stake_per_bet = 1.0
    merged["stake_eur"] = stake_per_bet
    merged["expected_profit_eur"] = pd.to_numeric(merged["best_ev"], errors="coerce") * stake_per_bet
    merged["expected_return_eur"] = merged["stake_eur"] + merged["expected_profit_eur"]
    merged["bet_selected"] = pd.to_numeric(merged["best_ev"], errors="coerce") > 0.0
    merged["bet_selected_ev_gt_2pct"] = pd.to_numeric(merged["best_ev"], errors="coerce") > 0.02
    merged["expected_bankroll_cumulative_eur"] = pd.to_numeric(
        merged["expected_profit_eur"], errors="coerce"
    ).fillna(0.0).cumsum()

    best_ev_series = pd.to_numeric(merged["best_ev"], errors="coerce")
    filter_ev_2 = best_ev_series > 0.02

    out_dir = root / "out" / "roi"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Snapshots de cuotas (proximas jornadas + siguiente jornada)
    snapshot_dir = out_dir / "odds_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%SZ")
    odds_snapshot_path = snapshot_dir / f"laliga_upcoming_odds_{timestamp}.csv"
    odds.to_csv(odds_snapshot_path, index=False)

    next_jornada_path = None
    next_jornada_keys: set[str] = set()
    odds_with_date = odds.copy()
    odds_with_date["date_dt"] = pd.to_datetime(odds_with_date["date"], errors="coerce")
    if odds_with_date["date_dt"].notna().any():
        first_date = odds_with_date.loc[odds_with_date["date_dt"].notna(), "date_dt"].min()
        date_limit = first_date + pd.Timedelta(days=3)
        next_jornada = odds_with_date[
            (odds_with_date["date_dt"] >= first_date) & (odds_with_date["date_dt"] <= date_limit)
        ].copy()
        if "_key" in next_jornada.columns:
            next_jornada_keys = set(next_jornada["_key"].dropna().astype(str).tolist())
        next_jornada = next_jornada.drop(columns=["date_dt"])
        next_jornada_path = snapshot_dir / f"next_jornada_odds_{timestamp}.csv"
        next_jornada.to_csv(next_jornada_path, index=False)

    merged["is_next_jornada"] = merged["_key"].astype(str).isin(next_jornada_keys)

    # Liquidacion real si hay marcadores finales en proximosPartidos.json
    realized_total_profit = None
    realized_roi = None
    settled_bets = 0
    settlement_df = merged.copy()
    settlement_df["actual_result"] = None
    settlement_df["bet_won"] = pd.Series([None] * len(settlement_df), dtype="object")
    settlement_df["realized_profit_eur"] = np.nan
    settlement_df["realized_bankroll_cumulative_eur"] = np.nan

    # Fuentes externas automaticas de resultados reales
    external_results_sportsdb = _fetch_external_results()
    external_results_odds_scores = _fetch_odds_api_scores_results()
    external_results = pd.concat(
        [external_results_odds_scores, external_results_sportsdb],
        ignore_index=True,
    )
    if not external_results.empty:
        external_results = external_results.drop_duplicates(subset=["_key"], keep="first")

    external_results_count = int(len(external_results))
    external_results_used = False
    if not external_results.empty:
        settlement_df = settlement_df.merge(
            external_results[["_key", "actual_result"]],
            on="_key",
            how="left",
            suffixes=("", "_external"),
        )
        if "actual_result_external" in settlement_df.columns:
            settlement_df["actual_result"] = settlement_df["actual_result"].where(
                settlement_df["actual_result"].notna(), settlement_df["actual_result_external"]
            )
            settlement_df = settlement_df.drop(columns=["actual_result_external"])
            external_results_used = settlement_df["actual_result"].notna().any()

    manual_results_path = root / "data/fixtures/proximosPartidos.json"
    if manual_results_path.exists():
        manual_payload = json.loads(manual_results_path.read_text(encoding="utf-8"))
        matches = manual_payload.get("matches", []) if isinstance(manual_payload, dict) else []
        manual_rows: list[dict[str, Any]] = []
        for match in matches:
            if not isinstance(match, dict):
                continue
            date_value = match.get("date")
            team1 = match.get("team1")
            team2 = match.get("team2")
            score = match.get("score") or {}
            ft = score.get("ft") if isinstance(score, dict) else None
            home_goals, away_goals = parse_score_ft(ft)
            actual_result = result_from_goals(home_goals, away_goals)
            if actual_result is None:
                continue
            date_iso = pd.to_datetime(date_value, errors="coerce")
            if pd.isna(date_iso):
                continue
            manual_rows.append(
                {
                    "_key": date_iso.strftime("%Y-%m-%d") + "|" + norm_team(team1) + "|" + norm_team(team2),
                    "actual_result": actual_result,
                }
            )

        if manual_rows:
            manual_df = pd.DataFrame(manual_rows).drop_duplicates(subset=["_key"], keep="last")
            settlement_df = settlement_df.merge(manual_df, on="_key", how="left", suffixes=("", "_manual"))
            if "actual_result_manual" in settlement_df.columns:
                settlement_df["actual_result"] = settlement_df["actual_result_manual"]
                settlement_df = settlement_df.drop(columns=["actual_result_manual"])

            settlement_mask = settlement_df["actual_result"].notna() & settlement_df["best_pick"].notna()
            settlement_df.loc[settlement_mask, "bet_won"] = (
                settlement_df.loc[settlement_mask, "best_pick"] == settlement_df.loc[settlement_mask, "actual_result"]
            )

            won_mask = settlement_mask & settlement_df["bet_won"].fillna(False).astype(bool)
            lost_mask = settlement_mask & (~settlement_df["bet_won"].fillna(False).astype(bool))

            settlement_df.loc[won_mask, "realized_profit_eur"] = np.select(
                [
                    settlement_df.loc[won_mask, "best_pick"] == "H",
                    settlement_df.loc[won_mask, "best_pick"] == "D",
                    settlement_df.loc[won_mask, "best_pick"] == "A",
                ],
                [
                    pd.to_numeric(settlement_df.loc[won_mask, "odds_avg_h"], errors="coerce") - 1.0,
                    pd.to_numeric(settlement_df.loc[won_mask, "odds_avg_d"], errors="coerce") - 1.0,
                    pd.to_numeric(settlement_df.loc[won_mask, "odds_avg_a"], errors="coerce") - 1.0,
                ],
                default=np.nan,
            )
            settlement_df.loc[lost_mask, "realized_profit_eur"] = -1.0

            settlement_df = settlement_df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
            settlement_df["realized_bankroll_cumulative_eur"] = pd.to_numeric(
                settlement_df["realized_profit_eur"], errors="coerce"
            ).fillna(0.0).cumsum()

            settled_bets = int(settlement_mask.sum())
            if settled_bets > 0:
                realized_total_profit = float(
                    pd.to_numeric(settlement_df.loc[settlement_mask, "realized_profit_eur"], errors="coerce").sum(skipna=True)
                )
                realized_roi = realized_total_profit / settled_bets

    next_jornada_mask = merged["is_next_jornada"].fillna(False).astype(bool)
    next_jornada_ev = pd.to_numeric(merged.loc[next_jornada_mask, "best_ev"], errors="coerce")
    next_jornada_expected_profit = float(next_jornada_ev.sum(skipna=True)) if int(next_jornada_mask.sum()) > 0 else 0.0
    next_jornada_expected_roi = float(next_jornada_ev.mean(skipna=True)) if int(next_jornada_mask.sum()) > 0 else None

    settled_next_jornada_mask = (
        settlement_df["is_next_jornada"].fillna(False).astype(bool)
        & settlement_df["actual_result"].notna()
        & settlement_df["best_pick"].notna()
    )
    settled_next_jornada = int(settled_next_jornada_mask.sum())
    next_jornada_realized_profit = (
        float(
            pd.to_numeric(
                settlement_df.loc[settled_next_jornada_mask, "realized_profit_eur"],
                errors="coerce",
            ).sum(skipna=True)
        )
        if settled_next_jornada > 0
        else None
    )
    next_jornada_realized_roi = (
        (next_jornada_realized_profit / settled_next_jornada)
        if settled_next_jornada > 0 and next_jornada_realized_profit is not None
        else None
    )

    summary = {
        "rows_upcoming_odds": int(len(odds)),
        "rows_with_model_prediction": int(merged["p_H"].notna().sum()),
        "stake_per_bet_eur": stake_per_bet,
        "strategy": "Apostar al signo con mayor EV (odds_avg x prob_model - 1)",
        "expected_total_profit_eur": float(best_ev_series.sum(skipna=True)),
        "expected_roi": float(best_ev_series.mean(skipna=True)),
        "bets_ev_gt_2pct": int(filter_ev_2.sum()),
        "expected_profit_ev_gt_2pct_eur": float(best_ev_series[filter_ev_2].sum(skipna=True)),
        "expected_roi_ev_gt_2pct": float(best_ev_series[filter_ev_2].mean(skipna=True)) if int(filter_ev_2.sum()) > 0 else None,
        "expected_bankroll_final_eur": float(
            pd.to_numeric(merged["expected_bankroll_cumulative_eur"], errors="coerce").iloc[-1]
        )
        if len(merged)
        else 0.0,
        "next_jornada_bets": int(next_jornada_mask.sum()),
        "next_jornada_expected_profit_eur": next_jornada_expected_profit,
        "next_jornada_expected_roi": next_jornada_expected_roi,
        "next_jornada_expected_bankroll_eur": next_jornada_expected_profit,
        "next_jornada_settled_bets": settled_next_jornada,
        "next_jornada_realized_profit_eur": next_jornada_realized_profit,
        "next_jornada_realized_roi": next_jornada_realized_roi,
        "next_jornada_realized_bankroll_eur": next_jornada_realized_profit,
        "external_results_used": bool(external_results_used),
        "external_results_rows": external_results_count,
        "odds_snapshot_path": str(odds_snapshot_path).replace("\\", "/"),
        "next_jornada_odds_snapshot_path": str(next_jornada_path).replace("\\", "/") if next_jornada_path else None,
        "settled_bets": settled_bets,
        "realized_total_profit_eur": realized_total_profit,
        "realized_roi": realized_roi,
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
    }

    (out_dir / "roi_upcoming_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    keep = [
        "date",
        "home_team",
        "away_team",
        "odds_avg_h",
        "odds_avg_d",
        "odds_avg_a",
        "p_H",
        "p_D",
        "p_A",
        "ev_H",
        "ev_D",
        "ev_A",
        "best_pick",
        "best_ev",
        "stake_eur",
        "expected_profit_eur",
        "expected_return_eur",
        "bet_selected",
        "bet_selected_ev_gt_2pct",
        "is_next_jornada",
        "expected_bankroll_cumulative_eur",
    ]
    merged[keep].to_csv(out_dir / "roi_upcoming_detail.csv", index=False)

    settlement_keep = [
        "date",
        "home_team",
        "away_team",
        "best_pick",
        "actual_result",
        "bet_won",
        "realized_profit_eur",
        "is_next_jornada",
        "realized_bankroll_cumulative_eur",
    ]
    settlement_keep = [column for column in settlement_keep if column in settlement_df.columns]
    settlement_df[settlement_keep].to_csv(out_dir / "settlement_realized.csv", index=False)

    # Grafico 1: beneficio esperado por partido
    plot_df = merged.copy()
    plot_df["fixture"] = plot_df["home_team"].astype(str) + " vs " + plot_df["away_team"].astype(str)
    plot_df["expected_profit_eur"] = pd.to_numeric(plot_df["expected_profit_eur"], errors="coerce")
    plot_df = plot_df.dropna(subset=["expected_profit_eur"])

    if not plot_df.empty:
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        colors = ["#2ca02c" if value >= 0 else "#d62728" for value in plot_df["expected_profit_eur"]]
        ax1.barh(plot_df["fixture"], plot_df["expected_profit_eur"], color=colors)
        ax1.axvline(0, color="black", linewidth=1)
        ax1.set_title("Beneficio esperado por partido (1€ stake)")
        ax1.set_xlabel("Beneficio esperado (€)")
        ax1.set_ylabel("Partido")
        fig1.tight_layout()
        fig1.savefig(out_dir / "roi_expected_profit_by_match.png", dpi=150)
        plt.close(fig1)

    # Grafico 2: bankroll acumulado esperado
    bankroll_df = merged.copy()
    bankroll_df["date_dt"] = pd.to_datetime(bankroll_df["date"], errors="coerce")
    bankroll_df = bankroll_df.sort_values(["date_dt", "home_team", "away_team"]).reset_index(drop=True)
    bankroll_df["bet_index"] = np.arange(1, len(bankroll_df) + 1)

    if not bankroll_df.empty:
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(
            bankroll_df["bet_index"],
            pd.to_numeric(bankroll_df["expected_bankroll_cumulative_eur"], errors="coerce").fillna(0.0),
            marker="o",
            linewidth=2,
        )
        ax2.axhline(0, color="black", linewidth=1)
        ax2.set_title("Bankroll acumulado esperado")
        ax2.set_xlabel("Número de apuesta")
        ax2.set_ylabel("Bankroll esperado (€)")
        fig2.tight_layout()
        fig2.savefig(out_dir / "roi_cumulative_bankroll.png", dpi=150)
        plt.close(fig2)

    write_roi_readme(out_dir)

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
