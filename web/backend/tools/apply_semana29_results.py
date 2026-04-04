import json
from pathlib import Path

import pandas as pd

root = Path(".")
detail_path = root / "out/roi/roi_upcoming_detail.csv"
settle_path = root / "out/roi/settlement_realized.csv"
summary_path = root / "out/roi/roi_upcoming_summary.json"
txt_detail_path = root / "out/roi/resultados_jornada_semana29_detalle.txt"

results = {
    ("Villarreal", "Real Sociedad"): (3, 1),
    ("Elche", "Mallorca"): (2, 1),
    ("Espanyol", "Getafe"): (1, 2),
    ("Osasuna", "Girona"): (1, 0),
    ("Levante", "Oviedo"): (4, 2),
    ("Sevilla", "Valencia"): (0, 2),
    ("Barcelona", "Rayo Vallecano"): (1, 0),
    ("Celta Vigo", "Alaves"): (3, 4),
    ("Athletic Club", "Real Betis"): (2, 1),
    ("Real Madrid", "Atletico Madrid"): (3, 2),
}


def result_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


detail_df = pd.read_csv(detail_path)
if "is_next_jornada" in detail_df.columns:
    jornada_df = detail_df[detail_df["is_next_jornada"] == True].copy()
else:
    jornada_df = detail_df.head(10).copy()

rows = []
for _, row in jornada_df.iterrows():
    home_team = str(row["home_team"])
    away_team = str(row["away_team"])
    best_pick = str(row["best_pick"])

    home_goals, away_goals = results[(home_team, away_team)]
    actual_result = result_label(home_goals, away_goals)
    won = best_pick == actual_result

    if best_pick == "H":
        pick_odds = float(row["odds_avg_h"])
    elif best_pick == "D":
        pick_odds = float(row["odds_avg_d"])
    else:
        pick_odds = float(row["odds_avg_a"])

    realized_profit = (pick_odds - 1.0) if won else -1.0

    rows.append(
        {
            "date": row["date"],
            "home_team": home_team,
            "away_team": away_team,
            "best_pick": best_pick,
            "actual_result": actual_result,
            "bet_won": bool(won),
            "pick_odds": float(pick_odds),
            "realized_profit_eur": float(realized_profit),
            "is_next_jornada": True,
        }
    )

jornada_settlement_df = pd.DataFrame(rows).sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
jornada_settlement_df["realized_bankroll_cumulative_eur"] = jornada_settlement_df["realized_profit_eur"].cumsum()

if settle_path.exists():
    settlement_df = pd.read_csv(settle_path)
    for column in ["date", "home_team", "away_team"]:
        settlement_df[column] = settlement_df[column].astype(str)

    settlement_df = settlement_df.merge(
        jornada_settlement_df[
            [
                "date",
                "home_team",
                "away_team",
                "actual_result",
                "bet_won",
                "realized_profit_eur",
                "is_next_jornada",
            ]
        ],
        on=["date", "home_team", "away_team"],
        how="left",
        suffixes=("", "_new"),
    )

    for column in ["actual_result", "bet_won", "realized_profit_eur", "is_next_jornada"]:
        new_column = f"{column}_new"
        if new_column in settlement_df.columns:
            settlement_df[column] = settlement_df[new_column].where(settlement_df[new_column].notna(), settlement_df.get(column))
            settlement_df = settlement_df.drop(columns=[new_column])

    settlement_df = settlement_df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    settlement_df["realized_bankroll_cumulative_eur"] = pd.to_numeric(
        settlement_df["realized_profit_eur"], errors="coerce"
    ).fillna(0.0).cumsum()
else:
    settlement_df = jornada_settlement_df.copy()

settlement_df.to_csv(settle_path, index=False)

settled = int(jornada_settlement_df["actual_result"].notna().sum())
wins = int(jornada_settlement_df["bet_won"].sum())
losses = settled - wins
profit = float(jornada_settlement_df["realized_profit_eur"].sum())
roi = float(profit / settled) if settled else None

summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["next_jornada_settled_bets"] = settled
summary["next_jornada_realized_profit_eur"] = profit
summary["next_jornada_realized_roi"] = roi
summary["next_jornada_realized_bankroll_eur"] = profit
summary["settled_bets"] = settled
summary["realized_total_profit_eur"] = profit
summary["realized_roi"] = roi
summary["generated_at_utc"] = pd.Timestamp.now("UTC").isoformat()
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "DETALLE RESULTADOS JORNADA (Semana 29)",
    f"Partidos liquidados: {settled}",
    f"Aciertos: {wins}",
    f"Fallos: {losses}",
    f"Beneficio neto: {profit:.4f} €",
    f"ROI: {roi * 100:.2f}%",
    f"Bankroll final (sobre 10€): {10 + profit:.4f} €",
    "",
    "Partido | Pick | Resultado | Cuota pick | Beneficio",
]

for _, row in jornada_settlement_df.iterrows():
    lines.append(
        f"{row['home_team']} vs {row['away_team']} | {row['best_pick']} | {row['actual_result']} | {row['pick_odds']:.4f} | {row['realized_profit_eur']:.4f} €"
    )

txt_detail_path.write_text("\n".join(lines), encoding="utf-8")

print("OK", txt_detail_path)
print("profit", round(profit, 4), "wins", wins, "settled", settled)
