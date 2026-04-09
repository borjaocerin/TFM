from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
METRICS_PATH = REPO / "data" / "out" / "model_metrics.txt"
OUT_PATH = REPO / "docs" / "screens" / "model_metrics_leaderboard.png"


def main() -> None:
    text = METRICS_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^\s*-\s*(?P<model>[a-zA-Z0-9_]+):\s*log_loss=(?P<log_loss>[0-9.]+),\s*"
        r"accuracy=(?P<accuracy>[0-9.]+),\s*brier=(?P<brier>[0-9.]+),\s*ece=(?P<ece>[0-9.]+),\s*"
        r"f1_macro=(?P<f1>[0-9.]+),\s*improvement_vs_baseline_pct=(?P<impr>[+-]?[0-9.]+)%\s*$"
    )

    rows = []
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            row = match.groupdict()
            row["log_loss"] = float(row["log_loss"])
            row["accuracy"] = float(row["accuracy"])
            row["f1"] = float(row["f1"])
            row["impr"] = float(row["impr"])
            rows.append(row)

    if not rows:
        raise SystemExit("No se pudieron parsear las filas del leaderboard")

    rows.sort(key=lambda item: item["log_loss"])
    models = [item["model"] for item in rows]
    log_loss = [item["log_loss"] for item in rows]
    accuracy = [item["accuracy"] for item in rows]
    f1 = [item["f1"] for item in rows]
    improvement = [item["impr"] for item in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    y = np.arange(len(models))
    colors = ["#1f77b4" if idx == 0 else "#9db8d3" for idx in range(len(models))]

    ax.barh(y, log_loss, color=colors, edgecolor="#17324d", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Log loss (menor es mejor)", fontsize=12)
    ax.set_title("Leaderboard del modelo 1X2 por log loss", fontsize=17, weight="bold", pad=14)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_xlim(0, max(log_loss) * 1.22)

    for idx, (ll, acc, f1_score, imp) in enumerate(zip(log_loss, accuracy, f1, improvement)):
        ax.text(
            ll + max(log_loss) * 0.015,
            idx,
            f"{ll:.3f}  |  acc {acc:.2f}  |  f1 {f1_score:.2f}  |  {imp:+.1f}%",
            va="center",
            ha="left",
            fontsize=9.5,
            color="#1f2933",
        )

    ax.text(
        0.99,
        -0.08,
        "Ordenado por log_loss sobre validacion temporal. La primera barra destaca el mejor modelo.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#4b5563",
    )

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
