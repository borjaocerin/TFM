from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = ROOT / "backend" / "app" / "models" / "store" / "metadata.json"
DEFAULT_DATASET = ROOT / "data" / "out" / "laliga_enriched_model.csv"
DEFAULT_OUTDIR = ROOT / "docs" / "screens"
LOWER_IS_BETTER_METRICS = {"log_loss", "brier", "ece"}


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No existe metadata de modelo en {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_model_leaderboard(metadata: dict[str, Any], outdir: Path) -> Path:
    leaderboard = metadata.get("leaderboard", [])
    if not leaderboard:
        raise ValueError("metadata.json no contiene leaderboard")

    names = [str(item.get("model", "unknown")) for item in leaderboard]
    accuracy = [float(item.get("accuracy", np.nan)) for item in leaderboard]
    f1_macro = [float(item.get("f1_macro", np.nan)) for item in leaderboard]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="#1f77b4")
    ax.bar(x + width / 2, f1_macro, width, label="F1 macro", color="#ff7f0e")

    ax.set_title("Comparativa de modelos (CV temporal)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(max(accuracy), max(f1_macro)) * 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    for idx, val in enumerate(accuracy):
        ax.text(idx - width / 2, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    for idx, val in enumerate(f1_macro):
        ax.text(idx + width / 2, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    out = outdir / "model_leaderboard_cv.png"
    _save(fig, out)
    return out


def plot_primary_metric_comparison(metadata: dict[str, Any], outdir: Path) -> Path:
    leaderboard = metadata.get("leaderboard", [])
    if not leaderboard:
        raise ValueError("metadata.json no contiene leaderboard")

    metric = str(metadata.get("selection_metric") or "log_loss")
    lower_better = metric in LOWER_IS_BETTER_METRICS
    ordered = sorted(
        leaderboard,
        key=lambda item: float(item.get(metric, np.inf)),
        reverse=not lower_better,
    )

    names = [str(item.get("model", "unknown")) for item in ordered]
    values = [float(item.get(metric, np.nan)) for item in ordered]
    colors = ["#0b6e4f" if idx == 0 else "#4e79a7" for idx in range(len(names))]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    bars = ax.bar(names, values, color=colors)
    ax.set_title(f"Ranking de modelos por {metric} ({'menor' if lower_better else 'mayor'} es mejor)")
    ax.set_ylabel(metric)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=15)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.002 if np.isfinite(val) else 0),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    out = outdir / "model_primary_metric_comparison.png"
    _save(fig, out)
    return out


def plot_cv_fit_comparison(metadata: dict[str, Any], outdir: Path) -> Path:
    cv = metadata.get("metrics", {})
    fit = metadata.get("fit_metrics", {})
    metric_order = ["accuracy", "f1_macro", "log_loss", "brier", "ece"]

    cv_vals = [float(cv.get(m, np.nan)) for m in metric_order]
    fit_vals = [float(fit.get(m, np.nan)) for m in metric_order]

    x = np.arange(len(metric_order))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, cv_vals, width, label="CV temporal", color="#2ca02c")
    ax.bar(x + width / 2, fit_vals, width, label="Fit in-sample", color="#9467bd")

    ax.set_title("Metricas CV vs Fit (modelo seleccionado)")
    ax.set_ylabel("Valor")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_order)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    for idx, val in enumerate(cv_vals):
        if np.isfinite(val):
            ax.text(idx - width / 2, val + 0.004, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    for idx, val in enumerate(fit_vals):
        if np.isfinite(val):
            ax.text(idx + width / 2, val + 0.004, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    out = outdir / "model_metrics_cv_vs_fit.png"
    _save(fig, out)
    return out


def plot_missing_by_season(dataset_path: Path, outdir: Path) -> Path:
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe dataset en {dataset_path}")

    df = pd.read_csv(dataset_path)
    if "season" not in df.columns:
        raise ValueError("El dataset no tiene columna season")

    cols = [
        c
        for c in ["xg_last5_home", "xg_last5_away", "poss_last5_home", "poss_last5_away"]
        if c in df.columns
    ]
    if not cols:
        raise ValueError("No hay columnas xG/posesion rolling para graficar missing")

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df.dropna(subset=["season"])

    coverage = (
        df.groupby("season")[cols]
        .apply(lambda frame: frame.notna().mean() * 100.0)
        .reset_index()
        .sort_values("season")
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    for col in cols:
        ax.plot(coverage["season"], coverage[col], marker="o", label=col)

    ax.set_title("Cobertura por temporada de features xG/posesion")
    ax.set_xlabel("Season")
    ax.set_ylabel("Cobertura no nula (%)")
    ax.set_ylim(0, 102)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    out = outdir / "eda_missing_coverage_by_season.png"
    _save(fig, out)
    return out


def _draw_box(ax: plt.Axes, x: float, y: float, text: str, w: float = 2.6, h: float = 0.85) -> tuple[float, float, float, float]:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.6,
        edgecolor="#1f4f7a",
        facecolor="#e8f2fb",
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    return (x, y, w, h)


def _arrow(ax: plt.Axes, a: tuple[float, float], b: tuple[float, float]) -> None:
    arr = FancyArrowPatch(a, b, arrowstyle="->", mutation_scale=12, linewidth=1.8, color="#2f2f2f")
    ax.add_patch(arr)


def plot_eda_flow(outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    b1 = _draw_box(ax, 0.6, 5.8, "Input historico\n(partidos / team-level)")
    b2 = _draw_box(ax, 3.7, 5.8, "Normalizacion columnas\ny nombres de equipos")
    b3 = _draw_box(ax, 6.8, 5.8, "Join football-data\n+ odds + ELO")
    b4 = _draw_box(ax, 9.9, 5.8, "Feature engineering\nrolling pre-partido")

    b5 = _draw_box(ax, 2.2, 3.9, "EDA missing\n(global + por temporada)")
    b6 = _draw_box(ax, 5.3, 3.9, "Filtro train\nmin_season / cobertura")
    b7 = _draw_box(ax, 8.4, 3.9, "Train + calibracion\n(CV temporal)")

    b8 = _draw_box(ax, 4.0, 2.0, "Artefactos\nmodel.pkl + metadata.json")
    b9 = _draw_box(ax, 7.1, 2.0, "Graficos memoria\nmetricas + cobertura")

    _arrow(ax, (b1[0] + b1[2], b1[1] + b1[3] / 2), (b2[0], b2[1] + b2[3] / 2))
    _arrow(ax, (b2[0] + b2[2], b2[1] + b2[3] / 2), (b3[0], b3[1] + b3[3] / 2))
    _arrow(ax, (b3[0] + b3[2], b3[1] + b3[3] / 2), (b4[0], b4[1] + b4[3] / 2))

    _arrow(ax, (b4[0] + b4[2] / 2, b4[1]), (b7[0] + b7[2] / 2, b7[1] + b7[3]))
    _arrow(ax, (b2[0] + b2[2] / 2, b2[1]), (b5[0] + b5[2] / 2, b5[1] + b5[3]))
    _arrow(ax, (b5[0] + b5[2], b5[1] + b5[3] / 2), (b6[0], b6[1] + b6[3] / 2))
    _arrow(ax, (b6[0] + b6[2], b6[1] + b6[3] / 2), (b7[0], b7[1] + b7[3] / 2))

    _arrow(ax, (b7[0] + b7[2] / 2, b7[1]), (b8[0] + b8[2] / 2, b8[1] + b8[3]))
    _arrow(ax, (b7[0] + b7[2] / 2, b7[1]), (b9[0] + b9[2] / 2, b9[1] + b9[3]))

    ax.text(0.6, 6.85, "Flujo EDA y entrenamiento del modelo 1X2", fontsize=15, weight="bold")

    out = outdir / "eda_pipeline_flow.png"
    _save(fig, out)
    return out


def main() -> None:
    metadata = _load_metadata(DEFAULT_METADATA)
    outdir = DEFAULT_OUTDIR

    generated = [
        plot_primary_metric_comparison(metadata, outdir),
        plot_model_leaderboard(metadata, outdir),
        plot_cv_fit_comparison(metadata, outdir),
        plot_missing_by_season(DEFAULT_DATASET, outdir),
        plot_eda_flow(outdir),
    ]

    print("Imagenes generadas:")
    for path in generated:
        print(str(path))


if __name__ == "__main__":
    main()
