from __future__ import annotations

import json
import re
import unicodedata
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


def plot_data_quality_five_metrics(dataset_path: Path, outdir: Path) -> Path:
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe dataset en {dataset_path}")

    df = pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError("El dataset esta vacio")

    # Integridad = filas con campos clave completos y sin contradiccion marcador-resultado.
    integrity = np.nan
    integrity_checks: list[pd.Series] = []

    key_cols = [c for c in ["date", "home_team", "away_team", "home_goals", "away_goals", "result"] if c in df.columns]
    if key_cols:
        integrity_checks.append(df[key_cols].notna().all(axis=1))

    if {"home_goals", "away_goals", "result"}.issubset(df.columns):
        gh = pd.to_numeric(df["home_goals"], errors="coerce")
        ga = pd.to_numeric(df["away_goals"], errors="coerce")
        result = df["result"].astype(str).str.upper().str[0]
        expected = np.where(gh > ga, "H", np.where(gh < ga, "A", "D"))
        integrity_checks.append(gh.notna() & ga.notna() & result.notna() & (result == expected))

    if integrity_checks:
        integrity = float(pd.concat(integrity_checks, axis=1).all(axis=1).mean() * 100.0)

    # Validez = reglas de dominio minimas sobre goles, posesion, tiros y penaltis.
    validity = np.nan
    validity_checks: list[pd.Series] = []

    if {"home_goals", "away_goals"}.issubset(df.columns):
        gh = pd.to_numeric(df["home_goals"], errors="coerce")
        ga = pd.to_numeric(df["away_goals"], errors="coerce")
        validity_checks.append(gh.notna() & (gh >= 0))
        validity_checks.append(ga.notna() & (ga >= 0))

    if {"poss_home", "poss_away"}.issubset(df.columns):
        poss_h = pd.to_numeric(df["poss_home"], errors="coerce")
        poss_a = pd.to_numeric(df["poss_away"], errors="coerce")
        validity_checks.append(poss_h.notna() & poss_h.between(0, 100))
        validity_checks.append(poss_a.notna() & poss_a.between(0, 100))

    if {"sh_home", "sh_away", "sot_home", "sot_away"}.issubset(df.columns):
        sh_h = pd.to_numeric(df["sh_home"], errors="coerce")
        sh_a = pd.to_numeric(df["sh_away"], errors="coerce")
        sot_h = pd.to_numeric(df["sot_home"], errors="coerce")
        sot_a = pd.to_numeric(df["sot_away"], errors="coerce")
        validity_checks.append(sh_h.notna() & (sh_h >= 0) & sot_h.notna() & (sot_h >= 0) & (sot_h <= sh_h))
        validity_checks.append(sh_a.notna() & (sh_a >= 0) & sot_a.notna() & (sot_a >= 0) & (sot_a <= sh_a))

    if {"pk_home", "pk_away", "pkatt_home", "pkatt_away"}.issubset(df.columns):
        pk_h = pd.to_numeric(df["pk_home"], errors="coerce")
        pk_a = pd.to_numeric(df["pk_away"], errors="coerce")
        pkatt_h = pd.to_numeric(df["pkatt_home"], errors="coerce")
        pkatt_a = pd.to_numeric(df["pkatt_away"], errors="coerce")
        validity_checks.append(pk_h.notna() & pkatt_h.notna() & (pk_h >= 0) & (pkatt_h >= 0) & (pk_h <= pkatt_h))
        validity_checks.append(pk_a.notna() & pkatt_a.notna() & (pk_a >= 0) & (pkatt_a >= 0) & (pk_a <= pkatt_a))

    if validity_checks:
        validity = float(pd.concat(validity_checks, axis=1).all(axis=1).mean() * 100.0)

    # Unicidad = porcentaje de claves de partido unicas (date, home_team, away_team).
    uniqueness = np.nan
    unique_cols = [c for c in ["date", "home_team", "away_team"] if c in df.columns]
    if len(unique_cols) == 3:
        unique_count = int(df[unique_cols].dropna().drop_duplicates().shape[0])
        uniqueness = float((unique_count / len(df)) * 100.0)

    # Consistencia = uniformidad de nomenclatura y formatos entre filas.
    consistency = np.nan
    consistency_checks: list[pd.Series] = []

    if "result" in df.columns:
        result = df["result"].astype(str).str.upper().str[0]
        consistency_checks.append(result.isin(["H", "D", "A"]))

    if "season" in df.columns:
        season_num = pd.to_numeric(df["season"], errors="coerce")
        consistency_checks.append(season_num.notna())

    if {"home_team", "away_team"}.issubset(df.columns):
        home_ok = df["home_team"].astype(str).str.strip().str.len() > 0
        away_ok = df["away_team"].astype(str).str.strip().str.len() > 0
        consistency_checks.append(home_ok & away_ok)

        # Penaliza variantes del mismo equipo (acentos, abreviaturas o grafias distintas)
        # tomando como canonica la forma mas frecuente por token normalizado.
        def _norm_team(name: str) -> str:
            txt = str(name).strip().lower()
            txt = "".join(ch for ch in unicodedata.normalize("NFKD", txt) if not unicodedata.combining(ch))
            txt = re.sub(r"[^a-z0-9 ]+", " ", txt)
            txt = re.sub(r"\s+", " ", txt).strip()
            return txt

        team_cells = pd.concat([df["home_team"], df["away_team"]], ignore_index=True).dropna().astype(str)
        if not team_cells.empty:
            norm_df = pd.DataFrame({"raw": team_cells})
            norm_df["norm"] = norm_df["raw"].map(_norm_team)
            canonical = norm_df.groupby("norm")["raw"].agg(lambda x: x.value_counts().idxmax())
            consistent_team_cells = norm_df.apply(lambda row: row["raw"] == canonical[row["norm"]], axis=1)
            team_name_consistency = float(consistent_team_cells.mean() * 100.0)
        else:
            team_name_consistency = np.nan
    else:
        team_name_consistency = np.nan

    if "date" in df.columns:
        parsed_date = pd.to_datetime(df["date"], errors="coerce")
        consistency_checks.append(parsed_date.notna())

    if consistency_checks:
        base_consistency = float(pd.concat(consistency_checks, axis=1).all(axis=1).mean() * 100.0)
        if np.isfinite(team_name_consistency):
            consistency = min(base_consistency, team_name_consistency)
        else:
            consistency = base_consistency

    # Exactitud = proxy de fidelidad al partido real: goles y resultado coinciden.
    accuracy = np.nan
    if {"home_goals", "away_goals", "result"}.issubset(df.columns):
        gh = pd.to_numeric(df["home_goals"], errors="coerce")
        ga = pd.to_numeric(df["away_goals"], errors="coerce")
        result = df["result"].astype(str).str.upper().str[0]
        evaluable = gh.notna() & ga.notna() & result.notna()
        expected = np.where(gh > ga, "H", np.where(gh < ga, "A", "D"))
        if evaluable.any():
            accuracy = float((result[evaluable] == expected[evaluable]).mean() * 100.0)

    metrics = [
        ("Integridad", integrity),
        ("Validez", validity),
        ("Unicidad", uniqueness),
        ("Consistencia", consistency),
        ("Exactitud", accuracy),
    ]

    def _color_for(pct: float) -> str:
        if not np.isfinite(pct):
            return "#9aa0a6"
        if pct >= 95.0:
            return "#2ca02c"
        if pct >= 85.0:
            return "#ffb000"
        return "#d62728"

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    for ax, (label, value) in zip(axes, metrics):
        pct = float(np.clip(value, 0.0, 100.0)) if np.isfinite(value) else 0.0
        color = _color_for(value)
        ax.pie(
            [pct, 100.0 - pct],
            startangle=90,
            colors=[color, "#e8eaed"],
            wedgeprops={"width": 0.34, "edgecolor": "white"},
            counterclock=False,
        )
        text = f"{value:.1f}%" if np.isfinite(value) else "N/A"
        ax.text(0, 0, text, ha="center", va="center", fontsize=12, weight="bold", color="#1f2933")
        ax.set_title(label, fontsize=11)

    fig.suptitle("Calidad de datos pre-EDA (porcentaje) - 5 dimensiones", fontsize=15, weight="bold")
    out = outdir / "eda_data_quality_5metrics_pre_eda.png"
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
