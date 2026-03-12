import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { type PredictUpcomingResponse, predictUpcomingFixture } from "../lib/api";
import {
  formatRoundLabel,
  getTeamImgSources,
  loadNextTeamLogo,
  sanitizeRound
} from "../lib/fixtures";
import { useAppStore } from "../lib/state/appStore";

const VALUE_BET_THRESHOLD = 0.02;

function formatProbability(value: number): string {
  if (!Number.isFinite(value)) {
    return "-";
  }

  return `${(value * 100).toFixed(1)}%`;
}

function formatSignedPercentage(value: number): string {
  if (!Number.isFinite(value)) {
    return "-";
  }

  const percentage = value * 100;
  const sign = percentage > 0 ? "+" : "";
  return `${sign}${percentage.toFixed(1)}%`;
}

function formatDecimalOdds(value: number): string {
  if (!Number.isFinite(value)) {
    return "-";
  }

  return value.toFixed(2);
}

export function MatchPredictionPage() {
  const [searchParams] = useSearchParams();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictUpcomingResponse | null>(null);
  const setToast = useAppStore((state) => state.setToast);

  const requestedDate = searchParams.get("date")?.trim() ?? "";
  const requestedHomeTeam = searchParams.get("home_team")?.trim() ?? "";
  const requestedAwayTeam = searchParams.get("away_team")?.trim() ?? "";
  const requestedRound = searchParams.get("round")?.trim() ?? "";
  const hasRequiredParams = requestedDate !== "" && requestedHomeTeam !== "" && requestedAwayTeam !== "";

  useEffect(() => {
    if (!hasRequiredParams) {
      setPredictionResult(null);
      setError("No se ha seleccionado un partido válido.");
      return;
    }

    let cancelled = false;

    const loadPrediction = async () => {
      setLoading(true);
      setError(null);
      setPredictionResult(null);

      try {
        const response = await predictUpcomingFixture({
          date: requestedDate,
          home_team: requestedHomeTeam,
          away_team: requestedAwayTeam
        });

        if (cancelled) {
          return;
        }

        setPredictionResult(response);
        setToast("Prediccion completada");
      } catch (loadError) {
        if (cancelled) {
          return;
        }

        const message = `Error al predecir: ${String(loadError)}`;
        setError(message);
        setToast(message);
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    void loadPrediction();

    return () => {
      cancelled = true;
    };
  }, [hasRequiredParams, requestedAwayTeam, requestedDate, requestedHomeTeam, setToast]);

  const selectedFixture = predictionResult?.selected_fixture ?? {
    date: requestedDate,
    home_team: requestedHomeTeam,
    away_team: requestedAwayTeam,
    round: requestedRound
  };

  const roundLabel = formatRoundLabel(selectedFixture.round ?? requestedRound);
  const predictionRecord = predictionResult?.prediction ?? {};
  const pH = Number(predictionRecord["p_H"] ?? Number.NaN);
  const pD = Number(predictionRecord["p_D"] ?? Number.NaN);
  const pA = Number(predictionRecord["p_A"] ?? Number.NaN);
  const oddsAvgH = Number(predictionRecord["odds_avg_h"] ?? Number.NaN);
  const oddsAvgD = Number(predictionRecord["odds_avg_d"] ?? Number.NaN);
  const oddsAvgA = Number(predictionRecord["odds_avg_a"] ?? Number.NaN);

  const marketComparison = useMemo(() => {
    const outcomes = [
      { key: "H", label: "1 · Local", modelProbability: pH, odds: oddsAvgH },
      { key: "D", label: "X · Empate", modelProbability: pD, odds: oddsAvgD },
      { key: "A", label: "2 · Visitante", modelProbability: pA, odds: oddsAvgA }
    ];

    if (
      outcomes.some(
        (outcome) => !Number.isFinite(outcome.modelProbability) || !Number.isFinite(outcome.odds)
      )
    ) {
      return null;
    }

    const normalizedTotal = outcomes.reduce(
      (total, outcome) => total + 1 / outcome.odds,
      0
    );

    if (!Number.isFinite(normalizedTotal) || normalizedTotal <= 0) {
      return null;
    }

    const rows = outcomes.map((outcome) => {
      const marketProbability = (1 / outcome.odds) / normalizedTotal;
      const expectedValue = outcome.modelProbability * outcome.odds - 1;

      return {
        ...outcome,
        marketProbability,
        expectedValue,
        isValueBet: expectedValue > VALUE_BET_THRESHOLD
      };
    });

    const bestOption = rows.reduce((best, current) =>
      current.expectedValue > best.expectedValue ? current : best
    );

    return {
      rows,
      bestOption,
      hasValueBet: bestOption.expectedValue > VALUE_BET_THRESHOLD
    };
  }, [oddsAvgA, oddsAvgD, oddsAvgH, pA, pD, pH]);

  return (
    <>
      <div className="panel match-hero">
        <Link className="page-link" to="/">
          Volver a partidos
        </Link>
        <p className="kicker">Detalle del partido</p>
        {sanitizeRound(roundLabel) && <div className="fixture-card-round">{roundLabel}</div>}
        <div className="match-hero-body">
          <div className="fixture-team">
            <img
              src={getTeamImgSources(selectedFixture.home_team)[0]}
              alt={selectedFixture.home_team}
              className="fixture-logo"
              data-logo-fallback-step="0"
              onError={(event) => {
                loadNextTeamLogo(event.currentTarget, selectedFixture.home_team);
              }}
            />
            <span className="fixture-team-name">{selectedFixture.home_team}</span>
          </div>

          <div className="match-hero-center">
            <span className="fixture-vs">vs</span>
            <span className="small-note">{selectedFixture.date}</span>
            <span className="small-note">Temporada: {predictionResult?.season_label ?? "-"}</span>
          </div>

          <div className="fixture-team">
            <img
              src={getTeamImgSources(selectedFixture.away_team)[0]}
              alt={selectedFixture.away_team}
              className="fixture-logo"
              data-logo-fallback-step="0"
              onError={(event) => {
                loadNextTeamLogo(event.currentTarget, selectedFixture.away_team);
              }}
            />
            <span className="fixture-team-name">{selectedFixture.away_team}</span>
          </div>
        </div>
      </div>

      {loading && (
        <div className="panel">
          <h3>Cargando predicción</h3>
          <p className="small-note">Se está calculando el pronóstico del partido seleccionado.</p>
        </div>
      )}

      {!loading && error && (
        <div className="panel">
          <h3>No se ha podido cargar la predicción</h3>
          <p>{error}</p>
        </div>
      )}

      {!loading && !error && predictionResult && (
        <>
          <div className="panel">
            <h3>Probabilidades 1X2</h3>
            <div className="metric-grid metric-grid-three">
              <div className="metric-card">
                <span>1 · Local</span>
                <strong>{formatProbability(pH)}</strong>
              </div>
              <div className="metric-card">
                <span>X · Empate</span>
                <strong>{formatProbability(pD)}</strong>
              </div>
              <div className="metric-card">
                <span>2 · Visitante</span>
                <strong>{formatProbability(pA)}</strong>
              </div>
            </div>
          </div>

          {marketComparison && (
            <div className="panel">
              <h3>Comparativa modelo vs mercado</h3>
              <div className="metric-grid metric-grid-three">
                {marketComparison.rows.map((row) => (
                  <div className="metric-card" key={row.key}>
                    <span>{row.label}</span>
                    <strong>{formatDecimalOdds(row.odds)}</strong>
                    <div className="metric-detail-list">
                      <span>Modelo: {formatProbability(row.modelProbability)}</span>
                      <span>Mercado: {formatProbability(row.marketProbability)}</span>
                      <span className={row.isValueBet ? "tag-value" : undefined}>
                        EV: {formatSignedPercentage(row.expectedValue)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
              <p className="comparison-summary">
                Mejor opción: <strong>{marketComparison.bestOption.label}</strong> con EV{" "}
                <strong>{formatSignedPercentage(marketComparison.bestOption.expectedValue)}</strong>
                {marketComparison.hasValueBet
                  ? `, supera el umbral de ${formatSignedPercentage(VALUE_BET_THRESHOLD)}.`
                  : `, no supera el umbral de ${formatSignedPercentage(VALUE_BET_THRESHOLD)}.`}
              </p>
            </div>
          )}

          {!marketComparison && (
            <div className="panel">
              <h3>Comparativa modelo vs mercado</h3>
              <p className="small-note">
                No hay cuotas disponibles para este partido, así que no se puede comparar el modelo con el mercado.
              </p>
            </div>
          )}
        </>
      )}
    </>
  );
}