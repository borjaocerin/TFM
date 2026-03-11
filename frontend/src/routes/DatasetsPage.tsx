import { useEffect, useMemo, useState } from "react";

import {
  PredictUpcomingResponse,
  UpcomingFixtureOption,
  getUpcomingFixtures,
  predictUpcomingFixture
} from "../lib/api";
import { useAppStore } from "../lib/state/appStore";

const LOGOS_LALIGA_ALIASES: Record<string, string> = {
  athletic_club: "athletic-club",
  atletico_madrid: "atletico-madrid",
  betis: "real-betis",
  elche: "elche",
  espanyol: "espanyol",
  las_palmas: "las-palmas",
  levante: "levante",
  rayo_vallecano: "rayo-vallecano",
  real_madrid: "real-madrid",
  real_oviedo: "real-oviedo",
  real_sociedad: "real-sociedad"
};

const ROOT_LOGO_ALIASES: Record<string, string> = {
  athletic_club: "athletic",
  elche_cf: "elche",
  espanol: "espanyol",
  espanyol: "espanyol",
  levante_ud: "levante",
  rcd_espanyol_de_barcelona: "espanyol"
};

function normalizeTeamToken(team: string, separator: "_" | "-") {
  const normalized = team
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, separator);

  if (separator === "_") {
    return normalized.replace(/_+/g, "_").replace(/^_+|_+$/g, "");
  }

  return normalized.replace(/-+/g, "-").replace(/^-+|-+$/g, "");
}

function getTeamImgSources(team: string): string[] {
  const cleanUnderscore = normalizeTeamToken(team, "_");
  const cleanHyphen = normalizeTeamToken(team, "-");
  const laligaToken = LOGOS_LALIGA_ALIASES[cleanUnderscore] ?? cleanHyphen;
  const rootToken = ROOT_LOGO_ALIASES[cleanUnderscore] ?? cleanUnderscore;

  const sources = [
    `/teams/${rootToken}.svg`,
    `/teams/logos_laliga/spain_${laligaToken}.football-logos.cc.svg`,
    `/teams/${rootToken}.png`
  ];

  return Array.from(new Set(sources));
}

function teamFallbackDataUri(team: string): string {
  const words = team.trim().split(/\s+/).filter(Boolean);
  const initials = words
    .slice(0, 2)
    .map((word) => word[0]?.toUpperCase() ?? "")
    .join("") || "?";
  const safeInitials = initials.slice(0, 2);

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><circle cx="32" cy="32" r="31" fill="#f3f4f6" stroke="#cbd5e1"/><text x="32" y="39" text-anchor="middle" font-size="21" font-family="Verdana,sans-serif" fill="#334155">${safeInitials}</text></svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

function loadNextTeamLogo(img: HTMLImageElement, team: string): void {
  const sources = getTeamImgSources(team);
  const currentStep = Number.parseInt(img.dataset.logoFallbackStep ?? "0", 10);
  const nextStep = Number.isNaN(currentStep) ? 1 : currentStep + 1;

  if (nextStep < sources.length) {
    img.dataset.logoFallbackStep = String(nextStep);
    img.src = sources[nextStep];
    return;
  }

  img.onerror = null;
  img.src = teamFallbackDataUri(team);
}

export function DatasetsPage() {
  const [loadingOptions, setLoadingOptions] = useState(false);
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [fixtures, setFixtures] = useState<UpcomingFixtureOption[]>([]);
  const [seasonLabel, setSeasonLabel] = useState<string>("");
  const [sourcePath, setSourcePath] = useState<string>("");
  const [selectedFixtureId, setSelectedFixtureId] = useState<string>("");
  const [predictionResult, setPredictionResult] = useState<PredictUpcomingResponse | null>(null);
  const [selectedRound, setSelectedRound] = useState<string>("");
  const [searchText, setSearchText] = useState<string>("");
  const setToast = useAppStore((state) => state.setToast);

  const selectedFixture = useMemo(
    () => fixtures.find((fixture) => fixture.fixture_id === selectedFixtureId) ?? null,
    [fixtures, selectedFixtureId]
  );

  const loadFixtures = async () => {
    setLoadingOptions(true);
    try {
      const response = await getUpcomingFixtures();
      setFixtures(response.fixtures);
      setSeasonLabel(response.season_label);
      setSourcePath(response.source_path);

      if (response.fixtures.length > 0) {
        setSelectedFixtureId(response.fixtures[0].fixture_id);
        const demoNote = response.source_path?.startsWith("demo:") ? " (datos historicos de demo)" : "";
        setToast(`Cargados ${response.fixtures.length} partidos${demoNote}`);
      } else {
        setSelectedFixtureId("");
        setToast(
          response.error
            ? `Sin partidos: ${response.error}`
            : "Sin partidos disponibles. Comprueba ODDS_API_KEY en el backend."
        );
      }
    } catch (error) {
      setToast(`Error cargando partidos: ${String(error)}`);
    } finally {
      setLoadingOptions(false);
    }
  };

  const handlePredictSelected = async () => {
    if (!selectedFixture) {
      setToast("Selecciona un partido para predecir");
      return;
    }

    setLoadingPredict(true);
    try {
      const response = await predictUpcomingFixture({
        date: selectedFixture.date,
        home_team: selectedFixture.home_team,
        away_team: selectedFixture.away_team
      });
      setPredictionResult(response);
      setToast("Prediccion completada");
    } catch (error) {
      setToast(`Error al predecir: ${String(error)}`);
    } finally {
      setLoadingPredict(false);
    }
  };

  useEffect(() => {
    void loadFixtures();
  }, []);

  const predictionRecord = predictionResult?.prediction ?? {};
  const pH = Number(predictionRecord["p_H"] ?? 0);
  const pD = Number(predictionRecord["p_D"] ?? 0);
  const pA = Number(predictionRecord["p_A"] ?? 0);
  const oddsAvgH = Number(predictionRecord["odds_avg_h"] ?? Number.NaN);
  const oddsAvgD = Number(predictionRecord["odds_avg_d"] ?? Number.NaN);
  const oddsAvgA = Number(predictionRecord["odds_avg_a"] ?? Number.NaN);

  const bestLabel =
    pH >= pD && pH >= pA
      ? "1 (Local)"
      : pD >= pH && pD >= pA
        ? "X (Empate)"
        : "2 (Visitante)";

  const rounds = useMemo(() => {
    const unique = Array.from(new Set(fixtures.map(f => (f.round || "").trim()).filter(r => r && r !== 'undefined' && r !== 'null')));
    return unique.length > 0 ? unique.sort() : [];
  }, [fixtures]);

  const filteredFixtures = useMemo(() => {
    let filtered = fixtures;
    if (selectedRound) {
      filtered = filtered.filter(f => (f.round || "").trim() === selectedRound);
    }
    if (searchText.trim()) {
      const search = searchText.trim().toLowerCase();
      filtered = filtered.filter(f => {
        const home = f.home_team.toLowerCase();
        const away = f.away_team.toLowerCase();
        const label = f.label?.toLowerCase() || "";
        return (
          home.includes(search) ||
          away.includes(search) ||
          label.includes(search) ||
          `${home} vs ${away}`.includes(search) ||
          `${away} vs ${home}`.includes(search)
        );
      });
    }
    return filtered;
  }, [fixtures, selectedRound, searchText]);

  return (
    <>
      <div className="panel">
        <h2>Predicción visual de partido</h2>
        <p className="small-note">
          Temporada activa: <strong>{seasonLabel || "-"}</strong>
        </p>
        <p className="small-note">
          Fuente de partidos: <code>{sourcePath || "(API sin respuesta)"}</code>
        </p>
        <div className="field">
          <label>Filtrar por jornada</label>
          <select
            value={selectedRound}
            onChange={e => setSelectedRound(e.target.value)}
            disabled={rounds.length === 0}
          >
            <option value="">Todas</option>
            {rounds.map(r => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Buscar partido</label>
          <input
            type="text"
            value={searchText}
            onChange={e => setSearchText(e.target.value)}
            placeholder="Buscar equipos o partido..."
            style={{ width: "100%", marginTop: "0.5rem" }}
          />
        </div>
        <div className="fixture-cards-grid">
          {filteredFixtures.length === 0 && (
            <div style={{padding:'1.5rem',textAlign:'center',color:'#b0b0b0'}}>No hay partidos disponibles</div>
          )}
          {filteredFixtures.map((fixture) => {
            const isSelected = fixture.fixture_id === selectedFixtureId;
            return (
              <div
                key={fixture.fixture_id}
                className={`fixture-card${isSelected ? ' selected' : ''}`}
                onClick={() => setSelectedFixtureId(fixture.fixture_id)}
                tabIndex={0}
                style={{cursor:'pointer'}}
              >
                <div className="fixture-card-date">{fixture.date}{fixture.round ? ` | Jornada ${fixture.round}` : ''}</div>
                <div className="fixture-card-body">
                  <div className="fixture-team">
                    <img
                      src={getTeamImgSources(fixture.home_team)[0]}
                      alt={fixture.home_team}
                      className="fixture-logo"
                      data-logo-fallback-step="0"
                      onError={e => {
                        loadNextTeamLogo(e.currentTarget, fixture.home_team);
                      }}
                    />
                    <span className="fixture-team-name">{fixture.home_team}</span>
                  </div>
                  <span className="fixture-vs">vs</span>
                  <div className="fixture-team">
                    <img
                      src={getTeamImgSources(fixture.away_team)[0]}
                      alt={fixture.away_team}
                      className="fixture-logo"
                      data-logo-fallback-step="0"
                      onError={e => {
                        loadNextTeamLogo(e.currentTarget, fixture.away_team);
                      }}
                    />
                    <span className="fixture-team-name">{fixture.away_team}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
        <div className="actions">
          <button className="btn" type="button" onClick={() => void loadFixtures()} disabled={loadingOptions}>
            {loadingOptions ? "Cargando..." : "Recargar partidos"}
          </button>
          <button
            className="btn btn-alt"
            type="button"
            onClick={() => void handlePredictSelected()}
            disabled={!selectedFixture || loadingPredict}
          >
            {loadingPredict ? "Prediciendo..." : "Predecir partido seleccionado"}
          </button>
        </div>
      </div>

      {predictionResult && (
        <>
          <div className="panel">
            <h3>Resultado de prediccion</h3>
            <p>
              <strong>{predictionResult.selected_fixture.home_team}</strong> vs{" "}
              <strong>{predictionResult.selected_fixture.away_team}</strong>
            </p>
            <p>Fecha: {predictionResult.selected_fixture.date}</p>
            <p>Pronostico principal: {bestLabel}</p>
            <p>CSV de salida: {predictionResult.output_csv}</p>
          </div>

          <div className="panel">
            <h3>Probabilidades 1X2</h3>
            <div className="metric-grid">
              <div className="metric-card">
                <span>p_H (1)</span>
                <strong>{pH.toFixed(3)}</strong>
              </div>
              <div className="metric-card">
                <span>p_D (X)</span>
                <strong>{pD.toFixed(3)}</strong>
              </div>
              <div className="metric-card">
                <span>p_A (2)</span>
                <strong>{pA.toFixed(3)}</strong>
              </div>
            </div>
          </div>

          {Number.isFinite(oddsAvgH) && Number.isFinite(oddsAvgD) && Number.isFinite(oddsAvgA) && (
            <div className="panel">
              <h3>Cuotas mercado (The Odds API)</h3>
              <div className="metric-grid">
                <div className="metric-card">
                  <span>Cuota 1</span>
                  <strong>{oddsAvgH.toFixed(2)}</strong>
                </div>
                <div className="metric-card">
                  <span>Cuota X</span>
                  <strong>{oddsAvgD.toFixed(2)}</strong>
                </div>
                <div className="metric-card">
                  <span>Cuota 2</span>
                  <strong>{oddsAvgA.toFixed(2)}</strong>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </>
  );
}
