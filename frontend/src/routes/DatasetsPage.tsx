import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import {
  type UpcomingFixtureOption,
  getUpcomingFixtures
} from "../lib/api";
import {
  formatRoundLabel,
  getRoundSortValue,
  getTeamImgSources,
  loadNextTeamLogo,
  sanitizeRound
} from "../lib/fixtures";
import { useAppStore } from "../lib/state/appStore";

const VALUE_BET_THRESHOLD = 0.02;
type FixturesSortMode = "date" | "value_desc" | "value_asc";

function formatSignedPercentage(value: number): string {
  if (!Number.isFinite(value)) {
    return "-";
  }

  const percentage = value * 100;
  const sign = percentage > 0 ? "+" : "";
  return `${sign}${percentage.toFixed(1)}%`;
}

function buildFixturePredictionPath(fixture: UpcomingFixtureOption): string {
  const params = new URLSearchParams({
    date: fixture.date,
    home_team: fixture.home_team,
    away_team: fixture.away_team
  });
  const round = sanitizeRound(fixture.round);

  if (round) {
    params.set("round", round);
  }

  const snapshotEntries: Array<[string, number | null | undefined]> = [
    ["snap_p_h", fixture.p_H],
    ["snap_p_d", fixture.p_D],
    ["snap_p_a", fixture.p_A],
    ["snap_odds_h", fixture.odds_avg_h],
    ["snap_odds_d", fixture.odds_avg_d],
    ["snap_odds_a", fixture.odds_avg_a]
  ];

  snapshotEntries.forEach(([key, value]) => {
    const numeric = Number(value ?? Number.NaN);
    if (Number.isFinite(numeric)) {
      params.set(key, String(numeric));
    }
  });

  return `/partido?${params.toString()}`;
}

export function DatasetsPage() {
  const [fixtures, setFixtures] = useState<UpcomingFixtureOption[]>([]);
  const [selectedRound, setSelectedRound] = useState<string>("");
  const [searchText, setSearchText] = useState<string>("");
  const [sortMode, setSortMode] = useState<FixturesSortMode>("date");
  const [valueRankingLoading, setValueRankingLoading] = useState(false);
  const [valueRankingLoaded, setValueRankingLoaded] = useState(false);
  const [valueRankingAttempted, setValueRankingAttempted] = useState(false);
  const setToast = useAppStore((state) => state.setToast);

  const loadFixtures = async (includeValue = false) => {
    if (includeValue) {
      setValueRankingLoading(true);
      setValueRankingAttempted(true);
    }

    try {
      const response = await getUpcomingFixtures(
        includeValue
          ? {
              includeValue: true,
              valueThreshold: VALUE_BET_THRESHOLD
            }
          : undefined
      );
      setFixtures(response.fixtures);
      if (includeValue) {
        setValueRankingLoaded(true);
      }

      if (response.fixtures.length > 0) {
        const demoNote = response.source_path?.startsWith("demo:") ? " (datos historicos de demo)" : "";
        setToast(`Cargados ${response.fixtures.length} partidos${demoNote}`);
      } else {
        setToast(
          response.error
            ? `Sin partidos: ${response.error}`
            : "Sin partidos disponibles. Comprueba ODDS_API_KEY en el backend."
        );
      }
    } catch (error) {
      setToast(`Error cargando partidos: ${String(error)}`);
    } finally {
      if (includeValue) {
        setValueRankingLoading(false);
      }
    }
  };

  useEffect(() => {
    void loadFixtures();
  }, []);

  useEffect(() => {
    if (sortMode === "date" || valueRankingLoaded || valueRankingLoading || valueRankingAttempted) {
      return;
    }

    void loadFixtures(true);
  }, [sortMode, valueRankingAttempted, valueRankingLoaded, valueRankingLoading]);

  const rounds = useMemo(() => {
    const unique: string[] = Array.from(
      new Set(fixtures.map((fixture: UpcomingFixtureOption) => sanitizeRound(fixture.round)).filter(Boolean))
    );

    return unique.sort((left: string, right: string) => {
      const numericDifference = getRoundSortValue(left) - getRoundSortValue(right);

      if (numericDifference !== 0) {
        return numericDifference;
      }

      return left.localeCompare(right, "es", { numeric: true, sensitivity: "base" });
    });
  }, [fixtures]);

  const filteredFixtures = useMemo(() => {
    let filtered: UpcomingFixtureOption[] = fixtures;
    if (selectedRound) {
      filtered = filtered.filter(
        (fixture: UpcomingFixtureOption) => sanitizeRound(fixture.round) === selectedRound
      );
    }
    if (searchText.trim()) {
      const search = searchText.trim().toLowerCase();
      filtered = filtered.filter((fixture: UpcomingFixtureOption) => {
        const home = fixture.home_team.toLowerCase();
        const away = fixture.away_team.toLowerCase();
        const label = fixture.label?.toLowerCase() || "";
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

  const sortedFixtures = useMemo(() => {
    const compareByDate = (left: UpcomingFixtureOption, right: UpcomingFixtureOption): number => {
      const byDate = String(left.date ?? "").localeCompare(String(right.date ?? ""));
      if (byDate !== 0) {
        return byDate;
      }

      const byHome = String(left.home_team ?? "").localeCompare(String(right.home_team ?? ""), "es", {
        sensitivity: "base"
      });
      if (byHome !== 0) {
        return byHome;
      }

      return String(left.away_team ?? "").localeCompare(String(right.away_team ?? ""), "es", {
        sensitivity: "base"
      });
    };

    const sortedByDate = [...filteredFixtures].sort(compareByDate);
    if (sortMode === "date") {
      return sortedByDate;
    }

    const descending = sortMode === "value_desc";
    return sortedByDate.sort((left, right) => {
      const leftEv = Number(left.best_ev ?? Number.NaN);
      const rightEv = Number(right.best_ev ?? Number.NaN);

      const leftHasEv = Number.isFinite(leftEv);
      const rightHasEv = Number.isFinite(rightEv);

      if (leftHasEv && rightHasEv && leftEv !== rightEv) {
        return descending ? rightEv - leftEv : leftEv - rightEv;
      }

      if (leftHasEv !== rightHasEv) {
        return leftHasEv ? -1 : 1;
      }

      return compareByDate(left, right);
    });
  }, [filteredFixtures, sortMode]);

  const handleSortChange = (nextSortMode: FixturesSortMode) => {
    setSortMode(nextSortMode);
  };

  return (
    <div className="panel">
      <h2>Predicción visual de partido</h2>
      <div className="field">
        <label>Filtrar por jornada</label>
        <select
          value={selectedRound}
          onChange={(event) => setSelectedRound(event.target.value)}
          disabled={rounds.length === 0}
        >
          <option value="">Todas</option>
          {rounds.map((round: string) => (
            <option key={round} value={round}>
              {formatRoundLabel(round)}
            </option>
          ))}
        </select>
      </div>
      <div className="field">
        <label>Buscar partido</label>
        <input
          type="text"
          value={searchText}
          onChange={(event) => setSearchText(event.target.value)}
          placeholder="Buscar equipos o partido..."
          style={{ width: "100%", marginTop: "0.5rem" }}
        />
      </div>
      <div className="field">
        <label>Ordenar partidos</label>
        <select value={sortMode} onChange={(event) => handleSortChange(event.target.value as FixturesSortMode)}>
          <option value="date">Fecha (próximo primero)</option>
          <option value="value_desc">Value bet (mejor EV → peor EV)</option>
          <option value="value_asc">Value bet (peor EV → mejor EV)</option>
        </select>
        {sortMode !== "date" && valueRankingLoading && (
          <p className="small-note">Calculando ranking de value bet para los partidos...</p>
        )}
      </div>
      <div className="fixture-cards-grid">
        {sortedFixtures.length === 0 && (
          <div style={{ padding: "1.5rem", textAlign: "center", color: "#b0b0b0" }}>
            No hay partidos disponibles
          </div>
        )}
        {sortedFixtures.map((fixture: UpcomingFixtureOption) => (
          <Link
            key={fixture.fixture_id}
            className="fixture-card fixture-card-link"
            to={buildFixturePredictionPath(fixture)}
          >
            {sanitizeRound(fixture.round) && (
              <div className="fixture-card-round">{formatRoundLabel(fixture.round)}</div>
            )}
            <div className="fixture-card-date">{fixture.date}</div>
            {Number.isFinite(Number(fixture.best_ev ?? Number.NaN)) && (
              <div className={fixture.value_bet ? "fixture-card-status tag-value" : "fixture-card-status"}>
                EV {formatSignedPercentage(Number(fixture.best_ev))} · Pick {String(fixture.best_ev_pick ?? "-")}
              </div>
            )}
            <div className="fixture-card-body">
              <div className="fixture-team">
                <img
                  src={getTeamImgSources(fixture.home_team)[0]}
                  alt={fixture.home_team}
                  className="fixture-logo"
                  data-logo-fallback-step="0"
                  onError={(event) => {
                    loadNextTeamLogo(event.currentTarget, fixture.home_team);
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
                  onError={(event) => {
                    loadNextTeamLogo(event.currentTarget, fixture.away_team);
                  }}
                />
                <span className="fixture-team-name">{fixture.away_team}</span>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
