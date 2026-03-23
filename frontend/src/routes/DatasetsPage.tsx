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

  return `/partido?${params.toString()}`;
}

export function DatasetsPage() {
  const [fixtures, setFixtures] = useState<UpcomingFixtureOption[]>([]);
  const [selectedRound, setSelectedRound] = useState<string>("");
  const [searchText, setSearchText] = useState<string>("");
  const setToast = useAppStore((state) => state.setToast);

  const loadFixtures = async () => {
    try {
      const response = await getUpcomingFixtures();
      setFixtures(response.fixtures);

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
    }
  };

  useEffect(() => {
    void loadFixtures();
  }, []);

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
    return [...filteredFixtures].sort(compareByDate);
  }, [filteredFixtures]);

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
