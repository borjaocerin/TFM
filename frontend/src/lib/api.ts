import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

export const BACKEND_ORIGIN = import.meta.env.VITE_BACKEND_ORIGIN ?? "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000
});

export type DatasetIngestPayload = {
  historical: string;
  football_data_dir: string;
  elo_csv?: string;
  team_map?: string;
  windows: number[];
};

export type DatasetIngestResponse = {
  rows_total: number;
  rows_by_season: Record<string, number>;
  missing_pct_by_column: Record<string, number>;
  columns: string[];
  output_all: string;
  output_model: string;
};

export type FixturesFeaturesPayload = {
  fixtures_csv: string;
  historical_csv?: string;
  elo_csv?: string;
  team_map?: string;
  windows: number[];
};

export type FixturesFeaturesResponse = {
  rows_total: number;
  generated_columns: string[];
  output_path: string;
};

export type TrainPayload = {
  dataset_path?: string;
  use_xgb: boolean;
  use_catboost?: boolean;
  calibration: "platt" | "isotonic";
};

export type TrainResponse = {
  best_model: string;
  metrics: Record<string, number>;
  leaderboard: Array<Record<string, string | number>>;
  model_path: string;
  metadata_path: string;
  reliability_plot?: string;
  metrics_report_path?: string;
};

export type UpcomingFixtureOption = {
  fixture_id: string;
  date: string;
  home_team: string;
  away_team: string;
  label: string;
  round?: string; // jornada
};

export type UpcomingFixturesResponse = {
  season_label: string;
  source_path: string;
  rows: number;
  fixtures: UpcomingFixtureOption[];
  error?: string | null;
};

export type PredictUpcomingPayload = {
  date: string;
  home_team: string;
  away_team: string;
};

export type PredictUpcomingResponse = {
  season_label: string;
  selected_fixture: {
    date: string;
    home_team: string;
    away_team: string;
    round?: string;
  };
  prediction: Record<string, unknown>;
  market_odds?: Record<string, unknown> | null;
  output_csv: string;
};

export type UpcomingOddsOption = {
  fixture_id: string;
  event_id: string;
  date: string;
  home_team: string;
  away_team: string;
  source: string;
  bookmakers: number;
  odds_avg_h?: number | null;
  odds_avg_d?: number | null;
  odds_avg_a?: number | null;
  odds_best_h?: number | null;
  odds_best_d?: number | null;
  odds_best_a?: number | null;
};

export type UpcomingOddsResponse = {
  sport_key: string;
  source_path: string;
  rows: number;
  requests_remaining: string;
  requests_used: string;
  odds: UpcomingOddsOption[];
};

export type PredictPayload = {
  fixtures_enriched_path?: string;
  fixtures?: Array<Record<string, unknown>>;
};

export type PredictResponse = {
  rows: number;
  output_csv: string;
  predictions: Array<Record<string, unknown>>;
};

export type OddsComparePayload = {
  predictions_csv?: string;
  predictions?: Array<Record<string, unknown>>;
  odds_kind: "odds_avg" | "odds_close";
  value_threshold: number;
};

export type OddsCompareResponse = {
  rows: number;
  metrics: Record<string, number | null>;
  value_bets: Array<Record<string, unknown>>;
  output_csv: string;
};

export async function ingestDatasets(
  payload: DatasetIngestPayload
): Promise<DatasetIngestResponse> {
  const response = await api.post<DatasetIngestResponse>("/datasets/ingest", payload);
  return response.data;
}

export async function buildFixturesFeatures(
  payload: FixturesFeaturesPayload
): Promise<FixturesFeaturesResponse> {
  const response = await api.post<FixturesFeaturesResponse>("/features/fixtures", payload);
  return response.data;
}

export async function trainModel(payload: TrainPayload): Promise<TrainResponse> {
  const response = await api.post<TrainResponse>("/model/train", payload);
  return response.data;
}

export async function getActiveModel(): Promise<{
  model_available: boolean;
  metadata: Record<string, unknown> | null;
}> {
  const response = await api.get("/model/active");
  return response.data;
}

export async function getUpcomingFixtures(): Promise<UpcomingFixturesResponse> {
  const response = await api.get<UpcomingFixturesResponse>("/predict/options/upcoming");
  return response.data;
}

export async function predictUpcomingFixture(
  payload: PredictUpcomingPayload
): Promise<PredictUpcomingResponse> {
  const response = await api.post<PredictUpcomingResponse>("/predict/upcoming", payload);
  return response.data;
}

export async function getUpcomingOdds(limit = 100): Promise<UpcomingOddsResponse> {
  const response = await api.get<UpcomingOddsResponse>("/odds/upcoming", {
    params: { limit }
  });
  return response.data;
}

export async function predict(payload: PredictPayload): Promise<PredictResponse> {
  const response = await api.post<PredictResponse>("/predict", payload);
  return response.data;
}

export async function compareOdds(payload: OddsComparePayload): Promise<OddsCompareResponse> {
  const response = await api.post<OddsCompareResponse>("/odds/compare", payload);
  return response.data;
}
