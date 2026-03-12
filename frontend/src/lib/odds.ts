export type OutcomeValues = {
  home: number;
  draw: number;
  away: number;
};

export type MarketComparisonRow = {
  key: "H" | "D" | "A";
  label: string;
  modelProbability: number;
  odds: number;
  marketProbability: number;
  expectedValue: number;
  isValueBet: boolean;
};

export type MarketComparison = {
  rows: MarketComparisonRow[];
  bestOption: MarketComparisonRow;
  hasValueBet: boolean;
};

export type RequiredOddsRow = {
  key: "H" | "D" | "A";
  label: string;
  modelProbability: number;
  breakEvenOdds: number;
  thresholdOdds: number;
};

export type RequiredOddsGuide = {
  rows: RequiredOddsRow[];
  bestOption: RequiredOddsRow;
};

const OUTCOME_DEFINITIONS = [
  { key: "H" as const, label: "1 · Local", field: "home" as const },
  { key: "D" as const, label: "X · Empate", field: "draw" as const },
  { key: "A" as const, label: "2 · Visitante", field: "away" as const }
];

export function buildMarketComparison(
  probabilities: OutcomeValues,
  odds: OutcomeValues,
  valueThreshold: number
): MarketComparison | null {
  const rows = OUTCOME_DEFINITIONS.map((definition) => ({
    key: definition.key,
    label: definition.label,
    modelProbability: probabilities[definition.field],
    odds: odds[definition.field]
  }));

  if (
    rows.some(
      (row) => !Number.isFinite(row.modelProbability) || !Number.isFinite(row.odds) || row.odds <= 0
    )
  ) {
    return null;
  }

  const normalizedTotal = rows.reduce((total, row) => total + 1 / row.odds, 0);
  if (!Number.isFinite(normalizedTotal) || normalizedTotal <= 0) {
    return null;
  }

  const comparisonRows = rows.map((row) => {
    const marketProbability = (1 / row.odds) / normalizedTotal;
    const expectedValue = row.modelProbability * row.odds - 1;
    return {
      ...row,
      marketProbability,
      expectedValue,
      isValueBet: expectedValue > valueThreshold
    };
  });

  const bestOption = comparisonRows.reduce((best, current) =>
    current.expectedValue > best.expectedValue ? current : best
  );

  return {
    rows: comparisonRows,
    bestOption,
    hasValueBet: bestOption.expectedValue > valueThreshold
  };
}

export function buildRequiredOddsGuide(
  probabilities: OutcomeValues,
  valueThreshold: number
): RequiredOddsGuide | null {
  const rows = OUTCOME_DEFINITIONS.map((definition) => ({
    key: definition.key,
    label: definition.label,
    modelProbability: probabilities[definition.field]
  }));

  if (
    rows.some(
      (row) =>
        !Number.isFinite(row.modelProbability) || row.modelProbability <= 0 || row.modelProbability >= 1
    )
  ) {
    return null;
  }

  const requiredRows = rows.map((row) => ({
    ...row,
    breakEvenOdds: 1 / row.modelProbability,
    thresholdOdds: (1 + valueThreshold) / row.modelProbability
  }));

  const bestOption = requiredRows.reduce((best, current) =>
    current.thresholdOdds < best.thresholdOdds ? current : best
  );

  return {
    rows: requiredRows,
    bestOption
  };
}
