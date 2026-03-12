import { describe, expect, it } from "vitest";

import { buildMarketComparison, buildRequiredOddsGuide } from "./odds";

describe("buildMarketComparison", () => {
  it("computes expected value and detects value bets", () => {
    const comparison = buildMarketComparison(
      { home: 0.55, draw: 0.25, away: 0.2 },
      { home: 2.3, draw: 3.4, away: 3.6 },
      0.02
    );

    expect(comparison).not.toBeNull();
    expect(comparison?.bestOption.key).toBe("H");
    expect(comparison?.bestOption.expectedValue).toBeCloseTo(0.265, 3);
    expect(comparison?.hasValueBet).toBe(true);
  });
});

describe("buildRequiredOddsGuide", () => {
  it("returns break-even and threshold odds for each outcome", () => {
    const guide = buildRequiredOddsGuide(
      { home: 0.5, draw: 0.3, away: 0.2 },
      0.02
    );

    expect(guide).not.toBeNull();
    expect(guide?.bestOption.key).toBe("H");
    expect(guide?.rows[0].breakEvenOdds).toBeCloseTo(2, 6);
    expect(guide?.rows[0].thresholdOdds).toBeCloseTo(2.04, 6);
    expect(guide?.rows[1].breakEvenOdds).toBeCloseTo(3.333333, 5);
  });
});
