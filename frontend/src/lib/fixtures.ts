export const LOGOS_LALIGA_ALIASES: Record<string, string> = {
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
  athletic_bilbao: "athletic",
  club_atletico_de_madrid: "atletico_madrid",
  deportivo_alaves: "alaves",
  elche_cf: "elche",
  espanol: "espanyol",
  espanyol: "espanyol",
  levante_ud: "levante",
  oviedo: "real_oviedo",
  rcd_espanyol_de_barcelona: "espanyol",
  rc_celta_de_vigo: "celta",
  real_betis: "betis",
  real_betis_balompie: "betis",
  real_oviedo: "real_oviedo",
  ca_osasuna: "osasuna",
  celta_vigo: "celta"
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

export function getTeamImgSources(team: string): string[] {
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

export function loadNextTeamLogo(img: HTMLImageElement, team: string): void {
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

export function sanitizeRound(round?: string | null): string {
  const value = String(round ?? "").trim();

  if (!value || ["undefined", "null", "none", "nan"].includes(value.toLowerCase())) {
    return "";
  }

  return value;
}

export function formatRoundLabel(round?: string | null): string {
  const value = sanitizeRound(round);

  if (!value) {
    return "";
  }

  if (/jornada|matchday|round/i.test(value)) {
    return value;
  }

  return `Jornada ${value}`;
}

export function getRoundSortValue(round: string): number {
  const match = round.match(/\d+/);

  if (!match) {
    return Number.MAX_SAFE_INTEGER;
  }

  return Number.parseInt(match[0], 10);
}