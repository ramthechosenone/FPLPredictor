export const POSITIONS = ["ALL", "GK", "DEF", "MID", "FWD"] as const;

export const TEAMS = [
  "All Teams",
  "Arsenal",
  "Aston Villa",
  "Bournemouth",
  "Brentford",
  "Brighton",
  "Chelsea",
  "Crystal Palace",
  "Everton",
  "Fulham",
  "Ipswich",
  "Leicester",
  "Liverpool",
  "Man City",
  "Man Utd",
  "Newcastle",
  "Nott'm Forest",
  "Southampton",
  "Spurs",
  "West Ham",
  "Wolves",
] as const;

export const POSITION_COLORS: Record<string, string> = {
  GK: "position-gk",
  DEF: "position-def",
  MID: "position-mid",
  FWD: "position-fwd",
};
