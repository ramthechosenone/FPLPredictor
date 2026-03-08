export interface Player {
  rank: number;
  player_id: number;
  name: string;
  position: "GK" | "DEF" | "MID" | "FWD";
  team: string;
  price: number;
  predicted_points: number;
}

export interface PredictionsResponse {
  count: number;
  players: Player[];
}

export interface HealthResponse {
  status: string;
  model: string;
  metrics: {
    MAE: number;
    RMSE: number;
    "R²": number;
  };
}
