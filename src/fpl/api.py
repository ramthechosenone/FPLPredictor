"""
FPL Prediction API

FastAPI server that serves predictions from the trained model.
"""

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from .predict import load_model, predict_next_gw

app = FastAPI(
    title="FPL Predictor API",
    description="Predict Fantasy Premier League player points using ML",
    version="1.0.0",
)

# Load model once at startup
model, metadata = load_model()
predictions_cache = None


def get_predictions():
    global predictions_cache
    if predictions_cache is None:
        predictions_cache = predict_next_gw(model=model, metadata=metadata)
    return predictions_cache


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": metadata["model_name"],
        "metrics": metadata["metrics"],
    }


@app.get("/predict")
def predict_player(player_id: int = Query(..., description="Player element ID")):
    """Predict next-GW points for a specific player."""
    df = get_predictions()
    player = df[df["player_id"] == player_id]
    if player.empty:
        return JSONResponse(
            status_code=404,
            content={"error": f"Player {player_id} not found or has insufficient data"},
        )
    row = player.iloc[0]
    return {
        "player_id": int(row["player_id"]),
        "name": row["web_name"],
        "position": row["position"],
        "team": row["team_name"],
        "price": float(row["price"]),
        "predicted_points": float(row["predicted_points"]),
        "based_on_gw": int(row["round"]),
    }


@app.get("/predict/top")
def predict_top(n: int = Query(15, ge=1, le=100, description="Number of top players")):
    """Get top N predicted players for next GW."""
    df = get_predictions().head(n)
    return {
        "count": len(df),
        "players": [
            {
                "rank": i + 1,
                "player_id": int(row["player_id"]),
                "name": row["web_name"],
                "position": row["position"],
                "team": row["team_name"],
                "price": float(row["price"]),
                "predicted_points": float(row["predicted_points"]),
            }
            for i, row in df.iterrows()
        ],
    }


@app.get("/predict/position/{position}")
def predict_by_position(
    position: str,
    n: int = Query(10, ge=1, le=50, description="Number of players"),
):
    """Get top predicted players for a specific position (GK, DEF, MID, FWD)."""
    position = position.upper()
    if position not in ("GK", "DEF", "MID", "FWD"):
        return JSONResponse(
            status_code=400,
            content={"error": "Position must be GK, DEF, MID, or FWD"},
        )
    df = get_predictions()
    pos_df = df[df["position"] == position].head(n)
    return {
        "position": position,
        "count": len(pos_df),
        "players": [
            {
                "rank": i + 1,
                "player_id": int(row["player_id"]),
                "name": row["web_name"],
                "team": row["team_name"],
                "price": float(row["price"]),
                "predicted_points": float(row["predicted_points"]),
            }
            for i, row in pos_df.iterrows()
        ],
    }
