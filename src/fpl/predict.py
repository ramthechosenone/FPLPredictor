"""
FPL Prediction Pipeline

Loads the trained model and generates predictions for players
based on their recent gameweek history.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def load_model(models_dir: Path = None):
    """Load the trained model and metadata."""
    if models_dir is None:
        models_dir = get_project_root() / "models"

    model = joblib.load(models_dir / "best_model.joblib")
    with open(models_dir / "model_metadata.json") as f:
        metadata = json.load(f)

    return model, metadata


def build_features(data_dir: Path = None) -> pd.DataFrame:
    """
    Build the feature matrix from raw data.

    Replicates the feature engineering from notebook 03,
    producing the same columns the model was trained on.
    """
    if data_dir is None:
        data_dir = get_project_root() / "data" / "raw"

    # Load bootstrap
    with open(data_dir / "bootstrap_static.json") as f:
        bootstrap = json.load(f)
    with open(data_dir / "fixtures.json") as f:
        fixtures_raw = json.load(f)

    players_meta = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])
    fixtures = pd.DataFrame(fixtures_raw)

    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    team_map = dict(zip(teams["id"], teams["name"]))
    players_meta["position"] = players_meta["element_type"].map(pos_map)
    players_meta["team_name"] = players_meta["team"].map(team_map)

    # Load all histories
    histories_dir = data_dir / "player_histories"
    rows = []
    for hist_file in histories_dir.glob("*.json"):
        pid = int(hist_file.stem.split("_")[1])
        with open(hist_file) as f:
            data = json.load(f)
        for gw in data["history"]:
            gw["player_id"] = pid
            rows.append(gw)

    gw_df = pd.DataFrame(rows)
    gw_df = gw_df.merge(
        players_meta[["id", "web_name", "position", "team_name", "element_type", "team"]],
        left_on="player_id", right_on="id", how="left"
    )

    # Type conversions
    float_cols = ["expected_goals", "expected_assists", "expected_goal_involvements",
                  "expected_goals_conceded", "influence", "creativity", "threat", "ict_index"]
    for col in float_cols:
        gw_df[col] = gw_df[col].astype(float)

    gw_df["price"] = gw_df["value"] / 10
    gw_df["kickoff_time"] = pd.to_datetime(gw_df["kickoff_time"])
    gw_df = gw_df.sort_values(["player_id", "round"]).reset_index(drop=True)

    # Rolling features
    rolling_cols = {
        "total_points": "pts", "minutes": "min", "goals_scored": "goals",
        "assists": "ast", "bonus": "bonus", "bps": "bps",
        "expected_goals": "xg", "expected_assists": "xa",
        "expected_goal_involvements": "xgi", "expected_goals_conceded": "xgc",
        "clean_sheets": "cs", "influence": "infl", "creativity": "crea",
        "threat": "thrt", "ict_index": "ict",
    }
    windows = [3, 5]

    for col, short in rolling_cols.items():
        shifted = gw_df.groupby("player_id")[col].shift(1)
        for w in windows:
            gw_df[f"{short}_roll{w}"] = (
                shifted.groupby(gw_df["player_id"])
                .transform(lambda x: x.rolling(w, min_periods=1).mean())
            )

    # Season-to-date
    season_cols = ["total_points", "minutes", "expected_goals", "expected_assists", "bonus"]
    for col in season_cols:
        short = rolling_cols.get(col, col)
        shifted = gw_df.groupby("player_id")[col].shift(1)
        gw_df[f"{short}_season_avg"] = (
            shifted.groupby(gw_df["player_id"])
            .transform(lambda x: x.expanding(min_periods=1).mean())
        )

    gw_df["games_played"] = gw_df.groupby("player_id").cumcount()

    # FDR
    fdr_rows = []
    for _, fix in fixtures.iterrows():
        if pd.isna(fix.get("event")):
            continue
        gw = int(fix["event"])
        fdr_rows.append({"team": fix["team_h"], "round": gw, "fdr": fix["team_h_difficulty"]})
        fdr_rows.append({"team": fix["team_a"], "round": gw, "fdr": fix["team_a_difficulty"]})

    fdr_df = pd.DataFrame(fdr_rows)
    gw_df = gw_df.merge(fdr_df[["team", "round", "fdr"]], on=["team", "round"], how="left")

    # Price momentum
    gw_df["price_prev"] = gw_df.groupby("player_id")["price"].shift(1)
    gw_df["price_change_1gw"] = gw_df["price"] - gw_df["price_prev"]
    price_3gw = gw_df.groupby("player_id")["price"].shift(3)
    gw_df["price_change_3gw"] = gw_df["price"] - price_3gw

    # Rest days
    gw_df["prev_kickoff"] = gw_df.groupby("player_id")["kickoff_time"].shift(1)
    gw_df["rest_days"] = (gw_df["kickoff_time"] - gw_df["prev_kickoff"]).dt.days

    gw_df["target"] = gw_df["total_points"]

    return gw_df


def predict_next_gw(model=None, metadata=None, data_dir: Path = None) -> pd.DataFrame:
    """
    Generate predictions for the latest gameweek's players.

    Returns a DataFrame with player info and predicted points,
    sorted by predicted points descending.
    """
    if model is None or metadata is None:
        model, metadata = load_model()

    feature_cols = metadata["feature_columns"]
    gw_df = build_features(data_dir)

    # Use each player's most recent row (latest GW with data)
    latest = gw_df.groupby("player_id").tail(1).copy()
    latest = latest.dropna(subset=feature_cols)

    latest["predicted_points"] = model.predict(latest[feature_cols])

    result = latest[["player_id", "web_name", "position", "team_name", "price",
                      "round", "predicted_points"]].copy()
    result = result.sort_values("predicted_points", ascending=False).reset_index(drop=True)
    result["predicted_points"] = result["predicted_points"].round(2)

    return result
