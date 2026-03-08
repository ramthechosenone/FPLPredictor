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


# ---------------------------------------------------------------------------
# Feature-building helpers
# ---------------------------------------------------------------------------

def _compute_rolling_features(gw_df: pd.DataFrame):
    """Compute rolling average features for all stat columns."""
    rolling_cols = {
        "total_points": "pts", "minutes": "min", "goals_scored": "goals",
        "assists": "ast", "bonus": "bonus", "bps": "bps",
        "expected_goals": "xg", "expected_assists": "xa",
        "expected_goal_involvements": "xgi", "expected_goals_conceded": "xgc",
        "clean_sheets": "cs", "influence": "infl", "creativity": "crea",
        "threat": "thrt", "ict_index": "ict",
        # Phase 1a: new rolling cols from unused FPL data
        "starts": "starts",
        "tackles": "tackles",
        "recoveries": "recov",
        "yellow_cards": "yc",
        "saves": "saves",
    }
    windows = [3, 5]

    for col, short in rolling_cols.items():
        shifted = gw_df.groupby("player_id")[col].shift(1)
        for w in windows:
            gw_df[f"{short}_roll{w}"] = (
                shifted.groupby(gw_df["player_id"])
                .transform(lambda x: x.rolling(w, min_periods=1).mean())
            )

    return gw_df, rolling_cols


def _compute_season_averages(gw_df, rolling_cols):
    """Compute expanding season-to-date averages."""
    season_cols = ["total_points", "minutes", "expected_goals", "expected_assists",
                   "bonus", "starts"]
    for col in season_cols:
        short = rolling_cols.get(col, col)
        shifted = gw_df.groupby("player_id")[col].shift(1)
        gw_df[f"{short}_season_avg"] = (
            shifted.groupby(gw_df["player_id"])
            .transform(lambda x: x.expanding(min_periods=1).mean())
        )
    return gw_df


def _compute_defensive_actions(gw_df):
    """Phase 1b: Composite defensive actions feature."""
    gw_df["def_actions"] = (
        gw_df["tackles"] + gw_df["recoveries"]
        + gw_df["clearances_blocks_interceptions"]
    )
    shifted = gw_df.groupby("player_id")["def_actions"].shift(1)
    for w in [3, 5]:
        gw_df[f"def_actions_roll{w}"] = (
            shifted.groupby(gw_df["player_id"])
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )
    return gw_df


def _compute_team_strength(gw_df):
    """Phase 1c: Team goals for/against from match scores."""
    gw_df["team_goals_for"] = np.where(
        gw_df["was_home"], gw_df["team_h_score"], gw_df["team_a_score"]
    )
    gw_df["team_goals_against"] = np.where(
        gw_df["was_home"], gw_df["team_a_score"], gw_df["team_h_score"]
    )
    for stat in ["team_goals_for", "team_goals_against"]:
        shifted = gw_df.groupby("player_id")[stat].shift(1)
        gw_df[f"{stat}_roll5"] = (
            shifted.groupby(gw_df["player_id"])
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )
    return gw_df


def _compute_h2h_features(gw_df):
    """Phase 1d: Head-to-head features (player vs specific opponent)."""
    gw_df = gw_df.sort_values(["player_id", "round"]).reset_index(drop=True)
    h2h_pts = []
    h2h_xg = []
    h2h_games = []

    for (pid, opp), grp in gw_df.groupby(["player_id", "opponent_team"]):
        pts_shifted = grp["total_points"].shift(1)
        xg_shifted = grp["expected_goals"].shift(1)
        avg_pts = pts_shifted.expanding(min_periods=1).mean()
        avg_xg = xg_shifted.expanding(min_periods=1).mean()
        games = pts_shifted.expanding(min_periods=1).count()
        h2h_pts.append(avg_pts)
        h2h_xg.append(avg_xg)
        h2h_games.append(games)

    if h2h_pts:
        gw_df["h2h_avg_pts"] = pd.concat(h2h_pts).reindex(gw_df.index)
        gw_df["h2h_avg_xg"] = pd.concat(h2h_xg).reindex(gw_df.index)
        gw_df["h2h_games"] = pd.concat(h2h_games).reindex(gw_df.index)
    else:
        gw_df["h2h_avg_pts"] = np.nan
        gw_df["h2h_avg_xg"] = np.nan
        gw_df["h2h_games"] = np.nan

    return gw_df


def _compute_transfer_momentum(gw_df):
    """Phase 1e: Transfer balance rolling mean and ownership signal."""
    shifted = gw_df.groupby("player_id")["transfers_balance"].shift(1)
    gw_df["transfers_balance_roll3"] = (
        shifted.groupby(gw_df["player_id"])
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    gw_df["log_selected"] = np.log1p(gw_df["selected"])
    return gw_df


def _compute_prev_season_features(data_dir):
    """Phase 1f: Previous season baseline from history_past."""
    histories_dir = data_dir / "player_histories"
    rows = []
    for hist_file in histories_dir.glob("*.json"):
        pid = int(hist_file.stem.split("_")[1])
        with open(hist_file) as f:
            data = json.load(f)
        past = data.get("history_past", [])
        if not past:
            continue
        latest_past = past[-1]
        minutes = latest_past.get("minutes", 0)
        if minutes and minutes > 0:
            nineties = minutes / 90
            pts = latest_past.get("total_points", 0)
            xg = float(latest_past.get("expected_goals", 0))
            rows.append({
                "player_id": pid,
                "prev_season_pts_per90": pts / nineties,
                "prev_season_xg_per90": xg / nineties,
                "prev_season_minutes": minutes,
            })

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["player_id", "prev_season_pts_per90",
                                  "prev_season_xg_per90", "prev_season_minutes"])


def _compute_vaastav_h2h(gw_df, data_dir):
    """Phase 2b: Multi-season H2H from vaastav historical data.
    
    Vaastav CSVs have 'name' (web_name) and 'opponent_team' columns.
    We match to current player_id via web_name from bootstrap.
    """
    ext_dir = data_dir.parent / "external" / "vaastav"
    if not ext_dir.exists():
        return gw_df

    bootstrap_path = data_dir / "bootstrap_static.json"
    if not bootstrap_path.exists():
        return gw_df
    with open(bootstrap_path) as f:
        bootstrap = json.load(f)
    # Map web_name -> player_id (current season)
    name_to_pid = {p["web_name"]: p["id"] for p in bootstrap["elements"]}

    past_rows = []
    for season_dir in ext_dir.iterdir():
        csv_path = season_dir / "merged_gw.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "name" not in df.columns:
            continue
        # Match by name — imperfect but covers most players
        df["player_id"] = df["name"].map(name_to_pid)
        df = df.dropna(subset=["player_id"])
        df["player_id"] = df["player_id"].astype(int)
        if "opponent_team" in df.columns and "total_points" in df.columns:
            past_rows.append(df[["player_id", "opponent_team", "total_points"]].copy())

    if not past_rows:
        return gw_df

    past_df = pd.concat(past_rows, ignore_index=True)
    prev_h2h = (past_df.groupby(["player_id", "opponent_team"])["total_points"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "h2h_prev_season_avg_pts", "count": "h2h_prev_season_games"})
                .reset_index())

    gw_df = gw_df.merge(prev_h2h, on=["player_id", "opponent_team"], how="left")
    return gw_df


def _compute_trajectory_features(gw_df):
    """Phase 2c: Season-over-season improvement trajectories."""
    if "prev_season_xg_per90" not in gw_df.columns:
        return gw_df

    shifted_xg = gw_df.groupby("player_id")["expected_goals"].shift(1)
    shifted_min = gw_df.groupby("player_id")["minutes"].shift(1)
    cum_xg = shifted_xg.groupby(gw_df["player_id"]).transform(lambda x: x.expanding(1).sum())
    cum_min = shifted_min.groupby(gw_df["player_id"]).transform(lambda x: x.expanding(1).sum())
    current_xg_per90 = cum_xg / (cum_min / 90).replace(0, np.nan)

    shifted_pts = gw_df.groupby("player_id")["total_points"].shift(1)
    cum_pts = shifted_pts.groupby(gw_df["player_id"]).transform(lambda x: x.expanding(1).sum())
    current_pts_per90 = cum_pts / (cum_min / 90).replace(0, np.nan)

    gw_df["xg_improvement"] = current_xg_per90 - gw_df["prev_season_xg_per90"]
    gw_df["pts_improvement"] = current_pts_per90 - gw_df["prev_season_pts_per90"]
    return gw_df


def _compute_fbref_features(gw_df, data_dir):
    """Phase 3: FBref advanced stats (graceful degradation if not available)."""
    fbref_dir = data_dir.parent / "external" / "fbref"
    if not fbref_dir.exists():
        return gw_df

    csvs = sorted(fbref_dir.glob("player_stats_*.csv"))
    if not csvs:
        return gw_df

    try:
        fbref_df = pd.read_csv(csvs[-1])
    except Exception:
        return gw_df

    if "player_id" not in fbref_df.columns:
        return gw_df

    fbref_cols = [c for c in ["touches", "passes_completed", "pass_pct",
                               "key_passes", "sca", "progressive_passes",
                               "progressive_carries", "aerials_won"]
                  if c in fbref_df.columns]

    if not fbref_cols or "round" not in fbref_df.columns:
        return gw_df

    gw_df = gw_df.merge(
        fbref_df[["player_id", "round"] + fbref_cols],
        on=["player_id", "round"], how="left"
    )

    for col in fbref_cols:
        shifted = gw_df.groupby("player_id")[col].shift(1)
        gw_df[f"{col}_roll5"] = (
            shifted.groupby(gw_df["player_id"])
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )

    return gw_df


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features(data_dir=None):
    """
    Build the feature matrix from raw data.

    Replicates the feature engineering from notebook 03,
    producing the same columns the model was trained on,
    plus enhanced features from Phases 1-3.
    """
    if data_dir is None:
        data_dir = get_project_root() / "data" / "raw"

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
        players_meta[["id", "code", "web_name", "position", "team_name", "element_type", "team"]],
        left_on="player_id", right_on="id", how="left"
    )

    # Type conversions
    float_cols = ["expected_goals", "expected_assists", "expected_goal_involvements",
                  "expected_goals_conceded", "influence", "creativity", "threat", "ict_index"]
    for col in float_cols:
        gw_df[col] = gw_df[col].astype(float)

    for col in ["tackles", "recoveries", "clearances_blocks_interceptions",
                "starts", "saves", "yellow_cards", "transfers_balance", "selected",
                "team_h_score", "team_a_score"]:
        if col in gw_df.columns:
            gw_df[col] = pd.to_numeric(gw_df[col], errors="coerce").fillna(0)

    gw_df["price"] = gw_df["value"] / 10
    gw_df["kickoff_time"] = pd.to_datetime(gw_df["kickoff_time"])
    gw_df = gw_df.sort_values(["player_id", "round"]).reset_index(drop=True)

    # Phase 1a: Rolling features
    gw_df, rolling_cols = _compute_rolling_features(gw_df)

    # Phase 1a: Season averages
    gw_df = _compute_season_averages(gw_df, rolling_cols)

    gw_df["games_played"] = gw_df.groupby("player_id").cumcount()

    # Phase 1b: Defensive actions
    gw_df = _compute_defensive_actions(gw_df)

    # Phase 1c: Team strength
    gw_df = _compute_team_strength(gw_df)

    # Phase 1d: Head-to-head
    gw_df = _compute_h2h_features(gw_df)

    # Phase 1e: Transfer momentum
    gw_df = _compute_transfer_momentum(gw_df)

    # Phase 1f: Previous season baseline
    prev_season_df = _compute_prev_season_features(data_dir)
    if not prev_season_df.empty:
        gw_df = gw_df.merge(prev_season_df, on="player_id", how="left")

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

    # Phase 2b: Multi-season H2H from vaastav
    gw_df = _compute_vaastav_h2h(gw_df, data_dir)

    # Phase 2c: Season-over-season trajectory
    gw_df = _compute_trajectory_features(gw_df)

    # Phase 3: FBref advanced stats
    gw_df = _compute_fbref_features(gw_df, data_dir)

    gw_df["target"] = gw_df["total_points"]

    return gw_df


def get_feature_columns():
    """Return the full list of feature columns for the enhanced model (v3)."""
    base_rolling = [
        "pts", "min", "goals", "ast", "bonus", "bps",
        "xg", "xa", "xgi", "xgc", "cs",
        "infl", "crea", "thrt", "ict",
        "starts", "tackles", "recov", "yc", "saves",
    ]
    rolling_features = [f"{s}_roll{w}" for s in base_rolling for w in [3, 5]]

    season_features = ["pts_season_avg", "min_season_avg", "xg_season_avg",
                       "xa_season_avg", "bonus_season_avg", "starts_season_avg"]

    defensive = ["def_actions_roll3", "def_actions_roll5"]

    team_strength = ["team_goals_for_roll5", "team_goals_against_roll5"]

    h2h = ["h2h_avg_pts", "h2h_avg_xg", "h2h_games"]

    transfer = ["transfers_balance_roll3", "log_selected"]

    prev_season = ["prev_season_pts_per90", "prev_season_xg_per90", "prev_season_minutes"]

    vaastav_h2h = ["h2h_prev_season_avg_pts", "h2h_prev_season_games"]

    trajectory = ["xg_improvement", "pts_improvement"]

    original_other = ["was_home", "fdr", "price", "price_change_1gw",
                      "price_change_3gw", "rest_days", "games_played", "element_type"]

    all_features = (rolling_features + season_features + defensive + team_strength
                    + h2h + transfer + prev_season + vaastav_h2h + trajectory
                    + original_other)
    return all_features


def predict_next_gw(model=None, metadata=None, data_dir=None):
    """
    Generate predictions for the latest gameweek's players.

    Returns a DataFrame with player info and predicted points,
    sorted by predicted points descending.
    """
    if model is None or metadata is None:
        model, metadata = load_model()

    feature_cols = metadata["feature_columns"]
    gw_df = build_features(data_dir)

    latest = gw_df.groupby("player_id").tail(1).copy()
    latest = latest[latest["minutes"] > 0]
    # Only require core features to be non-null; XGBoost handles NaN natively
    core_cols = [c for c in feature_cols if "h2h" not in c and "prev_season" not in c
                 and "improvement" not in c]
    latest = latest.dropna(subset=core_cols)

    latest["predicted_points"] = model.predict(latest[feature_cols])

    result = latest[["player_id", "code", "web_name", "position", "team_name", "price",
                      "round", "predicted_points"]].copy()
    result = result.sort_values("predicted_points", ascending=False).reset_index(drop=True)
    result["predicted_points"] = result["predicted_points"].round(2)

    return result
