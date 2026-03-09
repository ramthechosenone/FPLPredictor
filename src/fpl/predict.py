"""
FPL Prediction Pipeline v4

Loads the trained model and generates predictions for players
based on their recent gameweek history with next-GW fixture awareness.
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
# Feature-building helpers (v3 base)
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
    """Composite defensive actions feature."""
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
    """Team goals for/against from match scores."""
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
    """Head-to-head features (player vs specific opponent)."""
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
    """Transfer balance rolling mean."""
    shifted = gw_df.groupby("player_id")["transfers_balance"].shift(1)
    gw_df["transfers_balance_roll3"] = (
        shifted.groupby(gw_df["player_id"])
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    return gw_df


def _compute_prev_season_features(data_dir):
    """Previous season baseline from history_past."""
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
    """Multi-season H2H from vaastav historical data."""
    ext_dir = data_dir.parent / "external" / "vaastav"
    if not ext_dir.exists():
        return gw_df

    bootstrap_path = data_dir / "bootstrap_static.json"
    if not bootstrap_path.exists():
        return gw_df
    with open(bootstrap_path) as f:
        bootstrap = json.load(f)
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
    """Season-over-season improvement trajectories."""
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
    """FBref advanced stats (graceful degradation if not available)."""
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
# v4 NEW helpers: Opponent strength, EMA, Last-1, Form acceleration
# ---------------------------------------------------------------------------

def _get_team_strengths(data_dir: Path) -> Dict[int, Dict]:
    """Load team strength ratings from bootstrap data."""
    with open(data_dir / "bootstrap_static.json") as f:
        bootstrap = json.load(f)
    strengths = {}
    for t in bootstrap["teams"]:
        strengths[t["id"]] = {
            "attack_home": t["strength_attack_home"],
            "attack_away": t["strength_attack_away"],
            "defence_home": t["strength_defence_home"],
            "defence_away": t["strength_defence_away"],
        }
    return strengths


def _get_next_gw_number(data_dir: Path) -> int:
    """Return the GW number with is_next=True."""
    with open(data_dir / "bootstrap_static.json") as f:
        bootstrap = json.load(f)
    for e in bootstrap["events"]:
        if e.get("is_next"):
            return e["id"]
    finished = [e["id"] for e in bootstrap["events"] if e.get("finished")]
    return max(finished) + 1 if finished else 1


def _get_next_gw_fixtures(data_dir: Path, next_gw: int, team_strengths: Dict) -> Dict:
    """Build per-team fixture info for the next GW.

    Returns {team_id: {fdr, is_home, opp_attack, opp_defence, opp_team}}.
    """
    with open(data_dir / "fixtures.json") as f:
        fixtures = json.load(f)

    league_mean = 1200
    result = {}
    for fix in fixtures:
        if fix.get("event") != next_gw:
            continue

        th, ta = fix["team_h"], fix["team_a"]
        opp_a = team_strengths.get(ta, {})
        opp_h = team_strengths.get(th, {})

        # Home team faces away opponent
        result[th] = {
            "fdr": fix.get("team_h_difficulty", 3),
            "is_home": 1,
            "opp_attack": opp_a.get("attack_away", league_mean) - league_mean,
            "opp_defence": opp_a.get("defence_away", league_mean) - league_mean,
            "opp_team": ta,
        }
        # Away team faces home opponent
        result[ta] = {
            "fdr": fix.get("team_a_difficulty", 3),
            "is_home": 0,
            "opp_attack": opp_h.get("attack_home", league_mean) - league_mean,
            "opp_defence": opp_h.get("defence_home", league_mean) - league_mean,
            "opp_team": th,
        }
    return result


def _compute_opponent_strength(gw_df: pd.DataFrame, data_dir: Path):
    """v4 Phase 1: Add opponent strength features for training rows.

    For each row we know opponent_team and was_home. We look up the
    opponent's contextual strength ratings.
    """
    team_strengths = _get_team_strengths(data_dir)
    league_mean = 1200

    # Vectorized approach using map
    opp_att_home = gw_df["opponent_team"].map(lambda t: team_strengths.get(t, {}).get("attack_home", league_mean))
    opp_att_away = gw_df["opponent_team"].map(lambda t: team_strengths.get(t, {}).get("attack_away", league_mean))
    opp_def_home = gw_df["opponent_team"].map(lambda t: team_strengths.get(t, {}).get("defence_home", league_mean))
    opp_def_away = gw_df["opponent_team"].map(lambda t: team_strengths.get(t, {}).get("defence_away", league_mean))

    is_home = gw_df["was_home"].astype(bool)
    # If player is home, opponent plays away; if player is away, opponent plays at home
    gw_df["next_opp_attack"] = np.where(is_home, opp_att_away, opp_att_home) - league_mean
    gw_df["next_opp_defence"] = np.where(is_home, opp_def_away, opp_def_home) - league_mean
    gw_df["next_fdr"] = gw_df["fdr"]
    gw_df["is_home_next"] = gw_df["was_home"].astype(int)
    gw_df["opp_strength_diff"] = gw_df["next_opp_attack"] - gw_df["next_opp_defence"]

    # Player's own team strength
    own_att_home = gw_df["team"].map(lambda t: team_strengths.get(t, {}).get("attack_home", league_mean))
    own_att_away = gw_df["team"].map(lambda t: team_strengths.get(t, {}).get("attack_away", league_mean))
    own_def_home = gw_df["team"].map(lambda t: team_strengths.get(t, {}).get("defence_home", league_mean))
    own_def_away = gw_df["team"].map(lambda t: team_strengths.get(t, {}).get("defence_away", league_mean))

    own_attack = np.where(is_home, own_att_home, own_att_away)
    own_defence = np.where(is_home, own_def_home, own_def_away)
    opp_attack_raw = np.where(is_home, opp_att_away, opp_att_home)
    opp_defence_raw = np.where(is_home, opp_def_away, opp_def_home)

    own_strength = (own_attack + own_defence) / 2
    opp_strength = (opp_attack_raw + opp_defence_raw) / 2
    gw_df["team_vs_opp"] = own_strength - opp_strength

    return gw_df


def _compute_ema_features(gw_df: pd.DataFrame):
    """v4 Phase 2a: Exponential moving averages for recency weighting."""
    ema_cols = {
        "total_points": "pts", "expected_goals": "xg", "expected_assists": "xa",
        "expected_goal_involvements": "xgi", "bonus": "bonus", "bps": "bps",
        "ict_index": "ict",
    }
    for col, short in ema_cols.items():
        shifted = gw_df.groupby("player_id")[col].shift(1)
        for span in [3, 5]:
            gw_df[f"{short}_ema{span}"] = (
                shifted.groupby(gw_df["player_id"])
                .transform(lambda x: x.ewm(span=span, min_periods=1).mean())
            )
    return gw_df


def _compute_last1_features(gw_df: pd.DataFrame):
    """v4 Phase 2b: Last single-GW raw features."""
    last1_cols = {
        "total_points": "pts", "expected_goals": "xg", "expected_assists": "xa",
        "bonus": "bonus", "minutes": "min",
    }
    for col, short in last1_cols.items():
        gw_df[f"{short}_last1"] = gw_df.groupby("player_id")[col].shift(1)
    return gw_df


def _compute_form_acceleration(gw_df: pd.DataFrame):
    """v4 Phase 3: Form acceleration / trend features."""
    gw_df["pts_accel_3v5"] = gw_df["pts_roll3"] - gw_df["pts_roll5"]
    gw_df["xg_accel_3v5"] = gw_df["xg_roll3"] - gw_df["xg_roll5"]
    gw_df["xgi_accel_3v5"] = gw_df["xgi_roll3"] - gw_df["xgi_roll5"]
    gw_df["pts_vs_season"] = gw_df["pts_roll3"] - gw_df["pts_season_avg"]
    gw_df["xg_vs_season"] = gw_df["xg_roll3"] - gw_df["xg_season_avg"]
    gw_df["pts_spike"] = gw_df["pts_last1"] - gw_df["pts_roll5"]
    return gw_df


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features(data_dir=None):
    """
    Build the feature matrix from raw data.

    v4: Adds opponent strength, EMA, last-1, form acceleration features.
    Removes log_selected (ownership bias).
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

    # --- v3 features ---
    gw_df, rolling_cols = _compute_rolling_features(gw_df)
    gw_df = _compute_season_averages(gw_df, rolling_cols)
    gw_df["games_played"] = gw_df.groupby("player_id").cumcount()
    gw_df = _compute_defensive_actions(gw_df)
    gw_df = _compute_team_strength(gw_df)
    gw_df = _compute_h2h_features(gw_df)
    gw_df = _compute_transfer_momentum(gw_df)

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

    # Vaastav H2H
    gw_df = _compute_vaastav_h2h(gw_df, data_dir)

    # Trajectory
    gw_df = _compute_trajectory_features(gw_df)

    # FBref
    gw_df = _compute_fbref_features(gw_df, data_dir)

    # --- v4 NEW features ---
    # Phase 1: Opponent strength
    gw_df = _compute_opponent_strength(gw_df, data_dir)

    # Phase 2a: EMA features
    gw_df = _compute_ema_features(gw_df)

    # Phase 2b: Last-1-GW features
    gw_df = _compute_last1_features(gw_df)

    # Phase 3: Form acceleration (must come after rolling + last1)
    gw_df = _compute_form_acceleration(gw_df)

    # Phase 4: pts_per_price (replaces log_selected)
    gw_df["pts_per_price"] = gw_df["pts_roll3"] / gw_df["price"].replace(0, np.nan)

    gw_df["target"] = gw_df["total_points"]

    return gw_df


def get_feature_columns():
    """Return the full list of feature columns for the v4 model (99 features)."""
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

    transfer = ["transfers_balance_roll3"]  # v4: removed log_selected

    prev_season = ["prev_season_pts_per90", "prev_season_xg_per90", "prev_season_minutes"]

    vaastav_h2h = ["h2h_prev_season_avg_pts", "h2h_prev_season_games"]

    trajectory = ["xg_improvement", "pts_improvement"]

    original_other = ["was_home", "fdr", "price", "price_change_1gw",
                      "price_change_3gw", "rest_days", "games_played", "element_type"]

    # --- v4 NEW feature groups ---
    opponent_strength = ["next_opp_attack", "next_opp_defence", "next_fdr",
                         "is_home_next", "opp_strength_diff", "team_vs_opp"]

    ema_features = [f"{s}_ema{span}"
                    for s in ["pts", "xg", "xa", "xgi", "bonus", "bps", "ict"]
                    for span in [3, 5]]

    last1_features = ["pts_last1", "xg_last1", "xa_last1", "bonus_last1", "min_last1"]

    form_acceleration = ["pts_accel_3v5", "xg_accel_3v5", "xgi_accel_3v5",
                         "pts_vs_season", "xg_vs_season", "pts_spike"]

    bias_correction = ["pts_per_price"]

    all_features = (rolling_features + season_features + defensive + team_strength
                    + h2h + transfer + prev_season + vaastav_h2h + trajectory
                    + original_other + opponent_strength + ema_features
                    + last1_features + form_acceleration + bias_correction)
    return all_features


def _generate_reason(row: pd.Series) -> str:
    """Generate a human-readable reason explaining why a player scored high.

    Written for fans, not data scientists — uses plain language and
    includes actual numbers so users can judge for themselves.
    """
    reasons = []
    opp_name = row.get("next_opp_name", "")

    # 1. Recent form spike — they had a big GW
    pts_last1 = row.get("pts_last1")
    pts_roll5 = row.get("pts_roll5")
    if pd.notna(pts_last1) and pd.notna(pts_roll5) and pts_roll5 > 0 and pts_last1 >= pts_roll5 * 1.5:
        reasons.append(
            f"Returned {pts_last1:.0f} pts last gameweek — well above his "
            f"5-game average of {pts_roll5:.1f}"
        )

    # 2. Form trend — recent form better than longer window
    accel = row.get("pts_accel_3v5")
    pts_roll3 = row.get("pts_roll3")
    if pd.notna(accel) and accel > 0.5 and pd.notna(pts_roll3) and pd.notna(pts_roll5):
        reasons.append(
            f"Form is picking up — averaging {pts_roll3:.1f} pts over his "
            f"last 3 games vs {pts_roll5:.1f} over 5"
        )

    # 3. Fixture — name the opponent
    fdr = row.get("next_fdr")
    is_home = row.get("is_home_next")
    if pd.notna(fdr) and opp_name:
        venue = "at home" if (pd.notna(is_home) and is_home == 1) else "away"
        if fdr <= 2:
            reasons.append(f"Faces {opp_name} {venue} — a very favorable fixture")
        elif fdr == 3:
            reasons.append(f"Plays {opp_name} {venue} next")
    elif pd.notna(fdr) and fdr <= 2:
        reasons.append("Favorable fixture coming up")

    # 4. Home advantage (only if not already covered by fixture reason)
    if pd.notna(is_home) and is_home == 1 and not any("at home" in r for r in reasons):
        reasons.append("Has the home-ground advantage this week")

    # 5. xG — explain what it means
    xg = row.get("xg_roll3")
    if pd.notna(xg) and xg > 0.4:
        reasons.append(
            f"Creating {xg:.2f} expected goals per game recently — "
            f"the chances are falling to him"
        )

    # 6. Bonus magnet
    bonus = row.get("bonus_roll3")
    if pd.notna(bonus) and bonus >= 1.5:
        reasons.append(
            f"Picking up {bonus:.1f} bonus pts per game on average — "
            f"consistently one of the best players on the pitch"
        )

    # 7. Value for money
    ppp = row.get("pts_per_price")
    price = row.get("price")
    if pd.notna(ppp) and ppp > 0.8 and pd.notna(price):
        reasons.append(
            f"At £{price:.1f}m, he\'s delivering excellent value "
            f"relative to his cost"
        )

    # 8. H2H track record — name the opponent
    h2h_games = row.get("h2h_games")
    h2h_avg = row.get("h2h_avg_pts")
    if pd.notna(h2h_games) and pd.notna(h2h_avg) and h2h_games >= 3 and h2h_avg > 5:
        opp_str = f" against {opp_name}" if opp_name else " against this opponent"
        reasons.append(
            f"Averages {h2h_avg:.1f} pts across {int(h2h_games)} previous "
            f"meetings{opp_str}"
        )

    # 9. Team strength mismatch
    tvso = row.get("team_vs_opp")
    if pd.notna(tvso) and tvso > 30:
        if opp_name:
            reasons.append(f"His team is rated significantly stronger than {opp_name}")
        else:
            reasons.append("His team has a clear quality advantage in this matchup")

    # Pick up to 4, or fallback
    if not reasons:
        if pd.notna(pts_roll3):
            reasons.append(f"Averaging {pts_roll3:.1f} pts over his last 3 games")
        else:
            reasons.append("Steady performer based on recent data")

    return ". ".join(reasons[:4]) + "."


def predict_next_gw(model=None, metadata=None, data_dir=None):
    """
    Generate predictions for the next gameweek.

    v4: Injects forward-looking fixture data (opponent strength, FDR, home/away)
    for the upcoming GW, replacing the last-played values.
    """
    if model is None or metadata is None:
        model, metadata = load_model()

    if data_dir is None:
        data_dir = get_project_root() / "data" / "raw"

    feature_cols = metadata["feature_columns"]
    gw_df = build_features(data_dir)

    latest = gw_df.groupby("player_id").tail(1).copy()
    latest = latest[latest["minutes"] > 0]

    # --- v4: Inject NEXT GW fixture data ---
    team_strengths = _get_team_strengths(data_dir)
    next_gw = _get_next_gw_number(data_dir)
    next_fixtures = _get_next_gw_fixtures(data_dir, next_gw, team_strengths)

    # Build team-id-to-name map for readable reasons
    with open(data_dir / "bootstrap_static.json") as f:
        _bs = json.load(f)
    team_name_map = {t["id"]: t["name"] for t in _bs["teams"]}

    # Overwrite opponent strength features with next-GW values
    for idx, row in latest.iterrows():
        team_id = row["team"]
        fix_info = next_fixtures.get(team_id)
        if fix_info:
            latest.at[idx, "next_opp_attack"] = fix_info["opp_attack"]
            latest.at[idx, "next_opp_defence"] = fix_info["opp_defence"]
            latest.at[idx, "next_fdr"] = fix_info["fdr"]
            latest.at[idx, "is_home_next"] = fix_info["is_home"]
            latest.at[idx, "opp_strength_diff"] = fix_info["opp_attack"] - fix_info["opp_defence"]
            latest.at[idx, "next_opp_name"] = team_name_map.get(fix_info["opp_team"], "")
            # Recalculate team_vs_opp
            own = team_strengths.get(team_id, {})
            league_mean = 1200
            if fix_info["is_home"]:
                own_s = (own.get("attack_home", league_mean) + own.get("defence_home", league_mean)) / 2
            else:
                own_s = (own.get("attack_away", league_mean) + own.get("defence_away", league_mean)) / 2
            opp_s = (fix_info["opp_attack"] + league_mean + fix_info["opp_defence"] + league_mean) / 2
            latest.at[idx, "team_vs_opp"] = own_s - opp_s

    # Only require core features to be non-null; XGBoost handles NaN natively
    core_cols = [c for c in feature_cols if "h2h" not in c and "prev_season" not in c
                 and "improvement" not in c]
    latest = latest.dropna(subset=core_cols)

    latest["score"] = model.predict(latest[feature_cols])
    latest["reason"] = latest.apply(_generate_reason, axis=1)

    result = latest[["player_id", "code", "web_name", "position", "team_name", "price",
                      "round", "score", "reason"]].copy()
    result = result.sort_values("score", ascending=False).reset_index(drop=True)
    result["score"] = result["score"].round(2)
    # Backwards compat alias -- remove after frontend deploys
    result["predicted_points"] = result["score"]

    return result
