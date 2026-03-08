"""
FBref Data Fetching Module

Fetches advanced player statistics from FBref via the soccerdata library.
Provides fuzzy name matching between FPL and FBref player names.

This module degrades gracefully - if soccerdata or FBref data is unavailable,
the prediction pipeline works with FPL data alone.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def _load_name_mapping(mappings_dir: Path) -> Dict[str, str]:
    """Load manual FPL-to-FBref name mapping overrides."""
    path = mappings_dir / "fpl_to_fbref.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def build_name_mapping(bootstrap_elements, fbref_names, mappings_dir=None):
    """
    Build a mapping from FPL web_name to FBref player name using fuzzy matching.
    Manual overrides in data/mappings/fpl_to_fbref.json take priority.
    """
    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        logger.warning("rapidfuzz not installed - skipping FBref name matching")
        return {}

    if mappings_dir is None:
        mappings_dir = get_project_root() / "data" / "mappings"

    manual = _load_name_mapping(mappings_dir)
    mapping = {}

    for player in bootstrap_elements:
        fpl_name = player["web_name"]
        if fpl_name in manual:
            mapping[fpl_name] = manual[fpl_name]
            continue
        result = process.extractOne(
            fpl_name, fbref_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=75
        )
        if result:
            mapping[fpl_name] = result[0]

    logger.info(f"Matched {len(mapping)}/{len(bootstrap_elements)} FPL players to FBref names")
    return mapping


def fetch_fbref_stats(season="2025-2026", data_dir=None):
    """
    Fetch player match stats from FBref via soccerdata.
    Caches results to data/external/fbref/player_stats_{season}.csv
    """
    try:
        import soccerdata as sd
    except ImportError:
        logger.warning("soccerdata not installed - skipping FBref fetch")
        return None

    if data_dir is None:
        data_dir = get_project_root() / "data"

    cache_dir = data_dir / "external" / "fbref"
    cache_file = cache_dir / f"player_stats_{season}.csv"

    if cache_file.exists():
        logger.info(f"Loading cached FBref data from {cache_file}")
        return pd.read_csv(cache_file)

    logger.info(f"Fetching FBref stats for season {season}...")
    try:
        fbref = sd.FBref(leagues="ENG-Premier League", seasons=season)
        stats = fbref.read_player_season_stats(stat_type="standard")
        cache_dir.mkdir(parents=True, exist_ok=True)
        stats.to_csv(cache_file, index=True)
        logger.info(f"Saved FBref stats to {cache_file}")
        return stats
    except Exception as e:
        logger.warning(f"Failed to fetch FBref data: {e}")
        return None


def merge_fbref_with_fpl(fbref_df, bootstrap_elements, data_dir=None):
    """Match FBref player stats to FPL player IDs using fuzzy name matching."""
    if data_dir is None:
        data_dir = get_project_root() / "data"

    mappings_dir = data_dir / "mappings"

    if "player" in fbref_df.columns:
        fbref_names = fbref_df["player"].unique().tolist()
    elif fbref_df.index.name == "player" or "player" in (fbref_df.index.names or []):
        fbref_df = fbref_df.reset_index()
        fbref_names = fbref_df["player"].unique().tolist()
    else:
        logger.warning("Cannot find player name column in FBref data")
        return fbref_df

    name_map = build_name_mapping(bootstrap_elements, fbref_names, mappings_dir)
    fpl_name_to_id = {p["web_name"]: p["id"] for p in bootstrap_elements}
    fbref_to_fpl_id = {}
    for fpl_name, fbref_name in name_map.items():
        if fpl_name in fpl_name_to_id:
            fbref_to_fpl_id[fbref_name] = fpl_name_to_id[fpl_name]

    fbref_df["player_id"] = fbref_df["player"].map(fbref_to_fpl_id)
    matched = fbref_df["player_id"].notna().sum()
    logger.info(f"Matched {matched}/{len(fbref_df)} FBref rows to FPL player IDs")
    return fbref_df
