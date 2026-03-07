"""
FPL Data Fetching Module

This module provides functionality to fetch public Fantasy Premier League (FPL) data
from the official API and cache it locally as JSON files for reproducibility.

Endpoints fetched:
- bootstrap-static: Contains static data (players, teams, game settings, etc.)
- fixtures: Contains fixture information for the current season
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import requests


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# FPL API base URL
FPL_API_BASE = "https://fantasy.premierleague.com/api"


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Assumes the project root is 3 levels up from this file
    (src/fpl/fetch.py -> project root).
    
    Returns:
        Path: Path to the project root directory
    """
    return Path(__file__).parent.parent.parent


def ensure_directory_exists(directory: Path) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory: Path to the directory to create
    """
    directory.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")


def fetch_endpoint(url: str) -> Dict[str, Any]:
    """
    Fetch data from a given URL and return as JSON.
    
    Args:
        url: The URL to fetch data from
        
    Returns:
        Dict containing the JSON response
        
    Raises:
        requests.RequestException: If the request fails
    """
    logger.info(f"Fetching data from: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully fetched {len(str(data))} characters of data")
        return data
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        raise


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: The data dictionary to save
        filepath: Path where the JSON file should be saved
    """
    ensure_directory_exists(filepath.parent)
    
    logger.info(f"Saving data to: {filepath}")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Successfully saved {filepath}")


def fetch_bootstrap_static(output_path: Path) -> None:
    """
    Fetch bootstrap-static data from FPL API and save to JSON.
    
    The bootstrap-static endpoint contains static data including:
    - Players and their attributes
    - Teams
    - Game settings
    - Events (gameweeks)
    - Element types (positions)
    
    Args:
        output_path: Path where the JSON file should be saved
    """
    url = f"{FPL_API_BASE}/bootstrap-static/"
    data = fetch_endpoint(url)
    save_json(data, output_path)


def fetch_fixtures(output_path: Path) -> None:
    """
    Fetch fixtures data from FPL API and save to JSON.
    
    The fixtures endpoint contains fixture information for the current season,
    including match results, scheduled matches, and team information.
    
    Args:
        output_path: Path where the JSON file should be saved
    """
    url = f"{FPL_API_BASE}/fixtures/"
    data = fetch_endpoint(url)
    save_json(data, output_path)


def fetch_player_history(player_id: int, output_path: Path) -> Dict[str, Any]:
    """
    Fetch per-gameweek history for a single player.
    
    The element-summary endpoint returns:
    - history: per-gameweek stats (points, minutes, goals, assists, etc.)
    - fixtures: upcoming fixtures for the player
    - history_past: previous season summaries
    
    Args:
        player_id: The player's element ID
        output_path: Path where the JSON file should be saved
        
    Returns:
        Dict containing the player's summary data
    """
    url = f"{FPL_API_BASE}/element-summary/{player_id}/"
    data = fetch_endpoint(url)
    save_json(data, output_path)
    return data


def fetch_all_player_histories(data_dir: Path = None, player_ids: list = None) -> None:
    """
    Fetch gameweek histories for all (or specified) players.
    
    Fetches from bootstrap-static first to get the player list,
    then hits element-summary for each player. Includes a small
    delay between requests to be respectful to the API.
    
    Args:
        data_dir: Optional path to the data directory. If None, uses
                 project_root/data/raw/
        player_ids: Optional list of player IDs. If None, fetches all
                   players from bootstrap-static.
    """
    import time
    
    if data_dir is None:
        project_root = get_project_root()
        data_dir = project_root / "data" / "raw"
    
    if player_ids is None:
        bootstrap_path = data_dir / "bootstrap_static.json"
        if not bootstrap_path.exists():
            logger.info("bootstrap_static.json not found, fetching first...")
            fetch_bootstrap_static(bootstrap_path)
        with open(bootstrap_path, 'r') as f:
            bootstrap = json.load(f)
        player_ids = [p['id'] for p in bootstrap['elements']]
    
    histories_dir = data_dir / "player_histories"
    ensure_directory_exists(histories_dir)
    
    total = len(player_ids)
    logger.info(f"Fetching histories for {total} players...")
    
    for i, pid in enumerate(player_ids, 1):
        output_path = histories_dir / f"player_{pid}.json"
        if output_path.exists():
            logger.debug(f"Skipping player {pid} (already cached)")
            continue
        try:
            fetch_player_history(pid, output_path)
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{total} players fetched")
            time.sleep(0.3)  # rate-limit: ~3 requests/sec
        except Exception as e:
            logger.warning(f"Failed to fetch player {pid}: {e}")
            continue
    
    logger.info(f"Finished fetching player histories")


def fetch_all_data(data_dir: Path = None) -> None:
    """
    Fetch all FPL data endpoints and save them to the data directory.
    
    Args:
        data_dir: Optional path to the data directory. If None, uses
                 project_root/data/raw/
    """
    if data_dir is None:
        project_root = get_project_root()
        data_dir = project_root / "data" / "raw"
    
    logger.info(f"Fetching all FPL data to: {data_dir}")
    
    # Fetch bootstrap-static
    bootstrap_path = data_dir / "bootstrap_static.json"
    fetch_bootstrap_static(bootstrap_path)
    
    # Fetch fixtures
    fixtures_path = data_dir / "fixtures.json"
    fetch_fixtures(fixtures_path)
    
    logger.info("Successfully fetched all FPL data")


def main():
    """Main entry point for running the data fetch module."""
    logger.info("Starting FPL data fetch")
    try:
        fetch_all_data()
        logger.info("FPL data fetch completed successfully")
    except Exception as e:
        logger.error(f"FPL data fetch failed: {e}")
        raise


if __name__ == "__main__":
    main()

