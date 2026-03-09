# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FPL Predictor is a Python data analysis project for Fantasy Premier League. It fetches player/fixture data from the official FPL API, caches it locally as JSON, and explores it via Jupyter notebooks.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Common Commands

```bash
# Fetch fresh data from the FPL API (saves to data/raw/)
python -m src.fpl.fetch

# Fetch per-player gameweek histories (saves to data/raw/player_histories/)
python -c "from src.fpl.fetch import fetch_all_player_histories; fetch_all_player_histories()"

# Launch Jupyter for notebook exploration
jupyter notebook notebooks/

# Export a notebook to HTML
jupyter nbconvert --to html --execute notebooks/<notebook>.ipynb --output-dir exports/

# Run the API server locally
uvicorn src.fpl.api:app --reload --port 8080

# Docker: build and run
docker build -t fpl-predictor .
docker run -p 8080:8080 fpl-predictor

# Deploy to GCP Cloud Run
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/fpl-predictor
gcloud run deploy fpl-predictor \
  --image gcr.io/YOUR_PROJECT_ID/fpl-predictor \
  --platform managed --region us-central1 \
  --allow-unauthenticated --memory 512Mi --cpu 1
```

## Architecture

- **`src/fpl/fetch.py`** — Data fetching module. Pulls from `https://fantasy.premierleague.com/api` endpoints (`bootstrap-static`, `fixtures`, `element-summary/{id}`). Saves JSON to `data/raw/`.
- **`src/fpl/predict.py`** — Prediction pipeline. Loads the trained model, builds features from raw data, generates per-player predicted points.
- **`src/fpl/api.py`** — FastAPI server. Endpoints: `/health`, `/predict?player_id=`, `/predict/top?n=`, `/predict/position/{pos}`.
- **`data/raw/`** — Cached JSON responses (`bootstrap_static.json`, `fixtures.json`, `player_histories/`).
- **`models/`** — Trained model (`best_model.joblib`) and metadata (`model_metadata.json`).
- **`notebooks/`** — Exploratory analysis notebooks. Each maps to a blog chapter.
- **`exports/`** — HTML exports of notebooks for browser viewing.
- **`Dockerfile`** — Containerizes the API for deployment. Uses `python:3.13-slim`, exposes port 8080.
- **`.dockerignore`** — Excludes `venv/`, notebooks, caches from the Docker image.

## Key Conventions

- Python 3.13 with venv (not conda/poetry)
- Notebooks expect to be run from `notebooks/` directory — they reference `Path.cwd().parent` as project root
- Raw data files are committed to the repo (not gitignored)
- Player history fetcher skips already-cached files and rate-limits at ~3 req/sec
- API runs on port 8080 (Cloud Run default)

## Blog Sync

`BLOG_JOURNAL.md` is the source of truth for the build journal. A copy lives in the personal-site repo at `../Personal Website/app/fpl/blog/BLOG_JOURNAL.md`.

- **Auto-sync**: A post-commit hook copies the file whenever a commit touches `BLOG_JOURNAL.md`
- **Manual sync**: `./sync_blog.sh`
- After syncing, the personal-site repo still needs its own commit + push to trigger Vercel deploy
