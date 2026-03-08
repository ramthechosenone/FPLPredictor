# FPL Predictor — Blog Journal

A running log of decisions, discoveries, and learnings as we build an FPL points predictor from scratch.

---

## Chapter 1: Data Capture (Dec 29)

### What we built
- A Python module (`src/fpl/fetch.py`) that pulls data from the official FPL API
- Two endpoints captured:
  - **bootstrap-static** — the "everything" endpoint: all 700+ players with attributes (cost, form, points, ownership%), all 20 teams, gameweek metadata, and position types
  - **fixtures** — every match of the season with scores, difficulty ratings, and kickoff times

### Key discoveries
- The FPL API is public and unauthenticated — no API key needed
- `bootstrap-static` is a single massive JSON (~2.4 MB) containing nested lists: `elements` (players), `teams`, `events` (gameweeks), `element_types` (positions)
- Player costs are stored as integers (e.g., 130 = £13.0m) — need to divide by 10
- The `status` field flags availability: `a` (available), `i` (injured), `d` (doubtful), `u` (unavailable), `s` (suspended)

### Design decisions
- Cache raw JSON locally in `data/raw/` so notebooks are reproducible without hitting the API
- Committed data files to the repo (small enough, and useful for reproducibility)

---

## Chapter 2: Exploratory Analysis (Jan 18)

### What we built
- `notebooks/01_players_overview.ipynb` — first look at the player data

### Key discoveries
- Position mapping: `element_type` 1=GK, 2=DEF, 3=MID, 4=FWD
- Team IDs need to be joined from the `teams` list to get readable names
- Filtering to `total_points > 0 and status == 'a'` removes noise from players who haven't featured
- `form` is a string (not float) in the raw data — needs casting
- `points_per_game` is also a string — the API stores many numeric fields as strings
- `selected_by_percent` shows ownership — useful for finding differentials
- **Top performers snapshot**: Haaland (153 pts, £15.1m) dominates; Semenyo (108 pts, £7.7m) best midfield value; Everton defenders are budget gems

### What was missing
- Single snapshot in time — current totals but not week-to-week performance
- No fixture difficulty analysis
- No historical gameweek-level data per player

---

## Chapter 3: Player Histories (In Progress)

### What we're building
- Extended `fetch.py` with `fetch_player_history()` and `fetch_all_player_histories()` to pull per-player gameweek data from `element-summary/{id}/`
- Each player gets a cached JSON file in `data/raw/player_histories/player_{id}.json`

### Technical decisions
- Rate-limited at 0.3s between requests (~3 req/sec) to respect the API
- Skip already-cached files so re-runs are fast
- Log progress every 50 players
- Failures for individual players are warned and skipped, not fatal

### What this unlocks
- Week-by-week points, minutes, goals, assists, bonus, clean sheets per player
- Rolling averages and form trends for feature engineering
- Previous season summaries (`history_past`) for context

### Next: notebook `02_player_histories` to explore this data

### Findings from notebook 02

**Dataset:** 21,994 gameweek rows, 775 players, GW1-GW29. Each row has 40+ columns including expected stats (xG, xA, xGI, xGC).

**xG analysis — overperformers vs underperformers:**
- Semenyo (15 goals vs 9.8 xG) and Wilson (9 vs 4.7 xG) are massively overperforming — FPL darlings but regression risks
- Mateta (8 goals vs 11.9 xG) and Gordon (4 vs 7.2 xG) are getting chances but not finishing — potential value picks if they regress upward

**Home advantage:**
- DEF: +0.4 pts at home, GK: +0.38 pts — set-piece and clean sheet related
- FWD: basically no home/away difference — surprising

**Feature correlations with gameweek points (most promising for prediction):**
- BPS (0.91) — but this is calculated *during* the game, so can't use it to predict
- Goals scored (0.66), clean sheets (0.48) — historical versions (rolling averages) are usable
- xGI (0.43), xG (0.40) — these ARE usable as features since they represent underlying quality
- Minutes (0.39) — proxy for nailedness/rotation risk
- Home/away (0.05) — small but real, and known before the match

**Key insight for modeling:** We need to be careful about data leakage. Features like BPS, goals, and assists in the *same* gameweek can't be used to predict that gameweek's points. We need to use *lagged* versions (e.g., rolling 3-GW average of xG) as predictors.

---

## Chapter 4: Feature Engineering

### What we built
- `notebooks/03_feature_engineering.ipynb` — transforms raw gameweek data into 43 model-ready features
- Output: `data/processed/features.csv` (19,669 rows, GW4-29, 775 players)

### Features created

**Rolling averages (lagged, no leakage):**
- 3-GW and 5-GW rolling means for: points, minutes, goals, assists, bonus, BPS, xG, xA, xGI, xGC, clean sheets, influence, creativity, threat, ICT index
- Key technique: `.shift(1)` before `.rolling()` ensures we never peek at current-GW data

**Season-to-date:** expanding mean of points, minutes, xG, xA, bonus

**Match context:** fixture difficulty rating (FDR), home/away flag, rest days between matches

**Market signals:** current price, 1-GW and 3-GW price change

**Target:** current gameweek's `total_points`

### Key discoveries
- **Minutes played** (`min_roll3`, 0.55 correlation) is the single best predictor — knowing whether a player starts is more valuable than any performance metric
- **BPS and ICT index** rolling averages (~0.50) capture "underlying quality" well — these are composite stats the FPL system calculates
- **xGC** (expected goals conceded, 0.48) is surprisingly strong — rewards defenders/GKs who face weak attacks
- **FDR** (0.02) has almost no correlation on its own — fixture difficulty matters but only in combination with other features (a model should pick this up via interactions)
- **Price change** (~0.07) is a weak direct predictor but captures market wisdom about form changes

### Design decision: dropping early gameweeks
- First 3 GWs per player are dropped because rolling windows need warmup
- This costs ~2,300 rows (~10%) but ensures feature quality
- Alternative would be padding with season averages from previous year, but adds complexity

### What's next
- Model selection: train and compare candidate models (linear regression, random forest, XGBoost)
- Time-based train/test split (train on GW4-22, test on GW23-29) to simulate real prediction

### Blog note: correlation vs model weights
- The feature correlations (e.g., min_roll3 = 0.55) are NOT prediction weights — they only show individual linear relationships with the target
- A feature with low correlation (like FDR at 0.02) can still be valuable in a model through interactions with other features
- Features with high correlation to each other (multicollinearity) may share credit in a model — the model sorts this out
- The actual prediction formula/weights come from training a model, which is the next step

---

## Chapter 5: Model Selection — Training and Results

### What is a model, really?

A model is a **formula that learns patterns from data**. You show it thousands of examples ("this player had these stats and scored X points"), and it figures out the relationship between the inputs (features) and the output (points). This process is called **training**.

Once trained, you give it a player's current stats and it outputs a prediction: "I think this player will score ~4.2 points next gameweek."

### The type of learning: Supervised Learning

This is **supervised learning** — we have both the inputs (43 features like rolling xG, minutes, price) AND the correct answer (actual points scored). The model learns by comparing its predictions to the real answers and adjusting itself to minimize the error.

It's called "supervised" because we're essentially giving the model an answer key to learn from. The opposite would be *unsupervised learning*, where you have data but no correct answers (e.g., grouping players into clusters without knowing what the "right" groups are).

### The three models we compared

#### 1. Linear Regression — "Draw the best straight line"

The simplest model. It assumes each feature contributes to points in a straight-line (linear) way:

```
predicted_points = intercept + (weight1 x feature1) + (weight2 x feature2) + ...
```

For example, it might learn: `predicted_points = -0.5 + (0.03 x min_roll3) + (0.2 x xg_roll3) + ...`

**How it trains:** It finds the set of weights that minimizes the total squared error across all training examples. There's actually a mathematical formula for this — no iteration needed.

**Strengths:** Fast, interpretable (you can see exactly how each feature contributes), hard to overfit.

**Weaknesses:** Assumes relationships are linear. If "playing 45 minutes" is worth 1 point but "playing 90 minutes" is worth 3 points (not 2), linear regression can't capture that curve.

#### 2. Random Forest — "Ask 200 decision trees and take the average"

A decision tree is like a flowchart:
```
Is min_roll3 > 60?
  YES → Is xg_roll3 > 0.3?
    YES → predict 4.5 pts
    NO  → predict 2.1 pts
  NO  → predict 0.3 pts
```

A single tree is fragile — it might memorize quirks in the training data. A **Random Forest** fixes this by building 200 trees, each trained on a random subset of the data and features, then averaging their predictions. The randomness + averaging cancels out individual tree mistakes.

**How it trains:** For each of the 200 trees, it randomly samples rows and features, then finds the best "split points" (like min_roll3 > 60) that separate high-scoring from low-scoring players.

**Strengths:** Captures non-linear relationships and feature interactions automatically. Resistant to overfitting because of the averaging.

**Weaknesses:** Less interpretable than linear regression (200 trees are hard to read). Can't extrapolate beyond the range of training data.

#### 3. XGBoost — "Build trees that fix each other's mistakes"

XGBoost (eXtreme Gradient Boosting) also uses decision trees, but instead of building them independently (like Random Forest), it builds them **sequentially**. Each new tree focuses specifically on the examples the previous trees got wrong.

Think of it like this:
- Tree 1 makes predictions. Some are wrong.
- Tree 2 is trained specifically on Tree 1's errors.
- Tree 3 is trained on the remaining errors after Trees 1+2.
- ...and so on for 300 trees.

The final prediction is the sum of all trees' contributions.

**How it trains:** Uses gradient descent (a mathematical optimization technique) to figure out what each new tree should focus on. The "gradient" tells the algorithm which direction to adjust.

**Strengths:** Usually the most accurate model for tabular data. Handles missing values, feature interactions, and non-linear relationships well.

**Weaknesses:** More hyperparameters to tune (learning rate, tree depth, etc.). Can overfit if not carefully configured. Least interpretable of the three.

### What we found

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 1.026 | 1.895 | 0.321 |
| **Random Forest** | **1.003** | 1.905 | 0.314 |
| XGBoost | 1.033 | 1.943 | 0.286 |

**What the metrics mean:**
- **MAE (Mean Absolute Error):** On average, the prediction is off by ~1 point. If the model says "4 points," the actual is typically between 3-5.
- **RMSE (Root Mean Squared Error):** Like MAE but penalizes big misses more. An RMSE of 1.9 means occasional big errors (predicting 3 when a player scores 15).
- **R² (R-squared):** The model explains 31% of the variance in points. The other 69% is randomness — goals, assists, and bonus points are inherently unpredictable.

### Why Random Forest won (barely)

All three models performed almost identically. This tells us something important: **the bottleneck isn't the model, it's the data**. FPL points have a massive random component — a deflected goal, a last-minute penalty, a VAR decision — that no model can predict from historical stats.

### Feature importance — what the model actually learned

`min_roll3` (3-GW rolling average minutes) accounts for **62%** of the model's importance. The model's primary strategy is:

1. "Will this player play?" (minutes rolling avg)
2. If yes, "How good has he been recently?" (points, xG, BPS rolling averages)
3. Minor adjustments for price, fixture difficulty, and position

### Where the model fails

- **Explosive hauls** (15+ points): Palmer scoring 20 points in a gameweek is essentially a random event from the model's perspective
- **Red cards and own goals**: These cause negative scores that can't be predicted from form data
- **GW23 was a disaster**: 0/15 predicted top players appeared in the actual top 30 — some gameweeks are just chaotic

### The honest takeaway for the blog

A 31% R² is actually reasonable for FPL prediction. Academic papers on football prediction typically achieve 25-35% for individual player performance. The model is useful for identifying **likely starters who are in good form** (the bread-and-butter 2-6 point returns) but can't predict the difference between a 5-point and a 15-point week.

---

## Chapter 6: Dockerize & Deploy to GCP Cloud Run

### What is Docker?

Docker packages your app + its dependencies into a **container** — a lightweight, isolated environment that runs the same everywhere. Think of it as shipping your entire laptop setup (Python, libraries, code, model file) as a single runnable package.

**Key concepts:**
- **Image:** A blueprint/snapshot of your app (like a class in OOP)
- **Container:** A running instance of an image (like an object)
- **Dockerfile:** Instructions to build an image (like a recipe)
- **Registry:** A place to store images (like GitHub for Docker images)

### What we built

- A `Dockerfile` that packages the FastAPI prediction server with all dependencies, data, and the trained model
- A `.dockerignore` to keep the image lean (excludes `venv/`, notebooks, Jupyter files)

### How the Dockerfile works

```dockerfile
FROM python:3.13-slim          # Start from official Python image
WORKDIR /app                   # Set working directory inside container
COPY requirements.txt .        # Copy deps list first (Docker caches this layer)
RUN pip install ... -r ...     # Install deps (cached if requirements.txt unchanged)
COPY src/ data/ models/        # Copy our code, data, and trained model
EXPOSE 8080                    # Document which port the app uses
CMD ["uvicorn", ...]           # Default command when container starts
```

**Why copy requirements.txt separately?** Docker builds in layers. If your code changes but `requirements.txt` doesn't, Docker reuses the cached pip install layer — making rebuilds fast.

### Why GCP Cloud Run?

Cloud Run = "run my Docker container in the cloud." Key benefits:
- **No servers to manage** — fully managed by Google
- **Scales to zero** — no cost when idle, scales up automatically on traffic
- **Pay per request** — perfect for a side project API
- **Free tier** — 2 million requests/month free

### Deploying to Cloud Run

```bash
# Build & push to GCP's container registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/fpl-predictor

# Deploy to Cloud Run
gcloud run deploy fpl-predictor \
  --image gcr.io/YOUR_PROJECT_ID/fpl-predictor \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1
```

Cloud Run gives you a public URL like `https://fpl-predictor-xxxxx.run.app` — hit it with curl and you get predictions.

### Architecture: End-to-end

```
FPL API  -->  fetch.py  -->  data/raw/ (JSON)
                                |
                          notebooks (EDA + feature eng + model training)
                                |
                          models/best_model.joblib
                                |
                          predict.py (feature pipeline + inference)
                                |
                          api.py (FastAPI endpoints)
                                |
                          Dockerfile (containerize everything)
                                |
                          GCP Cloud Run (public URL)
```

### Key learnings
- Docker layer caching is powerful — order your COPY statements from least-changing to most-changing
- Cloud Run uses port 8080 by default (configurable via `$PORT` env var)
- The `--allow-unauthenticated` flag makes the API public; remove it for private APIs
- `python:3.13-slim` keeps the image small (~150 MB base vs ~1 GB for the full image)

### The messy reality: what actually happened

Tutorials make deployment look like 3 clean commands. Here's what it actually looked like for us — every stumble included, because this is what real deployment feels like when you're learning.

#### Problem 1: Installing `gcloud` CLI on macOS

We ran `brew install google-cloud-sdk` and immediately hit:

```
Error: Provided python path `/opt/homebrew/opt/python@3.13/libexec/bin/python3` does not exist.
```

The Homebrew formula expected Python at a specific symlink path that didn't exist. The fix was to first install Python 3.13 through Homebrew (`brew install python@3.13`) so the symlink would be created, then reinstall gcloud.

**Lesson:** Cloud tooling often has opinions about where Python lives. Homebrew's symlink structure can clash with what tools expect.

#### Problem 2: Network timeout during install

After fixing the Python path, the installer timed out trying to download components from `dl.google.com`:

```
ReadTimeoutError: HTTPSConnectionPool(host='dl.google.com', port=443): Read timed out.
```

Not a code problem — just a flaky network connection. The fix was simply retrying. But it's the kind of thing that makes you question everything when you're new ("Did I break something?"). You didn't. Just retry.

**Lesson:** Network errors during installs are common and usually transient. Don't panic, just retry.

#### Problem 3: GCP permissions error on first deploy

With `gcloud` finally installed, we ran `gcloud builds submit` and got:

```
ERROR: 176753204897-compute@developer.gserviceaccount.com does not have
storage.objects.get access to the Google Cloud Storage object.
```

Cloud Build needs permission to upload your source code to a Cloud Storage bucket before building. By default, the compute service account doesn't have this. The fix:

```bash
gcloud projects add-iam-policy-binding fpl-predictor-489517 \
  --member="serviceAccount:176753204897-compute@developer.gserviceaccount.com" \
  --role="roles/storage.admin"
```

**Lesson:** GCP's IAM (Identity and Access Management) is powerful but confusing for beginners. Services need explicit permissions to talk to each other. The error messages at least tell you *which* permission is missing — look for the `Permission 'X' denied` part and grant that role.

#### The honest truth about "deploy to the cloud"

What tutorials show:
```
gcloud builds submit ...   # 1 command
gcloud run deploy ...      # 1 command
# Done!
```

What actually happens:
1. Install CLI tool (3 attempts, Python path issues)
2. Authenticate (straightforward, thankfully)
3. Create project + enable APIs (web console clicking)
4. Fix permissions (IAM role binding)
5. *Then* the 2 deploy commands work

This is normal. Every developer goes through this. The first deploy to any cloud platform is always the hardest — after that, it's just the 2 commands.

### It's live!

**Deployed URL:** https://fpl-predictor-176753204897.us-central1.run.app

Verified endpoints working in production:
- `/health` — returns model info (Random Forest, MAE 1.003, R² 0.314)
- `/predict/top?n=3` — Virgil van Dijk (5.9 pts), O'Reilly (5.4 pts), Donnarumma (5.31 pts)
- `/predict?player_id=X` — individual player predictions
- `/predict/position/DEF` — top predicted players by position

From a Jupyter notebook on a laptop to a public API in the cloud. The whole journey: data capture, EDA, feature engineering, model training, API server, Docker container, Cloud Run deployment.

---

## Chapter 7: API Reference — What Goes In, What Comes Out

Our API has four endpoints. Here's exactly what each one does, what you send it, and what you get back.

### Base URL

```
https://fpl-predictor-176753204897.us-central1.run.app
```

---

### `GET /health`

**What it's for:** Check if the API is running and see which model is loaded.

**Input:** Nothing — just hit the URL.

**Example:**
```bash
curl https://fpl-predictor-176753204897.us-central1.run.app/health
```

**Response:**
```json
{
  "status": "ok",
  "model": "Random Forest",
  "metrics": {
    "MAE": 1.003,
    "RMSE": 1.905,
    "R²": 0.314
  }
}
```

**What the fields mean:**
| Field | Meaning |
|-------|---------|
| `status` | `"ok"` if the server is healthy and the model loaded successfully |
| `model` | Which ML model is serving predictions (we trained 3, Random Forest won) |
| `metrics.MAE` | Mean Absolute Error — on average, predictions are off by ~1 point |
| `metrics.RMSE` | Root Mean Squared Error — penalizes big misses more heavily (1.9 = occasional large errors) |
| `metrics.R²` | R-squared — the model explains 31.4% of the variance in points. The rest is randomness (goals, cards, bonus are inherently unpredictable) |

---

### `GET /predict?player_id={id}`

**What it's for:** Get the predicted next-gameweek points for a specific player.

**Input:** `player_id` (required) — the player's FPL element ID. You can find this on the FPL website: go to a player's page and the number in the URL is their ID.

**Example:**
```bash
curl "https://fpl-predictor-176753204897.us-central1.run.app/predict?player_id=328"
```

**Response (success):**
```json
{
  "player_id": 328,
  "name": "Sessegnon",
  "position": "MID",
  "team": "Fulham",
  "price": 5.4,
  "predicted_points": 3.82,
  "based_on_gw": 29
}
```

**Response (player not found):** `404`
```json
{
  "error": "Player 99999 not found or has insufficient data"
}
```

**What the fields mean:**
| Field | Meaning |
|-------|---------|
| `player_id` | The FPL element ID you queried |
| `name` | Player's `web_name` from FPL (usually surname) |
| `position` | GK, DEF, MID, or FWD |
| `team` | Premier League team name |
| `price` | Current FPL price in millions (e.g., 5.4 = £5.4m) |
| `predicted_points` | Model's prediction for next gameweek points (typically 1-6 range) |
| `based_on_gw` | The most recent gameweek the model has data for. Predictions are for the *next* GW after this |

**Why would a player return 404?** Players who haven't played enough matches lack the rolling-average features the model needs (we require at least 3 GWs of history). Bench warmers and new signings get filtered out.

---

### `GET /predict/top?n={count}`

**What it's for:** Get the top N predicted players across all positions — your "who should I captain / transfer in" list.

**Input:** `n` (optional, default 15, max 100) — how many players to return.

**Example:**
```bash
curl "https://fpl-predictor-176753204897.us-central1.run.app/predict/top?n=5"
```

**Response:**
```json
{
  "count": 5,
  "players": [
    {
      "rank": 1,
      "player_id": 373,
      "name": "Virgil",
      "position": "DEF",
      "team": "Liverpool",
      "price": 6.1,
      "predicted_points": 5.9
    },
    {
      "rank": 2,
      "player_id": 411,
      "name": "O'Reilly",
      "position": "DEF",
      "team": "Man City",
      "price": 5.1,
      "predicted_points": 5.4
    },
    {
      "rank": 3,
      "player_id": 736,
      "name": "Donnarumma",
      "position": "GK",
      "team": "Man City",
      "price": 5.6,
      "predicted_points": 5.31
    },
    {
      "rank": 4,
      "player_id": 374,
      "name": "Konaté",
      "position": "DEF",
      "team": "Liverpool",
      "price": 5.5,
      "predicted_points": 5.06
    },
    {
      "rank": 5,
      "player_id": 235,
      "name": "Palmer",
      "position": "MID",
      "team": "Chelsea",
      "price": 10.6,
      "predicted_points": 4.81
    }
  ]
}
```

**What the fields mean:**
| Field | Meaning |
|-------|---------|
| `count` | Number of players returned (matches your `n` parameter) |
| `rank` | Position in the overall predicted points ranking (1 = highest) |
| Other fields | Same as the `/predict` endpoint above |

**What to notice:** The model loves defenders and goalkeepers. That's because our Random Forest learned that **minutes played** is the strongest predictor, and defenders/GKs are the most nailed-on starters. They reliably return 2-6 points per week. Attackers have higher ceilings but more variance — the model plays it safe.

---

### `GET /predict/position/{position}?n={count}`

**What it's for:** Get the top predicted players filtered by position — useful for FPL squad building where you need specific positions.

**Input:**
- `position` (required, in URL) — one of: `GK`, `DEF`, `MID`, `FWD`
- `n` (optional, default 10, max 50) — how many to return

**Example:**
```bash
curl "https://fpl-predictor-176753204897.us-central1.run.app/predict/position/FWD?n=3"
```

**Response (success):**
```json
{
  "position": "FWD",
  "count": 3,
  "players": [
    {
      "rank": 13,
      "player_id": 661,
      "name": "Ekitiké",
      "team": "Liverpool",
      "price": 9.1,
      "predicted_points": 4.41
    },
    {
      "rank": 22,
      "player_id": 624,
      "name": "Bowen",
      "team": "West Ham",
      "price": 7.5,
      "predicted_points": 4.0
    },
    {
      "rank": 33,
      "player_id": 249,
      "name": "João Pedro",
      "team": "Chelsea",
      "price": 7.6,
      "predicted_points": 3.82
    }
  ]
}
```

**Response (invalid position):** `400`
```json
{
  "error": "Position must be GK, DEF, MID, or FWD"
}
```

**What the fields mean:**
| Field | Meaning |
|-------|---------|
| `position` | The position you filtered by |
| `rank` | The player's rank in the *overall* list (not within position). Ekitiké is rank 13 overall but rank 1 among forwards |
| Other fields | Same as above |

**Note on `rank`:** The rank is global, not positional. So the top FWD might be rank 13 overall because 12 defenders/midfielders/keepers are predicted higher. This is by design — it lets you compare across positions.

---

### How to use this for actual FPL decisions

1. **Captain pick:** Hit `/predict/top?n=5` and captain the highest predicted player you own
2. **Transfer targets:** Hit `/predict/position/MID?n=10` to find the best midfielders to bring in
3. **Bench decisions:** Compare two players with `/predict?player_id=X` for each — start the one with higher predicted points
4. **Differential hunting:** Look for high-predicted players with low FPL ownership (cross-reference `price` — cheap + high prediction = under-owned gem)

### Limitations to keep in mind

- Predictions are based on the **last cached data** (GW29). They don't auto-update when new gameweeks are played — we'd need to re-fetch data and rebuild
- The model predicts **expected returns** (2-6 pt range), not explosive hauls. A predicted 5.9 doesn't mean "will definitely get 6" — it means "most likely outcome is around 4-7"
- Injured/suspended players may still appear if they were playing when the data was last captured

---

## Chapter 8: Enhanced Features & External Data (Mar 8)

### The problem with v2

The v2 model (XGBoost with 43 features, trained on played-only data) achieved R² = 0.016 and predictions maxed out around 7.4 points. While it was better than v1 (which included zero-minute rows), the model was leaving signal on the table. The FPL API gives us ~40 fields per gameweek — but we were only using about half of them.

### What we built

We refactored `build_features()` in `predict.py` from a single monolithic function into 9 modular helper functions, and added 27 new features across three phases:

**Phase 1 — Exploiting unused FPL API data:**

- **Defensive actions** (starts, tackles, recoveries, yellow cards, saves): Rolling 3- and 5-GW averages, plus a composite `def_actions` metric that sums tackles + recoveries + clearances/blocks/interceptions. This turned out to be the **single most impactful feature group** — adding defensive features alone raised R² from 0.03 to 0.05.
- **Team strength**: Derived `team_goals_for` and `team_goals_against` from match scores (`team_h_score`, `team_a_score`, `was_home`), then computed 5-game rolling averages. Captures whether a player's team is on a scoring/defensive run.
- **Head-to-head features**: For each (player, opponent_team) pair, computed expanding mean of `total_points` and `xG` from prior encounters this season. Shifted to avoid leakage. About 34% of played gameweek rows had H2H data (i.e., the player had faced that opponent before).
- **Transfer momentum**: Rolling 3-GW mean of `transfers_balance` (net transfers in/out) plus `log_selected` (log of total ownership). Signals whether FPL managers are buying or selling the player.
- **Previous season baseline**: Parsed `history_past[]` from each player's JSON to extract `prev_season_pts_per90` and `prev_season_xg_per90`. Helps anchor predictions for established players.

**Phase 2 — External historical data (vaastav):**

Downloaded per-GW CSVs from the [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) GitHub repo for the 2023-24 and 2024-25 seasons. Used name-based matching to link historical player data to current-season player IDs. This enabled multi-season H2H features, though the name matching is imperfect (0.5% coverage — needs improvement).

**Phase 3 — FBref integration (scaffolded):**

Created `src/fpl/fbref.py` with FBref data fetching via `soccerdata` and fuzzy name matching via `rapidfuzz`. The pipeline degrades gracefully — if no FBref data exists, it simply skips those features. Data saved to `data/external/fbref/` when available.

### Key findings from the ablation study

We tested each feature group individually (added to the v2 baseline):

| Feature Group | Features | R² | vs v2 |
|--------------|----------|-----|-------|
| v2 baseline | 43 | 0.0305 | — |
| + defensive actions | 56 | **0.0503** | +65% |
| + transfer momentum | 45 | 0.0381 | +25% |
| + head-to-head | 48 | 0.0326 | +7% |
| + prev season | 48 | 0.0315 | +3% |
| + team strength | 45 | 0.0250 | -18% |
| + trajectory | 45 | 0.0288 | -6% |

**Defensive actions dominated.** Tackles, recoveries, and clearances/blocks/interceptions are strong predictors of BPS (bonus point system), which directly drives total points. Transfer momentum was the second most impactful group — FPL managers collectively provide useful signal about which players are in form.

### The overfitting trap

When we naively threw all 70 features into a model with 300 trees and max_depth=5, the R² actually **dropped** to 0.0087 — worse than v2! The model was overfitting to noise in the many sparse/NaN-heavy features (H2H is 66% NaN, trajectory is 58% NaN).

The fix: curated feature set (68 features, excluding the noisiest) + lower complexity (max_depth=3, 300 trees, learning_rate=0.03). This gave us the sweet spot.

### Final v3 results

| Metric | v2 | v3 | Change |
|--------|-----|-----|--------|
| R² | 0.0142 | **0.0596** | +320% |
| MAE | 2.241 | **2.173** | -3% |
| RMSE | 2.916 | **2.848** | -2.3% |
| Pred range | 0.93 – 8.51 | 1.14 – 7.34 | tighter, more realistic |

The top predicted players now look very reasonable: Virgil van Dijk (5.96), Rice (5.84), Saka (5.55), Gabriel (5.30), Salah (5.24), Haaland (5.12).

### Technical decisions

- **NaN handling**: XGBoost handles NaN natively, so we don't impute missing values. But `predict_next_gw()` was dropping rows where *any* feature was NaN — this caused the API to return only 2 players! Fixed by only requiring core features (rolling stats, price, FDR) to be non-null.
- **Feature modularity**: Each feature group is now its own function (`_compute_rolling_features()`, `_compute_h2h_features()`, etc.), making it easy to add/remove groups.
- **Graceful degradation**: If vaastav data or FBref data don't exist, the pipeline silently skips those features. The model works with FPL data alone.

### Files changed

| File | Change |
|------|--------|
| `src/fpl/predict.py` | Refactored into 9 helper functions, 27 new features |
| `src/fpl/fetch.py` | Added `fetch_vaastav_data()` |
| `src/fpl/fbref.py` | **New** — FBref fetching + fuzzy name matching |
| `requirements.txt` | Added `soccerdata`, `rapidfuzz`, `numpy` |
| `notebooks/06_enhanced_features.ipynb` | **New** — feature analysis, ablation, v3 training |
| `data/external/vaastav/` | **New** — cached historical season CSVs |
| `data/mappings/fpl_to_fbref.json` | **New** — manual name mapping overrides |
| `models/best_model.joblib` | Updated to v3 |
| `models/model_metadata.json` | 68 features, v3 metrics |

### Lessons learned

1. **More features ≠ better model.** The ablation study was essential. Without it, we'd have shipped a worse model.
2. **Defensive stats are underrated.** Every FPL model tutorial focuses on xG and xA, but tackles/recoveries/CBI are strong BPS predictors.
3. **Transfer momentum is real signal.** Millions of FPL managers collectively notice things that raw stats miss.
4. **NaN handling matters at inference.** Training might work fine with NaN, but inference pipelines that drop NaN rows can silently break.

### What's next

- Improve vaastav name matching (currently using exact web_name — could use fuzzy matching for better coverage)
- Fetch FBref advanced stats (SCA, progressive passes) for attacking feature signal
- Hyperparameter tuning (Bayesian optimization on the curated feature set)
- Per-position models (GK features differ fundamentally from FWD features)
- Blog page on the frontend website

---

## Chapter 9: Next-GW Prediction Accuracy — Model v4

*March 8, 2026*

### The problem with v3

The v3 model was a "tier list" — it ranked players by overall quality rather than predicting who would score highest *next gameweek*. The evidence was damning:

- **Top 15 prediction range was only 1.1 points** (4.86 – 5.96). The model couldn't differentiate between players.
- **Semenyo** (excellent current form) was buried at #15. Joao Pedro (hot streak) wasn't in the top 15 at all.
- **Heavily biased toward premium defenders** — Virgil, Rice, Saka, Gabriel dominated regardless of fixture difficulty.
- The model had no idea *who the player faces next*. A player facing Leicester at home got the same prediction as one facing Liverpool away.

Five root causes:

1. **No upcoming fixture info** — used last-played FDR, not the next opponent's strength
2. **No recency weighting** — simple rolling averages weight all games equally
3. **No form acceleration** — no features for "form is improving"
4. **No single-GW signal** — shortest window was roll3, diluting standout performances
5. **Ownership bias** — `log_selected` pushed popular players up regardless of form

### The fix: 4 phases of new features

**Phase 1 — Opponent Strength (6 features)**

The FPL API already provides team strength ratings on a 1000-1400 scale (`strength_attack_home`, `strength_attack_away`, `strength_defence_home`, `strength_defence_away`). We just weren't using them.

New features:
- `next_opp_attack` / `next_opp_defence` — opponent's contextual strength (home vs away), centered around 0
- `next_fdr` — fixture difficulty rating for the upcoming match
- `is_home_next` — home advantage flag
- `opp_strength_diff` — attack minus defence imbalance of opponent
- `team_vs_opp` — player's team strength minus opponent strength

The critical change in `predict_next_gw()`: at inference time, we **overwrite** these features with the *next* GW's fixture data rather than the last-played values. The model trains on historical opponent strength, then at prediction time receives forward-looking fixture info. No leakage — fixture schedules are public before the season starts.

**Phase 2 — Recency-Weighted Features (19 features)**

*2a. Exponential Moving Averages (14 features)*

EMA with span=3 gives ~50% weight to the most recent game, vs equal weighting in rolling averages. Added EMA variants for: `pts`, `xg`, `xa`, `xgi`, `bonus`, `bps`, `ict` — each with span 3 and 5.

*2b. Last-1-GW Raw Features (5 features)*

The most recent single-GW performance as standalone features: `pts_last1`, `xg_last1`, `xa_last1`, `bonus_last1`, `min_last1`. A 15-point haul last week is now a direct signal, not diluted over 3-5 games.

**Phase 3 — Form Acceleration (6 features)**

These are derived features — just column arithmetic on existing rolling stats:

- `pts_accel_3v5` = `pts_roll3 - pts_roll5` — is 3-game form above 5-game trend?
- `xg_accel_3v5` / `xgi_accel_3v5` — same for expected stats
- `pts_vs_season` = `pts_roll3 - pts_season_avg` — hot streak vs baseline
- `xg_vs_season` — same for xG
- `pts_spike` = `pts_last1 - pts_roll5` — single-GW breakout detection

These explicitly capture "form is improving" — the exact signal needed for next-GW prediction.

**Phase 4 — Bias Correction (1 removal, 1 addition)**

- **Removed `log_selected`** — ownership is a popularity signal, not a performance predictor. It creates a feedback loop where already-popular players get boosted.
- **Added `pts_per_price`** = `pts_roll3 / price` — captures "outperforming expectations for their price bracket."

### Hyperparameter search

Tested 81 combinations across:
- `max_depth`: [3, 4, 5]
- `learning_rate`: [0.01, 0.03, 0.05]
- `n_estimators`: [300, 500, 800]
- `colsample_bytree`: [0.6, 0.7, 0.8]

Best: `max_depth=3, lr=0.01, n_estimators=500, colsample_bytree=0.7`

Lower learning rate + more trees = better generalization with the expanded feature set. Depth stayed at 3 — the model doesn't need deeper trees, it needs better features.

### Ablation study

| Feature Group | R² without | Δ R² | Verdict |
|--------------|-----------|-------|---------|
| Full v4 model | 0.3484 | — | baseline |
| w/o last1_features | 0.3381 | -0.0103 | **most impactful** |
| w/o form_acceleration | 0.3468 | -0.0017 | contributes |
| w/o opponent_strength | 0.3470 | -0.0014 | contributes |
| w/o bias_correction | 0.3471 | -0.0013 | contributes |
| w/o ema_features | 0.3478 | -0.0006 | marginal |

Last-1-GW features had the biggest individual impact — the single most recent game is the strongest signal for next-GW performance. Every group contributed positively; none hurt the model.

### Feature importance

11 of the top 20 most important features are new v4 additions:

`min_last1`, `pts_ema3`, `ict_ema3`, `bps_ema3`, `bps_ema5`, `next_fdr`, `ict_ema5`, `xgi_ema3`, `xgi_ema5`, `xg_ema3`, `team_vs_opp`

The EMA features are dominating — the model strongly prefers recency-weighted stats over simple rolling averages.

### Results

| Metric | v3 | v4 | Change |
|--------|-----|-----|--------|
| R² | 0.0596 | **0.3484** | +484% |
| MAE | 2.173 | **0.949** | -56% |
| RMSE | 2.848 | **1.837** | -36% |
| Features | 68 | 101 | +33 |

### Prediction spot-check

**v4 Top 10 predictions for GW30:**

| # | Player | Pos | Team | Predicted |
|---|--------|-----|------|-----------|
| 1 | Semenyo | MID | Man City | 5.39 |
| 2 | Thiago | FWD | Brentford | 4.98 |
| 3 | Guéhi | DEF | Man City | 4.59 |
| 4 | Saka | MID | Arsenal | 4.54 |
| 5 | Virgil | DEF | Liverpool | 4.51 |
| 6 | Bowen | FWD | West Ham | 4.37 |
| 7 | Gabriel | DEF | Arsenal | 4.31 |
| 8 | Palmer | MID | Chelsea | 4.27 |
| 9 | M.Salah | MID | Liverpool | 4.25 |
| 10 | Konaté | DEF | Liverpool | 4.24 |

**Key wins:**
- **Semenyo jumped from #15 to #1** — form acceleration features are working
- **Better position diversity** — forwards now appear (Thiago #2, Bowen #6)
- **Fixture-aware** — players facing weaker defenses are boosted appropriately
- **Top-15 spread: 1.23 pts** (up from 1.1, with room to grow as model sees more diverse test data)

### Files changed

| File | Change |
|------|--------|
| `src/fpl/predict.py` | 6 new helper functions, 32 new features, forward-looking fixture injection |
| `notebooks/07_next_gw_features.ipynb` | **New** — v4 training, hyperparameter search, ablation |
| `models/best_model.joblib` | Updated to v4 |
| `models/model_metadata.json` | 101 features, v4 metrics |

### Lessons learned

1. **The biggest prediction gain came from the simplest feature.** `pts_last1` (literally "what did the player score last week?") was the most impactful addition. Sometimes the obvious signal is the right one.
2. **EMA > simple rolling averages.** Exponentially-weighted means give the model a natural recency bias without throwing away older data entirely.
3. **Fixture context transforms the model's purpose.** Adding opponent strength + forward-looking fixture injection changed the model from a "tier list" to an actual next-GW predictor.
4. **Removing a feature can improve results.** Dropping `log_selected` (ownership) removed a popularity bias that was masking form signals.
5. **R² jumped from 0.06 to 0.35 — not by adding complexity, but by adding the right information.** The model had the same depth (3) and similar tree count. It just had better features to learn from.

### What's next

- Per-position models (GK features differ fundamentally from outfield players)
- Ensemble with `ep_next` (FPL's own expected points) at inference time
- Auto-refresh pipeline (weekly cron: fetch data → retrain → deploy)
- Improve vaastav name matching for better multi-season H2H coverage
