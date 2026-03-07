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
