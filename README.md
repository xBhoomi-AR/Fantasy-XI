# FantasyXI

**AI-powered decision-support for Fantasy Premier League** — predicts player performance, reasons strategically about transfers and chips, and builds a legal, optimal squad, one gameweek at a time.

*Project X, Community of Coders (COC) @ VJTI*

📖 [View full documentation](https://xbhoomi-ar.github.io/Fantasy-XI/)

---

## Overview

Historical FPL data → predicted player points → strategic decision → optimized legal squad → starting XI → gameweek result → next gameweek. Repeated sequentially across a season.

1. **Prediction** — how many points will each player score?
2. **Strategy (RL)** — how aggressive should this gameweek be, and is a chip worth playing?
3. **Optimization (MILP)** — what's the best *legal* squad given that strategy?
4. **Sequential state** — carry the result into the next gameweek.

## Problem

Doing well in FPL means, every week for 38 weeks:

- Picking a strong 15-man squad under a fixed budget.
- Deciding when a transfer is worth a points hit.
- Timing season-long chips correctly.
- Doing all of this from incomplete, noisy information.

FantasyXI automates this: predict performance, then use a trained strategic policy plus a constraint solver to make legal, reasoned decisions every gameweek.

## Our Approach

Split the problem like a real manager would — prediction, strategy, and optimization as separate, swappable layers, so the prediction model, the strategic policy, and the squad-legality rules can each change independently.

## Key Features

- **Two swappable prediction models** — BiLSTM (default) and XGBoost (fallback), same canonical output schema.
- **PPO strategic layer** — learns transfer aggressiveness, budget usage, position bias, and chip timing.
- **MILP squad optimizer** — legal 15-player squad + starting XI under real FPL constraints.
- **Sequential season execution** — squad, bank, free transfers, and chips carry forward gameweek to gameweek.
- **All four FPL chips** — Wildcard, Free Hit, Bench Boost, Triple Captain — implemented end-to-end.
- **A working local demo** — FastAPI + browser UI visualizing the real pipeline.

## System Architecture

```
Historical FPL / player data
        ↓
Prediction layer (BiLSTM default, XGBoost fallback)
        ↓
RL decision layer (PPO) — strategic action
        ↓
MILP optimization — legal squad + starting XI + captain
        ↓
Gameweek result / reward
        ↓
Sequential season state → next gameweek
```
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/adf0a129-ab64-42f7-8922-a107498b8132" />

## Prediction Layer

Estimates expected FPL points per player per gameweek. Both models optimize for **ranking players well**, not exact scorelines.

### BiLSTM (default)

Dual-expert gate architecture: a BiLSTM + attention encoder feeds a low-band and a high-band specialist regressor, plus a haul-potential calibrator for explosive gameweeks.

| Metric (2025–26 test set) | Overall | 0–2 pts | 3–5 pts | 6–9 pts | 10+ pts |
|---|---|---|---|---|---|
| MAE | 2.2315 | 2.072 | 3.998 | 1.610 | 4.453 |
| RMSE | 3.6546 | 3.634 | 4.192 | 2.171 | 5.318 |
| Spearman | 0.7183 | — | — | — | — |

### XGBoost (fallback)

Position-specific regressors (GK/DEF/MID/FWD) on engineered historical FPL/fixture/team/Understat features.

| Metric (2025–26 test set) | Overall | GK | DEF | MID | FWD |
|---|---|---|---|---|---|
| MAE | 1.2689 | — | — | — | — |
| RMSE | 2.1067 | — | — | — | — |
| Spearman (ranking) | 0.7947 | 0.7021 | 0.7881 | 0.8133 | 0.8115 |

## Evaluation & Results
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/8b82a6bd-e837-42e1-9b10-e08c1920ed61" /><img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/4a1a1ed3-69d3-4edd-9047-e9ad52242c84" />

Both write to the same canonical schema (`rl_decision_layer/predictions/interface.py`) — interchangeable downstream.

## Decision & Optimization Layer

### PPO — strategy

A PPO agent (Stable-Baselines3) doesn't pick players — it picks a small **strategic action** each gameweek:

| Action component | What it controls |
|---|---|
| Aggressiveness | Willingness to take a transfer hit for extra points |
| Budget level | Fraction of available funds spent (85% / 95% / 100%) |
| Position bias | Lean toward attack, defence, or neutral |
| Chip choice | None, Wildcard, Free Hit, Bench Boost, Triple Captain |

PPO observes squad state, bank, free transfers, and chip availability, and its action feeds the MILP below. Trained artifact in use: `rl_decision_layer/ppo/models/ppo_fpl_v4.zip`.

### MILP — optimization

Turns PPO's strategy into a legal squad, maximizing predicted points under FPL's real constraints:

- Exactly 15 players: 2 GK, 5 DEF, 5 MID, 3 FWD.
- Max 3 players per club.
- Total cost within budget (bank + current squad value).
- Free transfers roll over (capped at 5); extra transfers cost 4 points each ("a hit"), only taken when worth it.

A second MILP picks the starting XI (1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD) and captain/vice (top two predicted scorers among starters). Captain points are doubled, and a 0-minute starter is auto-substituted — real FPL scoring rules.

### FPL Terminology

| Term | Meaning |
|---|---|
| GW | Gameweek |
| FT | Free Transfer |
| Hit | Extra transfer, costs 4 points |
| Bank | Remaining budget |
| Chip | One-per-season special action |
| PPO | Strategic decision layer |
| MILP | Squad-legality optimizer |

### Sequential gameweek execution

```
GW1 → decision → result/state → GW2 → decision → result/state → … → GW38
```

Squad, bank, free transfers, and chip usage carry forward — a player bought this week can be sold next week, and a used chip stays unavailable for the season.

## Project Journey / Mini Projects

- **`mini_projects/rl_sandbox/`** — DQN on a synthetic FPL-like environment, to learn RL mechanics first.
- **`mini_projects/PPO mini project/`** — PPO on portfolio management (not FPL), to learn PPO before applying it here.

Exploratory only, each with its own `requirements.txt` — not part of the root install.

## Results / Evaluation

**Prediction layer** — see BiLSTM/XGBoost tables above (both ranking-oriented, 2025–26 test season).

**PPO decision layer — 38-gameweek run** (`ppo_fpl_v4.zip`, real data from `rl_decision_layer/ppo/ppo_38gw_evaluation_results.csv`):

| Metric | Value |
|---|---|
| Season net points (38 GW) | 2,208.0 |
| Average points / GW | 58.11 |
| Transfer hits taken | 0 |
| Chips deployed | Free Hit (GW1) · Wildcard (GW2) · Triple Captain (GW3) · Bench Boost (GW4) |
| Season completion | GW1 → GW38, no infeasible gameweeks |

<img width="4200" height="1800" alt="image" src="https://github.com/user-attachments/assets/64ed3dc2-702c-49d5-97d8-0c856ab79bc6" />

**How do we know a recommended team is actually good?**

- **Prediction** — evaluated against real outcomes (MAE/RMSE/Spearman above).
- **Decision layer** — the run above is real and reproducible, but a single run, not yet baselined.
- **Baseline** — a PPO-free MILP-only baseline is implemented (`run_baseline.py` / `historical_loop.py`) but hasn't yet been run and compared against the result above.
- An earlier internal report claimed a specific baseline score, but that number is hardcoded in the report script, not a real run — not treated as verified evidence here.

<img width="4200" height="1800" alt="image" src="https://github.com/user-attachments/assets/b487d365-0ab1-49b1-833d-5dc631d90f79" />

## Frontend / Demo

```
Start Season → Gameweek 1 → Next Gameweek → Gameweek 2 → … → Season Complete
```

Squad, pitch-formation starting XI, captain/vice, transfers, chips, bank/free transfers — all rendered straight from the backend. **No prediction/RL/optimization logic lives in the frontend.**

See [Installation / Setup](#installation--setup) to run it.

## Project Structure

```text
FantasyXI/
├── models/
│   ├── BiLSTM_model/          # Dual-expert BiLSTM prediction model (default)
│   └── xgboost_model/         # Position-specific XGBoost model (fallback)
├── rl_decision_layer/
│   ├── predictions/           # Canonical prediction schema
│   ├── candidates/            # Per-position candidate shortlisting
│   ├── environment/           # Squad rules + historical gameweek stepping
│   ├── optimization/          # MILP squad selection, starting XI, scoring, chips
│   ├── ppo/                   # PPO observation/action/env, training, trained models
│   └── tests/                 # Test suite
├── backend/                   # FastAPI backend
├── frontend/                  # Plain HTML/CSS/JS demo UI
├── mini_projects/             # Exploratory RL projects
├── docs/                      # MkDocs documentation site source
├── mkdocs.yml                 # Documentation site config
├── requirements.txt           # Consolidated dependencies
└── README.md
```

## Future Scope

- **Live FPL data integration** — currently historical-data only.
- **A more adaptive PPO** — strategic action hasn't yet been shown to vary situationally.
- **Forward-looking MILP** — currently myopic (one gameweek at a time), which can cause late-season infeasibility.
- **A validated baseline comparison** — see Results/Evaluation above.

## Installation / Setup

**Prerequisites** — Python 3.10+ (tested on 3.12). No Node.js/npm/React needed.

**Full project:**
```bash
pip install -r requirements.txt
```
**Demo only (lighter, no retraining):**
```bash
pip install -r rl_decision_layer/requirements.txt -r backend/requirements.txt
```
Each component also keeps its own minimal `requirements.txt` if you only need one part.

## Usage / Quick Start

**Run the full local demo:**
```bash
uvicorn backend.app:app --reload --port 8000
```
Open **http://localhost:8000** → Start Season → Next Gameweek, through Gameweek 38.

**Run the sequential pipeline from the CLI:**
```bash
python -m rl_decision_layer.ppo.season_controller --start-gameweek 1 --num-gameweeks 5
```
**Single-gameweek query:**
```bash
python -m rl_decision_layer.ppo.show_squad --start-gameweek 1
```
**PPO-free MILP-only baseline:**
```bash
python -m rl_decision_layer.run_baseline
```

## Training / Retraining Options

Existing artifacts (`ppo_fpl_v4.zip`, BiLSTM checkpoints, XGBoost models) are enough to run everything above — retraining is optional.

**Retrain BiLSTM:**
```bash
python models/BiLSTM_model/gate_architecture/train.py
python models/BiLSTM_model/gate_architecture/predict.py
```
**Retrain XGBoost:**
```bash
python models/xgboost_model/scripts/train.py
```
**Retrain PPO:**
```bash
python -m rl_decision_layer.ppo.train --timesteps 50000
```
⚠️ Always overwrites its save path (default `ppo_fpl_pure_rl.zip`). Never pass `--save-name ppo_fpl_v4` — that's the frozen model this README's results describe.

More detail: `rl_decision_layer/README.md`, `models/xgboost_model/README.md`.

## Contributors

- [**Bhoomi Vaity**](https://github.com/xBhoomi-AR)
- [**Darshan Mahale**](https://github.com/d4rshnn)

## Mentors / Acknowledgements

- [**Ojas Alai**](https://github.com/ojasalai27)
- [**Kavish Nasta**](https://github.com/kavishnasta)

Built under **Project X, Community of Coders (COC) @ VJTI**.
