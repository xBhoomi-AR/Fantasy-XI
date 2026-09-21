# Fantasy-XI
Repository for documentation and implementation of the Fantasy XI project.

---

## ⚠️ Temporary Section — Third-Party Reproducibility / Quick Test

**This section is a temporary, factual reproducibility note, not the final project README.**
The final root README will be rewritten once the frontend, backend, MkDocs documentation,
and final product architecture exist. Everything below describes only what the repository
can actually do today, verified against the current code.

### 1. What FantasyXI currently does

Predicts Fantasy Premier League player points, builds a shortlist of realistic transfer
candidates, and uses a reinforcement-learning-guided optimizer to recommend a legal 15-player
squad, an 11-player starting XI, and a captain/vice-captain for a given historical gameweek.

### 2. Pipeline

```
BiLSTM / XGBoost predictions
        ↓
predictions/interface.py (canonical schema)
        ↓
candidate pool (per-position shortlist)
        ↓
PPO strategic action (aggressiveness, budget level, position bias, chip choice)
        ↓
transfer-aware MILP (select_squad) — legal 15-player squad, built from
        the CURRENT squad/bank/free-transfers, not from scratch
        ↓
MILP (pick_starting_xi) — 11 starters + captain/vice
        ↓
gameweek result / updated state (squad, bank, free transfers, chip used)
        ↓
state carried forward into the next gameweek's decision
        ↓
human-readable output (season_controller.py — sequential; show_squad.py — single gameweek)
```

### 3. Install dependencies

For the complete pipeline (both prediction models + the RL/PPO/MILP layer), from the repo root:
```bash
pip install -r requirements.txt
```
If you only need the RL/PPO/MILP layer against already-generated predictions (no retraining
of either prediction model), the lighter, scoped file is enough:
```bash
pip install -r rl_decision_layer/requirements.txt
```
`models/BiLSTM_model/requirements.txt` and `models/xgboost_model/requirements.txt` remain
as the minimal, authoritative dependency lists for each component individually.

### 4–5. Use what's already generated

The prediction files (BiLSTM and XGBoost outputs) and all raw/processed data are already
committed to this repository — no retraining of either prediction model is required to use
them.

**The final, authoritative trained PPO model is `ppo_fpl_v4.zip`** — a 71-dimensional
observation, `MultiDiscrete(3,3,3,5)` action (aggressiveness, budget level, position bias,
chip choice), `[64,64]` MLP policy trained for 50,176 timesteps over 38-gameweek episodes.
`.gitignore` explicitly tracks this file (`rl_decision_layer/ppo/models/ppo_fpl_v4.zip`)
while still ignoring internal/experimental artifacts in that directory — once committed and
pushed, a fresh `git clone` will contain it.

An older model, `ppo_fpl.zip` (67-dim observation, 2-action `(aggressiveness, budget_level)`
only, no chips), is also still present for historical reference but is **no longer
compatible with the current code** and should not be used — pass `--model ppo_fpl_v4`
(now the default for every script below) if you ever need to be explicit.

⚠️ **If you retrain, use a `--save-name` other than `ppo_fpl_v4`** (see §9) — the training
command always overwrites whatever file is already at its save path, and `ppo_fpl_v4.zip` is
the one you don't want to lose.

**Sequential multi-gameweek recommendation (the main product demo)**:
```bash
python -m rl_decision_layer.ppo.season_controller --start-gameweek 1 --num-gameweeks 5
```
Runs 5 consecutive gameweeks in one pass, carrying the squad, bank, free transfers, and chip
availability forward from each gameweek into the next — genuinely sequential, not
independent per-gameweek optimization (verified: a player bought in one gameweek's demo run
was sold again the very next gameweek, only possible if the system remembered what it just
did). For each gameweek it prints the PPO action (including chip choice), transfers IN/OUT
versus the previous gameweek's actual resulting squad, the full 15-player squad, starting XI,
captain, vice-captain, bench, chip used, transfers, hits, bank, free transfers, reward, and a
legality check.

**Single-gameweek query** (quick look at one gameweek, not the sequential product):
```bash
python -m rl_decision_layer.ppo.show_squad --start-gameweek 1
```
Prints the same per-gameweek detail as above for one isolated gameweek, using real player
names (from `players.csv`'s `player_name` column) and real team names.

### 5b. Localhost backend (`backend/`) — API for the future frontend

A minimal FastAPI service that exposes the same sequential pipeline above over HTTP, for a
web frontend to drive gameweek-by-gameweek instead of reading a terminal. It contains **no**
squad-selection/transfer/chip/scoring logic of its own — every endpoint calls
`backend/rl_bridge.py`, which calls the exact same `PPOEnv` + `SeasonState` machinery
`season_controller.py` already uses and `rl_decision_layer/tests/test_season_controller.py`
already validates. One saved `SeasonState` JSON file per demo session
(`backend/sessions/<session_id>.json`, gitignored) is the state store — the same file format
`--save-state`/`--load-state` already produce.

```bash
pip install -r rl_decision_layer/requirements.txt -r backend/requirements.txt
uvicorn backend.app:app --reload --port 8000
```

Endpoints:
- `POST /season/start` — starts a brand-new sequential season (fresh squad, GW1), returns a
  `session_id` plus GW1's full result (squad, starting XI, captain/vice, bench, transfers,
  bank, free transfers, chip used, reward, legality).
- `POST /season/{session_id}/next` — continues that exact session from its saved state and
  returns the next gameweek's full result. Squad/bank/free-transfers/chip-availability are
  carried forward from the previous call, not recomputed independently — calling this
  repeatedly is how a frontend reveals "Gameweek 1 → click Next → Gameweek 2 → ..."
- `GET /season/{session_id}/state` — the session's current saved state (next gameweek, bank,
  free transfers, chip availability, squad size) without advancing it.
- `GET /health` — liveness check.

Verified this session against the real `ppo_fpl_v4.zip`: GW1→GW2→GW3 via the API produced
numbers identical to the CLI's own known-good sequence (GW1 reward 65.0/free_hit, GW2 reward
101.0/wildcard/11 transfers, GW3 reward 91.0/5 transfers/4 hits with both chips correctly
shown as consumed) — the backend is a thin wrapper, not a second implementation.

### 6–7. Retrain / re-run BiLSTM (optional)

```bash
python models/BiLSTM_model/gate_architecture/train.py
python models/BiLSTM_model/gate_architecture/predict.py
```
Training consumes `models/BiLSTM_model/datasets/player_5gw_sequences.zip` (already tracked)
and writes checkpoints to `models/BiLSTM_model/models/` and `models/BiLSTM_model/scalers/`.
`predict.py` writes to `models/BiLSTM_model/predicted_points.csv` by default — the exact
path `predictions/interface.py` reads from. Verified working end-to-end this session
(172,743 predictions generated from a fresh inference run).

### 8. Retrain XGBoost (optional)

```bash
python models/xgboost_model/scripts/train.py
```
Consumes `models/xgboost_model/data/processed/model_features.csv` (already tracked) and
writes `models/xgboost_model/predictions/test_2025_26_predictions.csv` and
`final_predictions_latest_gameweek.csv` — the exact files `predictions/interface.py` reads
for the XGBoost path — plus `position_ensemble.joblib` (gitignored; only needed if you want
to reload the raw trained model object directly, not needed for the existing pipeline).
**Not run this session** — it has no safe way to redirect its output away from the
already-validated prediction files the whole pipeline is tested against, so this path is
structurally verified (clean imports, correct output schema, correct write targets) rather
than freshly executed.

### 9. Train PPO (optional — the final trained model, `ppo_fpl_v4.zip`, is already shipped)

```bash
python -m rl_decision_layer.ppo.train --timesteps 50000
```
Defaults now match `ppo_fpl_v4.zip`'s own recipe exactly: `--start-gameweek 1
--num-gameweeks 38 --device cpu`, 71-dim observation, `MultiDiscrete(3,3,3,5)` action,
`[64,64]` MLP, `n_steps=1024`, `batch_size=128`. Saves to
`rl_decision_layer/ppo/models/<save-name>.zip`, default save-name **`ppo_fpl_pure_rl`**
(deliberately not `ppo_fpl_v4` — that name is reserved for the final, frozen, evaluated
model, so retraining never silently overwrites it).

⚠️ **This command always trains a brand-new, randomly-initialized model and unconditionally
overwrites whatever file is already at its save path** (verified from `train.py`: without
`--resume` it builds a fresh `PPO(...)` regardless of what's already there, and
`model.save(path)` has no existence check). Never pass `--save-name ppo_fpl_v4`.

`--resume` (continue the *existing* model at `--save-name` instead of starting fresh) also
saves back to the same path when finished — it extends the lineage rather than replacing it
with a random one, but still overwrites the file on disk with the newly-extended version.

A model retrained this way will **not** be the same `ppo_fpl_v4.zip` this project's results
describe — training is stochastic and there is no guarantee of reproducing the same learned
behavior, even with identical hyperparameters.

### 10–11. Where outputs go / what consumes them

| Output | Location | Tracked in Git? | Consumed by |
|---|---|---|---|
| BiLSTM predictions | `models/BiLSTM_model/predicted_points.csv` | Yes | `predictions/interface.py` |
| XGBoost predictions | `models/xgboost_model/predictions/*.csv` | Yes | `predictions/interface.py` |
| PPO model (final, shipped) | `rl_decision_layer/ppo/models/ppo_fpl_v4.zip` | **Yes** | `show_squad.py`, `season_controller.py`, `evaluate.py` |
| PPO model (old, incompatible) | `rl_decision_layer/ppo/models/ppo_fpl.zip` | Yes (historical reference only — do not use) | — |
| PPO backup/experimental artifacts | `rl_decision_layer/ppo/models/*` (other files) | No — gitignored | internal development only |
| Final recommendation | printed to terminal | — | end user |

### 12. Important limitations

- No live FPL data integration, no live/current player prices, no automatic
  current-season/current-gameweek tracking. This is a **historical-data CLI demonstration**,
  not a hosted or live service.
- The MILP is myopic (optimizes one gameweek at a time) and always spends its full available
  budget, which can produce genuine infeasibility late in a long simulated run — documented
  behavior, not a crash.
- Real FPL sell-price rules aren't modeled (today's price is used as sell value).

### 13. Chips

**Implemented and verified end-to-end with `ppo_fpl_v4.zip`.** Wildcard, Free Hit, Bench
Boost, and Triple Captain are all real: chip availability (one-shot-per-season, per chip) is
part of the observation PPO sees, chip choice is the 4th component of PPO's action, and the
choice actually reaches the MILP (`chip_name`/`free_hit_or_wildcard` passed into
`select_squad()`) and scoring (`bench_boost`/`triple_captain` flags into `score_outcome()`).
Verified by direct execution: a wildcard used in one gameweek was correctly shown as
unavailable (silently downgraded to "none") when the policy tried to request it again in a
later gameweek of the same sequential run — chip state genuinely carries forward. A separate,
rule-based chip-timing heuristic also exists (`optimization/chip_strategy.py`) but is only
active if `PPOEnv` is constructed with `use_heuristic_chips=True` — the default path (used by
`show_squad.py`/`season_controller.py`) takes chip choice from PPO's own action, not the
heuristic.

### 14. PPO's current learned behavior — stated honestly

PPO is genuinely integrated and trained (`ppo_fpl_v4.zip`, 50,176 timesteps over 38-gameweek
episodes), and the full PPO → MILP → starting-XI → scoring pipeline, including chips and the
sequential multi-gameweek controller, is verified working and legal end-to-end. In the
gameweeks checked, the core strategic action (aggressiveness, budget level, position bias)
stayed fixed across the run, and chip choice repeated a request that had already been used
(and was correctly blocked) rather than switching to a different chip — i.e., it has not been
shown to adapt its strategy to the situation. This is stated plainly, not hidden: the
sequential squad/transfer/chip-state *mechanics* are real and verified; PPO's own strategic
*adaptiveness* is not yet demonstrated. See `rl_study/` for the full technical explanation.

### 15. What this is not

This is a CLI/local demonstration project, not a hosted live FPL service, not a web
application, and not a system with live gameweek/price updates.
