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
PPO (strategic action: transfer aggressiveness, budget level)
        ↓
MILP (select_squad) — legal 15-player squad
        ↓
MILP (pick_starting_xi) — 11 starters + captain/vice
        ↓
human-readable output (show_squad.py)
```

### 3. Install dependencies

```bash
pip install -r rl_decision_layer/requirements.txt
```
(For retraining the prediction models, see their own `requirements.txt` under
`models/BiLSTM_model/` and `models/xgboost_model/` — heavier, only needed if retraining.)

### 4–5. Use what's already generated — recommended path for most users

Every prediction file, PPO model, and metadata file needed for a recommendation is already
committed to this repository. No retraining is required.

```bash
python -m rl_decision_layer.ppo.show_squad --model ppo_fpl --start-gameweek 1
```
Prints the recommended 15-player squad, starting XI, captain, vice-captain, bench,
transfers, hits, bank, free transfers, reward, and a legality check, using real player names
(from `players.csv`'s `player_name` column) and real team names.

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

### 9. Train PPO (optional — a trained model already exists)

```bash
python -m rl_decision_layer.ppo.train --timesteps 500
```
Sensible defaults (`--start-gameweek 1 --num-gameweeks 3 --device cpu`). Saves to
`rl_decision_layer/ppo/models/ppo_fpl.zip`. **The currently frozen model
(`ppo_fpl.zip`, 66,048 lifetime timesteps) was not retrained or modified this session.**

### 10–11. Where outputs go / what consumes them

| Output | Location | Consumed by |
|---|---|---|
| BiLSTM predictions | `models/BiLSTM_model/predicted_points.csv` | `predictions/interface.py` |
| XGBoost predictions | `models/xgboost_model/predictions/*.csv` | `predictions/interface.py` |
| PPO model | `rl_decision_layer/ppo/models/*.zip` | `evaluate.py`, `show_squad.py` |
| Final recommendation | printed to terminal | end user |

### 12. Important limitations

- No live FPL data integration, no live/current player prices, no automatic
  current-season/current-gameweek tracking. This is a **historical-data CLI demonstration**,
  not a hosted or live service.
- The MILP is myopic (optimizes one gameweek at a time) and always spends its full available
  budget, which can produce genuine infeasibility late in a long simulated run — documented
  behavior, not a crash.
- Real FPL sell-price rules aren't modeled (today's price is used as sell value).

### 13. Chips

Wildcard, Free Hit, Bench Boost, and Triple Captain are **not implemented**. Future scope only.

### 14. PPO's current learned behavior — stated honestly

PPO is genuinely integrated and trained (66,048 timesteps across varied gameweek windows),
and the full PPO → MILP → starting-XI → scoring pipeline is verified working and legal
end-to-end. The current trained policy converges to a single strategic action
(`aggressiveness=1, budget_level=2`) that is numerically equivalent to the MILP baseline's
own default behavior across every tested evaluation window. It has not been shown to learn
a state-dependent strategy. See `rl_study/` for the full technical explanation of why.

### 15. What this is not

This is a CLI/local demonstration project, not a hosted live FPL service, not a web
application, and not a system with live gameweek/price updates.
