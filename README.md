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
PPO strategic action (transfer aggressiveness, budget level)
        ↓
transfer-aware MILP (select_squad) — legal 15-player squad,
        built from the CURRENT squad/bank/free-transfers, not from scratch
        ↓
MILP (pick_starting_xi) — 11 starters + captain/vice
        ↓
gameweek result / updated state (squad, bank, free transfers)
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

**The repository includes the frozen PPO model `ppo_fpl.zip` used for the documented
evaluation results.** `.gitignore` explicitly tracks this one file
(`rl_decision_layer/ppo/models/ppo_fpl.zip`) while still ignoring internal/experimental
artifacts in that directory (e.g. `ppo_fpl_2k_backup.zip`, future checkpoints) — once
committed and pushed, a fresh `git clone` will contain it. If for some reason it's absent
from your checkout, you can train a new one yourself (§9 below, a few minutes for a small
run), though that will **not** be the same frozen 66,048-timestep model this project's
results describe — it will be freshly, randomly initialized.

⚠️ **If you run the training command against the default save path, it will silently
overwrite the shipped `ppo_fpl.zip`** (see §9) — use a different `--save-name` if you want
to experiment without touching the shipped model.

## Sequential Gameweek Recommendation — the main product demonstration

`show_squad.py` (below) answers "what's the best squad for gameweek X in isolation." That is
**not** the intended product behavior. The actual intended behavior is a sequential FPL
manager: start with a squad, decide a gameweek, carry the resulting squad/bank/free-transfers
forward, then optimize *transfers relative to that existing squad* for the next gameweek, and
so on — exactly how a real FPL manager operates, never starting over from scratch each week.

This is implemented in `rl_decision_layer/ppo/season_controller.py`. It adds **no new
selection or transfer-optimization logic** — `PPOEnv` (unchanged) already carries
squad/bank/free-transfers from one `step()` to the next internally, and already accepts
`initial_squad`/`initial_bank`/`initial_free_transfers` to resume from a prior state; this
file only adds the outer sequential loop, JSON state persistence between separate runs, and
human-readable per-gameweek display (including a transfers-IN/transfers-OUT diff, computed
as a plain set-difference over the squad IDs the MILP already returned — not a second
transfer calculation).

**Run a fresh 5-gameweek sequential demo with the frozen model:**
```bash
python -m rl_decision_layer.ppo.season_controller --model ppo_fpl --start-gameweek 1 --num-gameweeks 5
```
For each gameweek this prints: the PPO action, transfers IN/OUT versus the *previous*
gameweek's actual resulting squad, the full 15-player squad (starting XI marked), starting
XI, captain, vice-captain, bench, transfers, hits, bank, free transfers, reward, and a
legality check — then proceeds to the next gameweek using the squad it just produced.

**Verified this session, concretely**: running GW1→GW5 continuously showed Willy Boly bought
in GW3's transfers and then sold again in GW4's transfers out — only possible if GW4's
decision is genuinely built on GW3's actual resulting squad, not an independently
re-optimized one.

**Resume a saved run across separate process invocations** (optional — a fresh single
continuous run above is the main demo path):
```bash
python -m rl_decision_layer.ppo.season_controller --model ppo_fpl --num-gameweeks 2 --save-state season_state.json
python -m rl_decision_layer.ppo.season_controller --model ppo_fpl --num-gameweeks 3 --load-state season_state.json --save-state season_state.json
```
`--load-state` reads `{gameweek, squad_ids, bank, free_transfers}` from the given JSON file
and resumes from exactly that point (verified this session: resuming produced a byte-for-byte
identical decision to the equivalent step of a single continuous run). `--save-state` writes
the state after the run's last successful gameweek.

### `show_squad.py` — single-gameweek query (still available, not the main demo)

Once a `ppo_fpl.zip` exists at that path:
```bash
python -m rl_decision_layer.ppo.show_squad --model ppo_fpl --start-gameweek 1
```
Prints the recommended 15-player squad, starting XI, captain, vice-captain, bench,
transfers, hits, bank, free transfers, reward, and a legality check for one gameweek only,
using real player names (from `players.csv`'s `player_name` column) and real team names.
Useful for a quick one-off look at a specific gameweek; use `season_controller.py` above to
see the actual sequential product behavior.

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

### 9. Train PPO (optional — a frozen trained model is already shipped)

```bash
python -m rl_decision_layer.ppo.train --timesteps 500
```
Sensible defaults (`--start-gameweek 1 --num-gameweeks 3 --device cpu`). Saves to
`rl_decision_layer/ppo/models/ppo_fpl.zip` by default.

⚠️ **This command always trains a brand-new, randomly-initialized model and unconditionally
overwrites whatever file is already at that save path — including the shipped frozen model**
(verified from `train.py`: without `--resume` it builds a fresh `PPO(...)` regardless of
what's already there, and `model.save(path)` has no existence check). If you want to
experiment without losing the shipped model, use `--save-name <something-else>`.

`--resume` (continue the *existing* model at `--save-name` instead of starting fresh) also
saves back to the same path when finished — it extends the lineage rather than replacing it
with a random one, but still overwrites the file on disk with the newly-extended version.

### 10–11. Where outputs go / what consumes them

| Output | Location | Tracked in Git? | Consumed by |
|---|---|---|---|
| BiLSTM predictions | `models/BiLSTM_model/predicted_points.csv` | Yes | `predictions/interface.py` |
| XGBoost predictions | `models/xgboost_model/predictions/*.csv` | Yes | `predictions/interface.py` |
| PPO model (frozen, shipped) | `rl_decision_layer/ppo/models/ppo_fpl.zip` | **Yes** (this file only) | `evaluate.py`, `show_squad.py` |
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

Wildcard, Free Hit, Bench Boost, and Triple Captain are **not implemented**. Future scope only.

### 14. PPO's current learned behavior — stated honestly

PPO is genuinely integrated and trained (66,048 timesteps across varied gameweek windows),
and the full PPO → MILP → starting-XI → scoring pipeline is verified working and legal
end-to-end, including the genuinely sequential multi-gameweek behavior in
`season_controller.py` above. **The sequential mechanics are real** — the squad, bank, free
transfers, and transfers genuinely carry from one gameweek to the next. **PPO's strategic
contribution to each of those decisions is currently not state-dependent**: the trained
policy outputs the same action (`aggressiveness=1, budget_level=2`) in every tested
gameweek, sequential or not, numerically equivalent to the MILP baseline's own default
behavior. Do not describe PPO as "learning a different strategy each gameweek" — it isn't,
yet. See `rl_study/` for the full technical explanation of why.

### 15. What this is not

This is a CLI/local demonstration project, not a hosted live FPL service, not a web
application, and not a system with live gameweek/price updates.
