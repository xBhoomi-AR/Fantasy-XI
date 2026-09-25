# Results & Evaluation

## Prediction layer

See [BiLSTM](../prediction/bilstm.md) and [XGBoost](../prediction/xgboost.md) for metrics — both evaluated on the real 2025–26 test season.

## PPO — 38-gameweek run

| Metric | Value |
|---|---|
| Season net points (38 GW) | 2,208.0 |
| Average points / GW | 58.11 |
| Transfer hits taken | 0 |
| Chips deployed | Free Hit (GW1) · Wildcard (GW2) · Triple Captain (GW3) · Bench Boost (GW4) |
| Season completion | GW1 → GW38, no infeasible gameweeks |

> **TODO:** Insert an RL/season-progression results graph here.

## Is the recommended team actually good?

- **Prediction** — evaluated against real outcomes. Clear, measurable answer.
- **Decision layer** — the run above is real and reproducible, but a single run, not yet baselined.
- **Baseline** — a PPO-free MILP-only baseline exists (`rl_decision_layer/run_baseline.py`) but hasn't been run and compared against the result above.

> **TODO:** Run the baseline and record a genuine comparison here.
