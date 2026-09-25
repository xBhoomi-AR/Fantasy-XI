# Running the Demo

## Full local demo (recommended)

```bash
uvicorn backend.app:app --reload --port 8000
```

Open **http://localhost:8000**, click **Start Season**, then **Next Gameweek** to step through the real 2025–26 season one gameweek at a time, up to Gameweek 38.

## Sequential pipeline from the command line

```bash
python -m rl_decision_layer.ppo.season_controller --start-gameweek 1 --num-gameweeks 5
```

Runs 5 consecutive gameweeks, carrying squad/bank/free-transfers/chips forward each step.

## Single-gameweek query

```bash
python -m rl_decision_layer.ppo.show_squad --start-gameweek 1
```

Prints one gameweek's recommendation in isolation (not sequential).

## PPO-free MILP-only baseline

```bash
python -m rl_decision_layer.run_baseline
```

Next: [Training & Retraining](training.md)
