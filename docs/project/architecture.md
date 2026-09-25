# Architecture

```
Historical FPL / player data
        ↓
Prediction layer (BiLSTM default, XGBoost fallback)
        ↓
RL decision layer (PPO) — strategic action
        ↓
MILP optimization — legal 15-player squad
        ↓
Starting XI + captain/vice
        ↓
Gameweek result / reward
        ↓
Sequential season state → next gameweek
```

> **TODO:** Insert a polished system architecture diagram here.

## Why layers, not one model?

- The **prediction** model can change (BiLSTM ↔ XGBoost) without touching transfer logic.
- **PPO** never has to know FPL's squad-legality rules — that's the optimizer's job.
- **MILP** guarantees a legal, optimal squad given whatever strategy PPO picks.

## Where to go next

- [Prediction Layer](../prediction/index.md)
- [Decision & Optimization](../decision/index.md)
- [Season Workflow](../implementation/workflow.md)
