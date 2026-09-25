# PPO — Strategic Layer

A PPO agent (Stable-Baselines3) doesn't pick players — it picks a small **strategic action** each gameweek.

## Action

| Component | Values | Effect |
|---|---|---|
| Aggressiveness | 0 / 1 / 2 | Willingness to take a transfer hit |
| Budget level | 0 / 1 / 2 | Fraction of funds spent (85% / 95% / 100%) |
| Position bias | 0 / 1 / 2 | Lean toward attack, defence, or neutral |
| Chip choice | 0–4 | None, Wildcard, Free Hit, Bench Boost, Triple Captain |

## Observation

A 71-dimensional state: each squad player's price/predicted points/form/fixture difficulty, bank, free transfers, gameweek, best available replacement per position, and chip availability.

## Reward

```
reward = starting XI points + captain bonus + chip bonus − 4 × hits
```

## Model in use

`rl_decision_layer/ppo/models/ppo_fpl_v4.zip` — used everywhere in this repository.

> **TODO:** Insert an RL training/results graph here.

## Why not the whole squad?

PPO's action space stays small and learnable. The actual constrained squad-selection problem goes to [MILP](milp.md), which solves it exactly.
