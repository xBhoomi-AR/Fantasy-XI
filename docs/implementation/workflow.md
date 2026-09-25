# Season Workflow

Every gameweek's decision starts from the *actual outcome* of the previous one — not an independent re-optimization.

```
GW1 squad, bank, free transfers, chips
        ↓  (PPO strategy + MILP transfers)
GW2 squad, bank, free transfers, chips
        ↓
GW3 …
```

## What carries forward

- Squad IDs
- Bank
- Free-transfer count
- Chips already used

`SeasonState` (`rl_decision_layer/ppo/season_controller.py`) stores this between gameweeks.

## Why this matters

- A player bought this week can be sold again next week.
- Free transfers accumulate when unused (capped at 5).
- A chip used once is unavailable for the rest of the season.

The same mechanism lets the [demo](demo.md) resume a season one click at a time.
