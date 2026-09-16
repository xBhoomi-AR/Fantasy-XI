# RL decision layer

Turns XGBoost's predictions into a candidate pool, steps a squad through
historical gameweeks, and now has a MILP that picks a legal squad from the
candidate pool. No PPO yet.

Only consumes `models/xgboost_model`'s output files (predictions CSV, raw
player/team data, pre-computed form columns) - never its training internals.
That keeps this swappable if the BiLSTM model becomes the prediction source
later.

## Layout

```
rl_decision_layer/
├── predictions/interface.py   # canonical prediction schema (the joins)
├── candidates/
│   ├── ranking.py              # rank each position separately
│   ├── candidate_pool.py       # top-K + squad + form, DGW-aware
│   └── pipeline.py             # get_candidates(gw, squad) - the entry point
├── environment/
│   ├── squad.py                 # 15-man squad rules + a deterministic test squad
│   └── historical_env.py        # steps a squad through real historical gameweeks
├── optimization/
│   └── squad_milp.py            # select_squad(): MILP-only baseline
└── tests/
    ├── test_day1_pipeline.py
    ├── test_environment.py
    └── test_milp.py
```

## Canonical prediction fields

`player_id, player_name, web_name, position, team_id, team_name,
opponent_team_id, season, gameweek, fixture_id, was_home_int,
fixture_difficulty, price, predicted_points, form_avg3/5/10/38`

Sources and join keys are documented in `predictions/interface.py`. `price`
is FPL's tenths-of-a-million convention (e.g. `53` = £5.3m), confirmed from
the 5-154 range across the whole dataset - no scale conversion applied,
just renamed from `value`.

## Candidate selection

Per position, the pool is the union of:
- prediction top-K (`DEFAULT_PREDICTION_TOP_K`)
- current squad (kept regardless of rank)
- recent form top-K by `form_avg5` (`DEFAULT_FORM_TOP_K`)

Defaults are sized against FPL's fixed squad slots (2 GK/5 DEF/5 MID/3 FWD),
not empirically tuned - override them per experiment.

**Double gameweeks**: a player with two fixtures in one gameweek is
collapsed into a single candidate row with predicted_points summed across
both fixtures (`fixture_count` records how many). Fixture-specific fields
(opponent, home/away) keep the first fixture's values, since a single row
can't represent two different fixtures - MILP will need to deal with DGWs
explicitly when it gets there.

## Leakage rule

Predictions and form for gameweek G only ever use data from before G - this
is enforced upstream in the XGBoost feature pipeline (shift-before-roll),
not recomputed here. The test suite independently recomputes `form_avg3`
from raw match stats for a few sampled players per gameweek and checks it
matches, rather than just trusting the code path.

## Entry point

```python
from rl_decision_layer.candidates.pipeline import get_candidates

pool = get_candidates(target_gameweek=20, current_squad_ids=[1, 2, 3, ...])
```

## Historical environment

`HistoricalEnv` (in `environment/historical_env.py`) walks a squad through
real historical gameweeks:

```python
from rl_decision_layer.environment.historical_env import HistoricalEnv
from rl_decision_layer.environment.squad import build_starting_squad
from rl_decision_layer.predictions.interface import build_canonical_predictions

squad = build_starting_squad(build_canonical_predictions(gameweek=10))
env = HistoricalEnv()
state = env.reset(start_gameweek=10, squad_ids=squad)   # decision-time info only
outcome, next_state = env.step()                         # scores GW10, advances to GW11
```

`reset()`/`step()` return a `DecisionState` (gameweek, squad, bank, free
transfers, candidates) with no actual points in it - that's the future
MILP/PPO's input. `step()` only returns an `Outcome` (actual points scored)
for the gameweek that was just played, after the decision is locked in.
`build_starting_squad` is a deterministic, legal, affordable squad built
from cheapest-per-position - not a real historical manager's team, since we
don't have that data.

## MILP squad selection

`select_squad(candidates)` in `optimization/squad_milp.py` picks the 15
players that maximize total `predicted_points`, subject to:
- exactly 15 players, 2 GK / 5 DEF / 5 MID / 3 FWD
- max 3 players from one club
- total price <= budget (defaults to the full 1000/£100m)

This is the MILP-only baseline - no PPO input yet. It's a fresh full-squad
pick, not a transfer optimizer: it doesn't cap how many players differ from
`current_squad_ids` or account for sell prices/transfer costs, since
`HistoricalEnv` doesn't track those (see below). `current_squad_ids` is only
used to report which selected players were already owned
(`SquadResult.already_owned`).

A few players are occasionally missing `price` in the source data (a gap in
`player_market_history` for that gameweek/player) - `select_squad` drops
them rather than guessing a price, so they're just not selectable.

Once PPO exists, it should influence this by adjusting what gets passed in -
e.g. a smaller/reweighted candidate pool, a tighter budget, or a modified
objective - not by bypassing the MILP's legality constraints.

**Still deferred**: transfer legality/cost, budget enforcement beyond
reporting `bank`, starting XI / captain / vice-captain selection (a separate
step once a 15-man squad exists), chips.

## Running tests

```
python rl_decision_layer/tests/test_day1_pipeline.py
python rl_decision_layer/tests/test_environment.py
python rl_decision_layer/tests/test_milp.py
```

`test_milp.py` needs `pulp` (`pip install -r rl_decision_layer/requirements.txt`).

Needs `model_features.csv` to have real content, not a Git LFS pointer -
regenerate locally with `python models/xgboost_model/scripts/build_features.py`
if needed.
