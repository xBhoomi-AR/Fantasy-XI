# RL decision layer

Turns XGBoost's predictions into a candidate pool, steps a squad through
historical gameweeks, and has a transfer-aware MILP that picks each
gameweek's squad (plus a starting XI/captain picker) from the candidate
pool. No PPO yet.

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
│   ├── squad_milp.py             # select_squad() + decide(state) bridge to HistoricalEnv
│   └── starting_xi.py            # pick_starting_xi(): XI + captain/vice from a selected squad
└── tests/
    ├── test_day1_pipeline.py
    ├── test_environment.py
    ├── test_milp.py
    ├── test_starting_xi.py
    └── test_decision_loop.py     # env + MILP wired together across real gameweeks
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

`select_squad(candidates)` in `optimization/squad_milp.py` picks 15 players
that maximize `predicted_points`, subject to: exactly 15 players, 2 GK / 5
DEF / 5 MID / 3 FWD, max 3 players from one club, total price within budget.

Two modes:
- **fresh pick** (`current_squad_ids` empty) - the original baseline, budget
  defaults to the full 1000/£100m. Unchanged from before.
- **transfer-aware** (`current_squad_ids` given) - budget becomes
  `bank + current squad's value at today's prices`, and each transfer beyond
  `free_transfers` costs 4 points, subtracted straight from the objective
  (via an integer `hits` variable, so the solver only takes hits when the
  points gained are actually worth it). `SquadResult` gets `transfers_made`
  and `hits` on top of the existing fields.

We don't have purchase-price/sell-price history anywhere in the data - only
`value`, today's price - so a sold player is valued at their current price,
not FPL's real 50%-of-profit-on-rise rule. That's a stated simplification,
not the real mechanic. If a current squad member has no row for the target
gameweek (no fixture, or missing price), their contribution to the old
squad's value is just skipped - understates the real bank in that edge case.

`decide(state)` bridges a `DecisionState` straight to a MILP call - this is
what `HistoricalEnv.step()` should be given:

```python
from rl_decision_layer.optimization.squad_milp import decide

state = env.reset(start_gameweek=20, squad_ids=squad, bank=10.0, free_transfers=1)
decision = decide(state)
outcome, next_state = env.step(decision.selected_ids)
```

`env.step()` itself is unchanged - it still just takes a list of player IDs
and doesn't know or care whether a human, the MILP, or (eventually) PPO
produced it. That's deliberate: PPO will call `decide()`-like logic too,
just with extra strategic parameters layered in later (a tighter/looser
budget, a reweighted objective, etc.) rather than a different interface.

A few players are occasionally missing `price` in the source data (a gap in
`player_market_history` for that gameweek/player) - `select_squad` drops
them rather than guessing a price.

## Starting XI / captain

`pick_starting_xi(squad_rows)` in `optimization/starting_xi.py` takes an
already-selected 15-man squad and picks 11 starters (1 GK, 3-5 DEF, 2-5 MID,
1-3 FWD - FPL's formation rules) that maximize predicted points, plus a
captain and vice-captain (highest and second-highest predicted points among
starters). Deliberately separate from squad selection - transfers and
captaincy are different decisions, and PPO should eventually be able to
influence captaincy without touching squad selection. Captain-doubling isn't
wired into `Outcome`'s actual scoring yet - that's reward-calculation
territory, not built here.

**Still deferred**: chips, and the real FPL sell-price rule (see above).

## Running tests

```
python rl_decision_layer/tests/test_day1_pipeline.py
python rl_decision_layer/tests/test_environment.py
python rl_decision_layer/tests/test_milp.py
python rl_decision_layer/tests/test_starting_xi.py
python rl_decision_layer/tests/test_decision_loop.py
```

The last three need `pulp` (`pip install -r rl_decision_layer/requirements.txt`).

Needs `model_features.csv` to have real content, not a Git LFS pointer -
regenerate locally with `python models/xgboost_model/scripts/build_features.py`
if needed.
