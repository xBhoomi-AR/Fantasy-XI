# RL decision layer

Turns XGBoost's predictions into a candidate pool, steps a squad through
historical gameweeks, and has a transfer-aware MILP that picks each
gameweek's squad, a starting XI/captain, and scores the actual outcome into
a reward. `historical_loop.py` runs this chronologically across a season -
the deterministic MILP-only baseline a PPO-guided version will be compared
against. The `ppo/` package has the observation/action/environment
interfaces a PPO agent will eventually use, but no agent is trained yet -
see "PPO interfaces" below.

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
│   ├── starting_xi.py            # pick_starting_xi(): XI + captain/vice from a selected squad
│   └── scoring.py                # score_outcome() + calculate_reward()
├── historical_loop.py            # run_backtest(): chronological MILP-only backtest
├── ppo/
│   ├── observation.py            # build_observation(state) -> fixed-size vector
│   ├── action.py                 # action_to_milp_kwargs(): strategic action -> MILP kwargs
│   └── env.py                    # PPOEnv: reset()/step(action) wrapper
└── tests/
    ├── test_candidate_pool.py
    ├── test_environment.py
    ├── test_milp.py
    ├── test_starting_xi.py
    ├── test_decision_loop.py     # env + MILP wired together across real gameweeks
    ├── test_scoring.py
    ├── test_historical_loop.py
    ├── test_free_transfers.py
    └── test_ppo_interface.py
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

`step(new_squad_ids, new_bank)` carries the decision's leftover budget into
the next gameweek's `DecisionState.bank` - pass `decision.remaining_budget`
from the MILP result.

`free_transfers` rolls over too: `step()` works out how many transfers were
made by diffing `new_squad_ids` against the squad it already had (no extra
parameter needed), then applies real FPL's rule - unused free transfers
carry forward, capped at 5:

```
used = min(transfers_made, free_transfers)
free_transfers = min(5, free_transfers - used + 1)
```

So making 0 transfers accrues one more (up to the cap), making exactly as
many transfers as you had free resets you to 1 next week, and hits still
come from `select_squad()`'s own count - the two stay in sync because both
compute `transfers_made` the same way (squad members not kept).

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
outcome, next_state = env.step(decision.selected_ids, new_bank=decision.remaining_budget)
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
influence captaincy without touching squad selection.

## Scoring and reward

`optimization/scoring.py`. `score_outcome(outcome, xi)` turns an `Outcome`
(actual points per squad player) into a `ScoredOutcome`: only the starting
XI counts, and the captain's points are added a second time (real FPL
captain doubling). No auto-subs (a starter who blanks isn't replaced by a
bench player) and no vice-captain fallback (vice only matters in real FPL if
the captain gets 0 minutes, which isn't tracked) - both are known gaps, not
built.

`calculate_reward(scored, hits)` is `scored.total_points - 4 * hits` - what
a real manager's gameweek score would actually show, since FPL's own score
is already net of transfer-cost deductions. Uses `decision.hits` from the
MILP result, since `ScoredOutcome` itself has no notion of transfers.

## Historical backtest loop

`historical_loop.py`, function `run_backtest()`, run directly with
`python -m rl_decision_layer.historical_loop`. Chains reset → decide → step
→ score → reward across consecutive real gameweeks, stopping (not faking a
result) if the MILP or starting XI ever comes back infeasible. This is the
deterministic MILP-only baseline that a future PPO-guided version will be
compared against.

A full-season run still hits genuine infeasibility partway through - not a
bug, and not fixed by free-transfer rollover. With rollover in place the run
gets to GW1-25 cleanly then goes infeasible at GW26 (a real double
gameweek): available budget (bank + squad value) is 593, but even the
cheapest legal combination in that gameweek's pool costs 641. Directly
tested this is unrelated to `free_transfers` - re-ran the same squad/budget
with free_transfers forced to 1, 3, and 5, all three come back `Infeasible`,
since the budget constraint in `select_squad()` doesn't depend on
`free_transfers` at all (only the hits penalty does).

The real cause is structural: this MILP is myopic (maximizes this gameweek
only, no notion of preserving budget for future gameweeks) and always
spends right up to its budget each week since holding cash back has no
value to it. Over a long enough run that can leave too little money to
field a legal squad in whatever a particular gameweek's candidate pool
happens to cost at the low end. Fixing this properly would mean giving the
MILP some forward-looking budget preference, which is a MILP redesign, not
a state-propagation fix - out of scope here.

## PPO interfaces

`ppo/` - no agent is trained here, this is the observation/action/env
scaffolding a future PPO agent will use. PPO doesn't pick players - the
existing MILP still does that (unchanged) - it picks a small strategic
action that changes what gets passed into `select_squad()`.

**Observation** (`observation.py`, `build_observation(state) -> np.ndarray`,
fixed size 67): 4 features (price, predicted_points, form_avg5,
fixture_difficulty) per squad player in position order (always 15, since
the squad is fixed size - the part of the state that varies, the candidate
pool, isn't flattened directly, see below), plus bank/free_transfers/
gameweek (3), plus the best available predicted_points per position among
non-squad candidates (4). No actual points anywhere in it.

**Action** (`action.py`, `(aggressiveness, budget_level)`, each 0/1/2, a
3x3 space): aggressiveness sets `hit_cost` passed to `select_squad()` -
lower cost, more willing to take transfer hits. budget_level sets what
fraction (0.85/0.95/1.0) of the squad's full value+bank gets passed as
`budget`. Both are existing `select_squad()` parameters - nothing about the
MILP's formulation changed. "Roll vs transfer" isn't a separate action, it
falls out of a conservative/low-budget setting naturally making 0
transfers. Positional priority and captaincy strategy aren't implemented -
they'd need a per-position objective weight in `select_squad()` that
doesn't exist and wasn't added.

**Env** (`env.py`, `PPOEnv`): plain `reset()`/`step(action)` wrapper around
`HistoricalEnv` + `select_squad()` + `pick_starting_xi()` +
`score_outcome()`/`calculate_reward()` - all reused unmodified. An
infeasible action (see below) ends the episode with a -100 reward rather
than faking a squad. Not a `gymnasium.Env` yet - no RL library is installed
in this project; wrapping it is a trivial follow-up once a training library
is chosen.

**Known interaction, not a bug**: `build_starting_squad()` already builds
the cheapest legal squad for its gameweek's pool. Asking for `budget_level`
0 or 1 (i.e. less than 100% of that squad's own value) from a fresh minimal
squad is often genuinely infeasible, since there's no legal squad cheaper
than the cheapest one already found - confirmed directly, and it doesn't
resolve after a step or two either, since a 0-transfer decision leaves the
squad exactly as minimal as it started. The budget-saving actions only make
sense once a squad's value has grown past the bare minimum through real
transfers over a season - `test_infeasible_action_is_handled_safely` uses
this exact case to prove the episode-ending safeguard works.

## Known limitations

- Real FPL sell-price rule (50% of any price rise) - can't be reconstructed
  honestly, no purchase-price history exists in the data. Sell value = the
  player's current price instead.
- The MILP is myopic (single-gameweek only) and always spends its full
  budget, which can eventually leave too little money to field a legal
  squad in a cheap-at-the-low-end gameweek's pool. Not something
  free-transfer rollover fixes - see "Historical backtest loop" above.
- Chips aren't implemented.
- No vice-captain fallback / auto-subs in scoring.
- A squad member missing from a gameweek's candidate pool (no fixture, or
  missing price) contributes 0 to that squad's valuation for budget
  purposes, understating the real available money.

## Running tests

```
python rl_decision_layer/tests/test_candidate_pool.py
python rl_decision_layer/tests/test_environment.py
python rl_decision_layer/tests/test_milp.py
python rl_decision_layer/tests/test_starting_xi.py
python rl_decision_layer/tests/test_decision_loop.py
python rl_decision_layer/tests/test_scoring.py
python rl_decision_layer/tests/test_historical_loop.py
python rl_decision_layer/tests/test_free_transfers.py
python rl_decision_layer/tests/test_ppo_interface.py
```

Most of these need `pulp` (`pip install -r rl_decision_layer/requirements.txt`).

Needs `model_features.csv` to have real content, not a Git LFS pointer -
regenerate locally with `python models/xgboost_model/scripts/build_features.py`
if needed. `predictions/interface.py` caches the raw CSV/`model_features.csv`
reads per process (`functools.lru_cache`) - without it, a full-season
backtest re-parses a ~250k row CSV on every single gameweek and is
unworkably slow; with it, a season completes in well under a minute.
