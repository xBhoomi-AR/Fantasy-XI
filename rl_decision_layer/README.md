# RL decision layer

Turns XGBoost's predictions into a candidate pool, steps a squad through
historical gameweeks, and has a transfer-aware MILP that picks each
gameweek's squad, a starting XI/captain, and scores the actual outcome into
a reward. `historical_loop.py` runs this chronologically across a season -
the deterministic MILP-only baseline a PPO-guided version will be compared
against. The `ppo/` package has the observation/action/environment
interfaces a PPO agent will eventually use, but no agent is trained yet -
see "PPO interfaces" below.

`predictions/interface.py` supports two prediction sources - BiLSTM
(`models/BiLSTM_model/predicted_points.csv`, the default) and XGBoost
(`models/xgboost_model/predictions/`), selected via `model="bilstm"` /
`model="xgboost"` wherever predictions are loaded. Only each model's output
files are consumed, never their training internals - the two are swappable
because both are joined into the same canonical schema below.

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
├── run_baseline.py                # small manual smoke test for the MILP-only baseline
├── ppo/
│   ├── observation.py            # build_observation(state) -> fixed-size vector
│   ├── action.py                 # action_to_milp_kwargs(): strategic action -> MILP kwargs
│   ├── env.py                    # PPOEnv(gymnasium.Env)
│   ├── train.py                  # trains and saves a PPO model
│   ├── evaluate.py               # loads a saved model, runs a short episode
│   └── models/                   # saved models (gitignored)
└── tests/
    ├── test_candidate_pool.py
    ├── test_environment.py
    ├── test_milp.py
    ├── test_starting_xi.py
    ├── test_decision_loop.py     # env + MILP wired together across real gameweeks
    ├── test_scoring.py
    ├── test_historical_loop.py
    ├── test_free_transfers.py
    ├── test_ppo_interface.py
    └── test_ppo_training.py
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
(actual points and actual minutes per squad player) into a `ScoredOutcome`:
only the starting XI counts, and the captain's points are added a second
time (real FPL captain doubling). A starter with 0 actual minutes is
auto-substituted by the first bench player with >0 minutes; if the captain
played 0 minutes, the armband falls back to the vice-captain (only if the
vice actually played) - both mirror real FPL's own rules.

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

## PPO

`ppo/` - a real, trained PPO agent (stable-baselines3), but PPO still doesn't
pick players - the existing MILP does that. PPO picks a small strategic
action that changes what gets passed into `select_squad()`. **The final,
authoritative trained model is `ppo_fpl_v4.zip`** - an older `ppo_fpl.zip`
also exists for historical reference but is no longer compatible with the
current observation/action code and should not be used.

**Observation** (`observation.py`, `build_observation(state, available_chips) ->
np.ndarray`, fixed size **71**): 4 features (price, predicted_points,
form_avg5, fixture_difficulty) per squad player in position order (always
15), plus bank/free_transfers/gameweek (3), plus the best available
predicted_points per position among non-squad candidates (4), plus 4 chip
availability flags (wildcard/free_hit/bench_boost/triple_captain, 1.0 if
still unused this season, 0.0 if already used). No actual points anywhere
in it. (A later, since-reverted experiment added 3 more "fixture signal"
features for a 74-dim observation - that version is not what `ppo_fpl_v4.zip`
was trained on and has been removed from the inference path; see `rl_study/`
for the full compatibility investigation.)

**Action** (`action.py`, `(aggressiveness, budget_level, position_bias,
chip_choice)`, `MultiDiscrete(3,3,3,5)`): aggressiveness sets `hit_cost`
passed to `select_squad()` (0/1/2 → 8.0/4.0/2.0 - lower cost, more willing
to take transfer hits). budget_level sets what fraction (0.85/0.95/1.0) of
the squad's full value+bank gets passed as `budget`. position_bias
reweights the objective toward attack (MID/FWD ×1.3) or defense (GK/DEF
×1.3), or stays neutral. chip_choice (0-4) picks none/wildcard/free_hit/
bench_boost/triple_captain - `PPOEnv` enforces each chip as usable at most
once per season (tracked in `self.available_chips`, reset each episode,
consumed on use); requesting an already-used chip is silently downgraded to
"none" rather than erroring.

**Env** (`env.py`, `PPOEnv(gymnasium.Env)`): `reset()`/`step(action)` wrapper
around `HistoricalEnv` + `select_squad()` + `pick_starting_xi()` +
`score_outcome()`/`calculate_reward()`, plus chip-availability tracking and
an optional `MILPCache` (`optimization/milp_cache.py`) that memoizes
`(gameweek, squad, action) -> SquadResult` so repeated/training-time steps
over the same state don't re-solve the MILP. Follows the real Gymnasium API.
Unlike earlier versions, an infeasible MILP decision no longer ends the
episode (`terminated` is always `False`) - it falls back first to a
standard re-solve, then to keeping the existing squad unchanged, so a long
training episode is never killed by one bad action.

**Chip decision path**: by default (`use_heuristic_chips=False`, what
`show_squad.py`/`season_controller.py`/`evaluate.py` use), chip choice comes
directly from PPO's own action. A separate, rule-based chip-timing heuristic
(`optimization/chip_strategy.py::get_recommended_chip()`, e.g. reserving
Free Hit for blank gameweeks, Bench Boost for fixture-congested weeks) exists
and is wired into `PPOEnv.step()`, but only takes over if `PPOEnv` is
constructed with `use_heuristic_chips=True` - an alternate mode, not PPO's
default behavior.

**Training** (`train.py`): builds a `PPOEnv` (38-gameweek episodes,
randomized start gameweek), a stable-baselines3 PPO (`net_arch=[64, 64]`,
`n_steps=1024`, `batch_size=128`, CPU), and calls
`model.learn(total_timesteps=...)` - these defaults match `ppo_fpl_v4.zip`'s
own training recipe exactly.

```
python -m rl_decision_layer.ppo.train --timesteps 50000
```

Training is timesteps-based (each timestep is one gameweek decision), not
episode-based, since stable-baselines3 PPO counts in timesteps.
`--start-gameweek`/`--num-gameweeks` control the training episode's
historical window, `--device` defaults to `cpu`. Default `--save-name` is
`ppo_fpl_pure_rl`, deliberately **not** `ppo_fpl_v4` - that name is reserved
for the final, frozen, evaluated model, so retraining never silently
overwrites it. Real training should be run by you, locally, not by Claude.

`--checkpoint-freq N` saves a snapshot to `ppo/models/checkpoints/` every N
timesteps (0/default = off). `--resume` continues training `--save-name`'s
existing saved model instead of starting fresh.

Models save to `ppo/models/<name>.zip` (gitignored, except `ppo_fpl_v4.zip`
and `ppo_fpl.zip` which are explicitly un-ignored - see the root
`.gitignore`). Load and run one with:

```
python -m rl_decision_layer.ppo.evaluate --model ppo_fpl_v4
python -m rl_decision_layer.ppo.show_squad --model ppo_fpl_v4 --start-gameweek 1
python -m rl_decision_layer.ppo.season_controller --model ppo_fpl_v4 --start-gameweek 1 --num-gameweeks 5
```

`evaluate.py` steps a saved model through a short historical episode and
prints each gameweek's action/reward. `show_squad.py` prints one gameweek's
full human-readable recommendation (names, not IDs). `season_controller.py`
is the actual sequential product - it carries the squad/bank/free-transfers/
chip-availability state from each gameweek into the next within one run, and
can optionally save/resume that state across separate process invocations
via `--save-state`/`--load-state`.

The MILP-only baseline (no PPO at all) still runs independently:

```
python -m rl_decision_layer.run_baseline
```

## Known limitations

- Real FPL sell-price rule (50% of any price rise) - can't be reconstructed
  honestly, no purchase-price history exists in the data. Sell value = the
  player's current price instead.
- The MILP is myopic (single-gameweek only) and always spends its full
  budget, which can eventually leave too little money to field a legal
  squad in a cheap-at-the-low-end gameweek's pool - see "Historical backtest
  loop" above. `PPOEnv` works around this at inference time with a
  keep-existing-squad fallback rather than ending the episode.
- Chips are implemented (see "PPO" above) but PPO's own choice of which
  chip to use has not been shown to adapt to the situation.
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
python rl_decision_layer/tests/test_ppo_training.py
python rl_decision_layer/tests/test_season_controller.py
python rl_decision_layer/tests/test_milp_cache.py
python -m rl_decision_layer.tests.test_full_season   # run as a module - needs the repo root on sys.path
```

Most of these need `pulp`, `gymnasium` and `stable-baselines3`
(`pip install -r rl_decision_layer/requirements.txt`). `test_ppo_training.py`
runs a real (tiny, ~64 timestep) training loop, so it's slower than the rest.

Needs `model_features.csv` to have real content, not a Git LFS pointer -
regenerate locally with `python models/xgboost_model/scripts/build_features.py`
if needed. `predictions/interface.py` caches the raw CSV/`model_features.csv`
reads per process (`functools.lru_cache`) - without it, a full-season
backtest re-parses a ~250k row CSV on every single gameweek and is
unworkably slow; with it, a season completes in well under a minute.
