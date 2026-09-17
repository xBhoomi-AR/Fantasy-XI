"""Converts a small strategic PPO action into MILP keyword arguments.

Action is (aggressiveness, budget_level), each 0/1/2:
- aggressiveness sets hit_cost - how reluctant the MILP is to take a
  transfer hit. Lower cost = more willing to take hits = more aggressive.
- budget_level sets how much of the available money gets spent this week -
  holding some back is a simple stand-in for risk preference / saving for a
  future gameweek, since unspent budget carries forward through
  HistoricalEnv's bank.

"Roll vs transfer" isn't a separate action - a conservative, low-budget
setting naturally ends up making 0 transfers when nothing's worth it.
Positional priority and captaincy strategy aren't implemented - they'd need
select_squad() to support a per-position objective weight, which doesn't
exist and isn't being added this session.
"""

from __future__ import annotations

from ..environment.squad import squad_value

ACTION_SHAPE = (3, 3)

HIT_COST_BY_AGGRESSIVENESS = {0: 8.0, 1: 4.0, 2: 2.0}
BUDGET_FRACTION_BY_LEVEL = {0: 0.85, 1: 0.95, 2: 1.0}


def action_to_milp_kwargs(action, state) -> dict:
    aggressiveness, budget_level = action
    available = state.bank + squad_value(state.candidates, state.squad_ids)
    return {
        "hit_cost": HIT_COST_BY_AGGRESSIVENESS[aggressiveness],
        "budget": available * BUDGET_FRACTION_BY_LEVEL[budget_level],
    }
