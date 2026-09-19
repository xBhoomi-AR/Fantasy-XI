"""Converts a strategic PPO action into MILP keyword arguments and chip controls.

Action is (aggressiveness, budget_level, position_bias, chip_choice):
- aggressiveness (0/1/2): hit_cost (0: 8.0/reluctant, 1: 4.0/normal, 2: 2.0/aggressive)
- budget_level (0/1/2): available money spent (0: 85%, 1: 95%, 2: 100%)
- position_bias (0/1/2): 0: neutral (1.0x), 1: attack-heavy (1.3x MID/FWD), 2: defense-heavy (1.3x GK/DEF)
- chip_choice (0/1/2/3/4): 0: none, 1: wildcard, 2: free_hit, 3: bench_boost, 4: triple_captain
"""

from __future__ import annotations

from ..environment.squad import squad_value

# Extended action space: (aggressiveness, budget_level, position_bias, chip_choice)
ACTION_SHAPE = (3, 3, 3, 5)

HIT_COST_BY_AGGRESSIVENESS = {0: 8.0, 1: 4.0, 2: 2.0}
BUDGET_FRACTION_BY_LEVEL = {0: 0.85, 1: 0.95, 2: 1.0}

POSITION_BIAS_WEIGHTS = {
    0: {"GK": 1.0, "DEF": 1.0, "MID": 1.0, "FWD": 1.0},
    1: {"GK": 0.8, "DEF": 0.8, "MID": 1.3, "FWD": 1.3},  # Attack Heavy
    2: {"GK": 1.3, "DEF": 1.3, "MID": 0.8, "FWD": 0.8},  # Defense Heavy
}

CHIP_NAMES = {
    0: "none",
    1: "wildcard",
    2: "free_hit",
    3: "bench_boost",
    4: "triple_captain",
}


def action_to_milp_kwargs(action, state) -> dict:
    import numpy as np
    act = np.asarray(action).flatten()
    if len(act) == 2:
        aggressiveness, budget_level = act
        pos_bias = 0
        chip_choice = 0
    else:
        aggressiveness, budget_level, pos_bias, chip_choice = act

    available = state.bank + squad_value(state.candidates, state.squad_ids)
    budget = available * BUDGET_FRACTION_BY_LEVEL[budget_level]
    hit_cost = HIT_COST_BY_AGGRESSIVENESS[aggressiveness]
    pos_weights = POSITION_BIAS_WEIGHTS.get(pos_bias, POSITION_BIAS_WEIGHTS[0])

    chip_name = CHIP_NAMES.get(chip_choice, "none")
    is_wc_or_fh = chip_name in ("wildcard", "free_hit")

    return {
        "hit_cost": hit_cost,
        "budget": budget,
        "position_weights": pos_weights,
        "free_hit_or_wildcard": is_wc_or_fh,
        "chip_name": chip_name,
    }
