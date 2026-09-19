"""Generates a precomputed MILP decision cache for faster RL training.

Run with:
    python -m rl_decision_layer.scripts.generate_milp_cache --season 2025-26
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from rl_decision_layer.candidates.pipeline import get_candidates
from rl_decision_layer.environment.historical_env import DecisionState
from rl_decision_layer.environment.squad import build_starting_squad
from rl_decision_layer.optimization.milp_cache import MILPCache
from rl_decision_layer.optimization.squad_milp import select_squad
from rl_decision_layer.ppo.action import ACTION_SHAPE, action_to_milp_kwargs
from rl_decision_layer.predictions.interface import build_canonical_predictions

CACHE_PATH = Path(__file__).resolve().parent.parent / "ppo" / "models" / "milp_cache.pkl"


def pregenerate_cache(season: str = "2025-26", output_path: Path = CACHE_PATH):
    cache = MILPCache()
    print(f"Generating MILP cache for season {season}...")

    # Start with initial GW1 squad baseline
    gw1_preds = build_canonical_predictions(season=season, gameweek=1)
    current_squad = build_starting_squad(gw1_preds)
    bank = 0.0
    free_transfers = 1

    # Action dimensions: aggressiveness (3), budget (3), position_bias (3), chip_choice (5)
    action_combos = list(itertools.product(*(range(dim) for dim in ACTION_SHAPE)))
    total_iterations = 38 * len(action_combos)
    count = 0

    for gw in range(1, 39):
        candidates = get_candidates(gw, current_squad_ids=tuple(current_squad), season=season)
        state = DecisionState(
            gameweek=gw,
            squad_ids=list(current_squad),
            bank=bank,
            free_transfers=free_transfers,
            candidates=candidates,
        )

        for act in action_combos:
            count += 1
            act_tuple = tuple(act)
            kwargs = action_to_milp_kwargs(act_tuple, state)
            chip_name = kwargs.pop("chip_name", "none")

            res = select_squad(
                candidates,
                current_squad_ids=current_squad,
                bank=bank,
                free_transfers=free_transfers,
                **kwargs,
            )

            cache.store(gw, current_squad, act_tuple, res)

            if count % 200 == 0 or count == total_iterations:
                print(f"Progress: [{count}/{total_iterations}] entries cached...")

        # Advance baseline squad using default action (1, 2, 0, 0) - normal hit cost, 100% budget, neutral, no chip
        def_kwargs = action_to_milp_kwargs((1, 2, 0, 0), state)
        def_kwargs.pop("chip_name", "none")
        def_res = select_squad(
            candidates,
            current_squad_ids=current_squad,
            bank=bank,
            free_transfers=free_transfers,
            **def_kwargs,
        )
        if def_res.status == "Optimal":
            current_squad = def_res.selected_ids
            bank = def_res.remaining_budget
            free_transfers = min(5, free_transfers + 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache.save(output_path)
    print(f"Successfully saved {len(cache)} cached MILP decisions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="2025-26")
    args = parser.parse_args()
    pregenerate_cache(season=args.season)
