"""Full 38-Gameweek evaluation runner for Fantasy-XI."""

from __future__ import annotations

import pandas as pd

from rl_decision_layer.environment.historical_env import HistoricalEnv
from rl_decision_layer.environment.squad import build_starting_squad
from rl_decision_layer.optimization.chip_strategy import get_recommended_chip
from rl_decision_layer.optimization.scoring import calculate_reward, score_outcome
from rl_decision_layer.optimization.squad_milp import select_squad
from rl_decision_layer.optimization.starting_xi import pick_starting_xi
from rl_decision_layer.predictions.interface import build_canonical_predictions


def run_full_season(max_gw: int = 38):
    print(f"\n======================================================================")
    print(f"RUNNING FULL-SEASON EVALUATION (GW1 -> GW{max_gw}) WITH RECALIBRATED PIPELINE")
    print(f"======================================================================\n")

    canonical_gw1 = build_canonical_predictions(gameweek=1)
    squad_ids = build_starting_squad(canonical_gw1)

    env = HistoricalEnv(season="2025-26")
    state = env.reset(start_gameweek=1, squad_ids=squad_ids, bank=0.0, free_transfers=1)

    available_chips = {"wildcard": True, "free_hit": True, "bench_boost": True, "triple_captain": True}
    total_raw_points = 0.0
    total_hits = 0

    print(f"{'GW':<5} | {'Chip Used':<16} | {'Actual Pts':<10} | {'Hits':<5} | {'Captain':<25}")
    print("-" * 72)

    for gw in range(1, max_gw + 1):
        # 1. MILP selection under recalibrated predictions
        decision = select_squad(
            state.candidates,
            current_squad_ids=state.squad_ids,
            bank=state.bank,
            free_transfers=state.free_transfers,
        )

        squad_rows = state.candidates[state.candidates["player_id"].isin(decision.selected_ids)]
        xi = pick_starting_xi(squad_rows)

        # 2. Rule-based chip selection
        chip_name = get_recommended_chip(
            gameweek=gw,
            squad_rows=squad_rows,
            xi_starting_ids=xi.starting_ids,
            xi_bench_ids=xi.bench_ids,
            captain_id=xi.captain_id,
            available_chips=available_chips,
        )
        if chip_name != "none":
            available_chips[chip_name] = False

        bench_boost = (chip_name == "bench_boost")
        triple_captain = (chip_name == "triple_captain")

        # 3. Environment Step
        prev_squad = list(state.squad_ids)
        outcome, next_state = env.step(decision.selected_ids, new_bank=decision.remaining_budget)

        if chip_name == "free_hit":
            next_state.squad_ids = prev_squad
            env.squad_ids = prev_squad

        scored = score_outcome(outcome, xi, bench_boost=bench_boost, triple_captain=triple_captain)
        gw_points = scored.total_points - 4.0 * decision.hits

        total_raw_points += gw_points
        total_hits += decision.hits

        cap_row = squad_rows[squad_rows["player_id"] == xi.captain_id]
        cap_name = cap_row["player_name"].values[0] if not cap_row.empty else "None"

        print(f"GW {gw:<3} | {chip_name:<16} | {gw_points:<10.1f} | {decision.hits:<5} | {cap_name:<25}")

        if next_state is not None:
            state = next_state

    print("-" * 72)
    print(f"TOTAL SEASON ACTUAL POINTS (GW1-GW{max_gw}): {total_raw_points:.1f}")
    print(f"TOTAL HITS TAKEN                     : {total_hits} (-{total_hits * 4} pts)")
    print(f"AVERAGE POINTS PER GAMEWEEK          : {total_raw_points / max_gw:.2f} pts/GW")
    print(f"======================================================================\n")



if __name__ == "__main__":
    run_full_season(38)
