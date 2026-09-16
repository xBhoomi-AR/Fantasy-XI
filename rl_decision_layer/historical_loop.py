"""Runs the deterministic MILP-only strategy chronologically across real
historical gameweeks. This is the reference baseline PPO will later be
compared against.

Run with:
    python -m rl_decision_layer.historical_loop
"""

from __future__ import annotations

from dataclasses import dataclass

from .environment.historical_env import HistoricalEnv
from .environment.squad import build_starting_squad
from .optimization.scoring import calculate_reward, score_outcome
from .optimization.squad_milp import decide
from .optimization.starting_xi import pick_starting_xi
from .predictions.interface import build_canonical_predictions


@dataclass
class GameweekLog:
    gameweek: int
    squad_ids: list[int]
    starting_ids: list[int]
    captain_id: int | None
    vice_captain_id: int | None
    transfers_made: int
    hits: int
    bank_after: float
    predicted_objective: float
    actual_points: float
    reward: float


def run_backtest(
    start_gameweek: int,
    num_gameweeks: int,
    season: str = "2025-26",
    initial_squad: list[int] | None = None,
    initial_bank: float = 0.0,
    free_transfers: int = 1,
) -> list[GameweekLog]:
    if initial_squad is None:
        initial_squad = build_starting_squad(build_canonical_predictions(season=season, gameweek=start_gameweek))

    env = HistoricalEnv(season=season)
    state = env.reset(start_gameweek, initial_squad, bank=initial_bank, free_transfers=free_transfers)

    logs = []
    for _ in range(num_gameweeks):
        decision = decide(state)
        if decision.status != "Optimal":
            print(f"GW{state.gameweek}: MILP decision infeasible ({decision.status}), stopping backtest here")
            break

        squad_rows = state.candidates[state.candidates["player_id"].isin(decision.selected_ids)]
        xi = pick_starting_xi(squad_rows)
        if xi.status != "Optimal":
            print(f"GW{state.gameweek}: starting XI infeasible, stopping backtest here")
            break

        outcome, next_state = env.step(decision.selected_ids, new_bank=decision.remaining_budget)
        scored = score_outcome(outcome, xi)
        reward = calculate_reward(scored, decision.hits)

        logs.append(GameweekLog(
            gameweek=outcome.gameweek,
            squad_ids=decision.selected_ids,
            starting_ids=xi.starting_ids,
            captain_id=xi.captain_id,
            vice_captain_id=xi.vice_captain_id,
            transfers_made=decision.transfers_made,
            hits=decision.hits,
            bank_after=next_state.bank,
            predicted_objective=decision.objective_value,
            actual_points=scored.total_points,
            reward=reward,
        ))

        state = next_state

    return logs


if __name__ == "__main__":
    logs = run_backtest(start_gameweek=1, num_gameweeks=38, initial_bank=0.0, free_transfers=1)

    print(f"\n{'GW':>3} {'transfers':>9} {'hits':>4} {'bank':>7} {'objective':>10} {'actual':>7} {'reward':>7}")
    for log in logs:
        print(f"{log.gameweek:>3} {log.transfers_made:>9} {log.hits:>4} {log.bank_after:>7.1f} "
              f"{log.predicted_objective:>10.2f} {log.actual_points:>7.1f} {log.reward:>7.1f}")

    total_reward = sum(log.reward for log in logs)
    print(f"\n{len(logs)} gameweeks completed, total reward: {total_reward:.1f}")
