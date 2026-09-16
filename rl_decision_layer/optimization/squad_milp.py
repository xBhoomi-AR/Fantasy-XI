"""MILP baseline: picks the 15-player squad that maximizes predicted_points,
subject to FPL's structural rules. No PPO input yet - this is the plain
MILP-only baseline we'll later compare a PPO-guided version against.

This is a fresh full-squad pick, not a transfer optimizer. It doesn't limit
how many players differ from an existing squad or account for sell prices /
transfer costs - the environment doesn't model those yet (see
environment/historical_env.py), so we're not pretending to enforce rules we
can't actually check. current_squad_ids is only used to flag which selected
players were already owned.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd
import pulp

from ..environment.squad import BUDGET, MAX_PER_CLUB, POSITION_COUNTS, SQUAD_SIZE


@dataclass
class SquadResult:
    status: str
    selected_ids: list[int]
    by_position: dict[str, list[int]]
    total_cost: float
    objective_value: float
    remaining_budget: float
    already_owned: list[int] = field(default_factory=list)


def select_squad(
    candidates: pd.DataFrame,
    budget: float = BUDGET,
    current_squad_ids: Sequence[int] = (),
) -> SquadResult:
    players = candidates.drop_duplicates("player_id").reset_index(drop=True)
    # a handful of players are missing `price` in the source data (e.g. a
    # gap in player_market_history for that gameweek) - can't cost them, so
    # they're not selectable rather than guessing a price
    players = players.dropna(subset=["price", "predicted_points"])

    prob = pulp.LpProblem("squad_selection", pulp.LpMaximize)
    pick = {row.player_id: pulp.LpVariable(f"pick_{row.player_id}", cat="Binary") for row in players.itertuples()}

    prob += pulp.lpSum(pick[row.player_id] * row.predicted_points for row in players.itertuples())

    prob += pulp.lpSum(pick.values()) == SQUAD_SIZE

    for position, count in POSITION_COUNTS.items():
        pos_ids = players.loc[players["position"] == position, "player_id"]
        prob += pulp.lpSum(pick[pid] for pid in pos_ids) == count

    for team_id, group in players.groupby("team_id"):
        prob += pulp.lpSum(pick[pid] for pid in group["player_id"]) <= MAX_PER_CLUB

    prob += pulp.lpSum(pick[row.player_id] * row.price for row in players.itertuples()) <= budget

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]

    if status != "Optimal":
        return SquadResult(status=status, selected_ids=[], by_position={}, total_cost=0.0,
                            objective_value=0.0, remaining_budget=budget)

    selected = players[players["player_id"].map(lambda pid: pick[pid].value() == 1)]
    by_position = {pos: selected.loc[selected["position"] == pos, "player_id"].tolist() for pos in POSITION_COUNTS}
    total_cost = float(selected["price"].sum())

    return SquadResult(
        status=status,
        selected_ids=selected["player_id"].tolist(),
        by_position=by_position,
        total_cost=total_cost,
        objective_value=float(pulp.value(prob.objective)),
        remaining_budget=budget - total_cost,
        already_owned=[pid for pid in selected["player_id"] if pid in set(current_squad_ids)],
    )
