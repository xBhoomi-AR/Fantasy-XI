"""MILP squad selector.

Two modes, both handled by select_squad():
- fresh pick (current_squad_ids empty): the original baseline - best legal 15
  under `budget` (defaults to the full £100m). No transfers or hits involved.
- transfer-aware (current_squad_ids given): budget becomes bank + whatever the
  current squad is worth at today's prices, and each transfer beyond
  free_transfers costs 4 points in the objective, same as real FPL.

We don't have any purchase-price/sell-price history anywhere in the data -
only `value`, today's market price - so a "sold" player is valued at their
current price, not FPL's real 50%-of-profit-on-rise rule. That's a
simplification we're stating outright, not pretending is the real mechanic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd
import pulp

from ..environment.squad import BUDGET, MAX_PER_CLUB, POSITION_COUNTS, SQUAD_SIZE, squad_value


@dataclass
class SquadResult:
    status: str
    selected_ids: list[int]
    by_position: dict[str, list[int]]
    total_cost: float
    objective_value: float
    remaining_budget: float
    already_owned: list[int] = field(default_factory=list)
    transfers_made: int = 0
    hits: int = 0


def select_squad(
    candidates: pd.DataFrame,
    budget: float | None = None,
    current_squad_ids: Sequence[int] = (),
    bank: float = 0.0,
    free_transfers: int = 1,
    hit_cost: float = 4.0,
    position_weights: dict[str, float] | None = None,
    free_hit_or_wildcard: bool = False,
    chip_name: str = "none",
    **kwargs,
) -> SquadResult:
    # If chip is wildcard or free_hit, treat as zero hits
    if chip_name in ("wildcard", "free_hit"):
        free_hit_or_wildcard = True
    players = candidates.drop_duplicates("player_id").reset_index(drop=True)
    players = players.dropna(subset=["price", "predicted_points"])

    transfer_aware = len(current_squad_ids) > 0 and not free_hit_or_wildcard
    if budget is None:
        budget = bank + squad_value(players, current_squad_ids) if len(current_squad_ids) > 0 else BUDGET
    elif budget <= 100.0:
        budget = budget * 10.0

    prob = pulp.LpProblem("squad_selection", pulp.LpMaximize)
    pick = {row.player_id: pulp.LpVariable(f"pick_{row.player_id}", cat="Binary") for row in players.itertuples()}

    pos_w = position_weights or {}
    objective = pulp.lpSum(
        pick[row.player_id] * (row.predicted_points * pos_w.get(row.position, 1.0))
        for row in players.itertuples()
    )

    hits_var = None
    if transfer_aware:
        kept_ids = [pid for pid in current_squad_ids if pid in pick]
        transfers = SQUAD_SIZE - pulp.lpSum(pick[pid] for pid in kept_ids)
        hits_var = pulp.LpVariable("hits", lowBound=0, cat="Integer")
        prob += hits_var >= transfers - free_transfers
        objective -= hit_cost * hits_var

    prob += objective

    prob += pulp.lpSum(pick.values()) == SQUAD_SIZE

    for position, count in POSITION_COUNTS.items():
        pos_ids = players.loc[players["position"] == position, "player_id"]
        prob += pulp.lpSum(pick[pid] for pid in pos_ids) == count

    for team_id, group in players.groupby("team_id"):
        prob += pulp.lpSum(pick[pid] for pid in group["player_id"]) <= MAX_PER_CLUB

    prob += pulp.lpSum(pick[row.player_id] * row.price for row in players.itertuples()) <= budget

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=1.0)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]

    if status != "Optimal":
        return SquadResult(status=status, selected_ids=[], by_position={}, total_cost=0.0,
                            objective_value=0.0, remaining_budget=budget)

    selected = players[players["player_id"].map(lambda pid: pick[pid].value() == 1)]
    by_position = {pos: selected.loc[selected["position"] == pos, "player_id"].tolist() for pos in POSITION_COUNTS}
    total_cost = float(selected["price"].sum())

    transfers_made = 0
    hits = 0
    if transfer_aware:
        kept = len(set(selected["player_id"]) & set(current_squad_ids))
        transfers_made = SQUAD_SIZE - kept
        hits = int(round(hits_var.value()))

    return SquadResult(
        status=status,
        selected_ids=selected["player_id"].tolist(),
        by_position=by_position,
        total_cost=total_cost,
        objective_value=float(pulp.value(prob.objective)),
        remaining_budget=budget - total_cost,
        already_owned=[pid for pid in selected["player_id"] if pid in set(current_squad_ids)],
        transfers_made=transfers_made,
        hits=hits,
    )


def decide(state) -> SquadResult:
    """Bridge from HistoricalEnv's DecisionState to a MILP decision - what
    env.step() should be given, and what PPO will eventually call instead of."""
    return select_squad(
        state.candidates,
        current_squad_ids=state.squad_ids,
        bank=state.bank,
        free_transfers=state.free_transfers,
    )
