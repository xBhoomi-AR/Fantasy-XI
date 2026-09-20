"""Picks a starting XI, captain and vice-captain from an already-selected
15-man squad. Separate from squad_milp.py on purpose - transfers and starting
XI are different decisions, and PPO will eventually want to influence
captaincy without touching the squad-selection logic.

FPL formation rules: 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD, 11 players total.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pulp

MIN_DEF, MAX_DEF = 3, 5
MIN_MID, MAX_MID = 2, 5
MIN_FWD, MAX_FWD = 1, 3
XI_SIZE = 11


@dataclass
class StartingXIResult:
    status: str
    starting_ids: list[int]
    bench_ids: list[int]
    captain_id: int | None
    vice_captain_id: int | None
    predicted_points: float


def pick_starting_xi(squad_rows: pd.DataFrame) -> StartingXIResult:
    """squad_rows must have exactly the 15 selected players, with
    player_id/position/predicted_points columns (same shape as a candidate
    pool filtered down to a squad)."""
    players = squad_rows.drop_duplicates("player_id").reset_index(drop=True)

    prob = pulp.LpProblem("starting_xi", pulp.LpMaximize)
    start = {row.player_id: pulp.LpVariable(f"start_{row.player_id}", cat="Binary") for row in players.itertuples()}

    prob += pulp.lpSum(start[row.player_id] * row.predicted_points for row in players.itertuples())
    prob += pulp.lpSum(start.values()) == XI_SIZE

    by_position = {pos: players.loc[players["position"] == pos, "player_id"] for pos in ("GK", "DEF", "MID", "FWD")}
    prob += pulp.lpSum(start[pid] for pid in by_position["GK"]) == 1
    prob += pulp.lpSum(start[pid] for pid in by_position["DEF"]) >= MIN_DEF
    prob += pulp.lpSum(start[pid] for pid in by_position["DEF"]) <= MAX_DEF
    prob += pulp.lpSum(start[pid] for pid in by_position["MID"]) >= MIN_MID
    prob += pulp.lpSum(start[pid] for pid in by_position["MID"]) <= MAX_MID
    prob += pulp.lpSum(start[pid] for pid in by_position["FWD"]) >= MIN_FWD
    prob += pulp.lpSum(start[pid] for pid in by_position["FWD"]) <= MAX_FWD

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]

    if status != "Optimal":
        # Safe heuristic fallback: select valid formation (1 GK, 3 DEF, 4 MID, 3 FWD) by predicted points
        gk = players[players["position"] == "GK"].sort_values("predicted_points", ascending=False).head(1)
        defn = players[players["position"] == "DEF"].sort_values("predicted_points", ascending=False).head(3)
        mid = players[players["position"] == "MID"].sort_values("predicted_points", ascending=False).head(4)
        fwd = players[players["position"] == "FWD"].sort_values("predicted_points", ascending=False).head(3)
        starting = pd.concat([gk, defn, mid, fwd]).drop_duplicates("player_id")
        if len(starting) < XI_SIZE:
            rem = players[~players["player_id"].isin(starting["player_id"])].sort_values("predicted_points", ascending=False)
            starting = pd.concat([starting, rem.head(XI_SIZE - len(starting))])
        bench = players[~players["player_id"].isin(starting["player_id"])]
        sorted_s = starting.sort_values("predicted_points", ascending=False)
        cap = int(sorted_s.iloc[0]["player_id"]) if len(sorted_s) > 0 else None
        vc = int(sorted_s.iloc[1]["player_id"]) if len(sorted_s) > 1 else cap
        return StartingXIResult(
            status="HeuristicFallback",
            starting_ids=starting["player_id"].tolist(),
            bench_ids=bench["player_id"].tolist(),
            captain_id=cap,
            vice_captain_id=vc,
            predicted_points=float(starting["predicted_points"].sum()),
        )

    starting = players[players["player_id"].map(lambda pid: start[pid].value() == 1)]
    bench = players[~players["player_id"].isin(starting["player_id"])]

    CAPTAIN_POS_WEIGHTS = {"FWD": 1.25, "MID": 1.20, "DEF": 0.85, "GK": 0.5}
    starting_copy = starting.copy()
    starting_copy["captain_score"] = starting_copy["predicted_points"] * starting_copy["position"].map(lambda p: CAPTAIN_POS_WEIGHTS.get(p, 1.0))

    by_points = starting_copy.sort_values("captain_score", ascending=False)
    captain_id = int(by_points.iloc[0]["player_id"])
    vice_captain_id = int(by_points.iloc[1]["player_id"]) if len(by_points) > 1 else captain_id

    return StartingXIResult(
        status=status,
        starting_ids=starting["player_id"].tolist(),
        bench_ids=bench["player_id"].tolist(),
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        predicted_points=float(starting["predicted_points"].sum()),
    )
