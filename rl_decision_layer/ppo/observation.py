"""Fixed-size observation built from a DecisionState.

PPO never sees individual candidates here - its action is a strategic knob
turn (see action.py), not a player pick, so it only needs its own squad's
state plus a small per-position summary of what else is available. The
squad is always exactly 15 players, which gives a naturally fixed-size
block; the candidate pool's size varies week to week, so it's summarized
into 4 numbers (best available predicted_points per position) instead of
being flattened directly.
"""

from __future__ import annotations

import numpy as np

from ..environment.squad import BUDGET, FREE_TRANSFER_CAP

SQUAD_SIZE = 15
PLAYER_FEATURES = 4  # price, predicted_points, form_avg5, fixture_difficulty
POSITIONS = ["GK", "DEF", "MID", "FWD"]

# 60 player features + 3 team state + 4 position best + 4 chip availability = 71
OBSERVATION_SIZE = SQUAD_SIZE * PLAYER_FEATURES + 3 + len(POSITIONS) + 4


def build_observation(state, available_chips: dict[str, bool] | None = None) -> np.ndarray:
    squad_rows = state.candidates[state.candidates["player_id"].isin(state.squad_ids)]
    squad_rows = squad_rows.drop_duplicates("player_id").sort_values(["position", "player_id"])

    player_features = []
    for _, row in squad_rows.iterrows():
        player_features += [row["price"] / 100.0, row["predicted_points"] / 10.0,
                             row["form_avg5"] / 10.0, row["fixture_difficulty"] / 5.0]
    # a squad member with no fixture this gameweek won't be in candidates at
    # all - pad with zeros rather than shrink the observation
    player_features += [0.0] * (SQUAD_SIZE * PLAYER_FEATURES - len(player_features))

    team_state = [state.bank / BUDGET, state.free_transfers / FREE_TRANSFER_CAP, state.gameweek / 38.0]

    non_squad = state.candidates[~state.candidates["player_id"].isin(state.squad_ids)]
    position_best = []
    for pos in POSITIONS:
        pos_rows = non_squad[non_squad["position"] == pos]
        best = pos_rows["predicted_points"].max() if len(pos_rows) else 0.0
        position_best.append(best / 10.0)

    # Chip availability flags (1.0 if available, 0.0 if already used)
    chips = available_chips or {"wildcard": True, "free_hit": True, "bench_boost": True, "triple_captain": True}
    chip_flags = [
        1.0 if chips.get("wildcard", True) else 0.0,
        1.0 if chips.get("free_hit", True) else 0.0,
        1.0 if chips.get("bench_boost", True) else 0.0,
        1.0 if chips.get("triple_captain", True) else 0.0,
    ]

    obs = np.array(player_features + team_state + position_best + chip_flags, dtype=np.float32)
    return np.nan_to_num(obs, nan=0.0)
