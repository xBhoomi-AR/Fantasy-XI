# Builds a fixed-size observation vector from squad state for the PPO agent.

from __future__ import annotations

import numpy as np

from ..environment.squad import BUDGET, FREE_TRANSFER_CAP

SQUAD_SIZE = 15
PLAYER_FEATURES = 4  # price, predicted_points, form_avg5, fixture_difficulty
POSITIONS = ["GK", "DEF", "MID", "FWD"]

# 60 player features + 3 team state + 4 position best + 4 chip availability + 3 fixture signals = 74
OBSERVATION_SIZE = SQUAD_SIZE * PLAYER_FEATURES + 3 + len(POSITIONS) + 4 + 3


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

    # Future fixture signals to allow neural PPO to learn DGW/BGW chip reservation
    max_dgw = float(squad_rows["fixture_count"].max()) if "fixture_count" in squad_rows.columns and len(squad_rows) else 1.0
    bgw_ratio = float((squad_rows["fixture_difficulty"].isna()).sum()) / 15.0 if "fixture_difficulty" in squad_rows.columns and len(squad_rows) else 0.0
    cap_ceil = float(squad_rows["predicted_points"].max()) / 10.0 if len(squad_rows) else 0.0
    fixture_signals = [max_dgw / 2.0, bgw_ratio, cap_ceil]

    obs = np.array(player_features + team_state + position_best + chip_flags + fixture_signals, dtype=np.float32)
    return np.nan_to_num(obs, nan=0.0)
