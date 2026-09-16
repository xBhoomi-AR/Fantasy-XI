"""15-player squad representation, validation, and a deterministic starting squad
for testing the environment. These are FPL's standard squad rules (not derived
from our data), so they're just hardcoded constants.
"""

from __future__ import annotations

import pandas as pd

POSITION_COUNTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
SQUAD_SIZE = 15
MAX_PER_CLUB = 3
BUDGET = 1000  # tenths-of-a-million, i.e. FPL's usual 100.0m cap, same units as `price`


def validate_squad(canonical_df: pd.DataFrame, squad_ids: list[int]) -> list[str]:
    """Checks squad size, position counts and the 3-per-club limit.
    Returns a list of violations - empty means the squad is legal.
    Doesn't check budget or transfer legality, that's for MILP.
    """
    rows = canonical_df[canonical_df["player_id"].isin(squad_ids)].drop_duplicates("player_id")
    errors = []

    if len(rows) != SQUAD_SIZE:
        errors.append(f"squad has {len(rows)} players, expected {SQUAD_SIZE}")

    counts = rows["position"].value_counts()
    for position, required in POSITION_COUNTS.items():
        actual = counts.get(position, 0)
        if actual != required:
            errors.append(f"{position}: {actual} players, expected {required}")

    club_counts = rows["team_id"].value_counts()
    for team_id, n in club_counts[club_counts > MAX_PER_CLUB].items():
        errors.append(f"team {team_id}: {n} players, max {MAX_PER_CLUB}")

    return errors


def squad_value(canonical_df: pd.DataFrame, squad_ids: list[int]) -> float:
    rows = canonical_df[canonical_df["player_id"].isin(squad_ids)].drop_duplicates("player_id")
    return float(rows["price"].sum())


def build_starting_squad(canonical_df: pd.DataFrame) -> list[int]:
    """Picks a legal, affordable 15-man squad from the cheapest players
    available per position, respecting the 3-per-club limit.

    This is NOT a real historical manager's squad - we don't have that data.
    It exists purely to give the environment a deterministic, legal starting
    point for testing. Cheapest-first naturally keeps it well under budget
    without needing any actual optimization.
    """
    squad = []
    club_counts: dict[int, int] = {}
    # a DGW player can have two rows for the same gameweek - dedupe first so
    # they aren't picked twice
    unique_players = canonical_df.drop_duplicates("player_id")

    for position, n in POSITION_COUNTS.items():
        candidates = unique_players[unique_players["position"] == position].sort_values(["price", "player_id"])
        picked = 0
        for _, row in candidates.iterrows():
            if picked == n:
                break
            team = row["team_id"]
            if club_counts.get(team, 0) >= MAX_PER_CLUB:
                continue
            squad.append(int(row["player_id"]))
            club_counts[team] = club_counts.get(team, 0) + 1
            picked += 1
        if picked < n:
            raise ValueError(f"could not fill {position} ({picked}/{n}) within the club limit")

    return squad
