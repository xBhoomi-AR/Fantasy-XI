"""Steps a squad through historical gameweeks.

Two separate pieces of state, deliberately kept apart:
  - DecisionState: everything available before gameweek G is played. This is
    what a future MILP/PPO decision would be based on - it never contains G's
    actual points.
  - Outcome: what actually happened in G, only produced by step() after a
    decision has been made.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..candidates.pipeline import get_candidates
from ..predictions.interface import DATA_RAW_DIR
from .squad import FREE_TRANSFER_CAP, SQUAD_SIZE


@dataclass
class DecisionState:
    gameweek: int
    squad_ids: list[int]
    bank: float
    free_transfers: int
    candidates: pd.DataFrame


@dataclass
class Outcome:
    gameweek: int
    squad_ids: list[int]
    actual_points: dict[int, float]
    squad_total_points: float = field(init=False)

    def __post_init__(self):
        self.squad_total_points = sum(self.actual_points.values())


def _load_actual_points(season: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW_DIR / "player_match_stats.csv", engine="python")
    df = df[df["season"].astype(str) == season].copy()
    df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce").fillna(0)
    # a DGW player has two rows for the same gameweek - sum them, same as
    # real FPL scoring and the same treatment candidate_pool.py applies to predictions
    return df.groupby(["player_id", "gameweek"], as_index=False)["total_points"].sum()


class HistoricalEnv:
    """Walks a squad through real historical gameweeks, one at a time."""

    def __init__(self, season: str = "2025-26"):
        self.season = season
        self._actuals = _load_actual_points(season)
        self.gameweek: int | None = None
        self.squad_ids: list[int] = []
        self.bank = 0.0
        self.free_transfers = 1

    def reset(self, start_gameweek: int, squad_ids: list[int], bank: float = 0.0, free_transfers: int = 1) -> DecisionState:
        self.gameweek = start_gameweek
        self.squad_ids = list(squad_ids)
        self.bank = bank
        self.free_transfers = free_transfers
        return self._decision_state()

    def _decision_state(self) -> DecisionState:
        candidates = get_candidates(self.gameweek, current_squad_ids=self.squad_ids)
        return DecisionState(
            gameweek=self.gameweek,
            squad_ids=list(self.squad_ids),
            bank=self.bank,
            free_transfers=self.free_transfers,
            candidates=candidates,
        )

    def step(self, new_squad_ids: list[int] | None = None, new_bank: float | None = None) -> tuple[Outcome, DecisionState]:
        """Applies a decision (or leaves the squad unchanged if none is
        given), scores the current gameweek against actual results, then
        advances to the next gameweek. `new_squad_ids`/`new_bank` are what
        MILP's decide() produces - pass decision.selected_ids and
        decision.remaining_budget here to carry the budget forward.
        """
        if self.gameweek is None:
            raise RuntimeError("call reset() first")

        transfers_made = 0
        if new_squad_ids is not None:
            kept = len(set(self.squad_ids) & set(new_squad_ids))
            transfers_made = SQUAD_SIZE - kept
            self.squad_ids = list(new_squad_ids)

        # unused free transfers roll over, capped at 5 - same rule select_squad()
        # uses to decide hits, kept in sync here
        used_free = min(transfers_made, self.free_transfers)
        self.free_transfers = min(FREE_TRANSFER_CAP, self.free_transfers - used_free + 1)

        if new_bank is not None:
            self.bank = new_bank

        gw_actuals = self._actuals[self._actuals["gameweek"] == self.gameweek].set_index("player_id")["total_points"]
        points = {pid: float(gw_actuals.get(pid, 0.0)) for pid in self.squad_ids}
        outcome = Outcome(gameweek=self.gameweek, squad_ids=list(self.squad_ids), actual_points=points)

        self.gameweek += 1
        return outcome, self._decision_state()
