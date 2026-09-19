"""MILP decision cache for accelerating RL environment steps.

Caches (gameweek, squad_hash, action_key) -> SquadResult/dict representation.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from .squad_milp import SquadResult


def compute_squad_hash(squad_ids: list[int] | tuple[int, ...]) -> int:
    """Order-independent hash for squad IDs set."""
    return hash(frozenset(squad_ids))


class MILPCache:
    def __init__(self, cache_dict: dict[tuple[int, int, tuple[Any, ...]], dict[str, Any]] | None = None):
        # key: (gameweek, squad_hash, action_tuple)
        # value: dict representation of SquadResult
        self._cache = cache_dict if cache_dict is not None else {}

    def lookup(
        self,
        gameweek: int,
        squad_ids: list[int] | tuple[int, ...],
        action_key: tuple[Any, ...],
    ) -> SquadResult | None:
        key = (gameweek, compute_squad_hash(squad_ids), action_key)
        val = self._cache.get(key)
        if val is None:
            return None
        return SquadResult(**val)

    def store(
        self,
        gameweek: int,
        squad_ids: list[int] | tuple[int, ...],
        action_key: tuple[Any, ...],
        result: SquadResult,
    ) -> None:
        key = (gameweek, compute_squad_hash(squad_ids), action_key)
        self._cache[key] = {
            "status": result.status,
            "selected_ids": list(result.selected_ids),
            "by_position": result.by_position,
            "total_cost": result.total_cost,
            "objective_value": result.objective_value,
            "remaining_budget": result.remaining_budget,
            "already_owned": list(result.already_owned),
            "transfers_made": result.transfers_made,
            "hits": result.hits,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> MILPCache:
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, "rb") as f:
            cache_dict = pickle.load(f)
        return cls(cache_dict=cache_dict)

    def __len__(self) -> int:
        return len(self._cache)
