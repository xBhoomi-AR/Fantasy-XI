"""Tests for MILPCache and its integration with PPOEnv.

Run with:
    python rl_decision_layer/tests/test_milp_cache.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_decision_layer.optimization.milp_cache import MILPCache
from rl_decision_layer.optimization.squad_milp import SquadResult
from rl_decision_layer.ppo.env import PPOEnv

failures = []


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def test_milp_cache_store_and_lookup() -> None:
    cache = MILPCache()
    squad_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    action_key = (1, 2, 0, 0)
    gw = 10

    mock_result = SquadResult(
        status="Optimal",
        selected_ids=squad_ids,
        by_position={"GK": [1, 2], "DEF": [3, 4, 5, 6, 7], "MID": [8, 9, 10, 11, 12], "FWD": [13, 14, 15]},
        total_cost=100.0,
        objective_value=50.0,
        remaining_budget=0.0,
        already_owned=squad_ids,
        transfers_made=0,
        hits=0,
    )

    check(cache.lookup(gw, squad_ids, action_key) is None, "Cache miss before store")
    cache.store(gw, squad_ids, action_key, mock_result)
    res = cache.lookup(gw, squad_ids, action_key)
    check(res is not None and res.selected_ids == squad_ids, "Cache hit after store")

    # Order independence check
    shuffled_squad = list(reversed(squad_ids))
    res_shuffled = cache.lookup(gw, shuffled_squad, action_key)
    check(res_shuffled is not None and res_shuffled.selected_ids == squad_ids, "Order-independent squad lookup")


def test_milp_cache_save_and_load() -> None:
    cache = MILPCache()
    squad_ids = [1, 2, 3]
    action_key = (0, 1, 2, 3)
    gw = 5

    mock_result = SquadResult(
        status="Optimal",
        selected_ids=squad_ids,
        by_position={},
        total_cost=20.0,
        objective_value=15.0,
        remaining_budget=5.0,
    )
    cache.store(gw, squad_ids, action_key, mock_result)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cache.save(tmp_path)
        loaded_cache = MILPCache.load(tmp_path)
        check(len(loaded_cache) == 1, "Loaded cache has correct length")
        res = loaded_cache.lookup(gw, squad_ids, action_key)
        check(res is not None and res.objective_value == 15.0, "Loaded cache retrieves correct data")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_ppo_env_with_cache() -> None:
    cache = MILPCache()
    env = PPOEnv(start_gameweek=15, num_gameweeks=2, milp_cache=cache, use_cache=True)
    obs, _ = env.reset()

    action = [1, 2, 0, 0]
    check(len(cache) == 0, "Cache starts empty")

    obs, reward, terminated, truncated, info = env.step(action)
    check(len(cache) == 1, "Cache populated after 1 step")

    # Step again with same state and action (reset to same state)
    env.reset()
    obs2, reward2, _, _, info2 = env.step(action)
    check(len(cache) == 1, "Cache size remains 1 (hit instead of store)")
    check(reward == reward2, "Cached step returns identical reward")


def main() -> None:
    test_milp_cache_store_and_lookup()
    test_milp_cache_save_and_load()
    test_ppo_env_with_cache()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
