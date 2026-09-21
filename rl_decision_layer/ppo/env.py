# Gymnasium environment wrapping HistoricalEnv + MILP for the PPO agent.

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pathlib import Path

from ..environment.historical_env import HistoricalEnv
from ..environment.squad import build_starting_squad, validate_squad
from ..optimization.milp_cache import MILPCache
from ..optimization.scoring import calculate_reward, score_outcome
from ..optimization.squad_milp import select_squad
from ..optimization.starting_xi import pick_starting_xi
from ..predictions.interface import build_canonical_predictions
from .action import ACTION_SHAPE
from .action import action_to_milp_kwargs as _action_to_milp_kwargs
from .observation import OBSERVATION_SIZE, build_observation

INFEASIBLE_PENALTY = -100.0
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "models" / "milp_cache.pkl"


class PPOEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, start_gameweek, num_gameweeks, season="2025-26",
                 initial_squad=None, initial_bank=0.0, initial_free_transfers=1,
                 initial_available_chips: dict | None = None,
                 milp_cache: MILPCache | None = None, use_cache: bool = True,
                 use_heuristic_chips: bool = False, randomize_start_gw: bool = False):
        super().__init__()
        self.start_gameweek = start_gameweek
        self.num_gameweeks = num_gameweeks
        self.season = season
        self.initial_squad = initial_squad
        self.initial_bank = initial_bank
        self.initial_free_transfers = initial_free_transfers
        self.initial_available_chips = initial_available_chips
        self.use_heuristic_chips = use_heuristic_chips
        self.randomize_start_gw = randomize_start_gw

        if use_cache:
            self.milp_cache = milp_cache or MILPCache.load(DEFAULT_CACHE_PATH)
        else:
            self.milp_cache = None

        self.observation_space = spaces.Box(low=0.0, high=5.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(ACTION_SHAPE)

        self._env = HistoricalEnv(season=season)
        self._state = None
        self._steps_taken = 0
        self.available_chips = {"wildcard": True, "free_hit": True, "bench_boost": True, "triple_captain": True}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.available_chips = dict(self.initial_available_chips) if self.initial_available_chips else \
            {"wildcard": True, "free_hit": True, "bench_boost": True, "triple_captain": True}
        start_gw = int(np.random.randint(1, 29)) if self.randomize_start_gw else self.start_gameweek
        squad = self.initial_squad or build_starting_squad(
            build_canonical_predictions(season=self.season, gameweek=start_gw))
        self._state = self._env.reset(start_gw, squad, self.initial_bank, self.initial_free_transfers)
        self._steps_taken = 0
        return build_observation(self._state, self.available_chips), {}

    def step(self, action):
        if self._state is None:
            obs, _ = self.reset()
            return obs, 0.0, False, True, {}

        act_tuple = tuple(action)
        kwargs = _action_to_milp_kwargs(action, self._state)
        chip_name = kwargs.pop("chip_name", "none")

        # In heuristic mode, strategic rules govern chip timing to prevent premature dumping
        if self.use_heuristic_chips:
            from ..optimization.chip_strategy import get_recommended_chip
            current_squad_rows = self._state.candidates[self._state.candidates["player_id"].isin(self._state.squad_ids)]
            current_xi = pick_starting_xi(current_squad_rows) if len(current_squad_rows) else None
            chip_name = get_recommended_chip(
                gameweek=self._state.gameweek,
                squad_rows=current_squad_rows,
                xi_starting_ids=current_xi.starting_ids if current_xi else [],
                xi_bench_ids=current_xi.bench_ids if current_xi else [],
                captain_id=current_xi.captain_id if current_xi else None,
                available_chips=self.available_chips,
            )
            if chip_name in ("wildcard", "free_hit"):
                kwargs["free_hit_or_wildcard"] = True

        # Enforce single-use chip availability constraint
        if chip_name != "none":
            if self.available_chips.get(chip_name, False):
                self.available_chips[chip_name] = False
            else:
                # chip already used this season - reject it, and make sure no
                # stale chip state leaks into select_squad(): a rejected
                # wildcard/free_hit must not disable normal hit accounting
                chip_name = "none"
                kwargs["free_hit_or_wildcard"] = False

        decision = None
        if self.milp_cache is not None and not self.use_heuristic_chips:
            decision = self.milp_cache.lookup(self._state.gameweek, self._state.squad_ids, act_tuple)

        if decision is None:
            decision = select_squad(
                self._state.candidates,
                current_squad_ids=self._state.squad_ids,
                bank=self._state.bank,
                free_transfers=self._state.free_transfers,
                **kwargs,
            )
            if self.milp_cache is not None and decision.status == "Optimal" and not self.use_heuristic_chips:
                self.milp_cache.store(self._state.gameweek, self._state.squad_ids, act_tuple, decision)

        if decision.status != "Optimal":
            # Safe fallback: retry with standard legal squad parameters so episode never crashes
            decision = select_squad(
                self._state.candidates,
                current_squad_ids=self._state.squad_ids,
                bank=self._state.bank,
                free_transfers=self._state.free_transfers,
                hit_cost=4.0,
                chip_name="none",
            )
            if decision.status != "Optimal":
                from ..optimization.squad_milp import SquadResult
                decision = SquadResult(
                    status="FallbackKeep",
                    selected_ids=list(self._state.squad_ids),
                    by_position={},
                    total_cost=0.0,
                    objective_value=0.0,
                    remaining_budget=self._state.bank,
                    transfers_made=0,
                    hits=0,
                )

        squad_rows = self._state.candidates[self._state.candidates["player_id"].isin(decision.selected_ids)]
        xi = pick_starting_xi(squad_rows)

        # Apply chip modifiers in scoring:
        bench_boost = (chip_name == "bench_boost")
        triple_captain = (chip_name == "triple_captain")

        # Step environment:
        prev_squad = list(self._state.squad_ids)
        outcome, next_state = self._env.step(decision.selected_ids, new_bank=decision.remaining_budget)
        
        # If Free Hit was played, squad reverts back to prev_squad for next gameweek
        if chip_name == "free_hit" and next_state is not None:
            next_state.squad_ids = prev_squad
            self._env.squad_ids = prev_squad

        scored = score_outcome(outcome, xi, bench_boost=bench_boost, triple_captain=triple_captain)
        reward = calculate_reward(scored, decision.hits)

        self._steps_taken += 1
        truncated = (self._steps_taken >= self.num_gameweeks) or (next_state is None)

        info = {
            "transfers": decision.transfers_made,
            "hits": decision.hits,
            "gameweek": outcome.gameweek,
            "squad_ids": decision.selected_ids,
            "squad_size": len(decision.selected_ids),
            "legality_violations": validate_squad(self._state.candidates, decision.selected_ids),
            "starting_ids": xi.starting_ids,
            "bench_ids": xi.bench_ids,
            "captain_id": xi.captain_id,
            "vice_captain_id": xi.vice_captain_id,
            "bank": next_state.bank if next_state is not None else 0.0,
            "free_transfers": next_state.free_transfers if next_state is not None else 1,
            "chip_used": chip_name,
        }
        obs = build_observation(next_state, self.available_chips) if next_state is not None else build_observation(self._state, self.available_chips)
        self._state = next_state
        return obs, reward, False, truncated, info

