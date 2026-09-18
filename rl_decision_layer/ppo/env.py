"""Gymnasium environment wrapping HistoricalEnv + the MILP for a strategic
PPO agent.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..environment.historical_env import HistoricalEnv
from ..environment.squad import build_starting_squad, validate_squad
from ..optimization.scoring import calculate_reward, score_outcome
from ..optimization.squad_milp import select_squad
from ..optimization.starting_xi import pick_starting_xi
from ..predictions.interface import build_canonical_predictions
from .action import ACTION_SHAPE
from .action import action_to_milp_kwargs as _action_to_milp_kwargs
from .observation import OBSERVATION_SIZE, build_observation

INFEASIBLE_PENALTY = -100.0


class PPOEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, start_gameweek, num_gameweeks, season="2025-26",
                 initial_squad=None, initial_bank=0.0, initial_free_transfers=1):
        super().__init__()
        self.start_gameweek = start_gameweek
        self.num_gameweeks = num_gameweeks
        self.season = season
        self.initial_squad = initial_squad
        self.initial_bank = initial_bank
        self.initial_free_transfers = initial_free_transfers

        self.observation_space = spaces.Box(low=0.0, high=5.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(ACTION_SHAPE)

        self._env = HistoricalEnv(season=season)
        self._state = None
        self._steps_taken = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        squad = self.initial_squad or build_starting_squad(
            build_canonical_predictions(season=self.season, gameweek=self.start_gameweek))
        self._state = self._env.reset(self.start_gameweek, squad, self.initial_bank, self.initial_free_transfers)
        self._steps_taken = 0
        return build_observation(self._state), {}

    def step(self, action):
        kwargs = _action_to_milp_kwargs(action, self._state)
        decision = select_squad(
            self._state.candidates,
            current_squad_ids=self._state.squad_ids,
            bank=self._state.bank,
            free_transfers=self._state.free_transfers,
            **kwargs,
        )

        if decision.status != "Optimal":
            # no legal squad at this budget - end the episode rather than fake one
            obs = build_observation(self._state)
            info = {"status": decision.status, "gameweek": self._state.gameweek}
            return obs, INFEASIBLE_PENALTY, True, False, info

        squad_rows = self._state.candidates[self._state.candidates["player_id"].isin(decision.selected_ids)]
        xi = pick_starting_xi(squad_rows)

        outcome, next_state = self._env.step(decision.selected_ids, new_bank=decision.remaining_budget)
        scored = score_outcome(outcome, xi)
        reward = calculate_reward(scored, decision.hits)

        self._steps_taken += 1
        truncated = self._steps_taken >= self.num_gameweeks

        # additive evaluation detail only - doesn't change obs/reward/done semantics.
        # validated against self._state (this gameweek's own candidate pool) before
        # it gets reassigned below - otherwise this would check the squad against
        # NEXT gameweek's candidates instead of the ones it was actually picked from.
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
            "bank": next_state.bank,
            "free_transfers": next_state.free_transfers,
        }
        self._state = next_state
        return build_observation(next_state), reward, False, truncated, info
