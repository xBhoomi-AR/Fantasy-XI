"""Gym-like wrapper around HistoricalEnv + the MILP for a strategic PPO
agent. Not a gymnasium.Env subclass yet - no RL library is installed in
this project, this just matches the standard reset()/step() shape so
wrapping it later is trivial.
"""

from __future__ import annotations

from ..environment.historical_env import HistoricalEnv
from ..environment.squad import build_starting_squad
from ..optimization.scoring import calculate_reward, score_outcome
from ..optimization.squad_milp import select_squad
from ..optimization.starting_xi import pick_starting_xi
from ..predictions.interface import build_canonical_predictions
from .action import ACTION_SHAPE
from .action import action_to_milp_kwargs as _action_to_milp_kwargs
from .observation import OBSERVATION_SIZE, build_observation

INFEASIBLE_PENALTY = -100.0


class PPOEnv:
    action_shape = ACTION_SHAPE
    observation_size = OBSERVATION_SIZE

    def __init__(self, start_gameweek, num_gameweeks, season="2025-26",
                 initial_squad=None, initial_bank=0.0, initial_free_transfers=1):
        self.start_gameweek = start_gameweek
        self.num_gameweeks = num_gameweeks
        self.season = season
        self.initial_squad = initial_squad
        self.initial_bank = initial_bank
        self.initial_free_transfers = initial_free_transfers
        self._env = HistoricalEnv(season=season)
        self._state = None
        self._steps_taken = 0

    def reset(self):
        squad = self.initial_squad or build_starting_squad(
            build_canonical_predictions(season=self.season, gameweek=self.start_gameweek))
        self._state = self._env.reset(self.start_gameweek, squad, self.initial_bank, self.initial_free_transfers)
        self._steps_taken = 0
        return build_observation(self._state)

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
            return build_observation(self._state), INFEASIBLE_PENALTY, True, {"status": decision.status}

        squad_rows = self._state.candidates[self._state.candidates["player_id"].isin(decision.selected_ids)]
        xi = pick_starting_xi(squad_rows)

        outcome, next_state = self._env.step(decision.selected_ids, new_bank=decision.remaining_budget)
        scored = score_outcome(outcome, xi)
        reward = calculate_reward(scored, decision.hits)

        self._steps_taken += 1
        done = self._steps_taken >= self.num_gameweeks
        self._state = next_state

        info = {"transfers": decision.transfers_made, "hits": decision.hits, "gameweek": outcome.gameweek}
        return build_observation(next_state), reward, done, info
