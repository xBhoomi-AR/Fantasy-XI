"""Scores a decision's actual outcome and turns it into a reward.

Only the starting XI counts, and the captain's points count twice - same as
real FPL. No auto-subs (a starter who blanks isn't replaced by a bench
player) and no vice-captain fallback (vice only matters in real FPL if the
captain gets 0 minutes, which we don't track) - both are known
simplifications, not implemented. No chip logic either.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..environment.historical_env import Outcome
from .starting_xi import StartingXIResult

DEFAULT_HIT_COST = 4.0


@dataclass
class ScoredOutcome:
    gameweek: int
    starting_points: float
    captain_points: float
    total_points: float


def score_outcome(outcome: Outcome, xi: StartingXIResult) -> ScoredOutcome:
    starting_points = sum(outcome.actual_points.get(pid, 0.0) for pid in xi.starting_ids)
    captain_points = outcome.actual_points.get(xi.captain_id, 0.0) if xi.captain_id is not None else 0.0
    return ScoredOutcome(
        gameweek=outcome.gameweek,
        starting_points=starting_points,
        captain_points=captain_points,
        total_points=starting_points + captain_points,  # captain counted twice overall
    )


def calculate_reward(scored: ScoredOutcome, hits: int, hit_cost: float = DEFAULT_HIT_COST) -> float:
    """Reward = actual gameweek score minus points given up for transfer
    hits - what a real manager's gameweek score would show. hits comes from
    the MILP decision (SquadResult.hits), not from the scored outcome, since
    the outcome itself has no notion of transfers."""
    return scored.total_points - hit_cost * hits
