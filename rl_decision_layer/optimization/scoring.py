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


def score_outcome(
    outcome: Outcome,
    xi: StartingXIResult,
    bench_boost: bool = False,
    triple_captain: bool = False,
    **kwargs,
) -> ScoredOutcome:
    starting_ids = list(xi.starting_ids)
    bench_ids = list(xi.bench_ids)
    
    # 1. Determine active captain (if captain played 0 mins, fallback to vice-captain)
    captain_id = xi.captain_id
    vice_id = xi.vice_captain_id
    
    active_captain = captain_id
    if captain_id is not None and outcome.actual_minutes.get(captain_id, 0.0) == 0:
        if vice_id is not None and outcome.actual_minutes.get(vice_id, 0.0) > 0:
            active_captain = vice_id

    # 2. Auto-substitutions: replace starting XI players with 0 mins from bench
    active_starters = []
    bench_available = list(bench_ids)
    
    for pid in starting_ids:
        if outcome.actual_minutes.get(pid, 0.0) > 0:
            active_starters.append(pid)
        else:
            # Substitute with first eligible bench player who played >0 mins
            sub_found = False
            for b_idx, b_pid in enumerate(bench_available):
                if outcome.actual_minutes.get(b_pid, 0.0) > 0:
                    active_starters.append(b_pid)
                    bench_available.pop(b_idx)
                    sub_found = True
                    break
            if not sub_found:
                active_starters.append(pid)  # No bench player available, keep 0-min starter

    starting_points = sum(outcome.actual_points.get(pid, 0.0) for pid in active_starters)
    captain_points = outcome.actual_points.get(active_captain, 0.0) if active_captain is not None else 0.0

    # Chip modifiers:
    # Triple Captain: captain bonus is 2x additional points (3x total)
    # Bench Boost: points from all bench players who actually played are also included
    captain_bonus = captain_points * (2.0 if triple_captain else 1.0)
    bench_boost_points = 0.0
    if bench_boost:
        # Include remaining bench players who played
        bench_boost_points = sum(outcome.actual_points.get(b_pid, 0.0) for b_pid in bench_ids if outcome.actual_minutes.get(b_pid, 0.0) > 0)

    total_points = starting_points + captain_bonus + bench_boost_points

    return ScoredOutcome(
        gameweek=outcome.gameweek,
        starting_points=starting_points,
        captain_points=captain_points,
        total_points=total_points,
    )


def calculate_reward(scored: ScoredOutcome, hits: int, hit_cost: float = DEFAULT_HIT_COST) -> float:
    """Reward = actual gameweek score minus points given up for transfer
    hits - what a real manager's gameweek score would show. hits comes from
    the MILP decision (SquadResult.hits), not from the scored outcome, since
    the outcome itself has no notion of transfers."""
    return scored.total_points - hit_cost * hits
