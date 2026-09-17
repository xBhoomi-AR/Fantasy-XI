import functools
from collections.abc import Sequence

import pandas as pd

from ..predictions.interface import build_canonical_predictions
from .candidate_pool import (
    DEFAULT_FORM_TOP_K,
    DEFAULT_FORM_WINDOW,
    DEFAULT_PREDICTION_TOP_K,
    build_candidate_pool,
)


@functools.lru_cache(maxsize=128)
def _get_canonical_predictions_cached(season: str, model: str, gameweek: int) -> pd.DataFrame:
    return build_canonical_predictions(season=season, model=model, gameweek=gameweek)


def get_candidates(
    target_gameweek: int,
    current_squad_ids: Sequence[int] = (),
    season: str = "2025-26",
    model: str = "bilstm",
    prediction_top_k: int | dict[str, int] = DEFAULT_PREDICTION_TOP_K,
    form_top_k: int | dict[str, int] = DEFAULT_FORM_TOP_K,
    form_window: str = DEFAULT_FORM_WINDOW,
) -> pd.DataFrame:
    """Build the candidate pool for target_gameweek.

    Loads cached canonical predictions for the chosen model ('bilstm' or 'xgboost')
    and builds the candidate pool.
    """
    predictions = _get_canonical_predictions_cached(season, model, target_gameweek)
    
    # Convert squad_ids to tuple for hashing if passed as list
    squad_tuple = tuple(current_squad_ids) if not isinstance(current_squad_ids, tuple) else current_squad_ids
    
    return build_candidate_pool(
        predictions,
        target_gameweek=target_gameweek,
        current_squad_ids=squad_tuple,
        prediction_top_k=prediction_top_k,
        form_top_k=form_top_k,
        form_window=form_window,
    )
