"""Single entry point for candidate generation: target gameweek + squad in,
candidate pool out. This is what the historical environment will call once
it exists - it shouldn't need to know about interface.py or ranking.py directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ..predictions.interface import build_canonical_predictions
from .candidate_pool import (
    DEFAULT_FORM_TOP_K,
    DEFAULT_FORM_WINDOW,
    DEFAULT_PREDICTION_TOP_K,
    build_candidate_pool,
)


def get_candidates(
    target_gameweek: int,
    current_squad_ids: Sequence[int] = (),
    season: str = "2025-26",
    prediction_top_k: int | dict[str, int] = DEFAULT_PREDICTION_TOP_K,
    form_top_k: int | dict[str, int] = DEFAULT_FORM_TOP_K,
    form_window: str = DEFAULT_FORM_WINDOW,
) -> pd.DataFrame:
    """Build the candidate pool for target_gameweek from scratch.

    Only loads data for target_gameweek, so it's inherently using
    information available before that gameweek - see predictions/interface.py
    for how the form/prediction columns are kept leakage-safe upstream.
    """
    predictions = build_canonical_predictions(season=season, gameweek=target_gameweek)
    return build_candidate_pool(
        predictions,
        target_gameweek=target_gameweek,
        current_squad_ids=current_squad_ids,
        prediction_top_k=prediction_top_k,
        form_top_k=form_top_k,
        form_window=form_window,
    )
