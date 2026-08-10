from __future__ import annotations

import pandas as pd

from heagital_mde.model.scoring.schema import clip_01


def compute_final_score(
    market_score: pd.Series,
    readiness_score: pd.Series,
    alpha: float,
    friction_score: pd.Series | None = None,
    friction_weight: float = 0.0,
) -> pd.Series:
    """Blend the pillars into the decision score.

        final = alpha * market + (1 - alpha) * readiness - friction_weight * friction

    ``alpha`` trades market attractiveness against adoption readiness.
    ``friction_weight`` is a penalty, not part of the blend, so it can push a
    score below zero when procurement friction is high.
    """
    a = clip_01(alpha)
    score = a * market_score + (1.0 - a) * readiness_score

    fw = clip_01(friction_weight)
    if friction_score is not None and fw > 0.0:
        score = score - fw * friction_score

    return score
