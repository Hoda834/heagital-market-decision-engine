from __future__ import annotations

import pandas as pd

from heagital_mde.model.scoring.schema import (
    READINESS_WEIGHT_KEYS,
    ReadinessWeightConfig,
    normalise_readiness_weights,
)


def compute_readiness_score(
    df: pd.DataFrame,
    w: ReadinessWeightConfig,
    prefix: str = "n_",
) -> pd.Series:
    """Weighted sum of the normalised readiness signals.

    Readiness signals are optional inputs. Any signal whose column is absent
    is dropped and the remaining weights are renormalised, so an ICB file
    without ``digital_maturity`` still produces a readiness score built from
    the signals that are actually present rather than silently scoring 0.
    """
    weights = w.as_dict()
    available = {k: v for k, v in weights.items() if f"{prefix}{k}" in df.columns}

    if not available:
        raise ValueError(
            f"No readiness signals available. Expected at least one of "
            f"{[f'{prefix}{k}' for k in READINESS_WEIGHT_KEYS]}."
        )

    if sum(available.values()) <= 0.0:
        # Every present signal carries zero weight; fall back to an equal split
        # across what is available rather than dividing by zero.
        available = {k: 1.0 for k in available}

    effective = normalise_readiness_weights(
        ReadinessWeightConfig(**{k: available.get(k, 0.0) for k in READINESS_WEIGHT_KEYS})
    ).as_dict()

    score = pd.Series(0.0, index=df.index, dtype=float)
    for key in available:
        score = score + float(effective[key]) * df[f"{prefix}{key}"]
    return score


def effective_readiness_weights(
    df: pd.DataFrame,
    w: ReadinessWeightConfig,
    prefix: str = "n_",
) -> dict[str, float]:
    """The readiness weights actually applied, after dropping absent signals."""
    weights = w.as_dict()
    available = {k: v for k, v in weights.items() if f"{prefix}{k}" in df.columns}
    if not available:
        return {k: 0.0 for k in READINESS_WEIGHT_KEYS}
    if sum(available.values()) <= 0.0:
        available = {k: 1.0 for k in available}
    effective = normalise_readiness_weights(
        ReadinessWeightConfig(**{k: available.get(k, 0.0) for k in READINESS_WEIGHT_KEYS})
    ).as_dict()
    return {k: (effective[k] if k in available else 0.0) for k in READINESS_WEIGHT_KEYS}
