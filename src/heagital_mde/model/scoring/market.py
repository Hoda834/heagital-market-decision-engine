from __future__ import annotations

import pandas as pd

from heagital_mde.model.scoring.schema import MARKET_WEIGHT_KEYS, MarketWeightConfig


def compute_market_score(df: pd.DataFrame, w: MarketWeightConfig, prefix: str = "n_") -> pd.Series:
    """Weighted sum of the normalised market signals.

    Expects ``w`` to already be renormalised to sum to 1, which
    ``load_scoring_config`` guarantees.
    """
    weights = w.as_dict()
    missing = [f"{prefix}{k}" for k in MARKET_WEIGHT_KEYS if f"{prefix}{k}" not in df.columns]
    if missing:
        raise ValueError(f"Missing normalised market columns: {missing}")

    score = pd.Series(0.0, index=df.index, dtype=float)
    for key in MARKET_WEIGHT_KEYS:
        score = score + float(weights[key]) * df[f"{prefix}{key}"]
    return score
