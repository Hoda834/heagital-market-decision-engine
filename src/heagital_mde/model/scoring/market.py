from __future__ import annotations

import pandas as pd

from heagital_mde.model.scoring.schema import MarketWeightConfig


def compute_market_score(df: pd.DataFrame, w: MarketWeightConfig) -> pd.Series:
    return (
        w.register * df["n_register"]
        + w.prevalence * df["n_prevalence"]
        + w.treatment_gap * df["n_treatment_gap"]
        + w.warfarin_proxy * df["n_warfarin_proxy"]
    )
