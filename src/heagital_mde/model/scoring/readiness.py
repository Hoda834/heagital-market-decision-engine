from __future__ import annotations

import pandas as pd

from heagital_mde.model.scoring.schema import ReadinessWeightConfig


def compute_readiness_score(df: pd.DataFrame, w: ReadinessWeightConfig) -> pd.Series:
    return w.treatment_gap * df["n_treatment_gap"] + w.warfarin_proxy * df["n_warfarin_proxy"]
