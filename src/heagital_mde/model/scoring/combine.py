from __future__ import annotations

import pandas as pd


def compute_final_score(market_score: pd.Series, readiness_score: pd.Series, alpha: float) -> pd.Series:
    a = max(0.0, min(1.0, float(alpha)))
    return a * market_score + (1.0 - a) * readiness_score
