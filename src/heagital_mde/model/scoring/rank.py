from __future__ import annotations

import pandas as pd


def rank_and_flag(df: pd.DataFrame, score_col: str, top_n: int) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(by=score_col, ascending=False, kind="mergesort").reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)
    out["recommended_cutoff_top_n"] = int(top_n)
    out["recommended_included"] = out["rank"] <= int(top_n)
    return out
