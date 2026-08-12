from __future__ import annotations

import pandas as pd


def rank_and_flag(
    df: pd.DataFrame,
    score_col: str,
    top_n: int,
    tie_breaker: str | None = "icb_code",
) -> pd.DataFrame:
    """Sort descending by ``score_col``, assign 1-based ranks and a cut-off flag.

    Ties are broken by ``tie_breaker`` (ascending) so that two ICBs with an
    identical score always land in the same order across runs. Without this the
    order depends on input row order, which makes rankings non-reproducible.
    """
    if score_col not in df.columns:
        raise ValueError(f"Cannot rank on missing column: {score_col}")
    if int(top_n) < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")

    out = df.copy()

    sort_cols = [score_col]
    ascending = [False]
    if tie_breaker and tie_breaker in out.columns:
        sort_cols.append(tie_breaker)
        ascending.append(True)

    out = out.sort_values(by=sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)

    if "rank" in out.columns:
        out = out.drop(columns=["rank"])
    out.insert(0, "rank", out.index + 1)

    out["recommended_cutoff_top_n"] = int(top_n)
    out["recommended_included"] = out["rank"] <= int(top_n)
    return out
