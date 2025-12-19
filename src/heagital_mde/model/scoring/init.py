from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from heagital_mde.model.normalise import normalise_columns
from heagital_mde.model.scoring.combine import compute_final_score
from heagital_mde.model.scoring.market import compute_market_score
from heagital_mde.model.scoring.rank import rank_and_flag
from heagital_mde.model.scoring.readiness import compute_readiness_score
from heagital_mde.model.scoring.schema import (
    MarketWeightConfig,
    ReadinessWeightConfig,
    ScoringConfig,
    load_scoring_config,
)


def _coerce_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _build_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Backwards compatibility: allow af_register as register
    if "register" not in out.columns and "af_register" in out.columns:
        out["register"] = out["af_register"]

    required = ["register", "prevalence", "treatment_gap", "warfarin_proxy"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for scoring: {missing}. "
            "Update load_icb.py to produce: register, prevalence, treatment_gap, warfarin_proxy."
        )

    out["register"] = _coerce_numeric(out, "register")
    out["prevalence"] = _coerce_numeric(out, "prevalence")
    out["treatment_gap"] = _coerce_numeric(out, "treatment_gap")
    out["warfarin_proxy"] = _coerce_numeric(out, "warfarin_proxy")

    out = out.dropna(subset=required).reset_index(drop=True)
    return out


def score_and_rank(
    df: pd.DataFrame,
    scoring_config_path: str | Path,
    market_weights_override: MarketWeightConfig | None = None,
    readiness_weights_override: ReadinessWeightConfig | None = None,
    alpha_override: float | None = None,
    top_n_override: int | None = None,
) -> pd.DataFrame:
    cfg: ScoringConfig = load_scoring_config(scoring_config_path)

    market_w = market_weights_override or cfg.market_weights
    readiness_w = readiness_weights_override or cfg.readiness_weights
    alpha = float(alpha_override) if alpha_override is not None else float(cfg.alpha)
    top_n = int(top_n_override) if top_n_override is not None else int(cfg.top_n)

    base = _build_signals(df)

    base = normalise_columns(
        base,
        columns=["register", "prevalence", "treatment_gap", "warfarin_proxy"],
        cfg=cfg.normalisation,
        prefix="n_",
    )

    base["market_score"] = compute_market_score(base, market_w)
    base["readiness_score"] = compute_readiness_score(base, readiness_w)
    base["final_score"] = compute_final_score(base["market_score"], base["readiness_score"], alpha)

    cols: list[str] = ["icb_code", "icb_name"]
    if "region" in base.columns:
        cols.append("region")

    cols += [
        "market_score",
        "readiness_score",
        "final_score",
        "n_register",
        "n_prevalence",
        "n_treatment_gap",
        "n_warfarin_proxy",
    ]

    out = base[cols].copy()
    out = rank_and_flag(out, score_col="final_score", top_n=top_n)

    return out
