from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml

from heagital_mde.model.normalise import NormalisationConfig, normalise_columns


@dataclass(frozen=True)
class MarketWeightConfig:
    register: float
    prevalence: float
    treatment_gap: float
    warfarin_proxy: float


@dataclass(frozen=True)
class ReadinessWeightConfig:
    treatment_gap: float
    warfarin_proxy: float


def _load_yaml(path: str | Path) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_scoring_config(
    path: str | Path,
) -> Tuple[MarketWeightConfig, ReadinessWeightConfig, NormalisationConfig, float, int]:
    cfg = _load_yaml(path)

    w = cfg.get("weights", {})

    market_cfg = w.get("market", {})
    market_weights = MarketWeightConfig(
        register=float(market_cfg.get("register", 0.30)),
        prevalence=float(market_cfg.get("prevalence", 0.20)),
        treatment_gap=float(market_cfg.get("treatment_gap", 0.30)),
        warfarin_proxy=float(market_cfg.get("warfarin_proxy", 0.20)),
    )

    readiness_cfg = w.get("readiness", {})
    readiness_weights = ReadinessWeightConfig(
        treatment_gap=float(readiness_cfg.get("treatment_gap", 0.50)),
        warfarin_proxy=float(readiness_cfg.get("warfarin_proxy", 0.50)),
    )

    n = cfg.get("normalisation", {})
    norm_cfg = NormalisationConfig(
        method=str(n.get("method", "minmax")),
        clip=bool(n.get("clip", True)),
    )

    alpha = float(cfg.get("alpha", 0.60))

    cutoff_cfg = cfg.get("cutoff", {})
    top_n = int(cutoff_cfg.get("top_n", 15))

    return market_weights, readiness_weights, norm_cfg, alpha, top_n


def _normalise_weights_to_one_market(w: MarketWeightConfig) -> MarketWeightConfig:
    total = w.register + w.prevalence + w.treatment_gap + w.warfarin_proxy
    if total <= 0:
        raise ValueError("Market weight sum must be greater than 0.")
    return MarketWeightConfig(
        register=w.register / total,
        prevalence=w.prevalence / total,
        treatment_gap=w.treatment_gap / total,
        warfarin_proxy=w.warfarin_proxy / total,
    )


def _normalise_weights_to_one_readiness(w: ReadinessWeightConfig) -> ReadinessWeightConfig:
    total = w.treatment_gap + w.warfarin_proxy
    if total <= 0:
        raise ValueError("Readiness weight sum must be greater than 0.")
    return ReadinessWeightConfig(
        treatment_gap=w.treatment_gap / total,
        warfarin_proxy=w.warfarin_proxy / total,
    )


def _build_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = ["register", "prevalence", "treatment_gap", "warfarin_proxy"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for scoring: {missing}")

    out["register"] = pd.to_numeric(out["register"], errors="coerce")
    out["prevalence"] = pd.to_numeric(out["prevalence"], errors="coerce")
    out["treatment_gap"] = pd.to_numeric(out["treatment_gap"], errors="coerce")
    out["warfarin_proxy"] = pd.to_numeric(out["warfarin_proxy"], errors="coerce")

    out = out.dropna(subset=["register", "prevalence", "treatment_gap", "warfarin_proxy"]).reset_index(drop=True)
    return out


def score_and_rank(
    df: pd.DataFrame,
    scoring_config_path: str | Path,
    market_weights_override: MarketWeightConfig | None = None,
    readiness_weights_override: ReadinessWeightConfig | None = None,
    alpha_override: float | None = None,
) -> pd.DataFrame:
    market_w, readiness_w, norm_cfg, alpha, top_n = load_scoring_config(scoring_config_path)

    if market_weights_override is not None:
        market_w = market_weights_override
    if readiness_weights_override is not None:
        readiness_w = readiness_weights_override
    if alpha_override is not None:
        alpha = float(alpha_override)

    alpha = max(0.0, min(1.0, float(alpha)))

    market_w = _normalise_weights_to_one_market(market_w)
    readiness_w = _normalise_weights_to_one_readiness(readiness_w)

    base = _build_signals(df)

    base = normalise_columns(
        base,
        columns=["register", "prevalence", "treatment_gap", "warfarin_proxy"],
        cfg=norm_cfg,
        prefix="n_",
    )

    base["market_score"] = (
        market_w.register * base["n_register"]
        + market_w.prevalence * base["n_prevalence"]
        + market_w.treatment_gap * base["n_treatment_gap"]
        + market_w.warfarin_proxy * base["n_warfarin_proxy"]
    )

    base["readiness_score"] = (
        readiness_w.treatment_gap * base["n_treatment_gap"]
        + readiness_w.warfarin_proxy * base["n_warfarin_proxy"]
    )

    base["final_score"] = alpha * base["market_score"] + (1.0 - alpha) * base["readiness_score"]

    cols = ["icb_code", "icb_name"]
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
    out = out.sort_values(by="final_score", ascending=False, kind="mergesort").reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)

    out["recommended_cutoff_top_n"] = top_n
    out["recommended_included"] = out["rank"] <= top_n

    return out
