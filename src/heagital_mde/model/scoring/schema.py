from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from heagital_mde.model.normalise import NormalisationConfig


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


@dataclass(frozen=True)
class ScoringConfig:
    market_weights: MarketWeightConfig
    readiness_weights: ReadinessWeightConfig
    alpha: float
    normalisation: NormalisationConfig
    top_n: int


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Scoring config must be a YAML mapping.")
    return data


def _clip_01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _normalise_market_weights(w: MarketWeightConfig) -> MarketWeightConfig:
    total = w.register + w.prevalence + w.treatment_gap + w.warfarin_proxy
    if total <= 0:
        raise ValueError("Market weight sum must be greater than 0.")
    return MarketWeightConfig(
        register=w.register / total,
        prevalence=w.prevalence / total,
        treatment_gap=w.treatment_gap / total,
        warfarin_proxy=w.warfarin_proxy / total,
    )


def _normalise_readiness_weights(w: ReadinessWeightConfig) -> ReadinessWeightConfig:
    total = w.treatment_gap + w.warfarin_proxy
    if total <= 0:
        raise ValueError("Readiness weight sum must be greater than 0.")
    return ReadinessWeightConfig(
        treatment_gap=w.treatment_gap / total,
        warfarin_proxy=w.warfarin_proxy / total,
    )


def load_scoring_config(path: str | Path) -> ScoringConfig:
    cfg = _load_yaml(path)

    n = cfg.get("normalisation") or cfg.get("normalization") or cfg.get("normalisation") or {}
    normalisation = NormalisationConfig(
        method=str(n.get("method", "minmax")),
        clip=bool(n.get("clip", True)),
    )

    cutoff_cfg = cfg.get("cutoff", {}) or {}
    top_n = int(cutoff_cfg.get("top_n", 15))

    alpha = _clip_01(float(cfg.get("alpha", 0.60)))

    weights = cfg.get("weights", {}) or {}

    # New format expected:
    # weights:
    #   market:
    #     register: ...
    #     prevalence: ...
    #     treatment_gap: ...
    #     warfarin_proxy: ...
    #   readiness:
    #     treatment_gap: ...
    #     warfarin_proxy: ...
    market_cfg = weights.get("market", {}) if isinstance(weights, dict) else {}
    readiness_cfg = weights.get("readiness", {}) if isinstance(weights, dict) else {}

    if isinstance(market_cfg, dict) and market_cfg:
        market_weights = MarketWeightConfig(
            register=float(market_cfg.get("register", 0.30)),
            prevalence=float(market_cfg.get("prevalence", 0.20)),
            treatment_gap=float(market_cfg.get("treatment_gap", 0.30)),
            warfarin_proxy=float(market_cfg.get("warfarin_proxy", 0.20)),
        )
        readiness_weights = ReadinessWeightConfig(
            treatment_gap=float(readiness_cfg.get("treatment_gap", 0.50)),
            warfarin_proxy=float(readiness_cfg.get("warfarin_proxy", 0.50)),
        )
        market_weights = _normalise_market_weights(market_weights)
        readiness_weights = _normalise_readiness_weights(readiness_weights)
        return ScoringConfig(
            market_weights=market_weights,
            readiness_weights=readiness_weights,
            alpha=alpha,
            normalisation=normalisation,
            top_n=top_n,
        )

    # Backwards compatible format (your old file):
    # weights:
    #   clinical_risk: 0.45
    #   adoption_readiness: 0.35
    #   procurement_friction: 0.20
    # We ignore procurement_friction and treat this as readiness-only scoring.
    if isinstance(weights, dict) and ("clinical_risk" in weights or "adoption_readiness" in weights):
        clinical = float(weights.get("clinical_risk", 0.55))
        adoption = float(weights.get("adoption_readiness", 0.45))
        readiness_weights = _normalise_readiness_weights(
            ReadinessWeightConfig(treatment_gap=clinical, warfarin_proxy=adoption)
        )
        market_weights = _normalise_market_weights(
            MarketWeightConfig(register=0.30, prevalence=0.20, treatment_gap=0.30, warfarin_proxy=0.20)
        )
        return ScoringConfig(
            market_weights=market_weights,
            readiness_weights=readiness_weights,
            alpha=0.0,
            normalisation=normalisation,
            top_n=top_n,
        )

    market_weights = _normalise_market_weights(
        MarketWeightConfig(register=0.30, prevalence=0.20, treatment_gap=0.30, warfarin_proxy=0.20)
    )
    readiness_weights = _normalise_readiness_weights(ReadinessWeightConfig(treatment_gap=0.50, warfarin_proxy=0.50))
    return ScoringConfig(
        market_weights=market_weights,
        readiness_weights=readiness_weights,
        alpha=alpha,
        normalisation=normalisation,
        top_n=top_n,
    )
