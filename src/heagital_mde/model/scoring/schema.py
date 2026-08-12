from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from heagital_mde.model.normalise import NormalisationConfig

#: Weight blocks that must sum to a positive number and are renormalised to 1.
MARKET_WEIGHT_KEYS = ("register", "prevalence", "treatment_gap", "warfarin_proxy")
READINESS_WEIGHT_KEYS = ("treatment_gap", "warfarin_proxy", "digital_maturity")

#: Legacy weight names from the pre-0.2 config format. Present only so that an
#: old config fails loudly with a migration message instead of being ignored.
LEGACY_WEIGHT_KEYS = ("clinical_risk", "adoption_readiness", "procurement_friction")


@dataclass(frozen=True)
class MarketWeightConfig:
    register: float = 0.30
    prevalence: float = 0.20
    treatment_gap: float = 0.30
    warfarin_proxy: float = 0.20

    def as_dict(self) -> Dict[str, float]:
        return {k: float(getattr(self, k)) for k in MARKET_WEIGHT_KEYS}


@dataclass(frozen=True)
class ReadinessWeightConfig:
    treatment_gap: float = 0.40
    warfarin_proxy: float = 0.30
    digital_maturity: float = 0.30

    def as_dict(self) -> Dict[str, float]:
        return {k: float(getattr(self, k)) for k in READINESS_WEIGHT_KEYS}


@dataclass(frozen=True)
class ScoringConfig:
    market_weights: MarketWeightConfig
    readiness_weights: ReadinessWeightConfig
    alpha: float
    friction_weight: float
    normalisation: NormalisationConfig
    top_n: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "market_weights": self.market_weights.as_dict(),
            "readiness_weights": self.readiness_weights.as_dict(),
            "alpha": float(self.alpha),
            "friction_weight": float(self.friction_weight),
            "normalisation": {
                "method": self.normalisation.method,
                "clip": bool(self.normalisation.clip),
                "winsorize": float(self.normalisation.winsorize),
                "constant_fill": float(self.normalisation.constant_fill),
            },
            "top_n": int(self.top_n),
        }


def clip_01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _renormalise(weights: Mapping[str, float], block: str) -> Dict[str, float]:
    negative = {k: v for k, v in weights.items() if float(v) < 0.0}
    if negative:
        raise ValueError(f"{block} weights must be non-negative, got {negative}.")
    total = float(sum(float(v) for v in weights.values()))
    if total <= 0.0:
        raise ValueError(f"{block} weight sum must be greater than 0, got {total}.")
    return {k: float(v) / total for k, v in weights.items()}


def normalise_market_weights(w: MarketWeightConfig) -> MarketWeightConfig:
    return MarketWeightConfig(**_renormalise(w.as_dict(), "Market"))


def normalise_readiness_weights(w: ReadinessWeightConfig) -> ReadinessWeightConfig:
    return ReadinessWeightConfig(**_renormalise(w.as_dict(), "Readiness"))


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Scoring config must be a YAML mapping, got {type(data).__name__}: {p}")
    return data


def _read_block(raw: Any, keys: tuple[str, ...], block: str, path: Path) -> Dict[str, float]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"weights.{block} must be a mapping in {path}, got {type(raw).__name__}.")

    unknown = sorted(set(raw) - set(keys))
    if unknown:
        raise ValueError(
            f"Unknown key(s) under weights.{block} in {path}: {unknown}. Expected any of {list(keys)}."
        )

    values: Dict[str, float] = {}
    for key in keys:
        if key not in raw:
            continue
        try:
            values[key] = float(raw[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"weights.{block}.{key} must be a number in {path}, got {raw[key]!r}.") from exc
    return values


def load_scoring_config(path: str | Path) -> ScoringConfig:
    """Read a scoring config.

    Every recognised key is applied and every unrecognised key is an error.
    A config that cannot be understood raises rather than falling back to
    built-in defaults, so a mis-typed weight can never silently change a
    published ranking.
    """
    p = Path(path)
    cfg = _load_yaml(p)

    known_top_level = {"weights", "alpha", "friction_weight", "normalisation", "cutoff"}
    unknown_top_level = sorted(set(cfg) - known_top_level)
    if unknown_top_level:
        raise ValueError(
            f"Unknown top-level key(s) in {p}: {unknown_top_level}. Expected any of {sorted(known_top_level)}."
        )

    weights = cfg.get("weights") or {}
    if not isinstance(weights, dict):
        raise ValueError(f"'weights' must be a mapping in {p}, got {type(weights).__name__}.")

    legacy_present = sorted(k for k in LEGACY_WEIGHT_KEYS if k in weights)
    if legacy_present:
        raise ValueError(
            f"{p} uses the pre-0.2 weight format ({legacy_present}), which this engine no longer "
            "understands. Migrate to:\n"
            "  weights:\n"
            "    market: {register, prevalence, treatment_gap, warfarin_proxy}\n"
            "    readiness: {treatment_gap, warfarin_proxy, digital_maturity}\n"
            "  alpha: <market vs readiness blend, 0-1>\n"
            "  friction_weight: <penalty applied to procurement_friction, 0-1>"
        )

    unknown_blocks = sorted(set(weights) - {"market", "readiness"})
    if unknown_blocks:
        raise ValueError(
            f"Unknown block(s) under 'weights' in {p}: {unknown_blocks}. Expected 'market' and/or 'readiness'."
        )

    market_weights = normalise_market_weights(
        MarketWeightConfig(**_read_block(weights.get("market"), MARKET_WEIGHT_KEYS, "market", p))
    )
    readiness_weights = normalise_readiness_weights(
        ReadinessWeightConfig(**_read_block(weights.get("readiness"), READINESS_WEIGHT_KEYS, "readiness", p))
    )

    norm_raw = cfg.get("normalisation") or {}
    if not isinstance(norm_raw, dict):
        raise ValueError(f"'normalisation' must be a mapping in {p}, got {type(norm_raw).__name__}.")
    unknown_norm = sorted(set(norm_raw) - {"method", "clip", "winsorize", "constant_fill"})
    if unknown_norm:
        raise ValueError(f"Unknown key(s) under 'normalisation' in {p}: {unknown_norm}.")
    normalisation = NormalisationConfig(
        method=str(norm_raw.get("method", "minmax")),
        clip=bool(norm_raw.get("clip", True)),
        winsorize=float(norm_raw.get("winsorize", 0.0)),
        constant_fill=float(norm_raw.get("constant_fill", 0.5)),
    )

    cutoff_raw = cfg.get("cutoff") or {}
    if not isinstance(cutoff_raw, dict):
        raise ValueError(f"'cutoff' must be a mapping in {p}, got {type(cutoff_raw).__name__}.")
    unknown_cutoff = sorted(set(cutoff_raw) - {"top_n"})
    if unknown_cutoff:
        raise ValueError(f"Unknown key(s) under 'cutoff' in {p}: {unknown_cutoff}.")
    top_n = int(cutoff_raw.get("top_n", 15))
    if top_n < 1:
        raise ValueError(f"cutoff.top_n must be >= 1 in {p}, got {top_n}.")

    alpha = float(cfg.get("alpha", 0.60))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1 in {p}, got {alpha}.")

    friction_weight = float(cfg.get("friction_weight", 0.0))
    if not 0.0 <= friction_weight <= 1.0:
        raise ValueError(f"friction_weight must be between 0 and 1 in {p}, got {friction_weight}.")

    return ScoringConfig(
        market_weights=market_weights,
        readiness_weights=readiness_weights,
        alpha=alpha,
        friction_weight=friction_weight,
        normalisation=normalisation,
        top_n=top_n,
    )


def with_overrides(
    cfg: ScoringConfig,
    market_weights: MarketWeightConfig | None = None,
    readiness_weights: ReadinessWeightConfig | None = None,
    alpha: float | None = None,
    friction_weight: float | None = None,
    top_n: int | None = None,
) -> ScoringConfig:
    """Return a copy of ``cfg`` with the supplied overrides applied and validated."""
    out = cfg
    if market_weights is not None:
        out = replace(out, market_weights=normalise_market_weights(market_weights))
    if readiness_weights is not None:
        out = replace(out, readiness_weights=normalise_readiness_weights(readiness_weights))
    if alpha is not None:
        out = replace(out, alpha=clip_01(alpha))
    if friction_weight is not None:
        out = replace(out, friction_weight=clip_01(friction_weight))
    if top_n is not None:
        if int(top_n) < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}.")
        out = replace(out, top_n=int(top_n))
    return out
