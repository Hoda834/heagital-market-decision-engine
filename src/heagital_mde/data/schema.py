from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


CANONICAL_COLUMNS: Tuple[str, ...] = (
    "icb_code",
    "icb_name",
    "af_register",
    "treatment_gap",
    "warfarin_proxy",
    "region",
)

REQUIRED_CANONICAL_COLUMNS: Tuple[str, ...] = (
    "icb_code",
    "icb_name",
    "af_register",
    "treatment_gap",
    "warfarin_proxy",
)

RAW_TO_CANONICAL_MAP: Dict[str, str] = {
    "ICB ODS code": "icb_code",
    "ICB name": "icb_name",
    "Register": "af_register",
    "Treatment Gap (%)": "treatment_gap",
    "Warfarin Item icb": "warfarin_proxy",
    "Region": "region",
}

RAW_REQUIRED_COLUMNS: Tuple[str, ...] = tuple(RAW_TO_CANONICAL_MAP.keys())


@dataclass(frozen=True)
class ColumnSpec:
    raw_required: Tuple[str, ...] = RAW_REQUIRED_COLUMNS
    raw_to_canonical: Dict[str, str] = None  # type: ignore[assignment]
    canonical_required: Tuple[str, ...] = REQUIRED_CANONICAL_COLUMNS

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_to_canonical", dict(RAW_TO_CANONICAL_MAP))


def list_expected_raw_columns() -> List[str]:
    return list(RAW_REQUIRED_COLUMNS)


def list_expected_canonical_columns() -> List[str]:
    return list(CANONICAL_COLUMNS)
