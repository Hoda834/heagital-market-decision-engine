from __future__ import annotations

from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Canonical (internal) column names produced by the loader.
#
# The loader is the single place where raw spreadsheet headers are translated
# into these names. Everything downstream speaks canonical only, so no module
# below io/ ever has to rename a column again.
# ---------------------------------------------------------------------------

REQUIRED_CANONICAL_COLUMNS: Tuple[str, ...] = (
    "icb_code",
    "icb_name",
    "register",
    "prevalence",
    "treatment_gap",
    "warfarin_proxy",
)

OPTIONAL_CANONICAL_COLUMNS: Tuple[str, ...] = (
    "region",
    "digital_maturity",
    "procurement_friction",
)

CANONICAL_COLUMNS: Tuple[str, ...] = REQUIRED_CANONICAL_COLUMNS + OPTIONAL_CANONICAL_COLUMNS

# Signal groups consumed by the scoring model.
MARKET_SIGNALS: Tuple[str, ...] = (
    "register",
    "prevalence",
    "treatment_gap",
    "warfarin_proxy",
)

READINESS_SIGNALS: Tuple[str, ...] = (
    "treatment_gap",
    "warfarin_proxy",
    "digital_maturity",
)

FRICTION_SIGNAL: str = "procurement_friction"

# ---------------------------------------------------------------------------
# Raw header -> canonical mapping.
#
# Keys are matched case-insensitively and whitespace-insensitively, so
# "ICB ODS Code", "icb ods code" and "ICB  ODS  code" all resolve.
# ---------------------------------------------------------------------------

RAW_TO_CANONICAL_MAP: Dict[str, str] = {
    "ICB ODS code": "icb_code",
    "ICB name": "icb_name",
    "Register": "register",
    "Prevalence (%)": "prevalence",
    "Treatment Gap (%)": "treatment_gap",
    "Warfarin Item icb": "warfarin_proxy",
    "Region": "region",
    "Digital Maturity (0-1)": "digital_maturity",
    "Procurement Friction (0-1)": "procurement_friction",
}

# Additional accepted spellings for the same canonical column. These exist so
# that files produced against earlier versions of the template keep loading.
RAW_ALIASES: Dict[str, str] = {
    "af register": "register",
    "af_register": "register",
    "register size": "register",
    "icb code": "icb_code",
    "icb ods": "icb_code",
    "prevalence": "prevalence",
    "prevalence %": "prevalence",
    "treatment gap": "treatment_gap",
    "treatment gap %": "treatment_gap",
    "warfarin item icb": "warfarin_proxy",
    "warfarin proxy": "warfarin_proxy",
    "digital maturity": "digital_maturity",
    "procurement friction": "procurement_friction",
}

REQUIRED_RAW_COLUMNS: Tuple[str, ...] = tuple(
    raw for raw, canonical in RAW_TO_CANONICAL_MAP.items() if canonical in REQUIRED_CANONICAL_COLUMNS
)

OPTIONAL_RAW_COLUMNS: Tuple[str, ...] = tuple(
    raw for raw, canonical in RAW_TO_CANONICAL_MAP.items() if canonical in OPTIONAL_CANONICAL_COLUMNS
)

# Columns that must be a proportion in [0, 1] once loaded.
UNIT_INTERVAL_COLUMNS: Tuple[str, ...] = (
    "treatment_gap",
    "digital_maturity",
    "procurement_friction",
)


def _normalise_header(header: object) -> str:
    """Fold a raw header to a comparable key: lowercase, single-spaced."""
    return " ".join(str(header).strip().lower().split())


#: Lookup table built once, keyed by folded header.
_HEADER_LOOKUP: Dict[str, str] = {}
for _raw, _canonical in RAW_TO_CANONICAL_MAP.items():
    _HEADER_LOOKUP[_normalise_header(_raw)] = _canonical
for _raw, _canonical in RAW_ALIASES.items():
    _HEADER_LOOKUP.setdefault(_normalise_header(_raw), _canonical)
# Canonical names are themselves valid headers.
for _canonical in CANONICAL_COLUMNS:
    _HEADER_LOOKUP.setdefault(_normalise_header(_canonical), _canonical)


def resolve_header(header: object) -> str | None:
    """Return the canonical name for a raw header, or None if unrecognised."""
    return _HEADER_LOOKUP.get(_normalise_header(header))


def list_expected_raw_columns() -> list[str]:
    return list(REQUIRED_RAW_COLUMNS)


def list_optional_raw_columns() -> list[str]:
    return list(OPTIONAL_RAW_COLUMNS)


def list_expected_canonical_columns() -> list[str]:
    return list(CANONICAL_COLUMNS)
