from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from heagital_mde import __version__  # noqa: E402
from heagital_mde.io.load_icb import load_icb_features  # noqa: E402
from heagital_mde.io.validate import ValidationError, check_icb_features  # noqa: E402
from heagital_mde.model.scenarios import build_sensitivity, load_scenarios, run_scenarios  # noqa: E402
from heagital_mde.model.scoring import ScoringResult, score_and_rank  # noqa: E402
from heagital_mde.model.scoring.schema import (  # noqa: E402
    MarketWeightConfig,
    ReadinessWeightConfig,
    load_scoring_config,
)

TEMPLATE_PATH = PROJECT_ROOT / "data" / "template" / "icb_input_template.csv"
SCORING_CONFIG_PATH = PROJECT_ROOT / "src" / "heagital_mde" / "config" / "scoring_config.yml"
SCENARIO_CONFIG_PATH = PROJECT_ROOT / "src" / "heagital_mde" / "config" / "scenarios.yml"
GEOJSON_PATH = PROJECT_ROOT / "data" / "geo" / "nhs_england_regions.geojson"

REGION_CENTROIDS: Dict[str, Dict[str, float]] = {
    "North East and Yorkshire": {"lat": 54.8, "lon": -1.8},
    "North West": {"lat": 54.0, "lon": -2.8},
    "Midlands": {"lat": 52.8, "lon": -1.5},
    "East of England": {"lat": 52.3, "lon": 0.5},
    "London": {"lat": 51.5, "lon": -0.1},
    "South East": {"lat": 51.2, "lon": 0.8},
    "South West": {"lat": 50.9, "lon": -3.5},
}

st.set_page_config(page_title="Heagital Market Decision Engine", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_template_bytes(template_path: Path) -> bytes:
    return template_path.read_bytes() if template_path.exists() else b""


def write_temp_csv(uploaded_bytes: bytes) -> Path:
    tmp_path = Path(tempfile.mkdtemp()) / "uploaded_data.csv"
    tmp_path.write_bytes(uploaded_bytes)
    return tmp_path


def normalise_region_label(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    return text if text else "Unknown"


def normalised_signal_columns(df: pd.DataFrame) -> list[str]:
    """The n_* columns actually produced for this run."""
    return [c for c in df.columns if c.startswith("n_")]


def build_region_pivot(df_view: pd.DataFrame) -> pd.DataFrame:
    """Per-region rollup of inclusion rate and mean scores.

    Aggregates over whichever normalised signals the run produced, so an input
    file without the optional readiness columns still summarises cleanly.
    """
    required = ["region", "icb_code", "recommended_included", "final_score"]
    missing = [c for c in required if c not in df_view.columns]
    if missing:
        raise ValueError(f"Cannot build regional summary. Missing columns: {missing}")

    df = df_view.copy()
    df["region"] = df["region"].map(normalise_region_label)
    df["recommended_included"] = df["recommended_included"].astype(bool)

    aggregations: Dict[str, tuple] = {
        "total_icbs": ("icb_code", "count"),
        "included_icbs": ("recommended_included", "sum"),
        "included_rate": ("recommended_included", "mean"),
        "avg_final_score": ("final_score", "mean"),
    }
    for score_col in ("market_score", "readiness_score"):
        if score_col in df.columns:
            aggregations[f"avg_{score_col}"] = (score_col, "mean")
    for signal_col in normalised_signal_columns(df):
        aggregations[f"avg_{signal_col}"] = (signal_col, "mean")

    pivot = df.groupby("region", dropna=False).agg(**aggregations).reset_index()
    pivot["included_rate"] = (pivot["included_rate"] * 100.0).round(1)

    return pivot.sort_values(
        by=["included_icbs", "avg_final_score", "total_icbs"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def build_region_scores(df_view: pd.DataFrame) -> pd.DataFrame:
    """Mean final score per region, highest first."""
    missing = [c for c in ("region", "final_score") if c not in df_view.columns]
    if missing:
        raise ValueError(f"Cannot build region score summary. Missing columns: {missing}")

    df = df_view.copy()
    df["region"] = df["region"].map(normalise_region_label)

    out = df.groupby("region", dropna=False).agg(avg_final_score=("final_score", "mean")).reset_index()
    out["avg_final_score"] = pd.to_numeric(out["avg_final_score"], errors="coerce")
    out = out.dropna(subset=["avg_final_score"]).reset_index(drop=True)
    return out.sort_values(by="avg_final_score", ascending=False, kind="mergesort").reset_index(drop=True)


def apply_region_centroids(region_scores: pd.DataFrame) -> pd.DataFrame:
    df = region_scores.copy()
    df["lat"] = df["region"].map(lambda r: REGION_CENTROIDS.get(str(r), {}).get("lat"))
    df["lon"] = df["region"].map(lambda r: REGION_CENTROIDS.get(str(r), {}).get("lon"))
    return df.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def load_region_geojson(path: Path) -> Optional[dict]:
    """Load region boundaries, or None when the file is absent or a placeholder."""
    if not path.exists() or path.stat().st_size < 32:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) and data.get("features") else None


def geojson_name_key(geojson: dict) -> Optional[str]:
    """Find the property holding the region name."""
    properties = geojson["features"][0].get("properties", {})
    for candidate in ("nhser22nm", "nhser21nm", "name", "NAME", "region", "Region"):
        if candidate in properties:
            return candidate
    for key, value in properties.items():
        if isinstance(value, str):
            return key
    return None


def render_map(region_scores: pd.DataFrame) -> None:
    """Choropleth when real boundaries exist, otherwise a centroid bubble map."""
    if region_scores.empty:
        st.info("No regional scores to map. Add a Region column to the input file.")
        return

    geojson = load_region_geojson(GEOJSON_PATH)
    name_key = geojson_name_key(geojson) if geojson else None

    # Plotly 6 renamed the Mapbox-backed figures and deprecated the old names.
    # Pick whichever this install provides so the app is warning-free on both.
    modern = hasattr(px, "scatter_map")
    style_kwarg = "map_style" if modern else "mapbox_style"

    if geojson and name_key:
        choropleth = px.choropleth_map if modern else px.choropleth_mapbox
        fig = choropleth(
            region_scores,
            geojson=geojson,
            locations="region",
            featureidkey=f"properties.{name_key}",
            color="avg_final_score",
            color_continuous_scale="Viridis",
            center={"lat": 52.8, "lon": -1.5},
            zoom=4.6,
            opacity=0.75,
            height=520,
            **{style_kwarg: "carto-positron"},
        )
    else:
        with_coords = apply_region_centroids(region_scores)
        if with_coords.empty:
            st.info(
                "Region names do not match the known NHS England regions, so the map cannot be drawn. "
                "The regional tables below are unaffected."
            )
            return
        scatter = px.scatter_map if modern else px.scatter_mapbox
        fig = scatter(
            with_coords,
            lat="lat",
            lon="lon",
            color="avg_final_score",
            size="avg_final_score",
            hover_name="region",
            hover_data={"avg_final_score": ":.3f", "lat": False, "lon": False},
            color_continuous_scale="Viridis",
            zoom=5,
            height=520,
            **{style_kwarg: "carto-positron"},
        )
        st.caption(
            "Showing region centroids. Drop a populated NHS England region GeoJSON at "
            "`data/geo/nhs_england_regions.geojson` to switch to boundary shading."
        )

    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, title="Average final score by NHS region")
    st.plotly_chart(fig, width="stretch")


def prettify_columns_for_display(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=mapping)


def humanise(column: str) -> str:
    label = column.replace("avg_n_", "Average normalised ").replace("avg_", "Average ")
    label = label.replace("n_", "Normalised ").replace("_", " ")
    return label[0].upper() + label[1:]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar(default_top_n: int, default_alpha: float, default_friction: float) -> dict:
    controls: dict = {}

    with st.sidebar:
        st.header("1. Input data")
        st.download_button(
            label="Download input template (CSV)",
            data=read_template_bytes(TEMPLATE_PATH),
            file_name="icb_input_template.csv",
            mime="text/csv",
            disabled=not TEMPLATE_PATH.exists(),
        )
        controls["uploaded"] = st.file_uploader("Upload completed ICB CSV", type=["csv"])
        controls["gap_units"] = st.selectbox(
            "Treatment gap units",
            options=("auto", "percent", "fraction"),
            index=0,
            help="'auto' infers from the observed range. Set explicitly if your file mixes conventions.",
        )

        # Explicit keys: the market and readiness blocks share signal names, and
        # identical labels would otherwise collide into one widget.
        st.header("2. Market weights")
        st.caption("Relative emphasis within the market pillar. Rescaled to sum to 1.")
        controls["w_register"] = st.slider("AF register size", 0.0, 1.0, 0.30, 0.01, key="mkt_register")
        controls["w_prevalence"] = st.slider("Prevalence", 0.0, 1.0, 0.20, 0.01, key="mkt_prevalence")
        controls["w_gap_market"] = st.slider("Treatment gap", 0.0, 1.0, 0.30, 0.01, key="mkt_gap")
        controls["w_warfarin_market"] = st.slider("Warfarin proxy", 0.0, 1.0, 0.20, 0.01, key="mkt_warfarin")

        st.header("3. Readiness weights")
        st.caption("Relative emphasis within the readiness pillar. Rescaled to sum to 1.")
        controls["w_gap_readiness"] = st.slider("Treatment gap", 0.0, 1.0, 0.35, 0.01, key="rdy_gap")
        controls["w_warfarin_readiness"] = st.slider("Warfarin proxy", 0.0, 1.0, 0.25, 0.01, key="rdy_warfarin")
        controls["w_digital"] = st.slider("Digital maturity", 0.0, 1.0, 0.40, 0.01, key="rdy_digital")

        st.header("4. Decision parameters")
        controls["alpha"] = st.slider(
            "Alpha: market vs readiness",
            0.0,
            1.0,
            float(default_alpha),
            0.01,
            help="1.0 ranks purely on market size, 0.0 purely on adoption readiness.",
        )
        controls["friction_weight"] = st.slider(
            "Procurement friction penalty",
            0.0,
            1.0,
            float(default_friction),
            0.01,
            help="Applied only when the input file has a Procurement Friction column.",
        )
        controls["top_n"] = int(
            st.number_input("Recommended cut-off (Top N)", min_value=1, max_value=100, value=int(default_top_n), step=1)
        )
        controls["run_scenarios"] = st.checkbox("Run scenario sensitivity", value=True)

        controls["run"] = st.button("Run ranking", type="primary", disabled=controls["uploaded"] is None)

    return controls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("Heagital Market Decision Engine")
    st.caption(
        f"v{__version__} — upload an ICB-level dataset using the provided template, "
        "set the decision parameters, then run the ranking."
    )

    if not SCORING_CONFIG_PATH.exists():
        st.error(f"Missing scoring config. Expected at {SCORING_CONFIG_PATH.relative_to(PROJECT_ROOT)}")
        return

    try:
        base_config = load_scoring_config(SCORING_CONFIG_PATH)
    except (ValueError, FileNotFoundError) as exc:
        st.error(f"Scoring config could not be loaded: {exc}")
        return

    controls = render_sidebar(base_config.top_n, base_config.alpha, base_config.friction_weight)

    if controls["uploaded"] is None:
        st.info("Download the template from the sidebar, fill it in, upload the CSV, then run the ranking.")
        st.subheader("How the score is built")
        st.markdown(
            "```\n"
            "final = alpha x market + (1 - alpha) x readiness - friction_weight x friction\n"
            "```\n"
            "- **market** — register size, prevalence, treatment gap, warfarin proxy\n"
            "- **readiness** — treatment gap, warfarin proxy, digital maturity *(optional column)*\n"
            "- **friction** — procurement friction *(optional column)*\n\n"
            "Optional columns may be left out of the file; the engine redistributes their weight and "
            "tells you when it has done so."
        )
        return

    if not controls["run"]:
        st.info("Parameters set. Press **Run ranking** in the sidebar.")
        return

    try:
        data_path = write_temp_csv(controls["uploaded"].getvalue())

        with warnings.catch_warnings(record=True) as load_warnings:
            warnings.simplefilter("always")
            df_in = load_icb_features(data_path, gap_units=controls["gap_units"])

        report = check_icb_features(df_in)
        report.raise_if_failed()

        market_weights = MarketWeightConfig(
            register=float(controls["w_register"]),
            prevalence=float(controls["w_prevalence"]),
            treatment_gap=float(controls["w_gap_market"]),
            warfarin_proxy=float(controls["w_warfarin_market"]),
        )
        readiness_weights = ReadinessWeightConfig(
            treatment_gap=float(controls["w_gap_readiness"]),
            warfarin_proxy=float(controls["w_warfarin_readiness"]),
            digital_maturity=float(controls["w_digital"]),
        )

        result = score_and_rank(
            df_in,
            config=base_config,
            market_weights_override=market_weights,
            readiness_weights_override=readiness_weights,
            alpha_override=float(controls["alpha"]),
            friction_weight_override=float(controls["friction_weight"]),
            top_n_override=int(controls["top_n"]),
            return_result=True,
        )
        assert isinstance(result, ScoringResult)
        df_ranked = result.ranking

        sensitivity = None
        if controls["run_scenarios"] and SCENARIO_CONFIG_PATH.exists():
            scenario_results = run_scenarios(df_in, result.config, load_scenarios(SCENARIO_CONFIG_PATH))
            base_name = "base_case" if "base_case" in scenario_results else next(iter(scenario_results))
            sensitivity = build_sensitivity(scenario_results, base_scenario=base_name)

        has_region = "region" in df_ranked.columns
        region_pivot = build_region_pivot(df_ranked) if has_region else None
        region_scores = build_region_scores(df_ranked) if has_region else None

    except ValidationError as exc:
        st.error("Input data failed validation.")
        st.code(str(exc))
        return
    except (ValueError, FileNotFoundError) as exc:
        st.error(f"Run failed: {exc}")
        return
    except Exception as exc:  # pragma: no cover - surfaced to the user, not swallowed
        st.error(f"Unexpected failure: {type(exc).__name__}: {exc}")
        st.exception(exc)
        return

    st.success(f"Ranking completed for {len(df_ranked)} ICBs.")

    for warning in load_warnings:
        st.warning(str(warning.message))
    for message in report.warnings:
        st.warning(message)
    for message in result.warnings:
        st.warning(message)
    if result.signals_absent:
        st.info(
            "Optional signals not present in the upload: "
            + ", ".join(result.signals_absent)
            + ". Their weight was redistributed across the signals that are present."
        )

    metric_cols = st.columns(4)
    metric_cols[0].metric("ICBs scored", len(df_ranked))
    metric_cols[1].metric("Included in Top N", int(df_ranked["recommended_included"].sum()))
    metric_cols[2].metric("Alpha applied", f"{result.config.alpha:.2f}")
    metric_cols[3].metric(
        "Market/readiness correlation",
        "n/a" if result.pillar_correlation is None else f"{result.pillar_correlation:.2f}",
    )

    display_ranked_cols = {
        "rank": "Rank",
        "icb_code": "ICB code",
        "icb_name": "ICB name",
        "region": "Region",
        "market_score": "Market score",
        "readiness_score": "Readiness score",
        "friction_score": "Friction score",
        "final_score": "Final score",
        "recommended_cutoff_top_n": "Top N cut-off",
        "recommended_included": "Included in Top N",
    }
    display_ranked_cols.update({c: humanise(c) for c in normalised_signal_columns(df_ranked)})

    tab_ranking, tab_regions, tab_sensitivity, tab_audit = st.tabs(
        ["Ranking", "Regions", "Sensitivity", "Audit"]
    )

    with tab_ranking:
        st.dataframe(
            prettify_columns_for_display(df_ranked, display_ranked_cols),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            label="Download ranking CSV",
            data=df_ranked.to_csv(index=False).encode("utf-8"),
            file_name="icb_opportunity_ranking.csv",
            mime="text/csv",
        )

    with tab_regions:
        if region_pivot is None or region_scores is None:
            st.info("The uploaded file has no Region column, so regional summaries are unavailable.")
        else:
            st.subheader("Regional summary")
            pivot_labels = {c: humanise(c) for c in region_pivot.columns}
            pivot_labels.update(
                {
                    "region": "Region",
                    "total_icbs": "Total ICBs",
                    "included_icbs": "ICBs in Top N",
                    "included_rate": "Inclusion rate (%)",
                }
            )
            st.dataframe(
                prettify_columns_for_display(region_pivot, pivot_labels),
                width="stretch",
                hide_index=True,
            )
            st.subheader("Opportunity map")
            render_map(region_scores)

    with tab_sensitivity:
        if sensitivity is None:
            st.info("Scenario sensitivity was not run. Enable it in the sidebar.")
        else:
            counts = sensitivity["stability"].value_counts().to_dict()
            cols = st.columns(3)
            cols[0].metric("Robust", int(counts.get("robust", 0)), help="Inside the cut-off under every scenario.")
            cols[1].metric("Fragile", int(counts.get("fragile", 0)), help="Inclusion depends on the scenario.")
            cols[2].metric("Excluded", int(counts.get("excluded", 0)), help="Outside the cut-off under every scenario.")
            st.dataframe(sensitivity, width="stretch", hide_index=True)
            st.download_button(
                label="Download sensitivity CSV",
                data=sensitivity.to_csv(index=False).encode("utf-8"),
                file_name="icb_sensitivity.csv",
                mime="text/csv",
            )

    with tab_audit:
        st.caption("Everything needed to reproduce this ranking.")
        st.json(result.audit())


if __name__ == "__main__":
    main()
