from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from heagital_mde.io.load_icb import load_icb_features
from heagital_mde.io.validate import validate_icb_features
from heagital_mde.model.scoring import score_and_rank

st.set_page_config(page_title="Heagital Market Decision Engine", layout="wide")


def _normalise_weights_to_one(w1: float, w2: float, w3: float) -> Tuple[float, float, float]:
    total = w1 + w2 + w3
    if total <= 0:
        return 0.45, 0.35, 0.20
    return w1 / total, w2 / total, w3 / total


def _read_template_bytes(template_path: Path) -> bytes:
    if not template_path.exists():
        return b""
    return template_path.read_bytes()


def _write_temp_csv(uploaded_bytes: bytes) -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / "uploaded_data.csv"
    tmp_path.write_bytes(uploaded_bytes)
    return tmp_path


def _ensure_dataframe(ranked: object) -> pd.DataFrame:
    if isinstance(ranked, pd.DataFrame):
        return ranked.copy()
    if isinstance(ranked, list):
        return pd.DataFrame(ranked)
    raise TypeError("Unsupported results type returned by score_and_rank.")


def _normalise_region_label(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "Unknown"
    return str(x).strip()


def _build_region_pivot(df_view: pd.DataFrame) -> pd.DataFrame:
    df = df_view.copy()

    required = {
        "region",
        "icb_code",
        "recommended_included",
        "opportunity_score",
        "n_clinical_risk",
        "n_adoption_readiness",
        "n_procurement_friction",
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot build regional pivot. Missing columns: {missing}")

    df["region"] = df["region"].map(_normalise_region_label)
    df["recommended_included"] = df["recommended_included"].astype(bool)

    pivot = (
        df.groupby("region", dropna=False)
        .agg(
            total_icbs=("icb_code", "count"),
            included_icbs=("recommended_included", "sum"),
            included_rate=("recommended_included", "mean"),
            avg_opportunity_score=("opportunity_score", "mean"),
            avg_n_clinical_risk=("n_clinical_risk", "mean"),
            avg_n_adoption_readiness=("n_adoption_readiness", "mean"),
            avg_n_procurement_friction=("n_procurement_friction", "mean"),
        )
        .reset_index()
    )

    pivot["included_rate"] = (pivot["included_rate"] * 100.0).round(1)

    pivot = pivot.sort_values(
        by=["included_icbs", "avg_opportunity_score", "total_icbs"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    return pivot


def _build_region_opportunity(df_view: pd.DataFrame) -> pd.DataFrame:
    df = df_view.copy()

    required = {"region", "opportunity_score"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot build region opportunity summary. Missing columns: {missing}")

    df["region"] = df["region"].map(_normalise_region_label)

    out = (
        df.groupby("region", dropna=False)
        .agg(avg_opportunity_score=("opportunity_score", "mean"))
        .reset_index()
    )

    out["avg_opportunity_score"] = pd.to_numeric(out["avg_opportunity_score"], errors="coerce")
    out = out.dropna(subset=["avg_opportunity_score"]).reset_index(drop=True)

    out = out.sort_values(by="avg_opportunity_score", ascending=False, kind="mergesort").reset_index(drop=True)
    return out


def _densify_ring_to_n_points(ring: List[List[float]], target_points: int) -> List[List[float]]:
    """
    Densifies a polygon ring so it has at least target_points vertices.
    This is a schematic densification, not a true cartographic boundary.
    """
    if not ring:
        return ring

    # Ensure ring is closed
    if ring[0] != ring[-1]:
        ring = ring + [ring[0]]

    n = len(ring) - 1  # number of edges
    if n <= 0:
        return ring

    # We want total vertices (including closing point) to be at least target_points + 1
    desired_total = max(target_points + 1, len(ring))
    extra_needed = desired_total - len(ring)
    if extra_needed <= 0:
        return ring

    base_add = extra_needed // n
    remainder = extra_needed % n

    new_ring: List[List[float]] = []
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]

        new_ring.append([float(x1), float(y1)])

        add_here = base_add + (1 if i < remainder else 0)
        for k in range(1, add_here + 1):
            t = k / (add_here + 1)
            xi = x1 + (x2 - x1) * t
            yi = y1 + (y2 - y1) * t
            new_ring.append([float(xi), float(yi)])

    # Close ring
    new_ring.append([float(ring[0][0]), float(ring[0][1])])
    return new_ring


def _densify_geojson_polygons(geo: dict, target_points: int = 20) -> dict:
    """
    Applies densification to Polygon and MultiPolygon geometries.
    """
    features = geo.get("features", [])
    for feat in features:
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")

        if gtype == "Polygon" and isinstance(coords, list) and coords:
            outer = coords[0]
            coords[0] = _densify_ring_to_n_points(outer, target_points)
            geom["coordinates"] = coords

        elif gtype == "MultiPolygon" and isinstance(coords, list):
            new_coords = []
            for poly in coords:
                if not poly:
                    new_coords.append(poly)
                    continue
                outer = poly[0]
                poly[0] = _densify_ring_to_n_points(outer, target_points)
                new_coords.append(poly)
            geom["coordinates"] = new_coords

        feat["geometry"] = geom

    geo["features"] = features
    return geo


def _render_uk_region_map(
    region_scores: pd.DataFrame,
    geojson_path: Path,
    featureidkey: str = "properties.region",
    target_points: int = 20,
) -> None:
    if not geojson_path.exists():
        st.error(f"Missing GeoJSON file. Expected at: {geojson_path}")
        return

    if region_scores.empty:
        st.info("No region scores available to plot.")
        return

    geo_raw = json.loads(geojson_path.read_text(encoding="utf-8"))
    geo = _densify_geojson_polygons(geo_raw, target_points=target_points)

    fig = px.choropleth(
        region_scores,
        geojson=geo,
        featureidkey=featureidkey,
        locations="region",
        color="avg_opportunity_score",
        projection="mercator",
        hover_name="region",
        hover_data={"avg_opportunity_score": ":.3f"},
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        center={"lat": 54.2, "lon": -2.5},
        projection_scale=7,
    )
    fig.update_layout(
        title="UK opportunity score by region",
        margin={"r": 0, "t": 45, "l": 0, "b": 0},
        height=520,
    )

    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("Heagital Market Decision Engine")
    st.caption("Upload an ICB level dataset using the provided template. Then set decision weights and run the ranking.")

    project_root = Path(__file__).resolve().parent
    template_path = project_root / "data" / "template" / "icb_input_template.csv"
    scoring_config_path = project_root / "src" / "heagital_mde" / "config" / "scoring_config.yml"

    geojson_path = project_root / "data" / "geo" / "nhs_england_regions.geojson"
    geo_featureidkey_default = "properties.region"

    with st.sidebar:
        st.header("Step 1  Download template")
        template_bytes = _read_template_bytes(template_path)

        if template_bytes:
            st.download_button(
                label="Download CSV template",
                data=template_bytes,
                file_name="icb_input_template.csv",
                mime="text/csv",
            )
        else:
            st.warning("Template file not found. Expected at data/template/icb_input_template.csv")

        st.divider()
        st.header("Step 2  Upload your data")
        uploaded = st.file_uploader("Upload completed CSV", type=["csv"])

        st.divider()
        st.header("Step 3  Decision weights")
        w1 = st.slider("Clinical risk weight", 0.0, 1.0, 0.45, 0.01)
        w2 = st.slider("Adoption readiness weight", 0.0, 1.0, 0.35, 0.01)
        w3 = st.slider("Procurement friction weight", 0.0, 1.0, 0.20, 0.01)

        w1n, w2n, w3n = _normalise_weights_to_one(w1, w2, w3)
        st.caption(f"Normalised weights: clinical {w1n:.2f}, adoption {w2n:.2f}, friction {w3n:.2f}")

        top_n = st.number_input("Recommended cut off Top N", min_value=1, max_value=100, value=15, step=1)

        st.divider()
        st.header("Map settings")
        geo_featureidkey = st.text_input("GeoJSON featureidkey", value=geo_featureidkey_default)
        densify_points = st.number_input("Polygon points per region (schematic)", min_value=20, max_value=200, value=20, step=5)

        run = st.button("Run ranking", type="primary", disabled=(uploaded is None))

if uploaded is None:
    st.info("Download the template, fill it, upload the CSV, then run the ranking.")
    return

if not scoring_config_path.exists():
    st.error("Missing scoring config. Expected at src/heagital_mde/config/scoring_config.yml")
    return

if not run:
    st.stop()

try:
    data_path = _write_temp_csv(uploaded.getvalue())

    df_in = load_icb_features(data_path)
    validate_icb_features(df_in)

    ranked = score_and_rank(df_in, scoring_config_path)
    df_ranked = _ensure_dataframe(ranked)

    df_ranked["recommended_cutoff_top_n"] = int(top_n)
    df_ranked["recommended_included"] = df_ranked["rank"].astype(int) <= int(top_n)

    csv_bytes = df_ranked.to_csv(index=False).encode("utf-8")
    included_count = int(df_ranked["recommended_included"].sum())

    region_pivot = _build_region_pivot(df_ranked)
    region_scores = _build_region_opportunity(df_ranked)

except Exception as e:
    st.error(f"Run failed: {e}")
    return

REGION_CENTROIDS = {
    "North East and Yorkshire": {"lat": 54.8, "lon": -1.8},
    "North West": {"lat": 54.0, "lon": -2.8},
    "Midlands": {"lat": 52.8, "lon": -1.5},
    "East of England": {"lat": 52.3, "lon": 0.5},
    "London": {"lat": 51.5, "lon": -0.1},
    "South East": {"lat": 51.2, "lon": 0.8},
    "South West": {"lat": 50.9, "lon": -3.5},
}

region_scores["lat"] = region_scores["region"].map(lambda r: REGION_CENTROIDS.get(str(r), {}).get("lat"))
region_scores["lon"] = region_scores["region"].map(lambda r: REGION_CENTROIDS.get(str(r), {}).get("lon"))
region_scores = region_scores.dropna(subset=["lat", "lon"]).reset_index(drop=True)

st.success("Ranking completed.")

c1, c2 = st.columns([2, 1], gap="large")

with c1:
    st.subheader("ICB ranking")
    st.dataframe(df_ranked, use_container_width=True, hide_index=True)

    st.subheader("Regional pivot summary")
    st.dataframe(region_pivot, use_container_width=True, hide_index=True)

    st.subheader("UK opportunity map (web)")
    render_web_map(region_scores)

    st.subheader("Region opportunity score summary")
    st.dataframe(region_scores, use_container_width=True, hide_index=True)

with c2:
    st.subheader("Download")
    st.download_button(
        label="Download ranking CSV",
        data=csv_bytes,
        file_name="icb_opportunity_ranking.csv",
        mime="text/csv",
    )
    st.metric("Included ICBs", included_count)



if __name__ == "__main__":
    main()
def render_web_map(region_scores: pd.DataFrame) -> None:
    fig = px.scatter_mapbox(
        region_scores,
        lat="lat",
        lon="lon",
        color="avg_opportunity_score",
        size="avg_opportunity_score",
        hover_name="region",
        hover_data={"avg_opportunity_score": ":.3f"},
        zoom=5,
        height=520,
        color_continuous_scale="Blues",
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        title="UK opportunity score by region",
    )

    st.plotly_chart(fig, use_container_width=True)
