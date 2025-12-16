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


def _to_csv_bytes(rows: List[Dict[str, object]]) -> bytes:
    import csv

    if not rows:
        return b""

    buffer = io.StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


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


def _extract_geo_regions(geo: dict, featureidkey: str) -> List[str]:
    parts = featureidkey.split(".", 1)
    if len(parts) != 2:
        return []
    _, key_path = parts
    keys = key_path.split(".")
    out: List[str] = []

    for feat in geo.get("features", []):
        cur = feat
        ok = True
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            out.append(_normalise_region_label(cur))
    return out


def _render_uk_region_map(
    region_scores: pd.DataFrame,
    geojson_path: Path,
    featureidkey: str = "properties.region",
) -> None:
    if not geojson_path.exists():
        st.error(f"Missing GeoJSON file. Expected at: {geojson_path}")
        return

    if region_scores.empty:
        st.info("No region scores available to plot.")
        return

    geo = json.loads(geojson_path.read_text(encoding="utf-8"))

    geo_regions = set(_extract_geo_regions(geo, featureidkey))
    data_regions = set(region_scores["region"].map(_normalise_region_label).tolist())

    missing_in_geo = sorted([r for r in data_regions if r not in geo_regions and r != "Unknown"])
    missing_in_data = sorted([r for r in geo_regions if r not in data_regions])

    if missing_in_geo:
        with st.expander("Map matching diagnostics"):
            st.write("These region labels exist in your data but were not found in the GeoJSON:")
            st.code("\n".join(missing_in_geo))

    if missing_in_data:
        with st.expander("Map matching diagnostics"):
            st.write("These region labels exist in the GeoJSON but were not found in your data:")
            st.code("\n".join(missing_in_data))

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

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        title="UK opportunity score by region",
        margin={"r": 0, "t": 45, "l": 0, "b": 0},
    )

    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("Heagital Market Decision Engine")
    st.caption("Upload an ICB level dataset using the provided template. Then set decision weights and run the ranking.")

    project_root = Path(__file__).resolve().parent
    template_path = project_root / "data" / "template" / "icb_input_template.csv"
    scoring_config_path = project_root / "src" / "heagital_mde" / "config" / "scoring_config.yml"

    geojson_path = project_root / "data" / "geo" / "uk_regions.geojson"
    geo_featureidkey = "properties.region"

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
        st.header("Outputs")
        st.caption("UK map is based on regional average opportunity score.")
        st.caption("Expected GeoJSON location: data/geo/uk_regions.geojson")
        geo_featureidkey = st.text_input("GeoJSON featureidkey", value=geo_featureidkey)

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

    st.success("Ranking completed.")

    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        st.subheader("ICB ranking")
        st.dataframe(df_ranked, use_container_width=True, hide_index=True)

        st.subheader("Regional pivot summary")
        st.dataframe(region_pivot, use_container_width=True, hide_index=True)

        st.subheader("UK region map")
        _render_uk_region_map(region_scores, geojson_path=geojson_path, featureidkey=geo_featureidkey)

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
