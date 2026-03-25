"""Soil Bankruptcy Predictor — Streamlit application entry point.

Run with:
    streamlit run src/app.py
"""

from __future__ import annotations

import io
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---------------------------------------------------------------------------
# Resolve data paths relative to the project root, not the working directory.
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(_project_root, "data")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from streamlit_folium import st_folium

from src.data_ingestion import DataIngestion
from src.geo_lookup_engine import GeoLookupEngine
from src.models import DistrictNotFoundError
from src.normalization_engine import NormalizationEngine
from src.sbrs_engine import SBRSEngine
from src.simulation_engine import SimulationEngine
from src.recommendation_engine import RecommendationEngine
from src.report_generator import ReportGenerator

# ---------------------------------------------------------------------------
# Farmer-friendly label maps
# ---------------------------------------------------------------------------

INTERVENTION_OPTIONS: dict[str, str] = {
    "drip_irrigation":      "💧 Drip Irrigation",
    "cover_cropping":       "🌿 Cover Cropping",
    "legume_rotation":      "🫘 Legume Rotation (e.g. chickpea, lentil)",
    "reduced_fertilizer":   "🧪 Reduce Chemical Fertilizer",
    "rainwater_harvesting": "🌧️ Rainwater Harvesting",
    "crop_diversification": "🌾 Grow More Crop Varieties",
}

# Plain-English names for soil parameters (used in tables, charts, error messages)
PARAM_LABELS: dict[str, str] = {
    "organic_carbon":           "Soil Richness (Organic Carbon %)",
    "nitrogen_depletion_rate":  "Nitrogen Loss (kg/ha/yr)",
    "rainfall_variability":     "Rainfall Unpredictability (0–1)",
    "crop_diversity_index":     "Crop Variety Score (0–1)",
    "irrigation_stress":        "Water Overuse Level (0–1)",
    "fertilizer_load":          "Chemical Fertilizer Use (kg/ha)",
}

# Technical column names → friendly upload instructions
UPLOAD_COL_HELP: dict[str, str] = {
    "district":                 "District name (e.g. Pune, Nashik)",
    "organic_carbon":           "Soil Richness — organic carbon % (e.g. 0.45)",
    "nitrogen_depletion_rate":  "Nitrogen Loss — kg lost per hectare per year (e.g. 12.5)",
    "rainfall_variability":     "Rainfall Unpredictability — 0 (steady) to 1 (very erratic)",
    "crop_diversity_index":     "Crop Variety Score — 0 (one crop only) to 1 (many crops)",
    "irrigation_stress":        "Water Overuse — 0 (no stress) to 1 (severe overuse)",
    "fertilizer_load":          "Chemical Fertilizer — kg used per hectare (e.g. 120)",
}

RISK_EMOJI = {
    "Healthy":          "🟢",
    "At Risk":          "🟠",
    "Critical":         "🔴",
    "Imminent Collapse":"🔵",
}

# ---------------------------------------------------------------------------
# Engine singletons
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_engines():
    return (
        DataIngestion(),
        NormalizationEngine(),
        SBRSEngine(),
        SimulationEngine(),
        RecommendationEngine(),
        ReportGenerator(),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _process_upload(uploaded_file) -> bool:
    ingestion, normalizer, sbrs_engine, _, _, _ = _get_engines()
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()
    df = pd.read_csv(io.BytesIO(file_bytes)) if name.endswith(".csv") else pd.read_excel(io.BytesIO(file_bytes))

    valid, missing = ingestion.validate_schema(df)
    if not valid:
        st.session_state["validation_errors"] = missing
        return False

    st.session_state["validation_errors"] = []
    df = ingestion.clean_data(df)
    st.session_state["single_district"] = len(df) == 1
    norm_df = normalizer.normalize(df)
    sbrs_df = sbrs_engine.compute_sbrs(norm_df)
    st.session_state["uploaded_df"] = df
    st.session_state["sbrs_df"] = sbrs_df
    return True


def _run_simulation(df: pd.DataFrame, years: int) -> pd.DataFrame:
    _, _, _, sim_engine, _, _ = _get_engines()
    projections = sim_engine.simulate(df, years=years)
    st.session_state["projections_df"] = projections
    return projections


def _dominant_label(raw: str) -> str:
    key = raw.replace("_norm", "")
    return PARAM_LABELS.get(key, raw.replace("_", " ").title())


def _risk_badge(category: str) -> str:
    return f"{RISK_EMOJI.get(category, '⚪')} {category}"


def _score_verdict(score: float) -> str:
    """Plain-English verdict for a soil health score."""
    if score < 30:
        return "✅ Your soil is healthy — keep up the good work!"
    if score < 60:
        return "🟠 Your soil needs attention — some practices should change soon."
    if score < 80:
        return "🔴 Your soil is in serious trouble — act this season."
    return "🚨 Your soil is at risk of complete failure — urgent action needed."


def _trend_label(slope: float) -> str:
    if slope < -0.5:
        return "✅ Getting Better"
    if slope > 0.5:
        return "⚠️ Getting Worse"
    return "➡️ Stable"


# ---------------------------------------------------------------------------
# Map helpers (shared between tabs)
# ---------------------------------------------------------------------------

_GEO_PATH  = os.path.join(_DATA_DIR, "districts.geojson")
_SOIL_PATH = os.path.join(_DATA_DIR, "soil_data.csv")


@st.cache_data(show_spinner=False)
def _load_all_sbrs() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Return (full_soil_df, all_sbrs_df) or (None, None) if files missing."""
    if not (os.path.exists(_GEO_PATH) and os.path.exists(_SOIL_PATH)):
        return None, None
    _, normalizer, sbrs_engine, _, _, _ = _get_engines()
    soil_df = pd.read_csv(_SOIL_PATH)
    norm_df = normalizer.normalize(soil_df)
    sbrs_df = sbrs_engine.compute_sbrs(norm_df)
    return soil_df, sbrs_df


def _build_map(all_sbrs_df: pd.DataFrame | None) -> folium.Map:
    m = folium.Map(location=[19.7, 75.7], zoom_start=7, tiles="OpenStreetMap")
    if all_sbrs_df is None or not os.path.exists(_GEO_PATH):
        return m

    import json as _json
    with open(_GEO_PATH) as f:
        gj = _json.load(f)

    sbrs_lkp = dict(zip(all_sbrs_df["district"], all_sbrs_df["sbrs"].round(1)))
    cat_lkp  = dict(zip(all_sbrs_df["district"], all_sbrs_df["risk_category"]))

    for feat in gj["features"]:
        d = feat["properties"].get("district", "")
        score = sbrs_lkp.get(d, 0)
        feat["properties"]["sbrs"]          = score
        feat["properties"]["risk_category"] = cat_lkp.get(d, "Unknown")
        # Show score as "lower = better" in tooltip
        feat["properties"]["soil_score"]    = f"{score:.0f} / 100 (lower = healthier)"

    def _style(feature):
        s = feature["properties"].get("sbrs", 0)
        color = "#8B0000" if s >= 80 else "#FF4444" if s >= 60 else "#FFA500" if s >= 30 else "#2ECC71"
        return {"fillColor": color, "color": "#444", "weight": 1.2, "fillOpacity": 0.7}

    choropleth = folium.FeatureGroup(name="🗺️ Soil Risk Map (colour by risk)", show=True)
    folium.GeoJson(
        gj,
        style_function=_style,
        tooltip=folium.GeoJsonTooltip(
            fields=["district", "soil_score", "risk_category"],
            aliases=["District", "Soil Health Score", "Status"],
        ),
    ).add_to(choropleth)
    choropleth.add_to(m)

    # Heatmap overlay
    try:
        from folium.plugins import HeatMap
        heat_data = []
        for feat in gj["features"]:
            geom = feat["geometry"]
            if geom["type"] == "Polygon":
                coords = geom["coordinates"][0]
                clat = sum(c[1] for c in coords) / len(coords)
                clng = sum(c[0] for c in coords) / len(coords)
                heat_data.append([clat, clng, feat["properties"]["sbrs"] / 100.0])
        hl = folium.FeatureGroup(name="🔥 Heat Intensity Map", show=False)
        HeatMap(heat_data, radius=35, blur=25, min_opacity=0.3).add_to(hl)
        hl.add_to(m)
    except Exception:
        pass

    # District name labels
    label_layer = folium.FeatureGroup(name="🏷️ District Name Labels", show=False)
    for feat in gj["features"]:
        d = feat["properties"].get("district", "")
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            coords = geom["coordinates"][0]
            clat = sum(c[1] for c in coords) / len(coords)
            clng = sum(c[0] for c in coords) / len(coords)
            s = sbrs_lkp.get(d, 0)
            color = "darkred" if s >= 80 else "red" if s >= 60 else "orange" if s >= 30 else "green"
            folium.Marker(
                location=[clat, clng],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:9px;font-weight:bold;color:{color};white-space:nowrap">{d}</div>',
                    icon_size=(80, 12),
                    icon_anchor=(40, 6),
                ),
            ).add_to(label_layer)
    label_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ---------------------------------------------------------------------------
# Tab 1 — "Check My District" (Future View)
# ---------------------------------------------------------------------------

def _render_future_view() -> None:
    ingestion, normalizer, sbrs_engine, sim_engine, rec_engine, report_gen = _get_engines()
    _, all_sbrs_df = _load_all_sbrs()
    geo_ok = os.path.exists(_GEO_PATH) and os.path.exists(_SOIL_PATH)

    # ── How to read this tab ─────────────────────────────────────────────────
    with st.expander("ℹ️ How to use this tab", expanded=False):
        st.markdown(
            """
            **This tab helps you answer: "Is my soil healthy, and what will happen in the future?"**

            1. **Click your district** on the map to instantly see its soil health score.
            2. **Upload your own data** (or use the sample) to check multiple districts at once.
            3. **See future predictions** — what happens if nothing changes over the next 5–10 years.
            4. **Pick a farming practice** to see how quickly it can bring your soil back to health.
            5. **Download a PDF report** to share with your bank or agriculture officer.

            > 🌱 **Soil Health Score:** 0 = perfect soil · 100 = soil is failing. Lower is always better.
            """
        )

    # ── Step 1: Map ──────────────────────────────────────────────────────────
    st.markdown("### 📍 Step 1 — Click Your District on the Map")
    st.caption(
        "Each district is coloured by how healthy its soil is right now. "
        "**Click any district** to see its score and whether your crops are at risk. "
        "Use the layer buttons (top-right of map) to switch views."
    )

    india_map = _build_map(all_sbrs_df)
    map_data = st_folium(india_map, width=700, height=520, use_container_width=True)

    # Legend
    c1, c2, c3, c4 = st.columns(4)
    c1.success("🟢 Healthy  (score < 30)")
    c2.warning("🟠 Needs Attention  (30–60)")
    c3.error("🔴 Serious Risk  (60–80)")
    c4.info("🔵 Danger — Act Now  (80+)")

    # Pin-drop result
    if map_data and map_data.get("last_clicked") and geo_ok and all_sbrs_df is not None:
        lat = map_data["last_clicked"]["lat"]
        lng = map_data["last_clicked"]["lng"]
        try:
            geo_engine = GeoLookupEngine(_GEO_PATH, _SOIL_PATH)
            result = geo_engine.lookup(lat, lng)
            dr = result.district_record

            sbrs_match = all_sbrs_df[all_sbrs_df["district"].str.lower() == result.district.lower()]
            sbrs_row = sbrs_match.iloc[0] if not sbrs_match.empty else sbrs_engine.compute_sbrs(
                normalizer.normalize(pd.DataFrame([{
                    "district": dr.district, "organic_carbon": dr.organic_carbon,
                    "nitrogen_depletion_rate": dr.nitrogen_depletion_rate,
                    "rainfall_variability": dr.rainfall_variability,
                    "crop_diversity_index": dr.crop_diversity_index,
                    "irrigation_stress": dr.irrigation_stress,
                    "fertilizer_load": dr.fertilizer_load, "year": dr.year,
                }]))
            ).iloc[0]

            score = float(sbrs_row["sbrs"])
            cat   = sbrs_row["risk_category"]
            dominant = _dominant_label(sbrs_row["dominant_factor"])

            st.markdown(f"---\n#### 📌 You clicked: **{result.district}**")

            # Plain-English verdict first — most important thing for a farmer
            st.info(_score_verdict(score))

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "🌱 Soil Health Score",
                f"{score:.0f} / 100",
                help="Lower is better. 0 = perfect soil, 100 = soil has completely failed.",
            )
            m2.metric("⚠️ Status", _risk_badge(cat))
            m3.metric(
                "🔍 Biggest Problem",
                dominant,
                help="This is the single factor hurting your soil the most right now.",
            )

            with st.expander("📊 See detailed soil readings for this district"):
                st.caption("These are the actual measured values for your district's soil.")
                detail = pd.DataFrame([{
                    "What we measured": k,
                    "Value": v,
                } for k, v in {
                    "Soil Richness (Organic Carbon %)": dr.organic_carbon,
                    "Nitrogen Loss (kg/ha/yr)": dr.nitrogen_depletion_rate,
                    "Rainfall Unpredictability (0–1, lower = steadier)": dr.rainfall_variability,
                    "Crop Variety Score (0–1, higher = better)": dr.crop_diversity_index,
                    "Water Overuse Level (0–1, lower = better)": dr.irrigation_stress,
                    "Chemical Fertilizer Use (kg/ha)": dr.fertilizer_load,
                }.items()])
                st.dataframe(detail, use_container_width=True, hide_index=True)

        except DistrictNotFoundError:
            st.warning("⚠️ Could not find a district at that spot. Try clicking inside Maharashtra.")
        except ValueError as exc:
            st.error(str(exc))

    st.divider()

    # ── Step 2: Upload your own data ─────────────────────────────────────────
    st.markdown("### 📂 Step 2 — Upload Your District Data (Optional)")
    st.caption(
        "Have a CSV or Excel file with soil readings for multiple districts? "
        "Upload it here to get scores and future predictions for all of them. "
        "Or click **Try with Sample Data** to see how it works with real Maharashtra data."
    )

    col_up, col_demo = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload your soil data file (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key="future_uploader",
            label_visibility="collapsed",
        )
    with col_demo:
        if st.button("🧪 Try with Sample Data", use_container_width=True):
            demo_df = pd.read_csv(_SOIL_PATH) if os.path.exists(_SOIL_PATH) else ingestion.generate_synthetic_dataset()
            norm_df = normalizer.normalize(demo_df)
            sbrs_df = sbrs_engine.compute_sbrs(norm_df)
            st.session_state["uploaded_df"] = demo_df
            st.session_state["sbrs_df"] = sbrs_df
            st.session_state["validation_errors"] = []
            st.session_state["single_district"] = False
            st.success("✅ Sample data loaded — scroll down to see results.")

    if uploaded_file is not None:
        if not _process_upload(uploaded_file):
            errors = st.session_state.get("validation_errors", [])
            friendly_missing = [UPLOAD_COL_HELP.get(c, c) for c in errors]
            st.error(
                f"❌ Your file is missing some required columns.\n\n"
                f"**Missing:** {', '.join(errors)}\n\n"
                "Your file needs these columns (one row per district):\n\n" +
                "\n".join(f"- **{col}** — {desc}" for col, desc in UPLOAD_COL_HELP.items())
            )
            return

    errors = st.session_state.get("validation_errors", [])
    if errors:
        st.error(f"❌ Missing columns: {', '.join(errors)}")
        return

    df: pd.DataFrame | None = st.session_state.get("uploaded_df")
    sbrs_df: pd.DataFrame | None = st.session_state.get("sbrs_df")

    if df is None or sbrs_df is None:
        return

    st.divider()

    # ── Step 3: Soil Health Scores ───────────────────────────────────────────
    st.markdown("### 🌾 Step 3 — Soil Health Scores for Your Districts")
    st.caption(
        "**Lower score = healthier soil.** "
        "A score above 60 means your soil needs urgent attention. "
        "The 'Biggest Problem' column shows what is hurting your soil the most."
    )

    display_df = sbrs_df[["district", "sbrs", "risk_category", "dominant_factor"]].copy()
    display_df["sbrs"] = display_df["sbrs"].round(1)
    display_df["risk_category"] = display_df["risk_category"].apply(_risk_badge)
    display_df["dominant_factor"] = display_df["dominant_factor"].apply(_dominant_label)
    display_df.columns = [
        "District",
        "Soil Health Score (0–100, lower = healthier)",
        "Status",
        "Biggest Problem",
    ]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Step 4: Future Prediction ────────────────────────────────────────────
    st.markdown("### 🔮 Step 4 — What Will Happen in the Future?")
    st.caption(
        "This shows how your soil health score will change over the next few years **if nothing changes**. "
        "A rising line means your soil is getting worse. "
        "The red dashed line at 80 is the danger level — soil above that may not support crops."
    )

    base_year = int(df["year"].max()) if "year" in df.columns else 2023
    horizon = st.radio(
        "How many years ahead do you want to see?",
        options=[5, 10],
        horizontal=True,
        format_func=lambda x: f"{x} years",
    )

    projections: pd.DataFrame | None = st.session_state.get("projections_df")
    if projections is None or st.session_state.get("last_horizon") != horizon:
        projections = _run_simulation(df, horizon)
        st.session_state["last_horizon"] = horizon

    # Replace "Year 0, 1, 2..." with actual calendar years
    proj_plot = projections.copy()
    proj_plot["year"] = proj_plot["year"] + base_year

    fig_traj = px.line(
        proj_plot, x="year", y="sbrs", color="district",
        labels={"year": "Year", "sbrs": "Soil Health Score (lower = healthier)", "district": "District"},
        title="How Soil Health Score Will Change Over Time (if nothing changes)",
    )
    fig_traj.add_hline(y=80, line_dash="dash", line_color="red",
                       annotation_text="⚠️ Danger level (80) — crops may fail above this")
    fig_traj.add_hline(y=30, line_dash="dot", line_color="orange",
                       annotation_text="Needs Attention (30)")
    st.plotly_chart(fig_traj, use_container_width=True)

    # Danger alerts
    any_danger = False
    for district in projections["district"].unique():
        bk_year = sim_engine.predict_bankruptcy_year(projections[projections["district"] == district])
        if bk_year is not None:
            calendar_year = base_year + bk_year
            any_danger = True
            st.error(f"🚨 **{district}** — soil may reach the danger level by **{calendar_year}**. Act now!")
    if not any_danger:
        st.success("✅ No district is predicted to reach the danger level within the selected period.")

    st.divider()

    # ── Step 5: What Can You Do? ─────────────────────────────────────────────
    st.markdown("### 💡 Step 5 — What Can You Do to Improve Your Soil?")
    st.caption(
        "Pick a farming practice below to see how many years it would take to bring your soil "
        "back to a **healthy level (score below 30)**."
    )

    intervention_label = st.selectbox(
        "Choose a farming practice:",
        options=list(INTERVENTION_OPTIONS.values()),
        key="intervention_select",
    )
    intervention_key = {v: k for k, v in INTERVENTION_OPTIONS.items()}[intervention_label]
    impact = rec_engine.get_intervention_impact(intervention_key)

    recovery_rows = []
    for _, row in sbrs_df.iterrows():
        yrs = sim_engine.compute_recovery_time(float(row["sbrs"]), impact)
        if yrs == 0:
            recovery_label = "Already healthy ✅"
        elif yrs >= 50:
            recovery_label = "50+ years ⚠️ — needs multiple interventions"
        else:
            recovery_label = f"{yrs} years to reach healthy soil"
        recovery_rows.append({
            "District": row["district"],
            "Current Score": round(float(row["sbrs"]), 1),
            "Years to Reach Healthy Soil": recovery_label,
        })
    st.dataframe(pd.DataFrame(recovery_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Step 6: Download Report ──────────────────────────────────────────────
    st.markdown("### 📄 Step 6 — Download Your Soil Report")
    st.caption("Get a PDF report you can print and share with your bank, agriculture officer, or cooperative.")

    if st.button("📥 Generate & Download PDF Report", use_container_width=True):
        try:
            recs = {
                row["district"]: rec_engine.generate(
                    district=row["district"],
                    sbrs=float(row["sbrs"]),
                    dominant_factor=row["dominant_factor"],
                )
                for _, row in sbrs_df.iterrows()
            }
            pdf_bytes = report_gen.generate_pdf(
                district_data=df, sbrs_results=sbrs_df,
                projections=projections, recommendations=recs,
            )
            st.download_button(
                label="⬇️ Click here to download the PDF",
                data=pdf_bytes,
                file_name="soil_health_report.pdf",
                mime="application/pdf",
            )
        except Exception as exc:
            st.error(f"Could not generate PDF: {exc}")
            st.download_button(
                label="⬇️ Download as CSV instead",
                data=sbrs_df.to_csv(index=False).encode("utf-8"),
                file_name="soil_scores.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Tab 2 — "Past Trends" (Historical View)
# ---------------------------------------------------------------------------

_HIST_PARAMS = [
    "organic_carbon", "nitrogen_depletion_rate", "rainfall_variability",
    "crop_diversity_index", "irrigation_stress", "fertilizer_load",
]
_HIST_REQUIRED_COLS = _HIST_PARAMS + ["district", "year"]

# Parameters that should be shown on a separate axis due to very different scales
# (e.g. fertilizer_load is 100–200, while organic_carbon is 0.3–0.8)
_LARGE_SCALE_PARAMS = {"fertilizer_load", "nitrogen_depletion_rate"}


def _compute_slope(series: pd.Series) -> float:
    import numpy as np
    y = series.values.astype(float)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    return 0.0 if denom == 0 else float(((x - xm) * (y - ym)).sum() / denom)


def _render_historical_view() -> None:
    ingestion, normalizer, sbrs_engine, _, _, _ = _get_engines()

    # ── How to read this tab ─────────────────────────────────────────────────
    with st.expander("ℹ️ How to use this tab", expanded=False):
        st.markdown(
            """
            **This tab helps you answer: "Has my soil been getting better or worse over the years?"**

            - Upload a file with soil readings from multiple years (needs a 'year' column).
            - Or click **Try with Sample Data** to see real Maharashtra data from 2018–2023.
            - You'll see a score chart, a ranking of which districts are declining fastest,
              and detailed readings for any district you pick.

            > 🌱 **Soil Health Score:** 0 = perfect soil · 100 = soil is failing. A rising line is bad.
            """
        )

    st.markdown("### 📅 See How Soil Health Changed Over the Years")
    st.caption(
        "Upload a file that has soil readings from multiple years. "
        "This will show you whether your district's soil is getting better or worse over time."
    )

    col_up, col_demo = st.columns([3, 1])
    with col_up:
        hist_file = st.file_uploader(
            "Upload multi-year soil data (CSV or Excel — must have a 'year' column)",
            type=["csv", "xlsx", "xls"],
            key="hist_uploader",
            label_visibility="collapsed",
        )
    with col_demo:
        if st.button("🧪 Try with Sample Data", key="load_hist_demo", use_container_width=True):
            _HIST_PATH = os.path.join(_DATA_DIR, "soil_data_historical.csv")
            if os.path.exists(_HIST_PATH):
                hist_demo_df = pd.read_csv(_HIST_PATH)
                st.session_state["hist_df"] = ingestion.clean_data(hist_demo_df)
                n_districts = hist_demo_df["district"].nunique()
                n_years = hist_demo_df["year"].nunique()
                st.success(f"✅ Loaded real Maharashtra data — {n_districts} districts × {n_years} years (2018–2023).")
            else:
                st.error("❌ Sample data file not found. Please upload your own CSV file.")

    if hist_file is not None:
        file_bytes = hist_file.read()
        name = hist_file.name.lower()
        raw_df = pd.read_csv(io.BytesIO(file_bytes)) if name.endswith(".csv") else pd.read_excel(io.BytesIO(file_bytes))
        missing = [c for c in _HIST_REQUIRED_COLS if c not in raw_df.columns]
        if missing:
            st.error(f"❌ File is missing columns: **{', '.join(missing)}**")
            return
        st.session_state["hist_df"] = ingestion.clean_data(raw_df)

    hist_df: pd.DataFrame | None = st.session_state.get("hist_df")
    if hist_df is None:
        st.info("👆 Upload a file or click 'Try with Sample Data' to get started.")
        return

    # Region filter
    if "region" in hist_df.columns:
        regions = ["All Regions"] + sorted(hist_df["region"].dropna().unique().tolist())
        selected_region = st.selectbox("🗺️ Filter by Region:", options=regions, key="hist_region_filter")
        if selected_region != "All Regions":
            hist_df = hist_df[hist_df["region"] == selected_region].copy()

    # Compute SBRS per year
    records = []
    for yr in sorted(hist_df["year"].unique()):
        yr_df = hist_df[hist_df["year"] == yr].copy()
        sbrs_yr = sbrs_engine.compute_sbrs(normalizer.normalize(yr_df))
        sbrs_yr["year"] = yr
        records.append(sbrs_yr)
    hist_sbrs_df = pd.concat(records, ignore_index=True)

    st.divider()

    # ── Soil health over time ────────────────────────────────────────────────
    st.markdown("#### 📈 Soil Health Score Over the Years")
    st.caption(
        "Watch how each district's score changed year by year. "
        "**A rising line means soil is getting worse.** "
        "Crossing the orange or red lines means the district has entered a risk zone."
    )
    fig_tl = px.line(
        hist_sbrs_df, x="year", y="sbrs", color="district",
        labels={"year": "Year", "sbrs": "Soil Health Score (lower = healthier)", "district": "District"},
        title="Soil Health Score by Year",
    )
    fig_tl.add_hline(y=30, line_dash="dot", line_color="orange", annotation_text="Needs Attention (30)")
    fig_tl.add_hline(y=60, line_dash="dot", line_color="red",    annotation_text="Serious Risk (60)")
    fig_tl.add_hline(y=80, line_dash="dash", line_color="darkred", annotation_text="Danger (80)")
    st.plotly_chart(fig_tl, use_container_width=True)

    # ── When did things get bad? ─────────────────────────────────────────────
    st.markdown("#### 🚨 When Did Each District First Enter the Risk Zone?")
    st.caption("This shows the first year each district's soil crossed into a warning level.")
    crossing_rows = []
    for district in hist_sbrs_df["district"].unique():
        d_df = hist_sbrs_df[hist_sbrs_df["district"] == district].sort_values("year")
        for threshold, label in [(30, "🟠 Needs Attention"), (60, "🔴 Serious Risk")]:
            crossed = d_df[d_df["sbrs"] >= threshold]
            if not crossed.empty:
                crossing_rows.append({
                    "District": district,
                    "Risk Level Reached": label,
                    "First Year This Happened": int(crossed.iloc[0]["year"]),
                })
    if crossing_rows:
        st.dataframe(pd.DataFrame(crossing_rows), use_container_width=True, hide_index=True)
    else:
        st.success("✅ No district has crossed into the At Risk zone in this data.")

    st.divider()

    # ── Per-district soil readings ───────────────────────────────────────────
    st.markdown("#### 🔬 Detailed Soil Readings for a Specific District")
    st.caption(
        "Some measurements use very different scales, so they are shown in two separate charts "
        "to make them easier to read."
    )
    all_districts = sorted(hist_sbrs_df["district"].unique().tolist())
    selected = st.selectbox("Pick a district to inspect:", options=all_districts, key="hist_district_select")

    dist_raw = hist_df[hist_df["district"] == selected].sort_values("year")
    if not dist_raw.empty:
        # Split params into two groups: small-scale (0–1 range) and large-scale (100s range)
        small_params = [p for p in _HIST_PARAMS if p in dist_raw.columns and p not in _LARGE_SCALE_PARAMS]
        large_params = [p for p in _HIST_PARAMS if p in dist_raw.columns and p in _LARGE_SCALE_PARAMS]

        if small_params:
            fig_small = go.Figure()
            for param in small_params:
                fig_small.add_trace(go.Scatter(
                    x=dist_raw["year"], y=dist_raw[param],
                    mode="lines+markers",
                    name=PARAM_LABELS.get(param, param),
                ))
            fig_small.update_layout(
                title=f"Soil Readings Over Time — {selected} (small-scale measures)",
                xaxis_title="Year", yaxis_title="Value (0–1 scale)",
                legend_title="What We Measured",
            )
            st.plotly_chart(fig_small, use_container_width=True)

        if large_params:
            fig_large = go.Figure()
            for param in large_params:
                fig_large.add_trace(go.Scatter(
                    x=dist_raw["year"], y=dist_raw[param],
                    mode="lines+markers",
                    name=PARAM_LABELS.get(param, param),
                ))
            fig_large.update_layout(
                title=f"Soil Readings Over Time — {selected} (large-scale measures)",
                xaxis_title="Year", yaxis_title="Value (kg/ha or kg/ha/yr)",
                legend_title="What We Measured",
            )
            st.plotly_chart(fig_large, use_container_width=True)

    st.divider()

    # ── District ranking ─────────────────────────────────────────────────────
    st.markdown("#### 🏆 Which Districts Are Getting Worse the Fastest?")
    st.caption("Sorted from most deteriorating to most improving. The 'Trend' column tells you in plain words.")
    comp_rows = []
    for district in all_districts:
        d_sbrs = hist_sbrs_df[hist_sbrs_df["district"] == district].sort_values("year")
        slope = _compute_slope(d_sbrs["sbrs"])
        latest = float(d_sbrs.iloc[-1]["sbrs"]) if not d_sbrs.empty else 0.0
        comp_rows.append({
            "District": district,
            "Latest Score": round(latest, 1),
            "Status": _risk_badge(d_sbrs.iloc[-1]["risk_category"] if not d_sbrs.empty else "Healthy"),
            "Trend": _trend_label(slope),
        })
    comp_df = pd.DataFrame(comp_rows).sort_values("Latest Score", ascending=False)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3 — "🌦️ Climate Patterns" (Seasonal Climate View)
# ---------------------------------------------------------------------------

_CLIMATE_PATH = os.path.join(_DATA_DIR, "climate_seasonal.csv")

_SEASON_MAP = {
    "Winter (Dec–Feb)":       [12, 1, 2],
    "Summer (Mar–May)":       [3, 4, 5],
    "Monsoon (Jun–Sep)":      [6, 7, 8, 9],
    "Post-Monsoon (Oct–Nov)": [10, 11],
}

_DIR_ARROW = {
    "NE":    "↗ North-East (dry, cool)",
    "NE-W":  "↗→← NE shifting to West",
    "W":     "← West",
    "SW":    "↙ South-West (brings monsoon rains)",
    "SW-SE": "↙→↘ SW shifting to South-East",
    "SE":    "↘ South-East",
}

# Friendly axis labels for the year-on-year trend chart
_TREND_METRIC_LABELS: dict[str, str] = {
    "temp_max_c":        "Peak Temperature (°C)",
    "heatwave_days":     "Heatwave Days",
    "rainfall_mm":       "Rainfall (mm)",
    "wind_speed_kmh":    "Wind Speed (km/h)",
    "cloud_cover_oktas": "Cloud Cover (0 = clear, 8 = fully cloudy)",
}


@st.cache_data(show_spinner=False)
def _load_climate() -> pd.DataFrame | None:
    if not os.path.exists(_CLIMATE_PATH):
        return None
    return pd.read_csv(_CLIMATE_PATH)


def _render_climate_view() -> None:
    climate_df = _load_climate()
    if climate_df is None:
        st.error(f"❌ Climate data file not found. Expected: {_CLIMATE_PATH}")
        return

    # ── How to read this tab ─────────────────────────────────────────────────
    with st.expander("ℹ️ How to use this tab", expanded=False):
        st.markdown(
            """
            **This tab helps you answer: "What is the weather like in my district across the year?"**

            - Pick your district and year, then filter by season if you want.
            - You'll see heatwave days, temperature ranges, rainfall, wind, and cloud cover.
            - Scroll down to compare all districts side-by-side on a heatwave map.

            > ☁️ **Cloud Cover** is measured in oktas: 0 = completely clear sky, 8 = sky fully covered by clouds.
            > 🔥 **Heatwave Days** = days when temperature crossed 40°C.
            """
        )

    st.markdown("### 🌦️ Seasonal Climate Patterns — Maharashtra Districts")
    st.caption(
        "Explore heatwaves, rainfall, wind speed, and cloud cover for any district across the year. "
        "Data covers 2018–2023 with real Maharashtra climate zone patterns."
    )

    # ── Controls ─────────────────────────────────────────────────────────────
    col_d, col_y, col_s = st.columns([2, 1, 2])
    with col_d:
        all_districts = sorted(climate_df["district"].unique().tolist())
        sel_district = st.selectbox("📍 Pick a district:", all_districts, key="clim_district")
    with col_y:
        all_years = sorted(climate_df["year"].unique().tolist())
        sel_year = st.selectbox("📅 Year:", all_years, key="clim_year")
    with col_s:
        season_filter = st.selectbox(
            "🗓️ Filter by season (or All Year):",
            ["All Year"] + list(_SEASON_MAP.keys()),
            key="clim_season",
        )

    dist_df = climate_df[
        (climate_df["district"] == sel_district) &
        (climate_df["year"] == sel_year)
    ].sort_values("month").copy()

    if season_filter != "All Year":
        months_in_season = _SEASON_MAP[season_filter]
        dist_df = dist_df[dist_df["month"].isin(months_in_season)]

    if dist_df.empty:
        st.warning("No data for this selection.")
        return

    # ── Summary cards ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📊 At a Glance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌡️ Peak Temp", f"{dist_df['temp_max_c'].max():.0f}°C",
              help="Highest daily max temperature in the selected period")
    c2.metric("🔥 Heatwave Days", f"{dist_df['heatwave_days'].sum():.0f}",
              help="Total days when temperature crossed 40°C — dangerous for crops and outdoor work")
    c3.metric("🌧️ Total Rainfall", f"{dist_df['rainfall_mm'].sum():.0f} mm",
              help="Total rainfall in the selected period")
    c4.metric("💨 Avg Wind", f"{dist_df['wind_speed_kmh'].mean():.0f} km/h",
              help="Average wind speed")
    c5.metric(
        "☁️ Avg Cloud Cover",
        f"{dist_df['cloud_cover_oktas'].mean():.1f} / 8",
        help="Cloud cover in oktas: 0 = completely clear sky, 8 = sky fully covered by clouds",
    )

    st.divider()

    # ── Heatwave chart ────────────────────────────────────────────────────────
    st.markdown("#### 🔥 Heatwave Days per Month")
    st.caption("Days when temperature crossed 40°C — dangerous for crops and farmers working outdoors.")
    fig_hw = px.bar(
        dist_df, x="month_name", y="heatwave_days",
        color="heatwave_days",
        color_continuous_scale=["#ffe0b2", "#ff6f00", "#b71c1c"],
        labels={"month_name": "Month", "heatwave_days": "Heatwave Days (temp ≥ 40°C)"},
        title=f"Heatwave Days — {sel_district} ({sel_year})",
    )
    fig_hw.update_layout(coloraxis_showscale=False, xaxis={"categoryorder": "array",
        "categoryarray": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]})
    st.plotly_chart(fig_hw, use_container_width=True)

    # ── Temperature band chart ────────────────────────────────────────────────
    st.markdown("#### 🌡️ Temperature Range per Month")
    st.caption("The shaded band shows the gap between the coldest nights and hottest days each month.")
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=dist_df["month_name"], y=dist_df["temp_max_c"],
        mode="lines+markers", name="Hottest Day (°C)", line=dict(color="#e53935", width=2),
    ))
    fig_temp.add_trace(go.Scatter(
        x=dist_df["month_name"], y=dist_df["temp_mean_c"],
        mode="lines+markers", name="Average (°C)", line=dict(color="#fb8c00", width=2, dash="dot"),
    ))
    fig_temp.add_trace(go.Scatter(
        x=dist_df["month_name"], y=dist_df["temp_min_c"],
        mode="lines+markers", name="Coldest Night (°C)", line=dict(color="#1e88e5", width=2),
        fill="tonexty", fillcolor="rgba(30,136,229,0.08)",
    ))
    fig_temp.add_hline(y=40, line_dash="dash", line_color="red",
                       annotation_text="⚠️ 40°C — heatwave threshold")
    fig_temp.update_layout(
        title=f"Temperature Range — {sel_district} ({sel_year})",
        xaxis_title="Month", yaxis_title="Temperature (°C)",
        xaxis={"categoryorder": "array",
               "categoryarray": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]},
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    st.divider()

    # ── Rainfall chart ────────────────────────────────────────────────────────
    st.markdown("#### 🌧️ Rainfall Pattern")
    st.caption("How much rain fell each month (bars) and how many days it rained (line).")
    fig_rain = go.Figure()
    fig_rain.add_trace(go.Bar(
        x=dist_df["month_name"], y=dist_df["rainfall_mm"],
        name="Rainfall (mm)", marker_color="#1565c0", opacity=0.8,
    ))
    fig_rain.add_trace(go.Scatter(
        x=dist_df["month_name"], y=dist_df["rainy_days"],
        name="Rainy Days", mode="lines+markers",
        line=dict(color="#00acc1", width=2), yaxis="y2",
    ))
    fig_rain.update_layout(
        title=f"Rainfall — {sel_district} ({sel_year})",
        xaxis_title="Month",
        yaxis=dict(title="Rainfall (mm)", side="left"),
        yaxis2=dict(title="Number of Rainy Days", side="right", overlaying="y", range=[0, 31]),
        legend=dict(x=0.01, y=0.99),
        xaxis={"categoryorder": "array",
               "categoryarray": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]},
    )
    st.plotly_chart(fig_rain, use_container_width=True)

    st.divider()

    # ── Wind & Cloud chart ────────────────────────────────────────────────────
    st.markdown("#### 💨 Wind Speed & ☁️ Cloud Cover")
    st.caption(
        "Wind speed peaks during the monsoon (Jun–Sep) as south-west winds bring rain. "
        "Cloud cover goes from **0 (completely clear sky)** to **8 (sky fully covered by clouds)**."
    )
    fig_wc = go.Figure()
    fig_wc.add_trace(go.Bar(
        x=dist_df["month_name"], y=dist_df["wind_speed_kmh"],
        name="Wind Speed (km/h)", marker_color="#7b1fa2", opacity=0.7,
    ))
    fig_wc.add_trace(go.Scatter(
        x=dist_df["month_name"], y=dist_df["cloud_cover_oktas"],
        name="Cloud Cover (0 = clear, 8 = fully cloudy)", mode="lines+markers",
        line=dict(color="#546e7a", width=2), yaxis="y2",
    ))
    fig_wc.update_layout(
        title=f"Wind & Cloud Cover — {sel_district} ({sel_year})",
        xaxis_title="Month",
        yaxis=dict(title="Wind Speed (km/h)", side="left"),
        yaxis2=dict(title="Cloud Cover (0 = clear sky → 8 = fully cloudy)", side="right",
                    overlaying="y", range=[0, 8.5]),
        legend=dict(x=0.01, y=0.99),
        xaxis={"categoryorder": "array",
               "categoryarray": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]},
    )
    st.plotly_chart(fig_wc, use_container_width=True)

    # ── Cloud movement direction table ────────────────────────────────────────
    st.markdown("#### 🌬️ Wind & Cloud Movement Direction by Month")
    st.caption(
        "Shows which direction winds and clouds come from each month. "
        "**South-West (SW) winds bring the monsoon rains.** "
        "North-East (NE) winds bring dry, cool winters."
    )
    dir_rows = []
    for _, row in dist_df.iterrows():
        raw_dir = row["cloud_movement_dir"]
        dir_rows.append({
            "Month": row["month_name"],
            "Wind / Cloud Direction": _DIR_ARROW.get(raw_dir, raw_dir),
            "Season": (
                "❄️ Winter" if row["month"] in [12,1,2] else
                "☀️ Summer" if row["month"] in [3,4,5] else
                "🌧️ Monsoon" if row["month"] in [6,7,8,9] else
                "🍂 Post-Monsoon"
            ),
        })
    st.dataframe(pd.DataFrame(dir_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Multi-year trend for selected district ────────────────────────────────
    st.markdown("#### 📈 How Has Climate Changed Year-on-Year?")
    st.caption("Compare the same season across different years to see if things are getting hotter or drier.")

    trend_metric = st.selectbox(
        "What do you want to track?",
        options=list(_TREND_METRIC_LABELS.keys()),
        format_func=lambda x: {
            "temp_max_c":        "🌡️ Peak Temperature (°C)",
            "heatwave_days":     "🔥 Heatwave Days",
            "rainfall_mm":       "🌧️ Rainfall (mm)",
            "wind_speed_kmh":    "💨 Wind Speed (km/h)",
            "cloud_cover_oktas": "☁️ Cloud Cover (0 = clear, 8 = fully cloudy)",
        }[x],
        key="clim_trend_metric",
    )

    trend_df = climate_df[climate_df["district"] == sel_district].copy()
    if season_filter != "All Year":
        trend_df = trend_df[trend_df["month"].isin(_SEASON_MAP[season_filter])]
    trend_agg = trend_df.groupby("year")[trend_metric].mean().reset_index()
    trend_agg[trend_metric] = trend_agg[trend_metric].round(2)

    friendly_label = _TREND_METRIC_LABELS.get(trend_metric, trend_metric)
    fig_trend = px.line(
        trend_agg, x="year", y=trend_metric,
        markers=True,
        labels={"year": "Year", trend_metric: friendly_label},
        title=f"Year-on-Year Trend: {friendly_label} — {sel_district}",
    )
    fig_trend.update_traces(line=dict(width=2.5))
    st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()

    # ── Region comparison heatmap ─────────────────────────────────────────────
    st.markdown("#### 🗺️ Compare All Districts — Heatwave Days by Month")
    st.caption(
        f"Which districts suffer the most heatwave days? (Year: {sel_year}) "
        "Darker red = more days above 40°C."
    )

    hw_pivot = climate_df[climate_df["year"] == sel_year].pivot_table(
        index="district", columns="month_name", values="heatwave_days", aggfunc="sum"
    ).reindex(columns=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])

    fig_hm = px.imshow(
        hw_pivot,
        color_continuous_scale=["#ffffff", "#ffe0b2", "#ff6f00", "#b71c1c"],
        labels=dict(x="Month", y="District", color="Heatwave Days (temp ≥ 40°C)"),
        title=f"Heatwave Days — All Districts ({sel_year})",
        aspect="auto",
    )
    fig_hm.update_layout(height=700)
    st.plotly_chart(fig_hm, use_container_width=True)


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Soil Health Predictor",
        page_icon="🌱",
        layout="wide",
    )

    # Header
    st.markdown(
        """
        <div style='text-align:center; padding: 10px 0 4px 0'>
            <h1 style='font-size:2.2rem; margin-bottom:0'>🌱 Soil Health Predictor</h1>
            <p style='color:gray; font-size:1rem; margin-top:4px'>
                Know your soil's health today — and what will happen in the future
            </p>
            <p style='color:#c0392b; font-size:0.9rem; margin-top:2px'>
                ⚠️ <b>Soil Bankruptcy</b> means your soil has lost so much fertility that it can no longer
                support crops — like a bank account that has run out of money.
                This tool helps you see how close your soil is to that point.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "🔮 Check My District & Future Prediction",
        "📅 See Past Trends (Historical Data)",
        "🌦️ Climate Patterns",
    ])

    with tab1:
        _render_future_view()

    with tab2:
        _render_historical_view()

    with tab3:
        _render_climate_view()


if __name__ == "__main__":
    main()
