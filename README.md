# 🌱 Soil Health Predictor

A farmer-friendly web app that predicts soil degradation risk across Maharashtra districts,
shows historical trends, and gives actionable recovery recommendations.

---

## What it does

| Tab | What you can do |
|---|---|
| 🔮 Check My District | Click any district on the map to see its soil health score. Upload your own data to get future predictions and a downloadable PDF report. |
| 📅 See Past Trends | See how soil health changed from 2018 to 2023. Filter by region. Find out when districts first entered the danger zone. |
| 🌦️ Climate Patterns | Explore monthly heatwave days, rainfall, wind speed, and cloud cover for any district. Compare all districts in a heatmap. |

---

## How to run

```bash
streamlit run src/app.py
```

Requires Python 3.9+. Install dependencies first:

```bash
pip install -r requirements.txt
```

---

## Project structure

```
src/
  app.py                  — Streamlit UI (all three tabs)
  models.py               — Shared data classes and exceptions
  data_ingestion.py       — File loading, validation, and cleaning
  normalization_engine.py — Scales raw soil values to 0–1 risk scores
  sbrs_engine.py          — Computes the Soil Bankruptcy Risk Score (0–100)
  simulation_engine.py    — Projects soil health into the future
  recommendation_engine.py — Generates farming intervention advice
  geo_lookup_engine.py    — Resolves map clicks to district soil data
  report_generator.py     — Produces downloadable PDF reports

data/
  soil_data.csv           — 2023 baseline soil data, 36 districts
  soil_data_historical.csv — 2018–2023 soil data, 36 districts × 6 years
  climate_seasonal.csv    — Monthly climate data, 36 districts × 6 years × 12 months
  districts.geojson       — Maharashtra district boundary polygons
  demo_upload.csv         — Sample file for testing the upload feature
  README.md               — Detailed column descriptions for all data files

tests/
  test_checkpoint_1.py    — Unit tests for core engines
  test_checkpoint_2.py    — Integration and edge-case tests
```

---

## Understanding the Soil Bankruptcy Risk Score (SBRS)

| Score | Status | What it means |
|---|---|---|
| 0–29 | 🟢 Healthy | Soil is in good condition |
| 30–59 | 🟠 At Risk | Soil needs attention — act within 5 years |
| 60–79 | 🔴 Critical | Serious degradation — act urgently |
| 80–100 | 🔵 Imminent Collapse | Soil may become unproductive within 1–3 years |

The score is a weighted combination of six soil parameters:
- **Organic Carbon** (30%) — the most important indicator of soil health
- **Irrigation Stress** (25%) — groundwater overuse damages soil structure
- **Fertilizer Load** (20%) — excess chemicals kill the soil microbiome
- **Crop Diversity** (15%) — monocultures accelerate degradation
- **Rainfall Variability** (10%) — unpredictable rain stresses soil moisture

---

## Running tests

```bash
python -m pytest tests/ -q
```

All 69 tests should pass.
