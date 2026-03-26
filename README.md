<div align="center">

<img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/Folium-77B829?style=for-the-badge&logo=leaflet&logoColor=white"/>
<img src="https://img.shields.io/badge/Tests-69%20passing-brightgreen?style=for-the-badge"/>

# 🌱 Soil Quality Map
### *Predict soil degradation. Protect your harvest.*

A farmer-friendly web application that maps soil health across all **36 Maharashtra districts**, predicts future degradation, surfaces historical trends, and recommends actionable farming interventions — all in plain language, no jargon.

</div>

---

## 🖼️ What It Looks Like

| Tab | Preview |
|-----|---------|
| 🔮 **Check My District** | Interactive choropleth map — click any district to instantly see its soil health score, plain-English verdict, and biggest problem factor |
| 📅 **Past Trends** | Year-by-year soil health charts (2018–2023), district ranking by deterioration speed, and split-scale parameter charts |
| 🌦️ **Climate Patterns** | Monthly heatwave days, temperature bands, rainfall, wind speed, cloud cover, and a full all-district heatmap |

---

## ⚡ Quick Start Guide

```bash
# 1. Clone the repo
git clone https://github.com/Pratikkk0566/Soil-Quality-Map.git
cd Soil-Quality-Map

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run src/app.py
```

> Requires **Python 3.9+**. The app opens at `http://localhost:8501`.

---

## 🧠 How It Works

The core of the app is the **Soil Bankruptcy Risk Score (SBRS)** — a single number from 0 to 100 that tells you how close a district's soil is to becoming unproductive.

```
SBRS = weighted combination of 6 soil parameters
```

| Parameter | Weight | What it measures |
|-----------|--------|-----------------|
| Organic Carbon | 30% | Soil fertility — the most critical indicator |
| Irrigation Stress | 25% | Groundwater overuse damaging soil structure |
| Fertilizer Load | 20% | Excess chemicals killing the soil microbiome |
| Crop Diversity | 15% | Monocultures accelerating degradation |
| Rainfall Variability | 10% | Unpredictable rain stressing soil moisture |

### Score Interpretation

| Score | Status | What it means for a farmer |
|-------|--------|---------------------------|
| 0 – 29 | 🟢 Healthy | Soil is in good shape — keep current practices |
| 30 – 59 | 🟠 Needs Attention | Soil is declining — change practices within 5 years |
| 60 – 79 | 🔴 Serious Risk | Significant degradation — act this season |
| 80 – 100 | 🔵 Danger | Soil may become unproductive within 1–3 years |

> **"Soil Bankruptcy"** = when soil loses so much fertility it can no longer support crops — like a bank account that has run out of money.

---

## 🗂️ Project Structure

```
Soil-Quality-Map/
│
├── src/
│   ├── app.py                   # Streamlit UI — all three tabs
│   ├── models.py                # Shared dataclasses and exceptions
│   ├── data_ingestion.py        # File loading, validation, cleaning
│   ├── normalization_engine.py  # Scales raw soil values to 0–1 risk scores
│   ├── sbrs_engine.py           # Computes the SBRS (0–100)
│   ├── simulation_engine.py     # Projects soil health into the future
│   ├── recommendation_engine.py # Generates farming intervention advice
│   ├── geo_lookup_engine.py     # Resolves map clicks → district soil data
│   └── report_generator.py      # Produces downloadable PDF reports
│
├── data/
│   ├── soil_data.csv            # 2023 baseline — 36 districts
│   ├── soil_data_historical.csv # 2018–2023 — 36 districts × 6 years
│   ├── climate_seasonal.csv     # Monthly climate — 36 districts × 6 years × 12 months
│   ├── districts.geojson        # Real Maharashtra district boundary polygons
│   ├── demo_upload.csv          # Sample file for testing the upload feature
│   └── README.md                # Detailed column descriptions for all data files
│
├── tests/
│   ├── test_checkpoint_1.py     # Unit tests for core engines (32 tests)
│   └── test_checkpoint_2.py     # Integration and edge-case tests (37 tests)
│
├── requirements.txt
└── setup.cfg
```

---

## 🌾 Features

- **Interactive choropleth map** — real Census 2011 Maharashtra district boundaries, coloured by soil risk
- **Pin-drop lookup** — click any point on the map to get that district's full soil profile
- **Future simulation** — linear regression model projects SBRS up to 10 years ahead with actual calendar years on the x-axis
- **Intervention calculator** — pick a farming practice (drip irrigation, cover cropping, etc.) and see how many years it takes to reach healthy soil
- **Historical trend analysis** — split-scale charts so small-range and large-range parameters don't overlap
- **Seasonal climate explorer** — heatwave days, temperature bands, rainfall, wind, cloud cover per month
- **All-district heatmap** — compare every district's heatwave exposure in one view
- **PDF report generator** — downloadable report for banks, agriculture officers, or cooperatives
- **CSV/Excel upload** — bring your own district data for instant analysis

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -q
```

```
69 passed in 0.95s
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `pandas` / `numpy` | Data processing |
| `plotly` | Interactive charts |
| `folium` + `streamlit-folium` | Interactive map (pinned to `0.19.5` / `0.22.0`) |
| `geopandas` + `shapely` | District boundary geometry |
| `scikit-learn` | Linear regression for future simulation |
| `fpdf2` | PDF report generation |
| `openpyxl` | Excel file upload support |

---

## 📍 Data Coverage

- **36 districts** — all of Maharashtra (Census 2011 boundaries)
- **6 years** — 2018 to 2023
- **12 months** — full seasonal climate data per district per year
- **2,592 climate rows** — 36 × 6 × 12

---

<div align="center">
Made with 🌱 for Maharashtra's farmers
</div>
