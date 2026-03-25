# Data Directory

This folder holds all preloaded datasets used by the Soil Health Predictor app.
**Do not rename or move these files** — the app expects them at these exact paths.

---

## `soil_data.csv`
Current-year (2023) soil parameters for all 36 Maharashtra districts.
Used by the **"Check My District"** tab for the map overlay and pin-drop lookup.

| Column | What it means | Unit / Range |
|---|---|---|
| `district` | District name | text |
| `organic_carbon` | Organic matter in topsoil — higher is healthier | % (0–5) |
| `nitrogen_depletion_rate` | Nitrogen lost from soil per year — lower is better | kg/ha/yr |
| `rainfall_variability` | How unpredictable the rainfall is — lower is better | 0–1 |
| `crop_diversity_index` | Variety of crops grown — higher is better | 0–1 |
| `irrigation_stress` | Groundwater overuse — lower is better | 0–1 |
| `fertilizer_load` | Chemical fertilizer applied — lower is better | kg/ha/yr |
| `year` | Data year | 2023 |

---

## `soil_data_historical.csv`
Multi-year soil data (2018–2023) for all 36 districts.
Used by the **"See Past Trends"** tab to show how soil health changed over time.

Same columns as `soil_data.csv`, plus:
- `region` — Maharashtra region (Marathwada, Vidarbha, Konkan, etc.) for filtering

---

## `climate_seasonal.csv`
Monthly climate data (2018–2023) for all 36 districts.
Used by the **"Climate Patterns"** tab for heatwave, rainfall, wind, and cloud charts.

| Column | What it means |
|---|---|
| `month` / `month_name` | Month number (1–12) and name (Jan–Dec) |
| `temp_mean_c` / `temp_max_c` / `temp_min_c` | Average / hottest / coldest temperature (°C) |
| `heatwave_days` | Days in the month where temperature reached ≥ 40°C |
| `rainfall_mm` | Total rainfall for the month (mm) |
| `rainy_days` | Number of days it rained |
| `wind_speed_kmh` | Average wind speed (km/h) |
| `cloud_cover_oktas` | Cloud cover on a 0–8 scale (0 = clear sky, 8 = fully overcast) |
| `cloud_movement_dir` | Prevailing wind/cloud direction (NE in winter, SW during monsoon) |

---

## `districts.geojson`
GeoJSON polygon boundaries for all 36 Maharashtra districts (Census 2011).
Used by the map to draw district shapes and for pin-drop reverse geocoding.

Each feature must have a `district` property matching the names in `soil_data.csv`.

---

## `demo_upload.csv`
A sample 12-district CSV you can upload to test the "Upload Your Data" feature.
Contains the same columns as `soil_data.csv`.
