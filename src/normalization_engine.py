"""NormalizationEngine — converts raw soil measurements to a 0–1 risk scale.

Why normalization?
  Raw soil parameters have very different units and ranges
  (e.g. organic_carbon is 0.1–5%, fertilizer_load is 20–500 kg/ha).
  To combine them into a single score, we scale each one to [0, 1]
  where 0 = best possible (lowest risk) and 1 = worst possible (highest risk).

How it works:
  - Uses fixed reference ranges derived from real Maharashtra district data.
    This means a single-district input still gets a meaningful score
    (without fixed ranges, a single row would always normalize to 0.5).
  - Parameters where higher = better (organic_carbon, crop_diversity_index)
    are inverted so that 1 still means "worst".

Usage:
  normalizer = NormalizationEngine()
  norm_df = normalizer.normalize(soil_df)
  # Adds columns: organic_carbon_norm, nitrogen_depletion_rate_norm, etc.
"""

from __future__ import annotations

import pandas as pd

# Columns to normalize and whether to invert them
# invert=True means higher raw value = lower risk (e.g. organic_carbon)
_COLUMNS_CONFIG: dict[str, bool] = {
    "organic_carbon": True,
    "nitrogen_depletion_rate": False,
    "rainfall_variability": False,
    "crop_diversity_index": True,
    "irrigation_stress": False,
    "fertilizer_load": False,
}

# Fixed reference ranges derived from real Maharashtra district data.
# Using these ensures single-row normalization doesn't collapse to 0.5.
_REF_RANGES: dict[str, tuple[float, float]] = {
    "organic_carbon":          (0.18, 1.20),
    "nitrogen_depletion_rate": (10.0, 100.0),
    "rainfall_variability":    (0.13, 0.60),
    "crop_diversity_index":    (0.07, 0.76),
    "irrigation_stress":       (0.19, 0.95),
    "fertilizer_load":         (45.0, 230.0),
}


class NormalizationEngine:
    """Normalize raw soil parameters to a [0, 1] risk scale.

    0 = best health (lowest risk), 1 = worst health (highest risk).
    """

    def normalize_column(self, series: pd.Series, invert: bool = False, col_name: str = "") -> pd.Series:
        """Min-max normalize a single column to [0, 1].

        Uses fixed reference ranges when the series has only one unique value
        (single-row input) to avoid collapsing to 0.5.
        """
        min_val = series.min()
        max_val = series.max()

        # Use reference ranges for single-row or constant-column inputs
        if min_val == max_val and col_name in _REF_RANGES:
            ref_min, ref_max = _REF_RANGES[col_name]
            min_val, max_val = ref_min, ref_max

        if min_val == max_val:
            norm = pd.Series([0.5] * len(series), index=series.index, dtype=float)
        else:
            norm = (series - min_val) / (max_val - min_val)
            norm = norm.clip(0.0, 1.0)

        if invert:
            norm = 1.0 - norm

        return norm

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all soil parameters and add ``_norm``-suffixed columns."""
        result = df.copy()

        for col, invert in _COLUMNS_CONFIG.items():
            result[f"{col}_norm"] = self.normalize_column(result[col], invert=invert, col_name=col)

        return result
