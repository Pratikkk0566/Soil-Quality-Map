"""SBRSEngine — computes the Soil Bankruptcy Risk Score (SBRS).

What is SBRS?
  A single number from 0 to 100 that summarises how close a district's soil
  is to "bankruptcy" (irreversible degradation).
    0–29  = Healthy       (green)
   30–59  = At Risk       (orange)
   60–79  = Critical      (red)
   80–100 = Imminent Collapse (dark red)

How is it calculated?
  SBRS = weighted sum of normalised soil parameters × 100
  Weights (must sum to 1.0):
    organic_carbon        30%  — most important: carbon is the foundation of soil health
    irrigation_stress     25%  — overuse of groundwater degrades soil structure
    fertilizer_load       20%  — excess chemicals kill soil microbiome
    crop_diversity_index  15%  — monocultures accelerate degradation
    rainfall_variability  10%  — unpredictable rain stresses soil moisture balance

Usage:
  engine = SBRSEngine()
  sbrs_df = engine.compute_sbrs(normalized_df)
  # Returns: district, sbrs, risk_category, dominant_factor, component_scores
"""

from __future__ import annotations

import pandas as pd


class SBRSEngine:
    """Compute the Soil Bankruptcy Risk Score (SBRS) using a weighted sum formula."""

    WEIGHTS: dict[str, float] = {
        "organic_carbon_norm": 0.30,
        "irrigation_stress_norm": 0.25,
        "fertilizer_load_norm": 0.20,
        "crop_diversity_index_norm": 0.15,
        "rainfall_variability_norm": 0.10,
    }

    def compute_sbrs(self, normalized_df: pd.DataFrame) -> pd.DataFrame:
        """Compute SBRS score (0–100) for each district row.

        Args:
            normalized_df: DataFrame with _norm columns in [0, 1] and a 'district' column.

        Returns:
            New DataFrame with columns: district, sbrs, risk_category,
            dominant_factor, component_scores.
        """
        results = []

        for _, row in normalized_df.iterrows():
            component_scores: dict[str, float] = {}
            weighted_sum = 0.0

            for param, weight in self.WEIGHTS.items():
                contribution = weight * row[param]
                component_scores[param] = contribution
                weighted_sum += contribution

            sbrs = weighted_sum * 100.0
            risk_category = self.classify_risk(sbrs)
            dominant_factor = self.get_dominant_risk_factor(row)

            results.append(
                {
                    "district": row["district"],
                    "sbrs": sbrs,
                    "risk_category": risk_category,
                    "dominant_factor": dominant_factor,
                    "component_scores": component_scores,
                }
            )

        return pd.DataFrame(results)

    def classify_risk(self, score: float) -> str:
        """Map SBRS score to a risk category.

        Args:
            score: SBRS value in [0, 100].

        Returns:
            One of: "Healthy", "At Risk", "Critical", "Imminent Collapse".
        """
        if score < 30:
            return "Healthy"
        if score < 60:
            return "At Risk"
        if score < 80:
            return "Critical"
        return "Imminent Collapse"

    def get_dominant_risk_factor(self, row: pd.Series) -> str:
        """Return the parameter with the highest weighted contribution.

        Args:
            row: A single district row containing all _norm columns.

        Returns:
            The _norm column name whose weight × value is highest.
        """
        return max(self.WEIGHTS, key=lambda param: self.WEIGHTS[param] * row[param])
