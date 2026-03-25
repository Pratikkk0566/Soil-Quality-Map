"""RecommendationEngine — generates actionable farming advice based on SBRS results.

What it does:
  Given a district's dominant risk factor (the parameter contributing most to
  its SBRS score), it returns a prioritised list of practical interventions
  a farmer or agriculture officer can take.

It also provides quantified impact estimates for each intervention — used by
the simulation engine to calculate how many years it would take to recover.

Supported interventions and their estimated annual improvements:
  drip_irrigation       — reduces irrigation stress by ~8% per year
  cover_cropping        — improves organic carbon, reduces fertilizer need
  legume_rotation       — nitrogen fixation reduces synthetic fertilizer load
  reduced_fertilizer    — direct 20% reduction in chemical load per year
  rainwater_harvesting  — reduces groundwater dependency
  crop_diversification  — directly improves crop diversity index

Usage:
  engine = RecommendationEngine()
  recs = engine.generate(district="Latur", sbrs=84.2, dominant_factor="organic_carbon_norm")
  impact = engine.get_intervention_impact("drip_irrigation")
"""

from __future__ import annotations


class RecommendationEngine:
    """Generate prioritized intervention recommendations and quantified impact estimates."""

    # Curated recommendations per dominant risk factor (_norm column name)
    _RECOMMENDATIONS: dict[str, list[str]] = {
        "organic_carbon_norm": [
            "Apply compost at 5 tonnes/ha annually to rebuild organic matter and restore carbon levels",
            "Introduce cover cropping with legumes between main crop cycles to add 0.3% organic carbon per year",
            "Reduce tillage frequency to minimum-till or no-till to prevent carbon oxidation losses",
            "Incorporate crop residues back into soil instead of burning to retain 0.2% carbon per season",
        ],
        "irrigation_stress_norm": [
            "Switch to drip irrigation to reduce groundwater extraction by 30% while maintaining yield",
            "Install soil moisture sensors to irrigate only when soil moisture falls below field capacity threshold",
            "Adopt rainwater harvesting structures (farm ponds, check dams) to reduce dependence on groundwater by 25%",
            "Schedule irrigation during early morning hours to cut evaporation losses by up to 20%",
        ],
        "fertilizer_load_norm": [
            "Conduct soil nutrient testing annually and apply fertilizers only at recommended doses to cut excess load by 20%",
            "Introduce legume rotation (chickpea, lentil) to fix 80–120 kg N/ha/year and reduce synthetic nitrogen use",
            "Replace 30% of synthetic fertilizer with vermicompost or farmyard manure to lower chemical load",
            "Adopt split fertilizer application (3–4 doses per season) to improve uptake efficiency and reduce runoff",
        ],
        "crop_diversity_index_norm": [
            "Introduce at least 3 crop species in rotation annually to raise Simpson's diversity index by 0.15",
            "Intercrop cereals with legumes (e.g., sorghum + cowpea) to improve biodiversity and soil nitrogen",
            "Establish agroforestry strips with native trees to diversify the farming system and improve microclimate",
            "Adopt polyculture vegetable plots on 20% of farm area to increase crop diversity index by 0.10 per year",
        ],
        "rainfall_variability_norm": [
            "Construct farm-level water storage (percolation tanks, farm ponds) to buffer against rainfall variability",
            "Plant drought-tolerant crop varieties certified for the local agro-climatic zone to reduce yield risk",
            "Install micro-watershed management structures (contour bunds, stone walls) to retain 40% more runoff",
            "Adopt mulching with crop residues or plastic film to conserve soil moisture during dry spells",
        ],
    }

    # Estimated annual parameter improvement deltas per intervention
    # All values are non-negative (positive = improvement toward lower risk)
    _INTERVENTION_IMPACTS: dict[str, dict[str, float]] = {
        "drip_irrigation": {
            "irrigation_stress": 0.08,   # reduces groundwater extraction stress
            "rainfall_variability": 0.01, # marginal improvement in water use stability
        },
        "cover_cropping": {
            "organic_carbon": 0.15,       # adds organic matter per year
            "fertilizer_load": 0.10,      # reduces synthetic fertilizer need
            "crop_diversity_index": 0.05, # increases diversity
        },
        "legume_rotation": {
            "fertilizer_load": 0.15,      # nitrogen fixation reduces synthetic N load
            "organic_carbon": 0.10,       # root biomass adds carbon
            "crop_diversity_index": 0.10, # improves diversity index
        },
        "reduced_fertilizer": {
            "fertilizer_load": 0.20,      # direct reduction in chemical load
            "organic_carbon": 0.05,       # less chemical burn improves carbon retention
        },
        "rainwater_harvesting": {
            "irrigation_stress": 0.10,    # reduces groundwater dependency
            "rainfall_variability": 0.05, # buffers against variability
        },
        "crop_diversification": {
            "crop_diversity_index": 0.15, # direct improvement to diversity index
            "organic_carbon": 0.05,       # diverse root systems improve carbon
            "fertilizer_load": 0.05,      # diverse crops reduce uniform fertilizer demand
        },
    }

    def generate(self, district: str, sbrs: float, dominant_factor: str) -> list[str]:
        """Return a prioritized list of intervention recommendations for a district.

        Args:
            district: Name of the district (used for context; does not filter recommendations).
            sbrs: Current Soil Bankruptcy Risk Score (0–100).
            dominant_factor: The _norm column name of the dominant risk factor.

        Returns:
            A non-empty list of actionable recommendation strings, ordered by priority.
            Falls back to generic recommendations if the dominant_factor is unrecognized.
        """
        recommendations = self._RECOMMENDATIONS.get(dominant_factor)

        if recommendations is None:
            # Fallback: return a generic set covering the most impactful interventions
            recommendations = [
                "Conduct a comprehensive soil health audit to identify the primary degradation driver",
                "Introduce cover cropping to improve organic carbon and reduce fertilizer dependency",
                "Adopt water-efficient irrigation practices to reduce groundwater extraction stress",
                "Diversify crop rotation with at least 3 species to improve soil biodiversity",
            ]

        return list(recommendations)

    def get_intervention_impact(self, intervention: str) -> dict[str, float]:
        """Return estimated annual parameter improvement deltas for a given intervention.

        Args:
            intervention: Intervention identifier string (e.g., "drip_irrigation").

        Returns:
            A dict mapping soil parameter names to non-negative annual improvement deltas.
            Returns an empty dict if the intervention is not recognized.
        """
        return dict(self._INTERVENTION_IMPACTS.get(intervention, {}))
