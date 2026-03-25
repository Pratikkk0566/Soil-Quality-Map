"""SimulationEngine — Projects soil health trajectories into the future.

How it works:
  Each district starts from its current measured soil values.
  Every year, the "decay model" applies realistic annual degradation rates
  (e.g. organic carbon drops, irrigation stress rises).
  SBRS scores are computed at each step using fixed reference ranges so that
  single-district projections are meaningful (not collapsed to 50).

Key classes:
  SimulationEngine  — main class; call simulate() to get a projection DataFrame.

Usage:
  engine = SimulationEngine()
  projections = engine.simulate(soil_df, years=10)
  # Returns a long-format DataFrame: district × year × sbrs × risk_category
"""

from __future__ import annotations

import pandas as pd
import numpy as np

try:
    from src.normalization_engine import NormalizationEngine
    from src.sbrs_engine import SBRSEngine
except ModuleNotFoundError:
    from normalization_engine import NormalizationEngine
    from sbrs_engine import SBRSEngine

# Output columns for the projection DataFrame
_OUTPUT_COLS = [
    "district",
    "year",
    "organic_carbon",
    "nitrogen_depletion_rate",
    "rainfall_variability",
    "crop_diversity_index",
    "irrigation_stress",
    "fertilizer_load",
    "sbrs",
    "risk_category",
]

# Maximum years to simulate when computing recovery time
MAX_RECOVERY_HORIZON = 50


class SimulationEngine:
    """Project soil parameter trajectories and SBRS scores over a multi-year horizon."""

    # Reference min/max for each parameter derived from real Maharashtra data.
    # Used to normalize single-district projections without collapsing to 0.5.
    _REF_RANGES: dict[str, tuple[float, float]] = {
        "organic_carbon":          (0.18, 1.20),   # % — Latur worst to Gadchiroli best
        "nitrogen_depletion_rate": (10.0, 100.0),  # kg/ha/yr
        "rainfall_variability":    (0.13, 0.60),   # 0–1 index
        "crop_diversity_index":    (0.07, 0.76),   # 0–1 index (inverted: higher = better)
        "irrigation_stress":       (0.19, 0.95),   # 0–1 index
        "fertilizer_load":         (45.0, 230.0),  # kg/ha
    }

    def __init__(self) -> None:
        self._normalizer = NormalizationEngine()
        self._sbrs_engine = SBRSEngine()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_decay(self, state: dict[str, float]) -> dict[str, float]:
        """Return a new state dict with one year of decay applied."""
        s = state.copy()
        s["organic_carbon"] = max(0.0, s["organic_carbon"] - 0.05)
        s["crop_diversity_index"] = max(0.0, s["crop_diversity_index"] - 0.02)
        s["irrigation_stress"] = min(1.0, s["irrigation_stress"] + 0.03)
        s["fertilizer_load"] = s["fertilizer_load"] * 1.01
        s["rainfall_variability"] = min(1.0, s["rainfall_variability"] + 0.005)
        return s

    def _sbrs_from_state(self, state: dict[str, float], nitrogen: float) -> tuple[float, str]:
        """Compute SBRS for a single state using fixed reference ranges (avoids single-row 0.5 collapse)."""
        _INVERT = {"organic_carbon", "crop_diversity_index"}
        norm_vals: dict[str, float] = {}
        for param, (lo, hi) in self._REF_RANGES.items():
            val = state.get(param, nitrogen if param == "nitrogen_depletion_rate" else 0.0)
            if hi == lo:
                n = 0.5
            else:
                n = (val - lo) / (hi - lo)
                n = max(0.0, min(1.0, n))
            if param in _INVERT:
                n = 1.0 - n
            norm_vals[f"{param}_norm"] = n

        sbrs = sum(self._sbrs_engine.WEIGHTS[k] * norm_vals[k]
                   for k in self._sbrs_engine.WEIGHTS if k in norm_vals) * 100.0
        sbrs = max(0.0, min(100.0, sbrs))

        if sbrs < 30:
            risk = "Healthy"
        elif sbrs < 60:
            risk = "At Risk"
        elif sbrs < 80:
            risk = "Critical"
        else:
            risk = "Imminent Collapse"
        return round(sbrs, 4), risk

    def _states_to_projection_df(
        self,
        district: str,
        states: list[dict[str, float]],
        years: list[int],
        nitrogen_depletion_rate: float,
    ) -> pd.DataFrame:
        """Convert a list of per-year states into a projection DataFrame for one district."""
        records = []
        for year, state in zip(years, states):
            sbrs, risk_category = self._sbrs_from_state(state, nitrogen_depletion_rate)
            records.append({
                "district": district,
                "year": year,
                "organic_carbon": state["organic_carbon"],
                "nitrogen_depletion_rate": nitrogen_depletion_rate,
                "rainfall_variability": state["rainfall_variability"],
                "crop_diversity_index": state["crop_diversity_index"],
                "irrigation_stress": state["irrigation_stress"],
                "fertilizer_load": state["fertilizer_load"],
                "sbrs": sbrs,
                "risk_category": risk_category,
            })
        return pd.DataFrame(records, columns=_OUTPUT_COLS)

    def _simulate_decay_for_district(
        self, row: pd.Series, years: int
    ) -> pd.DataFrame:
        """Run the decay model for a single district row."""
        state = {
            "organic_carbon": float(row["organic_carbon"]),
            "crop_diversity_index": float(row["crop_diversity_index"]),
            "irrigation_stress": float(row["irrigation_stress"]),
            "fertilizer_load": float(row["fertilizer_load"]),
            "rainfall_variability": float(row["rainfall_variability"]),
        }
        nitrogen = float(row["nitrogen_depletion_rate"])
        district = str(row["district"])

        states: list[dict[str, float]] = []
        year_labels: list[int] = []

        for t in range(years + 1):
            states.append(state.copy())
            year_labels.append(t)
            if t < years:
                state = self._apply_decay(state)

        return self._states_to_projection_df(district, states, year_labels, nitrogen)

    def _simulate_linear_regression_for_district(
        self, district_df: pd.DataFrame, years: int
    ) -> pd.DataFrame | None:
        """Project parameters forward using linear regression on historical data.

        Fits one LinearRegression model per soil parameter, then extrapolates
        forward from the most recent year. Falls back to None if sklearn is
        unavailable or if there are fewer than 2 distinct years of data.

        Args:
            district_df: Historical rows for a single district (must have 'year' column).
            years: Number of years to project forward.

        Returns:
            Projection DataFrame, or None if regression cannot be applied.
        """
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError:
            return None

        if "year" not in district_df.columns or district_df["year"].nunique() < 2:
            return None

        district = str(district_df["district"].iloc[0])
        nitrogen = float(district_df["nitrogen_depletion_rate"].mean())
        base_year = int(district_df["year"].max())

        # Parameters to project (nitrogen is held constant — it's not in the decay model)
        params_to_fit = [
            "organic_carbon", "crop_diversity_index",
            "irrigation_stress", "fertilizer_load", "rainfall_variability",
        ]
        # Physical bounds: (min, max) — None means no bound on that side
        bounds = {
            "organic_carbon":          (0.0, None),
            "crop_diversity_index":    (0.0, 1.0),
            "irrigation_stress":       (0.0, 1.0),
            "fertilizer_load":         (0.0, None),
            "rainfall_variability":    (0.0, 1.0),
        }

        X = district_df["year"].values.reshape(-1, 1)
        models: dict[str, LinearRegression] = {}
        for param in params_to_fit:
            if param not in district_df.columns:
                return None
            lr = LinearRegression()
            lr.fit(X, district_df[param].values)
            models[param] = lr

        states: list[dict[str, float]] = []
        year_labels: list[int] = []
        for t in range(years + 1):
            future_year = base_year + t
            state: dict[str, float] = {}
            for param, lr in models.items():
                val = float(lr.predict([[future_year]])[0])
                lo, hi = bounds.get(param, (None, None))
                if lo is not None:
                    val = max(lo, val)
                if hi is not None:
                    val = min(hi, val)
                state[param] = val
            states.append(state)
            year_labels.append(t)

        return self._states_to_projection_df(district, states, year_labels, nitrogen)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(
        self,
        df: pd.DataFrame,
        years: int = 10,
        model: str = "decay",
    ) -> pd.DataFrame:
        """Simulate future soil state for each district over `years` years.

        Args:
            df: Input DataFrame with current-year soil parameters per district.
            years: Number of years to project (default 10). Output has years+1 rows per district.
            model: Simulation model — "decay" (default) or "linear_regression".

        Returns:
            Long-format DataFrame with columns:
            district, year, organic_carbon, nitrogen_depletion_rate,
            rainfall_variability, crop_diversity_index, irrigation_stress,
            fertilizer_load, sbrs, risk_category.
            Year 0 row matches input df values exactly.
        """
        all_projections: list[pd.DataFrame] = []

        if model == "linear_regression" and "year" in df.columns:
            for district, district_df in df.groupby("district"):
                proj = self._simulate_linear_regression_for_district(district_df, years)
                if proj is not None:
                    all_projections.append(proj)
                else:
                    # Fall back to decay for this district
                    row = district_df.iloc[0]
                    all_projections.append(self._simulate_decay_for_district(row, years))
        else:
            # Default: decay model (also fallback for linear_regression with insufficient data)
            for _, row in df.iterrows():
                all_projections.append(self._simulate_decay_for_district(row, years))

        if not all_projections:
            return pd.DataFrame(columns=_OUTPUT_COLS)

        result = pd.concat(all_projections, ignore_index=True)
        return result

    def predict_bankruptcy_year(self, district_projection: pd.DataFrame) -> int | None:
        """Return the first year where SBRS >= 80, or None if no such year exists.

        Args:
            district_projection: Projection DataFrame for a single district
                                  with columns 'year' and 'sbrs'.

        Returns:
            First year (int) where sbrs >= 80, or None.
        """
        bankruptcy_rows = district_projection[district_projection["sbrs"] >= 80]
        if bankruptcy_rows.empty:
            return None
        return int(bankruptcy_rows["year"].min())

    def compute_recovery_time(
        self,
        current_sbrs: float,
        intervention: dict[str, float],
    ) -> int:
        """Estimate years to reach SBRS < 30 given an intervention scenario.

        Args:
            current_sbrs: Current SBRS value in [0, 100].
            intervention: Mapping of parameter names to annual improvement deltas
                          (positive = improvement, i.e. reduction in risk).

        Returns:
            0 if current_sbrs < 30.
            50 if recovery not achievable within 50 years.
            Otherwise the first year SBRS drops below 30.
        """
        if current_sbrs < 30:
            return 0

        # Work directly in normalized [0, 1] risk space to avoid single-row
        # normalization collapsing all values to 0.5.
        # Start each norm parameter at a value consistent with current_sbrs.
        # Distribute current_sbrs / 100 evenly across all weighted parameters.
        initial_norm = (current_sbrs / 100.0)
        norm_state = {param: initial_norm for param in self._sbrs_engine.WEIGHTS}

        for year in range(1, MAX_RECOVERY_HORIZON + 1):
            # Apply intervention improvements directly in norm space
            for param, delta in intervention.items():
                norm_key = f"{param}_norm" if not param.endswith("_norm") else param
                if norm_key in norm_state:
                    norm_state[norm_key] = max(0.0, norm_state[norm_key] - delta)

            # Compute SBRS directly from weighted norm values (no re-normalization needed)
            sbrs = sum(
                self._sbrs_engine.WEIGHTS[p] * norm_state[p]
                for p in self._sbrs_engine.WEIGHTS
            ) * 100.0

            if sbrs < 30:
                return year

        return MAX_RECOVERY_HORIZON
