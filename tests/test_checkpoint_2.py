"""Checkpoint 2 — Unit tests for SimulationEngine, RecommendationEngine, and ReportGenerator."""

import pandas as pd
import numpy as np
import pytest

from src.simulation_engine import SimulationEngine
from src.recommendation_engine import RecommendationEngine
from src.report_generator import ReportGenerator, sanitize_filename


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_single_district_df(
    district: str = "TestDistrict",
    organic_carbon: float = 2.0,
    nitrogen_depletion_rate: float = 50.0,
    rainfall_variability: float = 0.3,
    crop_diversity_index: float = 0.6,
    irrigation_stress: float = 0.4,
    fertilizer_load: float = 150.0,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "district": district,
        "organic_carbon": organic_carbon,
        "nitrogen_depletion_rate": nitrogen_depletion_rate,
        "rainfall_variability": rainfall_variability,
        "crop_diversity_index": crop_diversity_index,
        "irrigation_stress": irrigation_stress,
        "fertilizer_load": fertilizer_load,
    }])


def _make_multi_district_df(n: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "district": [f"D{i}" for i in range(n)],
        "organic_carbon": rng.uniform(0.5, 3.5, n),
        "nitrogen_depletion_rate": rng.uniform(20.0, 120.0, n),
        "rainfall_variability": rng.uniform(0.1, 0.8, n),
        "crop_diversity_index": rng.uniform(0.1, 0.9, n),
        "irrigation_stress": rng.uniform(0.1, 0.9, n),
        "fertilizer_load": rng.uniform(50.0, 300.0, n),
    })


# ---------------------------------------------------------------------------
# SimulationEngine tests
# ---------------------------------------------------------------------------

class TestSimulateRowCount:
    def test_single_district_returns_years_plus_1_rows(self):
        engine = SimulationEngine()
        df = _make_single_district_df()
        result = engine.simulate(df, years=10)
        assert len(result) == 11  # year 0 through year 10

    def test_multiple_districts_each_returns_years_plus_1_rows(self):
        engine = SimulationEngine()
        n = 4
        df = _make_multi_district_df(n)
        years = 5
        result = engine.simulate(df, years=years)
        assert len(result) == n * (years + 1)

    def test_years_1_returns_2_rows_per_district(self):
        engine = SimulationEngine()
        df = _make_single_district_df()
        result = engine.simulate(df, years=1)
        assert len(result) == 2


class TestSimulateYear0MatchesInput:
    def test_year_0_organic_carbon_matches_input(self):
        engine = SimulationEngine()
        oc_value = 2.75
        df = _make_single_district_df(organic_carbon=oc_value)
        result = engine.simulate(df, years=5)
        year_0 = result[result["year"] == 0].iloc[0]
        assert year_0["organic_carbon"] == pytest.approx(oc_value)

    def test_year_0_all_params_match_input(self):
        engine = SimulationEngine()
        df = _make_single_district_df(
            organic_carbon=1.5,
            nitrogen_depletion_rate=80.0,
            rainfall_variability=0.45,
            crop_diversity_index=0.55,
            irrigation_stress=0.35,
            fertilizer_load=200.0,
        )
        result = engine.simulate(df, years=3)
        year_0 = result[result["year"] == 0].iloc[0]
        assert year_0["organic_carbon"] == pytest.approx(1.5)
        assert year_0["nitrogen_depletion_rate"] == pytest.approx(80.0)
        assert year_0["rainfall_variability"] == pytest.approx(0.45)
        assert year_0["crop_diversity_index"] == pytest.approx(0.55)
        assert year_0["irrigation_stress"] == pytest.approx(0.35)
        assert year_0["fertilizer_load"] == pytest.approx(200.0)


class TestPredictBankruptcyYear:
    def test_returns_none_when_no_year_crosses_80(self):
        engine = SimulationEngine()
        # Build a projection where SBRS never reaches 80
        proj = pd.DataFrame({
            "district": ["D0"] * 5,
            "year": [0, 1, 2, 3, 4],
            "sbrs": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        result = engine.predict_bankruptcy_year(proj)
        assert result is None

    def test_returns_correct_year_when_sbrs_crosses_80(self):
        engine = SimulationEngine()
        proj = pd.DataFrame({
            "district": ["D0"] * 6,
            "year": [0, 1, 2, 3, 4, 5],
            "sbrs": [50.0, 60.0, 70.0, 80.0, 85.0, 90.0],
        })
        result = engine.predict_bankruptcy_year(proj)
        assert result == 3

    def test_returns_first_year_when_multiple_cross_80(self):
        engine = SimulationEngine()
        proj = pd.DataFrame({
            "district": ["D0"] * 5,
            "year": [0, 1, 2, 3, 4],
            "sbrs": [75.0, 82.0, 85.0, 90.0, 95.0],
        })
        result = engine.predict_bankruptcy_year(proj)
        assert result == 1

    def test_exactly_80_counts_as_bankruptcy(self):
        engine = SimulationEngine()
        proj = pd.DataFrame({
            "district": ["D0"] * 3,
            "year": [0, 1, 2],
            "sbrs": [70.0, 80.0, 90.0],
        })
        result = engine.predict_bankruptcy_year(proj)
        assert result == 1


class TestComputeRecoveryTime:
    def test_returns_0_when_current_sbrs_below_30(self):
        engine = SimulationEngine()
        result = engine.compute_recovery_time(current_sbrs=20.0, intervention={})
        assert result == 0

    def test_returns_0_when_current_sbrs_is_0(self):
        engine = SimulationEngine()
        result = engine.compute_recovery_time(current_sbrs=0.0, intervention={})
        assert result == 0

    def test_returns_50_when_unachievable_empty_intervention_high_sbrs(self):
        engine = SimulationEngine()
        # With no intervention and a high SBRS, recovery is not achievable
        result = engine.compute_recovery_time(current_sbrs=90.0, intervention={})
        assert result == 50

    def test_returns_positive_years_with_effective_intervention(self):
        engine = SimulationEngine()
        # A strong intervention should achieve recovery in fewer than 50 years
        # Keys use _norm suffix to match the normalized state space
        intervention = {
            "organic_carbon_norm": 0.5,
            "irrigation_stress_norm": 0.5,
            "fertilizer_load_norm": 0.5,
            "crop_diversity_index_norm": 0.5,
            "rainfall_variability_norm": 0.5,
        }
        result = engine.compute_recovery_time(current_sbrs=50.0, intervention=intervention)
        assert 0 < result < 50


# ---------------------------------------------------------------------------
# RecommendationEngine tests
# ---------------------------------------------------------------------------

class TestGenerateRecommendations:
    _DOMINANT_FACTORS = [
        "organic_carbon_norm",
        "irrigation_stress_norm",
        "fertilizer_load_norm",
        "crop_diversity_index_norm",
        "rainfall_variability_norm",
    ]

    @pytest.mark.parametrize("factor", _DOMINANT_FACTORS)
    def test_generate_returns_non_empty_list_for_each_dominant_factor(self, factor):
        engine = RecommendationEngine()
        result = engine.generate(district="TestDistrict", sbrs=65.0, dominant_factor=factor)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_generate_returns_fallback_for_unknown_dominant_factor(self):
        engine = RecommendationEngine()
        result = engine.generate(
            district="TestDistrict",
            sbrs=50.0,
            dominant_factor="unknown_factor_norm",
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_generate_returns_strings(self):
        engine = RecommendationEngine()
        result = engine.generate("D1", 40.0, "organic_carbon_norm")
        assert all(isinstance(r, str) for r in result)


class TestGetInterventionImpact:
    _INTERVENTIONS = [
        "drip_irrigation",
        "cover_cropping",
        "legume_rotation",
        "reduced_fertilizer",
        "rainwater_harvesting",
        "crop_diversification",
    ]

    @pytest.mark.parametrize("intervention", _INTERVENTIONS)
    def test_returns_dict_with_non_negative_values(self, intervention):
        engine = RecommendationEngine()
        result = engine.get_intervention_impact(intervention)
        assert isinstance(result, dict)
        assert len(result) > 0
        for key, value in result.items():
            assert value >= 0, f"Negative impact value for {key} in {intervention}"

    def test_returns_empty_dict_for_unknown_intervention(self):
        engine = RecommendationEngine()
        result = engine.get_intervention_impact("nonexistent_intervention")
        assert result == {}


# ---------------------------------------------------------------------------
# ReportGenerator tests
# ---------------------------------------------------------------------------

class TestSanitizeFilename:
    def test_removes_path_traversal_dotdot_slash(self):
        result = sanitize_filename("../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_removes_forward_slash(self):
        result = sanitize_filename("district/name")
        assert "/" not in result

    def test_removes_backslash(self):
        result = sanitize_filename("district\\name")
        assert "\\" not in result

    def test_handles_null_bytes(self):
        result = sanitize_filename("safe\x00name")
        assert "\x00" not in result

    def test_safe_name_unchanged(self):
        result = sanitize_filename("safe_name-01")
        assert result == "safe_name-01"

    def test_removes_nested_path_traversal(self):
        result = sanitize_filename("../../etc/shadow")
        assert ".." not in result
        assert "/" not in result

    def test_empty_string_returns_empty(self):
        result = sanitize_filename("")
        assert result == ""


class TestGeneratePdf:
    def _make_sbrs_results(self, district: str = "TestDistrict") -> pd.DataFrame:
        return pd.DataFrame([{
            "district": district,
            "sbrs": 55.0,
            "risk_category": "At Risk",
            "dominant_factor": "organic_carbon_norm",
            "component_scores": {
                "organic_carbon_norm": 0.15,
                "irrigation_stress_norm": 0.10,
                "fertilizer_load_norm": 0.08,
                "crop_diversity_index_norm": 0.07,
                "rainfall_variability_norm": 0.05,
            },
            "bankruptcy_year": None,
        }])

    def _make_projections(self, district: str = "TestDistrict") -> pd.DataFrame:
        return pd.DataFrame({
            "district": [district] * 5,
            "year": [0, 1, 2, 3, 4],
            "sbrs": [55.0, 58.0, 61.0, 65.0, 70.0],
        })

    def test_generate_pdf_returns_bytes(self):
        rg = ReportGenerator()
        district_data = _make_single_district_df()
        sbrs_results = self._make_sbrs_results()
        projections = self._make_projections()
        recommendations = {"TestDistrict": ["Apply compost annually", "Reduce tillage"]}

        result = rg.generate_pdf(district_data, sbrs_results, projections, recommendations)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_generate_pdf_returns_valid_pdf_header(self):
        rg = ReportGenerator()
        district_data = _make_single_district_df()
        sbrs_results = self._make_sbrs_results()
        projections = self._make_projections()
        recommendations = {"TestDistrict": []}

        result = rg.generate_pdf(district_data, sbrs_results, projections, recommendations)
        # PDF files start with %PDF
        assert result[:4] == b"%PDF"

    def test_generate_pdf_with_bankruptcy_year(self):
        rg = ReportGenerator()
        district_data = _make_single_district_df()
        sbrs_results = self._make_sbrs_results()
        sbrs_results.loc[0, "bankruptcy_year"] = 7
        projections = self._make_projections()
        recommendations = {"TestDistrict": ["Test recommendation"]}

        result = rg.generate_pdf(district_data, sbrs_results, projections, recommendations)
        assert isinstance(result, bytes)
        assert len(result) > 0
