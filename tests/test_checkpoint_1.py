"""Checkpoint 1 — Basic unit tests for DataIngestion, NormalizationEngine, and SBRSEngine."""

import pandas as pd
import numpy as np
import pytest

from src.data_ingestion import DataIngestion, REQUIRED_COLUMNS
from src.normalization_engine import NormalizationEngine
from src.sbrs_engine import SBRSEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_df(n: int = 5) -> pd.DataFrame:
    """Return a minimal valid DataFrame with all required columns."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "district": [f"D{i}" for i in range(n)],
        "organic_carbon": rng.uniform(0.1, 4.5, n),
        "nitrogen_depletion_rate": rng.uniform(5.0, 180.0, n),
        "rainfall_variability": rng.uniform(0.05, 0.95, n),
        "crop_diversity_index": rng.uniform(0.05, 0.95, n),
        "irrigation_stress": rng.uniform(0.05, 0.95, n),
        "fertilizer_load": rng.uniform(20.0, 450.0, n),
    })


def _make_normalized_df(n: int = 5) -> pd.DataFrame:
    """Return a DataFrame with all _norm columns in [0, 1]."""
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "district": [f"D{i}" for i in range(n)],
        "organic_carbon_norm": rng.uniform(0, 1, n),
        "nitrogen_depletion_rate_norm": rng.uniform(0, 1, n),
        "rainfall_variability_norm": rng.uniform(0, 1, n),
        "crop_diversity_index_norm": rng.uniform(0, 1, n),
        "irrigation_stress_norm": rng.uniform(0, 1, n),
        "fertilizer_load_norm": rng.uniform(0, 1, n),
    })


# ---------------------------------------------------------------------------
# DataIngestion tests
# ---------------------------------------------------------------------------

class TestValidateSchema:
    def test_valid_df_returns_true_empty_list(self):
        di = DataIngestion()
        df = _make_valid_df()
        ok, missing = di.validate_schema(df)
        assert ok is True
        assert missing == []

    def test_missing_columns_returns_false_with_names(self):
        di = DataIngestion()
        df = _make_valid_df().drop(columns=["organic_carbon", "fertilizer_load"])
        ok, missing = di.validate_schema(df)
        assert ok is False
        assert "organic_carbon" in missing
        assert "fertilizer_load" in missing

    def test_single_missing_column(self):
        di = DataIngestion()
        df = _make_valid_df().drop(columns=["district"])
        ok, missing = di.validate_schema(df)
        assert ok is False
        assert missing == ["district"]


class TestCleanData:
    def test_imputes_median_for_missing_values(self):
        di = DataIngestion()
        df = _make_valid_df(10)
        # Introduce NaN in organic_carbon
        df.loc[0, "organic_carbon"] = np.nan
        expected_median = df["organic_carbon"].median()
        cleaned = di.clean_data(df)
        assert cleaned.loc[0, "organic_carbon"] == pytest.approx(expected_median)

    def test_no_nulls_after_clean(self):
        di = DataIngestion()
        df = _make_valid_df(10)
        df.loc[2, "fertilizer_load"] = np.nan
        df.loc[4, "rainfall_variability"] = np.nan
        cleaned = di.clean_data(df)
        assert cleaned.isnull().sum().sum() == 0

    def test_original_df_not_mutated(self):
        di = DataIngestion()
        df = _make_valid_df(5)
        df.loc[0, "organic_carbon"] = np.nan
        di.clean_data(df)
        assert np.isnan(df.loc[0, "organic_carbon"])


class TestGenerateSyntheticDataset:
    def test_returns_at_least_50_rows(self):
        di = DataIngestion()
        df = di.generate_synthetic_dataset()
        assert len(df) >= 50

    def test_has_required_columns(self):
        di = DataIngestion()
        df = di.generate_synthetic_dataset()
        for col in REQUIRED_COLUMNS:
            assert col in df.columns

    def test_organic_carbon_range(self):
        di = DataIngestion()
        df = di.generate_synthetic_dataset()
        assert df["organic_carbon"].between(0.0, 5.0).all()

    def test_rainfall_variability_range(self):
        di = DataIngestion()
        df = di.generate_synthetic_dataset()
        assert df["rainfall_variability"].between(0.0, 1.0).all()

    def test_crop_diversity_index_range(self):
        di = DataIngestion()
        df = di.generate_synthetic_dataset()
        assert df["crop_diversity_index"].between(0.0, 1.0).all()

    def test_irrigation_stress_range(self):
        di = DataIngestion()
        df = di.generate_synthetic_dataset()
        assert df["irrigation_stress"].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# NormalizationEngine tests
# ---------------------------------------------------------------------------

class TestNormalizeColumn:
    def test_values_in_0_1(self):
        ne = NormalizationEngine()
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ne.normalize_column(series)
        assert result.between(0.0, 1.0).all()

    def test_min_maps_to_0_max_maps_to_1(self):
        ne = NormalizationEngine()
        series = pd.Series([10.0, 20.0, 30.0])
        result = ne.normalize_column(series)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_invert_flips_values(self):
        ne = NormalizationEngine()
        series = pd.Series([0.0, 0.5, 1.0])
        normal = ne.normalize_column(series, invert=False)
        inverted = ne.normalize_column(series, invert=True)
        assert inverted.tolist() == pytest.approx((1.0 - normal).tolist())

    def test_constant_column_returns_0_5(self):
        ne = NormalizationEngine()
        series = pd.Series([3.0, 3.0, 3.0])
        result = ne.normalize_column(series)
        assert (result == 0.5).all()


class TestNormalizeInversion:
    def test_organic_carbon_inverted(self):
        """Higher organic_carbon → lower risk → norm near 0."""
        ne = NormalizationEngine()
        df = _make_valid_df(10)
        # Force a clear high and low value
        df.loc[0, "organic_carbon"] = 0.1   # low carbon → high risk
        df.loc[1, "organic_carbon"] = 4.5   # high carbon → low risk
        result = ne.normalize(df)
        assert result.loc[0, "organic_carbon_norm"] > result.loc[1, "organic_carbon_norm"]

    def test_crop_diversity_index_inverted(self):
        """Higher crop_diversity_index → lower risk → norm near 0."""
        ne = NormalizationEngine()
        df = _make_valid_df(10)
        df.loc[0, "crop_diversity_index"] = 0.05   # low diversity → high risk
        df.loc[1, "crop_diversity_index"] = 0.95   # high diversity → low risk
        result = ne.normalize(df)
        assert result.loc[0, "crop_diversity_index_norm"] > result.loc[1, "crop_diversity_index_norm"]

    def test_norm_columns_in_0_1(self):
        ne = NormalizationEngine()
        df = _make_valid_df(20)
        result = ne.normalize(df)
        norm_cols = [c for c in result.columns if c.endswith("_norm")]
        for col in norm_cols:
            assert result[col].between(0.0, 1.0).all(), f"{col} out of [0,1]"


# ---------------------------------------------------------------------------
# SBRSEngine tests
# ---------------------------------------------------------------------------

class TestComputeSBRS:
    def test_sbrs_in_0_100(self):
        engine = SBRSEngine()
        df = _make_normalized_df(10)
        result = engine.compute_sbrs(df)
        assert result["sbrs"].between(0.0, 100.0).all()

    def test_output_has_required_columns(self):
        engine = SBRSEngine()
        df = _make_normalized_df(5)
        result = engine.compute_sbrs(df)
        for col in ["district", "sbrs", "risk_category", "dominant_factor", "component_scores"]:
            assert col in result.columns

    def test_all_zeros_gives_sbrs_0(self):
        engine = SBRSEngine()
        df = pd.DataFrame({
            "district": ["D0"],
            **{col: [0.0] for col in SBRSEngine.WEIGHTS},
        })
        result = engine.compute_sbrs(df)
        assert result.loc[0, "sbrs"] == pytest.approx(0.0)

    def test_all_ones_gives_sbrs_100(self):
        engine = SBRSEngine()
        df = pd.DataFrame({
            "district": ["D0"],
            **{col: [1.0] for col in SBRSEngine.WEIGHTS},
        })
        result = engine.compute_sbrs(df)
        assert result.loc[0, "sbrs"] == pytest.approx(100.0)


class TestClassifyRisk:
    def test_score_0_is_healthy(self):
        assert SBRSEngine().classify_risk(0) == "Healthy"

    def test_score_29_is_healthy(self):
        assert SBRSEngine().classify_risk(29.9) == "Healthy"

    def test_score_30_is_at_risk(self):
        assert SBRSEngine().classify_risk(30) == "At Risk"

    def test_score_59_is_at_risk(self):
        assert SBRSEngine().classify_risk(59.9) == "At Risk"

    def test_score_60_is_critical(self):
        assert SBRSEngine().classify_risk(60) == "Critical"

    def test_score_79_is_critical(self):
        assert SBRSEngine().classify_risk(79.9) == "Critical"

    def test_score_80_is_imminent_collapse(self):
        assert SBRSEngine().classify_risk(80) == "Imminent Collapse"

    def test_score_100_is_imminent_collapse(self):
        assert SBRSEngine().classify_risk(100) == "Imminent Collapse"


class TestWeights:
    def test_weights_sum_to_1(self):
        total = sum(SBRSEngine.WEIGHTS.values())
        assert total == pytest.approx(1.0)
