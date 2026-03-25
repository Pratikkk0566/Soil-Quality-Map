"""DataIngestion — loads, validates, and cleans district soil datasets.

Responsibilities:
  - Load CSV or Excel files uploaded by the user (up to 10 MB)
  - Check that all required columns are present
  - Fill in missing values using column medians (safe imputation)
  - Generate synthetic test data when no real file is available

Required columns in any uploaded file:
  district, organic_carbon, nitrogen_depletion_rate, rainfall_variability,
  crop_diversity_index, irrigation_stress, fertilizer_load

Usage:
  ingestion = DataIngestion()
  df = ingestion.load_file("my_data.csv")
  valid, missing = ingestion.validate_schema(df)
  df = ingestion.clean_data(df)
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

try:
    from src.models import DataQualityError
except ModuleNotFoundError:
    from models import DataQualityError

# Required columns for a valid district soil dataset
REQUIRED_COLUMNS = [
    "district",
    "organic_carbon",
    "nitrogen_depletion_rate",
    "rainfall_variability",
    "crop_diversity_index",
    "irrigation_stress",
    "fertilizer_load",
]

# Numeric columns that must not be entirely null
NUMERIC_REQUIRED = [
    "organic_carbon",
    "nitrogen_depletion_rate",
    "rainfall_variability",
    "crop_diversity_index",
    "irrigation_stress",
    "fertilizer_load",
]

# 10 MB limit in bytes
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Maharashtra district names for synthetic data
MAHARASHTRA_DISTRICTS = [
    "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed",
    "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli",
    "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur",
    "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded",
    "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani",
    "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara",
    "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim",
    "Yavatmal",
]


class DataIngestion:
    """Load, validate, and clean district-level soil parameter datasets."""

    def load_file(self, file_path: str) -> pd.DataFrame:
        """Load a CSV or Excel file into a DataFrame.

        Args:
            file_path: Path to the CSV or Excel file.

        Returns:
            DataFrame containing the file contents.

        Raises:
            ValueError: If the file exceeds 10 MB or has an unsupported extension.
            FileNotFoundError: If the file does not exist.
        """
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File size {file_size} bytes exceeds the 10 MB limit "
                f"({MAX_FILE_SIZE_BYTES} bytes)."
            )

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(file_path)
        else:
            raise ValueError(
                f"Unsupported file format '{ext}'. Only CSV and Excel files are supported."
            )

    def validate_schema(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """Validate that all required columns are present in the DataFrame.

        Args:
            df: Input DataFrame to validate.

        Returns:
            (True, []) if all required columns are present.
            (False, [list of missing column names]) if any are absent.
        """
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            return False, missing
        return True, []

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing numeric values with column medians.

        Args:
            df: Input DataFrame (should have passed validate_schema).

        Returns:
            Cleaned DataFrame with missing numeric values imputed.

        Raises:
            DataQualityError: If any required numeric column is entirely null.
        """
        df = df.copy()

        # Check for entirely-null required numeric columns
        for col in NUMERIC_REQUIRED:
            if col in df.columns and df[col].isna().all():
                raise DataQualityError(
                    f"Required numeric column '{col}' contains only null values "
                    "and cannot be imputed."
                )

        # Impute missing numeric values with column median
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # Warn if dataset has only one district row
        if len(df) == 1:
            warnings.warn(
                "Single-district dataset — normalization uses neutral values. "
                "Results are indicative only.",
                UserWarning,
                stacklevel=2,
            )

        return df

    def generate_synthetic_dataset(self, n_districts: int = 50) -> pd.DataFrame:
        """Generate a synthetic dataset with valid parameter ranges.

        Args:
            n_districts: Number of district rows to generate (default 50).

        Returns:
            DataFrame with columns matching the required schema and valid ranges.
        """
        rng = np.random.default_rng(seed=42)

        # Build district names: use Maharashtra names first, then generic names
        districts: list[str] = []
        mh_names = MAHARASHTRA_DISTRICTS.copy()
        for i in range(n_districts):
            if i < len(mh_names):
                districts.append(mh_names[i])
            else:
                districts.append(f"District_{i + 1}")

        data = {
            "district": districts,
            "organic_carbon": rng.uniform(0.1, 4.5, n_districts),
            "nitrogen_depletion_rate": rng.uniform(5.0, 180.0, n_districts),
            "rainfall_variability": rng.uniform(0.05, 0.95, n_districts),
            "crop_diversity_index": rng.uniform(0.05, 0.95, n_districts),
            "irrigation_stress": rng.uniform(0.05, 0.95, n_districts),
            "fertilizer_load": rng.uniform(20.0, 450.0, n_districts),
            "year": 2023,
        }

        return pd.DataFrame(data)
