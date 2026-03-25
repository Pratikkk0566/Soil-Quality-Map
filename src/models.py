"""Data models and custom exceptions for the Soil Bankruptcy Predictor.

This file defines all shared data structures used across the project.
Think of it as the "vocabulary" — every other module imports from here.

Key dataclasses:
  GeoPoint        — a lat/lng coordinate (from a map pin-drop)
  DistrictRecord  — all soil parameters for one district (from the CSV)
  LocationResult  — result of a pin-drop lookup (district + soil data)
  SBRSResult      — full SBRS output for one district
  ProjectionRecord — one row in a multi-year simulation output

Custom exceptions:
  DataQualityError     — raised when uploaded data is unrecoverable
  DistrictNotFoundError — raised when a map click can't be matched to a district
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Geo models
# ---------------------------------------------------------------------------

@dataclass
class GeoPoint:
    lat: float  # Latitude in decimal degrees (-90 to 90)
    lng: float  # Longitude in decimal degrees (-180 to 180)


@dataclass
class DistrictRecord:
    district: str                  # District name (unique identifier)
    organic_carbon: float          # % organic carbon in topsoil (0.0–5.0)
    nitrogen_depletion_rate: float # kg N/ha/year lost (0–200)
    rainfall_variability: float    # Coefficient of variation of annual rainfall (0–1)
    crop_diversity_index: float    # Simpson's diversity index (0–1, higher = more diverse)
    irrigation_stress: float       # Groundwater extraction ratio (0–1)
    fertilizer_load: float         # kg NPK/ha/year (0–500)
    year: int                      # Data collection year


@dataclass
class LocationResult:
    point: GeoPoint                    # The original pin-drop coordinate
    district: str                      # Resolved district name
    district_record: DistrictRecord    # Preloaded soil parameters for the district
    match_method: str                  # "point_in_polygon" | "nearest_centroid"


# ---------------------------------------------------------------------------
# SBRS models
# ---------------------------------------------------------------------------

@dataclass
class SBRSResult:
    district: str
    sbrs: float                        # 0–100
    risk_category: str                 # "Healthy" | "At Risk" | "Critical" | "Imminent Collapse"
    dominant_factor: str               # Parameter with highest weighted contribution
    component_scores: dict             # Per-parameter weighted contributions
    bankruptcy_year: int | None        # Projected year of collapse, or None


@dataclass
class ProjectionRecord:
    district: str
    year: int
    organic_carbon: float
    irrigation_stress: float
    fertilizer_load: float
    crop_diversity_index: float
    rainfall_variability: float
    sbrs: float
    risk_category: str


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DataQualityError(Exception):
    """Raised when a dataset has unrecoverable quality issues (e.g. all-null column)."""


class DistrictNotFoundError(Exception):
    """Raised when a coordinate cannot be resolved to any known district."""
