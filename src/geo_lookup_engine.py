"""GeoLookupEngine — resolves a map pin-drop to a district and its soil data.

What it does:
  When a user clicks on the Maharashtra map, this engine:
  1. Checks which district polygon contains the clicked point (point-in-polygon)
  2. If the click is outside all polygons, finds the nearest district centroid
  3. Returns the district name + its preloaded soil parameters from soil_data.csv

Required files (must exist before creating a GeoLookupEngine instance):
  data/districts.geojson  — Maharashtra district boundary polygons
  data/soil_data.csv      — preloaded soil parameters for each district

Usage:
  engine = GeoLookupEngine("data/districts.geojson", "data/soil_data.csv")
  result = engine.lookup(lat=18.5, lng=73.8)
  print(result.district)          # "Pune"
  print(result.district_record)   # DistrictRecord(organic_carbon=0.54, ...)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:
    from src.models import DistrictNotFoundError, DistrictRecord, GeoPoint, LocationResult
except ModuleNotFoundError:
    from models import DistrictNotFoundError, DistrictRecord, GeoPoint, LocationResult

# Property name candidates for district name in GeoJSON features (in priority order)
DISTRICT_NAME_KEYS = ["district", "dtname", "NAME_2", "NAME_1", "name", "District"]


class GeoLookupEngine:
    """Resolve map pin-drop coordinates to district soil data.

    Loads a bundled GeoJSON boundary file and soil CSV dataset at init time,
    then supports point-in-polygon and nearest-centroid reverse geocoding.
    """

    def __init__(self, geojson_path: str, soil_csv_path: str) -> None:
        """Load and cache the GeoJSON boundary file and soil CSV dataset.

        Args:
            geojson_path: Path to the GeoJSON district boundary file.
            soil_csv_path: Path to the bundled soil parameters CSV file.

        Raises:
            FileNotFoundError: If either file does not exist.
        """
        if not os.path.exists(geojson_path):
            raise FileNotFoundError(
                f"GeoJSON boundary file not found: '{geojson_path}'. "
                "Please place the district boundary file at the specified path."
            )
        if not os.path.exists(soil_csv_path):
            raise FileNotFoundError(
                f"Soil CSV dataset not found: '{soil_csv_path}'. "
                "Please place the soil parameters CSV at the specified path."
            )

        # Import geopandas here so the rest of the module can be imported without it
        import geopandas as gpd

        self._gdf = gpd.read_file(geojson_path)
        self._soil_df = pd.read_csv(soil_csv_path)

        # Pre-compute centroids for nearest-centroid fallback
        # Suppress CRS warning — centroid accuracy is sufficient for district-level fallback
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            self._centroids = self._gdf.geometry.centroid

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, lat: float, lng: float) -> LocationResult:
        """Resolve a coordinate to a district and return its preloaded soil data.

        Args:
            lat: Latitude in decimal degrees.
            lng: Longitude in decimal degrees.

        Returns:
            LocationResult with point, district name, DistrictRecord, and match method.

        Raises:
            ValueError: If lat/lng are outside valid bounds.
            DistrictNotFoundError: If the coordinate cannot be resolved to any district.
        """
        if not (-90 <= lat <= 90):
            raise ValueError(
                f"Invalid latitude {lat}: must be in [-90, 90]."
            )
        if not (-180 <= lng <= 180):
            raise ValueError(
                f"Invalid longitude {lng}: must be in [-180, 180]."
            )

        point = GeoPoint(lat=lat, lng=lng)
        district_name, match_method = self.reverse_geocode(lat, lng)
        district_record = self.fetch_district_data(district_name)

        return LocationResult(
            point=point,
            district=district_name,
            district_record=district_record,
            match_method=match_method,
        )

    def reverse_geocode(self, lat: float, lng: float) -> tuple[str, str]:
        """Map (lat, lng) to a district name using the bundled GeoJSON boundary file.

        Tries point-in-polygon first; falls back to nearest centroid if the point
        lies outside all polygons.

        Args:
            lat: Latitude in decimal degrees.
            lng: Longitude in decimal degrees.

        Returns:
            (district_name, match_method) where match_method is
            "point_in_polygon" or "nearest_centroid".

        Raises:
            DistrictNotFoundError: If no district can be resolved.
        """
        from shapely.geometry import Point

        shapely_point = Point(lng, lat)  # shapely uses (x=lng, y=lat)

        # --- Point-in-polygon ---
        mask = self._gdf.geometry.contains(shapely_point)
        matches = self._gdf[mask]
        if not matches.empty:
            row = matches.iloc[0]
            district_name = self._extract_district_name(row)
            if district_name:
                return district_name, "point_in_polygon"

        # --- Nearest-centroid fallback ---
        if self._centroids.empty:
            raise DistrictNotFoundError(
                f"No district polygons available to resolve coordinate ({lat}, {lng})."
            )

        distances = self._centroids.distance(shapely_point)
        nearest_idx = distances.idxmin()
        nearest_row = self._gdf.loc[nearest_idx]
        district_name = self._extract_district_name(nearest_row)

        if not district_name:
            raise DistrictNotFoundError(
                f"Could not resolve coordinate ({lat}, {lng}) to any known district "
                "after point-in-polygon and nearest-centroid attempts."
            )

        return district_name, "nearest_centroid"

    def fetch_district_data(self, district: str) -> DistrictRecord:
        """Retrieve preloaded soil parameters for a named district from the bundled CSV.

        Args:
            district: District name to look up (case-insensitive).

        Returns:
            DistrictRecord populated from the CSV row.

        Raises:
            DistrictNotFoundError: If the district is not found in the CSV.
        """
        lower_district = district.lower()
        mask = self._soil_df["district"].str.lower() == lower_district
        matches = self._soil_df[mask]

        if matches.empty:
            raise DistrictNotFoundError(
                f"District '{district}' not found in the soil dataset. "
                f"Available districts: {self.list_available_districts()[:10]}..."
            )

        row = matches.iloc[0]
        return DistrictRecord(
            district=str(row["district"]),
            organic_carbon=float(row["organic_carbon"]),
            nitrogen_depletion_rate=float(row["nitrogen_depletion_rate"]),
            rainfall_variability=float(row["rainfall_variability"]),
            crop_diversity_index=float(row["crop_diversity_index"]),
            irrigation_stress=float(row["irrigation_stress"]),
            fertilizer_load=float(row["fertilizer_load"]),
            year=int(row["year"]),
        )

    def list_available_districts(self) -> list[str]:
        """Return all district names from the bundled soil CSV.

        Returns:
            List of district name strings.
        """
        return self._soil_df["district"].tolist()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_district_name(self, row: "pd.Series") -> str | None:
        """Extract the district name from a GeoDataFrame row using known property keys.

        Tries keys in priority order: "district", "NAME_2", "NAME_1", "name", "District".

        Args:
            row: A row from the GeoDataFrame.

        Returns:
            District name string, or None if no matching key is found.
        """
        for key in DISTRICT_NAME_KEYS:
            if key in row.index and pd.notna(row[key]) and str(row[key]).strip():
                return str(row[key]).strip()
        return None
