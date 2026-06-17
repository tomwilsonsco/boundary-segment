"""Shared utility for loading and validating the prediction mask geometry.

Used by scripts that accept a `--prediction-mask` argument to constrain
outputs. Usually extent parcels applicable for boundary predictions.
"""

import logging
from pathlib import Path
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

EXPECTED_CRS = "EPSG:27700"


def load_prediction_mask(mask_path: Path) -> shape:
    """Load the prediction mask file and validate its CRS.

    Args:
        mask_path: Path to a .gpkg or .shp file containing the prediction mask
            polygon(s).

    Returns:
        A single Shapely geometry (the dissolved union of all mask features)
        representing the area of interest.

    Raises:
        FileNotFoundError: If the mask file does not exist.
        ValueError: If the CRS of the mask is not EPSG:27700.
    """
    if not mask_path.exists():
        raise FileNotFoundError(f"Prediction mask not found: {mask_path}")

    gdf = gpd.read_file(mask_path)

    crs_str = str(gdf.crs).upper()
    logger.info(f"Prediction mask CRS: {gdf.crs}")

    # Normalise the expected CRS string for comparison
    expected_norm = "EPSG:27700"
    if crs_str != expected_norm:
        raise ValueError(
            f"Prediction mask CRS is {crs_str}. "
            f"Expected {expected_norm}. Aborting."
        )

    # Dissolve all features into a single union geometry
    if len(gdf) == 0:
        raise ValueError("Prediction mask contains no features.")

    union_geom = unary_union(gdf.geometry.values)
    logger.info(
        f"Prediction mask loaded: {len(gdf)} features, "
        f"dissolved to {union_geom.geom_type}"
    )

    return union_geom