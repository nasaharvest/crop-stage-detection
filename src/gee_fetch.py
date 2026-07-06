"""
gee_fetch.py
------------
Optional Google Earth Engine (GEE) layer for crop-stage detection.

This module fetches Sentinel-2 and Landsat NDVI time series for a polygon
using GEE, then passes the result to ``crop_stage.py`` for stage estimation.

Authentication
--------------
You must initialize GEE BEFORE importing or calling any function in this module:

    import ee
    ee.Initialize()          # uses your local credentials (gcloud auth / service account)
    from gee_fetch import fetch_ndvi, run_crop_stage_from_gee

See: https://developers.google.com/earth-engine/guides/auth

Optional dependencies (not needed for the BYOD workflow in crop_stage.py):
    earthengine-api >= 0.1.370
    geopandas >= 0.14
    shapely >= 2.0
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Union

import ee
import geopandas as gpd
import pandas as pd
import shapely.geometry
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from crop_stage import smooth_daily_interpolate_ndvi, estimate_stage_adaptive

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_utm_crs_from_lonlat(lon: float, lat: float) -> int:
    """Return the EPSG code of the UTM zone that contains (lon, lat)."""
    zone_number = min(int((lon + 180) / 6) + 1, 60)
    hemisphere = "326" if lat >= 0 else "327"
    return int(f"{hemisphere}{zone_number:02d}")


def _try_infer_wgs84(geom: BaseGeometry) -> bool:
    minx, miny, maxx, maxy = geom.bounds
    return -180.0 <= minx <= maxx <= 180.0 and -90.0 <= miny <= maxy <= 90.0


def _polygon_input_to_gdf(polygon_input) -> tuple[gpd.GeoDataFrame | None, str]:
    """
    Normalize any supported polygon input into a (single-row GeoDataFrame, label) pair.

    Supported input types
    ---------------------
    str | Path          file path (GeoJSON, Shapefile, GeoPackage, …)
    shapely Geometry    CRS inferred from coordinate bounds (must look like WGS84)
    gpd.GeoDataFrame    first row is used
    gpd.GeoSeries       first element is used
    dict                GeoJSON Feature / FeatureCollection / bare Geometry dict
    list                ring of [lon, lat] pairs → Polygon
    """
    if isinstance(polygon_input, (str, Path)):
        label = Path(polygon_input).stem
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            gdf = gpd.read_file(polygon_input)
        if gdf.crs is None:
            geom = gdf.geometry.iloc[0]
            if _try_infer_wgs84(geom):
                logger.warning(f"[CRS] No CRS in file '{label}'; inferring EPSG:4326.")
                gdf = gdf.set_crs("EPSG:4326")
            else:
                logger.error(f"[SKIP] No CRS in file '{label}' and coords don't look like WGS84.")
                return None, label
        return gdf, label

    if isinstance(polygon_input, gpd.GeoDataFrame):
        label = str(polygon_input.index[0])
        gdf = polygon_input.iloc[[0]].copy()
        if gdf.crs is None:
            geom = gdf.geometry.iloc[0]
            if _try_infer_wgs84(geom):
                logger.warning(f"[CRS] GeoDataFrame row '{label}' has no CRS; inferring EPSG:4326.")
                gdf = gdf.set_crs("EPSG:4326")
            else:
                logger.error(f"[SKIP] GeoDataFrame row '{label}' has no CRS and coords don't look like WGS84.")
                return None, label
        return gdf, label

    if isinstance(polygon_input, gpd.GeoSeries):
        label = str(polygon_input.index[0])
        geom = polygon_input.iloc[0]
        if polygon_input.crs is not None:
            gdf = gpd.GeoDataFrame(geometry=[geom], crs=polygon_input.crs)
        elif _try_infer_wgs84(geom):
            logger.warning(f"[CRS] GeoSeries element '{label}' has no CRS; inferring EPSG:4326.")
            gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        else:
            logger.error(f"[SKIP] GeoSeries element '{label}' has no CRS and coords don't look like WGS84.")
            return None, label
        return gdf, label

    if isinstance(polygon_input, BaseGeometry):
        label = "unnamed_polygon"
        if _try_infer_wgs84(polygon_input):
            logger.warning("[CRS] Shapely geometry has no CRS; inferring EPSG:4326.")
            gdf = gpd.GeoDataFrame(geometry=[polygon_input], crs="EPSG:4326")
        else:
            logger.error("[SKIP] Shapely geometry has no CRS and coords don't look like WGS84.")
            return None, label
        return gdf, label

    if isinstance(polygon_input, dict):
        geojson_type = polygon_input.get("type", "")
        if geojson_type == "FeatureCollection":
            features_list = polygon_input.get("features", [])
            if not features_list:
                logger.error("[SKIP] GeoJSON FeatureCollection has no features.")
                return None, "unnamed_polygon"
            feature = features_list[0]
        elif geojson_type == "Feature":
            feature = polygon_input
        else:
            feature = {"type": "Feature", "geometry": polygon_input, "properties": {}}
        geom = shapely.geometry.shape(feature["geometry"])
        props = feature.get("properties") or {}
        label = str(feature.get("id", props.get("id", "unnamed_polygon")))
        gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        return gdf, label

    if isinstance(polygon_input, list):
        geom = shapely.geometry.Polygon(polygon_input)
        label = "unnamed_polygon"
        if _try_infer_wgs84(geom):
            logger.warning("[CRS] List of coordinates has no CRS; inferring EPSG:4326.")
            gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        else:
            logger.error("[SKIP] List of coordinates doesn't look like WGS84.")
            return None, label
        return gdf, label

    raise TypeError(
        f"Unsupported polygon_input type: {type(polygon_input).__name__}. "
        "Expected str, Path, shapely Geometry, GeoDataFrame, GeoSeries, dict (GeoJSON), or list."
    )


def prepare_polygon_for_gee(
    polygon_input,
    poly_name: str | None = None,
    min_area_ha: float = 0.09,
    max_area_ha: float = 350,
    buffer_m: float = 0,
    split_multipolygon: bool = False,
) -> tuple[Optional[Union[ee.Geometry, List[ee.Geometry]]], str]:
    """
    Validate a polygon and convert it to an ``ee.Geometry`` for GEE queries.

    Parameters
    ----------
    polygon_input
        Any format accepted by ``_polygon_input_to_gdf``.
    poly_name : str, optional
        Label used in logs and returned to the caller. Auto-derived when None.
    min_area_ha, max_area_ha : float
        Area bounds (ha, after buffering). Polygons outside this range are skipped.
    buffer_m : float
        Buffer distance in metres applied before the area check. Use negative
        values to inset the polygon boundary (e.g. ``buffer_m=-10`` removes
        a 10-m ring to avoid edge pixels).
    split_multipolygon : bool
        When True, MultiPolygons are split into a list of individual
        ``ee.Geometry.Polygon`` objects instead of a single MultiPolygon.

    Returns
    -------
    tuple[ee.Geometry | list[ee.Geometry] | None, str]
        (ee_geometry, label) — geometry is None if the polygon was filtered out.
    """
    try:
        gdf, derived_name = _polygon_input_to_gdf(polygon_input)
        label = poly_name if poly_name is not None else derived_name

        if gdf is None:
            return None, label

        geom = gdf.geometry.iloc[0]

        if geom.is_empty:
            logger.info(f"[SKIP] Empty geometry: {label}")
            return None, label

        if geom.geom_type not in {"Polygon", "MultiPolygon"}:
            logger.info(f"[SKIP] Unsupported geometry type '{geom.geom_type}': {label}")
            return None, label

        if gdf.crs.to_epsg() == 4326:
            rp = geom.representative_point()
            utm_crs = get_utm_crs_from_lonlat(rp.x, rp.y)
            gdf = gdf.to_crs(utm_crs)

        gdf["geometry"] = gdf.buffer(buffer_m)
        geom_utm = gdf.geometry.iloc[0]

        if geom_utm.is_empty:
            logger.info(f"[SKIP] Geometry empty after buffer: {label}")
            return None, label

        area_ha = geom_utm.area / 10_000
        if area_ha < min_area_ha:
            logger.info(f"[SKIP] Area {area_ha:.3f} ha < min {min_area_ha} ha: {label}")
            return None, label
        if area_ha > max_area_ha:
            logger.info(f"[SKIP] Area {area_ha:.3f} ha > max {max_area_ha} ha: {label}")
            return None, label

        geom_wgs84 = (
            gpd.GeoSeries([geom_utm], crs=gdf.crs).to_crs("EPSG:4326").iloc[0]
        )
        polygons = (
            [geom_wgs84] if geom_wgs84.geom_type == "Polygon" else list(geom_wgs84.geoms)
        )

        if geom.geom_type == "Polygon":
            if len(polygons) == 1:
                return ee.Geometry.Polygon([list(polygons[0].exterior.coords)]), label
            coords = [[list(p.exterior.coords)] for p in polygons]
            return ee.Geometry.MultiPolygon(coords), label

        if not split_multipolygon:
            coords = [[list(p.exterior.coords)] for p in polygons]
            return ee.Geometry.MultiPolygon(coords), label

        return [ee.Geometry.Polygon([list(p.exterior.coords)]) for p in polygons], label

    except Exception as e:
        label = poly_name or "unnamed_polygon"
        logger.error(f"[ERROR] Failed to process '{label}': {e}", exc_info=True)
        return None, label


# ---------------------------------------------------------------------------
# GEE data extraction helpers (internal)
# ---------------------------------------------------------------------------

def _convert_raw_gee_data_to_df(
    raw_data: Dict,
    desired_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    try:
        features = raw_data.get("features", [])
        if not features:
            return pd.DataFrame()

        all_keys: set = set()
        for f in features:
            all_keys.update(f.get("properties", {}).keys())

        cols = [c for c in desired_cols if c in all_keys] if desired_cols else list(all_keys)

        records = [
            {col: f.get("properties", {}).get(col) for col in cols}
            for f in features
        ]
        df = pd.DataFrame(records)

        for col in df.select_dtypes(include="number").columns:
            df[col] = df[col].astype(float).round(3)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
            df = df.sort_values("date").reset_index(drop=True)

        return df
    except Exception as e:
        logger.error(f"[ERROR] Failed to parse GEE response: {e}")
        raise


def _calculate_sentinel2_data(
    s2_img: ee.Image,
    cs_img: ee.Image,
    ee_geom: ee.Geometry,
    poly_name: str,
    cs_threshold: float,
    bands_list: list,
) -> ee.Image:
    """Compute Sentinel-2 band means masked by Cloud Score+."""
    date = s2_img.date().format("YYYY-MM-dd")
    cs_band = cs_img.select("cs")
    cs_mask = cs_band.gt(cs_threshold)
    s2_masked = s2_img.updateMask(cs_mask)

    scl_band = s2_img.select("SCL")
    reducer_args = {"reducer": ee.Reducer.count(), "geometry": ee_geom, "scale": 20}
    total_px = ee.Number(scl_band.reduceRegion(**reducer_args).get("SCL"))
    cloud_mask = scl_band.eq(2).Or(scl_band.eq(4)).Or(scl_band.eq(5)).Not()
    cloud_px = ee.Number(scl_band.updateMask(cloud_mask).reduceRegion(**reducer_args).get("SCL"))
    scl_pct = ee.Number(ee.Algorithms.If(total_px.gt(0), cloud_px.divide(total_px).multiply(100).round(), 0))

    result = s2_img.set({"date": date, "poly_name": poly_name, "scl_clouds_percent": scl_pct})
    result = result.set(cs_band.reduceRegion(reducer=ee.Reducer.percentile([10]), geometry=ee_geom, scale=10).rename(["cs_p10"], ["p10_cs"]))

    optical_bands = [b for b in bands_list if "B" in b]
    if optical_bands:
        result = result.set(s2_masked.select(optical_bands).divide(10_000).reduceRegion(reducer=ee.Reducer.mean(), geometry=ee_geom, scale=10))
    if "NDVI" in bands_list:
        ndvi = s2_masked.normalizedDifference(["B8", "B4"]).rename("NDVI")
        result = result.set(ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=ee_geom, scale=10))

    return result


def _calculate_sentinel2_data_without_cs(
    s2_img: ee.Image,
    ee_geom: ee.Geometry,
    poly_name: str,
    bands_list: list,
) -> ee.Image:
    """Compute Sentinel-2 band means WITHOUT Cloud Score+ (fallback)."""
    date = s2_img.date().format("YYYY-MM-dd")
    scl_band = s2_img.select("SCL")
    reducer_args = {"reducer": ee.Reducer.count(), "geometry": ee_geom, "scale": 20}
    total_px = ee.Number(scl_band.reduceRegion(**reducer_args).get("SCL"))
    cloud_mask = scl_band.eq(2).Or(scl_band.eq(4)).Or(scl_band.eq(5)).Not()
    cloud_px = ee.Number(scl_band.updateMask(cloud_mask).reduceRegion(**reducer_args).get("SCL"))
    scl_pct = ee.Number(ee.Algorithms.If(total_px.gt(0), cloud_px.divide(total_px).multiply(100).round(), 0))

    result = s2_img.set({"date": date, "poly_name": poly_name, "scl_clouds_percent": scl_pct})
    optical_bands = [b for b in bands_list if "B" in b]
    if optical_bands:
        result = result.set(s2_img.select(optical_bands).divide(10_000).reduceRegion(reducer=ee.Reducer.mean(), geometry=ee_geom, scale=10))
    if "NDVI" in bands_list:
        ndvi = s2_img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        result = result.set(ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=ee_geom, scale=10))

    return result


def _apply_cs_or_fallback(img, cloud_score, ee_geom, poly_name, bands_list, cs_threshold):
    idx = img.get("system:index")
    cs_for_img = cloud_score.filter(ee.Filter.equals("system:index", idx))
    has_cs = cs_for_img.size().gt(0)
    return ee.Image(
        ee.Algorithms.If(
            has_cs,
            _calculate_sentinel2_data(img, ee.Image(cs_for_img.first()), ee_geom, poly_name, cs_threshold, bands_list),
            _calculate_sentinel2_data_without_cs(img, ee_geom, poly_name, bands_list),
        )
    )


def _calculate_landsat_data(image: ee.Image, ee_geom: ee.Geometry, poly_name: str, bands_list: list) -> ee.Image:
    """Compute Landsat band means with cloud masking via QA_PIXEL."""
    date = image.date().format("YYYY-MM-dd")
    optical_bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
    scaled = image.select(optical_bands).multiply(0.0000275).add(-0.2)
    image = image.addBands(scaled, overwrite=True)
    thermal_c = image.select("ST_B10").multiply(0.00341802).add(149).subtract(273.15)
    image = image.addBands(thermal_c.rename("ST_B10_C"), overwrite=True)

    valid_mask = image.select("QA_PIXEL").remap([21824, 21888], ee.List.repeat(1, 2), 0)
    img_masked = image.updateMask(valid_mask)

    qa_reducer = {"reducer": ee.Reducer.count(), "geometry": ee_geom, "scale": 30}
    total_px = image.select("QA_PIXEL").reduceRegion(**qa_reducer).rename(["QA_PIXEL"], ["total_pixels"])
    valid_px = img_masked.select("QA_PIXEL").reduceRegion(**qa_reducer).rename(["QA_PIXEL"], ["valid_pixels"])

    result = image.set({"date": date, "poly_name": poly_name}).set(total_px).set(valid_px)

    sr_bands = [b for b in bands_list if b.startswith("SR_") or b.startswith("ST_")]
    if sr_bands:
        result = result.set(img_masked.select(sr_bands).reduceRegion(reducer=ee.Reducer.mean(), geometry=ee_geom, scale=30))
    if "NDVI" in bands_list:
        ndvi = img_masked.expression("(NIR - RED) / (NIR + RED)", {"NIR": img_masked.select("SR_B5"), "RED": img_masked.select("SR_B4")}).rename("NDVI")
        result = result.set(ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=ee_geom, scale=30))

    return result


# ---------------------------------------------------------------------------
# Public GEE query functions
# ---------------------------------------------------------------------------

def get_sentinel2_ndvi(
    ee_geom: ee.Geometry,
    start_date: str,
    end_date: str,
    cs_threshold: float = 0.6,
    poly_name: str = "unnamed_polygon",
) -> pd.DataFrame:
    """
    Fetch Sentinel-2 NDVI for a polygon from GEE.

    Parameters
    ----------
    ee_geom : ee.Geometry   Polygon geometry (from ``prepare_polygon_for_gee``).
    start_date, end_date : str   Date range ``"YYYY-MM-DD"``.
    cs_threshold : float   Per-pixel Cloud Score+ threshold (default 0.6).
    poly_name : str   Label for logging.

    Returns
    -------
    pd.DataFrame  Columns: [``poly_name``, ``date``, ``NDVI``, ``p10_cs``, ``scl_clouds_percent``, ``sensor``].
                  Empty DataFrame if no valid images were found.
    """
    start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    bands_list = ["NDVI"]

    try:
        s2_sr = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start_date, end_date)
            .filterBounds(ee_geom)
            .filter(ee.Filter.listContains("system:band_names", "SCL"))
        )
        cloud_score = (
            ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
            .filterDate(start_date, end_date)
            .filterBounds(ee_geom)
        )
        result_collection = s2_sr.map(
            lambda img: _apply_cs_or_fallback(img, cloud_score, ee_geom, poly_name, bands_list, cs_threshold)
        )
        raw_data = result_collection.getInfo()

        if not raw_data.get("features"):
            return pd.DataFrame()

        df = _convert_raw_gee_data_to_df(raw_data, desired_cols=["poly_name", "date", "p10_cs", "scl_clouds_percent", "NDVI"])
        df["sensor"] = "sn2"
        return df

    except Exception as e:
        logger.error(f"[ERROR] Sentinel-2 fetch failed for '{poly_name}': {e}")
        return pd.DataFrame()


def remove_clouds_sentinel2(
    df: pd.DataFrame,
    p10_cs: float = 0.6,
    scl_clouds_percent: float = 90.0,
    min_ndvi: float = 0.0,
    max_ndvi: float = 1.0,
) -> pd.DataFrame:
    """
    Filter cloud-contaminated Sentinel-2 observations from a raw NDVI DataFrame.

    Parameters
    ----------
    df : pd.DataFrame   Raw Sentinel-2 DataFrame (from ``get_sentinel2_ndvi``).
    p10_cs : float   Minimum acceptable 10th-percentile Cloud Score+ (default 0.6).
    scl_clouds_percent : float   Maximum acceptable SCL cloud percentage (default 90).
    min_ndvi, max_ndvi : float   NDVI validity range.

    Returns
    -------
    pd.DataFrame  Filtered DataFrame, one row per date (clearest image kept).
    """
    if df.empty:
        return df
    df = df.copy()
    if "NDVI" in df.columns:
        df = df.dropna(subset=["NDVI"])
        df = df[(df["NDVI"] >= min_ndvi) & (df["NDVI"] <= max_ndvi)]
    if "scl_clouds_percent" in df.columns:
        df = df[df["scl_clouds_percent"] <= scl_clouds_percent]
    if "p10_cs" in df.columns:
        df = df[df["p10_cs"] >= p10_cs]
        df = df.sort_values("p10_cs", ascending=False).drop_duplicates("date", keep="first")
    return df.sort_values("date").reset_index(drop=True)


def get_landsat_ndvi(
    ee_geom: ee.Geometry,
    start_date: str,
    end_date: str,
    landsat: str = "LC08",
    poly_name: str = "unnamed_polygon",
) -> pd.DataFrame:
    """
    Fetch Landsat 8 or 9 NDVI for a polygon from GEE.

    Parameters
    ----------
    ee_geom : ee.Geometry
    start_date, end_date : str   Date range ``"YYYY-MM-DD"``.
    landsat : str   ``"LC08"`` (Landsat 8) or ``"LC09"`` (Landsat 9).
    poly_name : str   Label for logging.

    Returns
    -------
    pd.DataFrame  Columns: [``poly_name``, ``date``, ``NDVI``, ``total_pixels``, ``valid_pixels``, ``sensor``].
    """
    start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    bands_list = ["NDVI"]

    try:
        collection = (
            ee.ImageCollection(f"LANDSAT/{landsat}/C02/T1_L2")
            .filterDate(start_date, end_date)
            .filterBounds(ee_geom)
        )
        result_collection = collection.map(
            lambda img: _calculate_landsat_data(img, ee_geom, poly_name, bands_list)
        )
        raw_data = result_collection.getInfo()

        if not raw_data.get("features"):
            return pd.DataFrame()

        df = _convert_raw_gee_data_to_df(raw_data, desired_cols=["poly_name", "date", "total_pixels", "valid_pixels", "NDVI"])
        df["sensor"] = landsat.lower()
        return df

    except Exception as e:
        logger.error(f"[ERROR] Landsat {landsat} fetch failed for '{poly_name}': {e}")
        return pd.DataFrame()


def remove_clouds_landsat(
    df: pd.DataFrame,
    min_pct_clear: float = 50.0,
) -> pd.DataFrame:
    """
    Filter cloud-contaminated Landsat observations.

    Parameters
    ----------
    df : pd.DataFrame   Raw Landsat DataFrame (from ``get_landsat_ndvi``).
    min_pct_clear : float   Minimum percentage of clear pixels per image (default 50).

    Returns
    -------
    pd.DataFrame  Filtered, one row per date (clearest image kept).
    """
    if df.empty:
        return df
    df = df.copy().dropna(subset=["NDVI"])
    df["pct_clear"] = (100 * df["valid_pixels"] / df["total_pixels"]).round()
    df = df[df["pct_clear"] > min_pct_clear]
    df = (
        df.sort_values("pct_clear", ascending=False)
        .drop_duplicates("date", keep="first")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df


def fetch_ndvi(
    polygon_input,
    start_date: str,
    end_date: str,
    poly_name: str | None = None,
    min_area_ha: float = 0.04,
    max_area_ha: float = 200,
    buffer_m: float = -10,
    sn2_p10_cs: float = 0.5,
    sn2_cs_threshold: float = 0.6,
) -> pd.DataFrame:
    """
    Fetch a merged Sentinel-2 + Landsat 8/9 NDVI time series for one polygon.

    This is the main entry point for the GEE workflow. Call ``ee.Initialize()``
    before using this function.

    Parameters
    ----------
    polygon_input
        Polygon in any supported format (file path, GeoDataFrame, GeoJSON dict, …).
    start_date, end_date : str   Date range ``"YYYY-MM-DD"``.
    poly_name : str, optional
        Label used in the output ``poly_name`` column and in log messages.
        Auto-derived from the input when None.
    min_area_ha, max_area_ha : float   Area bounds (ha, after buffering).
    buffer_m : float   Buffer in metres (negative = inset; default -10 avoids boundary pixels).
    sn2_p10_cs : float   Minimum Cloud Score+ 10th percentile for Sentinel-2 (default 0.5).
    sn2_cs_threshold : float   Per-pixel Cloud Score+ mask threshold (default 0.6).

    Returns
    -------
    pd.DataFrame
        Columns: ``[poly_name, date, NDVI, sensor]``.
        Sentinel-2 is preferred when both S2 and Landsat observe the same date.
        Returns an empty DataFrame if the polygon is out-of-range or no valid
        images were found.

    Example
    -------
    >>> import ee
    >>> ee.Initialize()
    >>> from gee_fetch import fetch_ndvi
    >>> df = fetch_ndvi(
    ...     "my_field.geojson",
    ...     start_date="2023-03-01",
    ...     end_date="2023-10-01",
    ... )
    >>> print(df.head())
    """
    ee_geom, resolved_name = prepare_polygon_for_gee(
        polygon_input,
        poly_name=poly_name,
        min_area_ha=min_area_ha,
        max_area_ha=max_area_ha,
        buffer_m=buffer_m,
    )

    if ee_geom is None:
        logger.info(f"Polygon '{resolved_name}' filtered out — returning empty DataFrame.")
        return pd.DataFrame()

    sn2 = remove_clouds_sentinel2(
        get_sentinel2_ndvi(ee_geom, start_date, end_date, sn2_cs_threshold, resolved_name),
        p10_cs=sn2_p10_cs,
    )
    ls8 = remove_clouds_landsat(get_landsat_ndvi(ee_geom, start_date, end_date, "LC08", resolved_name))
    ls9 = remove_clouds_landsat(get_landsat_ndvi(ee_geom, start_date, end_date, "LC09", resolved_name))

    parts = [d[["poly_name", "date", "NDVI", "sensor"]].copy() for d in [sn2, ls8, ls9] if not d.empty]
    if not parts:
        logger.info(f"No valid images for '{resolved_name}'.")
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    df = (
        df.sort_values(["date", "sensor"])
        .drop_duplicates(subset=["date"], keep="last")  # S2 preferred (sorted last alphabetically vs lc08/lc09)
        .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# Full GEE → stage pipeline (batch over many polygons)
# ---------------------------------------------------------------------------

def _run_single_gee(single_row_gdf, field_id, end_date, lookback_days, id_col):
    start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    no_result = {id_col: field_id, "crop_stage": None, "stage_description": None, "peak_date": None, "days_since_peak": None, "last_date": None}
    try:
        ndvi_df = fetch_ndvi(single_row_gdf, start_date=start_date, end_date=end_date, poly_name=str(field_id))
        if ndvi_df.empty or len(ndvi_df) < 3:
            return no_result
        df_smooth = smooth_daily_interpolate_ndvi(ndvi_df)
        result = estimate_stage_adaptive(df_smooth["NDVI_smooth"].to_numpy(), dates=df_smooth["date"])
        return {
            id_col:              field_id,
            "crop_stage":        result["Stage"],
            "stage_description": result["Stage_description"],
            "peak_date":         result.get("Peak_date"),
            "days_since_peak":   result.get("Days_since_peak"),
            "last_date":         result.get("Last_date"),
        }
    except Exception as e:
        logger.error(f"Crop stage failed for {field_id}: {e}")
        return no_result


def run_crop_stage_from_gee(
    gdf: gpd.GeoDataFrame,
    id_col: str = "field_id",
    lookback_days: int = 150,
    max_workers: int = 16,
    end_date: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Fetch GEE NDVI and estimate the crop stage for every polygon in a GeoDataFrame.

    This is the full end-to-end GEE pipeline. Call ``ee.Initialize()`` first.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        One row per field. Must contain ``id_col`` and a geometry column.
    id_col : str   Column name for the unique field identifier (default ``"field_id"``).
    lookback_days : int   Days of NDVI history to fetch (default 150).
    max_workers : int   Thread pool size (default 16).
    end_date : str or None
        End date ``"YYYY-MM-DD"`` for the NDVI fetch window. Defaults to today
        when None. Set to a past date for historical analysis.

    Returns
    -------
    gpd.GeoDataFrame
        Input GeoDataFrame with added columns:
        ``[crop_stage, stage_description, peak_date, days_since_peak, last_date]``.

    Example
    -------
    >>> import ee, geopandas as gpd
    >>> ee.Initialize()
    >>> from gee_fetch import run_crop_stage_from_gee
    >>> gdf = gpd.read_file("my_fields.geojson")
    >>> results = run_crop_stage_from_gee(gdf, id_col="FID", lookback_days=180)
    >>> print(results[["FID", "crop_stage", "days_since_peak"]])
    """
    end_date = (
        (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if end_date is None
        else end_date
    )
    logger.info(f"Running GEE crop stage for {len(gdf)} fields (lookback={lookback_days} days)")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        futures = {
            exc.submit(_run_single_gee, gdf.loc[[idx]], row[id_col], end_date, lookback_days, id_col): row[id_col]
            for idx, row in gdf.iterrows()
        }
        for f in tqdm(as_completed(futures), total=len(futures)):
            results.append(f.result())

    results_df = pd.DataFrame(results).set_index(id_col)
    gdf = gdf.set_index(id_col).copy()
    for col in ["crop_stage", "stage_description", "peak_date", "days_since_peak", "last_date"]:
        if col in results_df.columns:
            gdf[col] = results_df[col]
    return gdf.reset_index()
