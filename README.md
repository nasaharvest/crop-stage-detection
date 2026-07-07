# Crop Stage Detection

A crop-agnostic model for estimating the crop stage from an NDVI time series — no crop-specific calibration needed.

![Crop Stage Detection Demo](docs/crop_stage_detection.gif)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nasaharvest/crop-stage-detection/blob/main/examples/01_byod.ipynb)

## How it works

Most row crops and vegetables follow the same characteristic NDVI trajectory
over a growing season: a rise from bare soil through rapid greenup to peak
vegetative vigor, followed by senescence and harvest. The duration, steepness,
and peak height vary by crop, region, and season — but the shape is consistent
enough that a single curve-relative model can locate "where on the curve" a
given observation falls, without knowing which crop it is.

![Crop Stages Legend](docs/crop_stages_legend.png)

The curve is divided into five generic stages:

| Stage | Name | Description |
|-------|------|-------------|
| **A** | Bare soil / planting / emergence | NDVI below lower threshold, no prior peak |
| **B** | Greenup / rapid growth | NDVI in mid-range and rising |
| **C** | Peak maturity | NDVI at or above upper threshold |
| **D** | Senescence | NDVI in mid-range and falling, peak confirmed |
| **E** | Post-harvest / residue / bare soil | NDVI below lower threshold, peak confirmed |

**Two entry points — pick the one that fits your workflow:**

- **GEE** — `run_crop_stage_from_gee(polygon, ...)` takes a single field polygon in any supported format, fetches Sentinel-2 and Landsat NDVI from Google Earth Engine, and returns the crop stage for the last available observation.
- **Bring Your Own Data** — `run_crop_stage_from_dataframe(df, ...)` takes any NDVI time series in tabular format from any source. Pass `id_col` to process multiple fields in one call.

Both functions work on one field at a time. For the GEE function, use a loop or `ThreadPoolExecutor` for multiple fields (see `examples/02_gee.ipynb`, Section 5).

Both functions handle preprocessing internally: observations are resampled to a daily grid, gap-filled with PCHIP interpolation, and smoothed with a Whittaker filter. If your data is already daily or pre-smoothed, the effect is negligible. The stage is then assigned from three signals on the smoothed series: where the current NDVI falls relative to adaptive thresholds, whether it is rising or falling (estimated via a Kalman filter), and whether a seasonal peak has already been confirmed.

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/nasaharvest/crop-stage-detection.git
cd crop-stage-detection
```

**Option A — Bring Your Own Data (no GEE)**

If you already have an NDVI time series and just want to run the stage model:

```bash
pip install -r requirements.txt
```

**Option B — Full GEE workflow**

If you want to fetch NDVI from Google Earth Engine as well, install both files — `requirements-gee.txt` adds the GEE-specific packages on top of the core ones:

```bash
pip install -r requirements.txt
pip install -r requirements-gee.txt
earthengine authenticate
```

For GEE authentication, see the [Earth Engine authentication guide](https://developers.google.com/earth-engine/guides/auth).

To verify the installation:

```bash
python test_basic.py
```

## Quick start — Bring Your Own Data (no GEE)

```python
import sys
sys.path.insert(0, "src")

import pandas as pd
from crop_stage import run_crop_stage_from_dataframe

df = pd.read_csv("sample_data/sample_ndvi.csv", parse_dates=["date"])
result = run_crop_stage_from_dataframe(df)
print(result["Stage"], "—", result["Stage_description"])
```

**Input format** — your DataFrame needs at minimum:

| Column | Type | Notes |
|--------|------|-------|
| `date` | datetime | Any parseable format, irregular spacing is fine |
| `NDVI` | float [0–1] | Cloud-free observations only |

The sample CSV also contains `sensor` and `field_id` columns — these are ignored in single-field mode and used only in the multi-field batch example.

## Quick start — GEE fetch

```python
import sys
sys.path.insert(0, "src")

import ee
ee.Initialize()   # must come first

from gee_fetch import run_crop_stage_from_gee

result = run_crop_stage_from_gee("my_field.geojson", lookback_days=180)
print(result["Stage"], "—", result["Stage_description"])
if result.get("Peak_date"):
    print(f"Peak: {result['Peak_date'].date()}  |  Days since peak: {result['Days_since_peak']}")
```

## Supported polygon input formats (GEE workflow)

`fetch_ndvi` and `run_crop_stage_from_gee` accept the polygon in any of these formats:

| Format | UTM supported | Where CRS is set |
|--------|:---:|---|
| File path — `.shp`, `.gpkg` | Yes | Embedded in file (`.prj` / file metadata) |
| File path — `.geojson`, `.kml` | No | WGS84 by spec |
| `gpd.GeoDataFrame` / `gpd.GeoSeries` | Yes | `.crs` attribute on the object |
| `shapely.geometry.Polygon` | No | No CRS — WGS84 inferred from bounds |
| GeoJSON `dict` | No | WGS84 by spec (RFC 7946) |
| `list` of `[lon, lat]` pairs | No | WGS84 assumed |

If no CRS is found but the coordinates look like lon/lat (−180..180, −90..90), WGS84 is inferred automatically with a log warning. If no CRS is found and the coordinates don't look like WGS84, the polygon is rejected.

## Multiple fields

```python
# BYOD — pass id_col to process all fields in one call
from crop_stage import run_crop_stage_from_dataframe
results = run_crop_stage_from_dataframe(df, id_col="field_id")
# returns a DataFrame with one row per field

# GEE — run_crop_stage_from_gee handles one polygon at a time; loop for many
import geopandas as gpd
from gee_fetch import run_crop_stage_from_gee
gdf = gpd.read_file("my_fields.geojson")
results = []
for _, row in gdf.iterrows():
    field = gpd.GeoDataFrame([row], geometry="geometry", crs=gdf.crs)
    r = run_crop_stage_from_gee(field, lookback_days=180)
    results.append({"field_id": row["field_id"], **r})
```

## Project structure

```
crop-stage-detection/
├── src/
│   ├── crop_stage.py       Core model (no GEE dependency)
│   └── gee_fetch.py        Optional GEE layer
├── examples/
│   ├── 01_byod.ipynb       BYOD walkthrough with visualisation
│   └── 02_gee.ipynb        Full GEE pipeline
├── sample_data/
│   └── sample_ndvi.csv     One field × one season (works out of the box)
├── docs/
│   ├── crop_stages_legend.png
│   └── crop_stage_detection.gif
├── requirements.txt        Core dependencies (BYOD)
├── requirements-gee.txt    Additional dependencies for GEE workflow
├── CITATION.cff
├── LICENSE
└── test_basic.py           Smoke test — run after install to verify
```

## Output fields

`estimate_stage_adaptive` returns a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `Stage` | str | Stage label: A, B, C, D, E, or "Insufficient Data" |
| `Stage_description` | str | Human-readable stage name |
| `Value` | float | NDVI at the last observation |
| `Velocity` | float or None | Kalman-estimated rate of change (NDVI/day); None for stage C and "Insufficient Data" |
| `Last_date` | Timestamp or None | Date of the last observation |
| `Peak_date` | Timestamp or None | Date of the detected seasonal peak (None if no peak confirmed) |
| `Days_since_peak` | int | Days elapsed since peak (or None) |
| `Upper_threshold` | float or None | Adaptive upper threshold used; None for "Insufficient Data" |
| `Lower_threshold` | float | Fixed lower threshold used |

## Key parameters

All thresholds and filter settings are configurable and can be passed directly
to `estimate_stage_adaptive` or `run_crop_stage_from_dataframe`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `upper_percentile` | 90 | Percentile of the series used as upper threshold |
| `min_peak_ndvi` | 0.50 | Hard floor for the upper threshold |
| `lower_threshold` | 0.35 | Fixed lower threshold |
| `min_peak_width` | 5 | Minimum consecutive days above upper threshold to confirm a peak |
| `min_observations` | 30 | Minimum series length for a reliable estimate |
| `date_col` | `"date"` | Column name for dates in the input DataFrame (``run_crop_stage_from_dataframe``) |
| `ndvi_col` | `"NDVI"` | Column name for NDVI values in the input DataFrame (``run_crop_stage_from_dataframe``) |
| `stage_descriptions` | built-in | Custom ``{label: description}`` dict to rename or translate stage labels |

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

If you use this model in your research, please cite:

```bibtex
@software{pelta2026cropstage,
  author       = {Pelta, Ran},
  title        = {Crop Stage Detection},
  year         = {2026},
  publisher    = {GitHub},
  organization = {NASA Harvest / Agmatix},
  url          = {https://github.com/nasaharvest/crop-stage-detection}
}
```

## Authors

Developed by [Ran Pelta](https://www.linkedin.com/in/ran-pelta) (Agmatix / GrowersTech, NASA Harvest)
