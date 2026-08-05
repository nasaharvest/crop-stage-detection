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

## Inputs

The model expects an NDVI time series in tabular format. Two entry points — pick the one that fits your workflow:

- **Bring Your Own Data** — `run_crop_stage_from_dataframe(df, ...)` takes NDVI time series in tabular format and returns the crop stage for the last observation in the input series. The required columns are `date` and `NDVI`. If `id_col` is provided, the model runs by unique `id_col` groups (groupby); if `id_col` is not provided, it treats the input as a single time series.
- **Google Earth Engine (GEE)** — `run_crop_stage_from_gee(polygon, ...)` fetches Sentinel-2 and Landsat NDVI from GEE and returns the crop stage for the last clouds free observation. `polygon` is the only required argument and accepts any supported format (see below). By default, the lookback window covers the previous 150 days up to the current date. Both the lookback window and the end date are configurable through the `lookback_days` and `end_date` parameters. You can also pass the same crop-stage tuning parameters shown in the [Key parameters](#key-parameters) section through this wrapper. The GEE function works on one polygon at a time; for multiple fields, use a loop or `ThreadPoolExecutor` (see [examples/02_gee.ipynb](examples/02_gee.ipynb), Section 5).

Both functions handle preprocessing internally: observations are resampled to one value per day, gap-filled with PCHIP interpolation, and smoothed with a Whittaker filter. Input observations are first converted into a daily NDVI series. If the input observed date range is shorter than 30 days, the processed series will contain fewer than 30 points and the model will return an "Insufficient Data" result. If your data is already daily or pre-smoothed, the effect on your NDVI values is negligible. The stage is then assigned from three signals on the smoothed series:

| Signal | Description |
|--------|-------------|
| NDVI level | Where the current value falls relative to adaptive lower and upper thresholds |
| Trend | Whether NDVI is rising or falling (estimated via a Kalman filter) |
| Peak history | Whether a seasonal peak has already been confirmed |

Stage assignment:

| Stage | NDVI level | Trend | Peak confirmed |
|-------|------------|-------|:--------------:|
| A — Bare soil / planting / emergence | Below lower threshold | Any | No |
| B — Greenup / rapid growth | Between thresholds | Rising | No |
| C — Peak maturity | At or above upper threshold | Any | — |
| D — Senescence | Between thresholds | Falling | Yes |
| E — Post-harvest / residue / bare soil | Below lower threshold | Any | Yes |

## Outputs

The output returns a dict with:

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

### Supported polygon input formats (GEE workflow)

The polygon can be provided in any of these formats. Sample files for each format are in [sample_data](sample_data/) and demonstrated in [examples/02_gee.ipynb](examples/02_gee.ipynb).

| Format | UTM supported | Where CRS is set |
|--------|:---:|---|
| File path — `.shp`, `.gpkg` | Yes | Embedded in file (`.prj` / file metadata) |
| File path — `.zip` (zipped shapefile) | Yes | `.prj` sidecar inside the zip |
| File path — `.geojson`, `.kml`, `.kmz` | No | WGS84 by spec |
| `gpd.GeoDataFrame` / `gpd.GeoSeries` | Yes | `.crs` attribute on the object |
| `shapely.geometry.Polygon` | No | No CRS — WGS84 inferred from bounds |
| GeoJSON `dict` | No | WGS84 by spec (RFC 7946) |
| `list` of `[lon, lat]` pairs | No | WGS84 assumed |

If no CRS is found but the coordinates look like lon/lat (−180..180, −90..90), WGS84 is inferred automatically with a log warning. If no CRS is found and the coordinates don't look like WGS84, the polygon is rejected.

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/nasaharvest/crop-stage-detection.git
cd crop-stage-detection
```

**Option A — Bring Your Own Data (no GEE)**

If you already have an NDVI time series and just want to run the stage model:

```bash
python -m pip install -r requirements.txt
```

**Option B — Full GEE workflow**

If you want to fetch NDVI from GEE as well, install both files — [requirements-gee.txt](requirements-gee.txt) adds the GEE-specific packages on top of the core ones:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-gee.txt
earthengine authenticate
```

The GEE workflow requires Earth Engine authentication before you can run the example or API calls. See the [Earth Engine authentication guide](https://developers.google.com/earth-engine/guides/auth).

To verify the installation:

```bash
python test_basic.py
```

## Examples

Worked examples are available in the notebooks:

- [examples/01_byod.ipynb](examples/01_byod.ipynb) — bring-your-own-data workflow with a sample NDVI time series
- [examples/02_gee.ipynb](examples/02_gee.ipynb) — full Google Earth Engine workflow

## Key parameters

All thresholds and filter settings are configurable and can be passed to the
DataFrame and GEE workflows, where they are forwarded to
`estimate_stage_adaptive` internally.

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
