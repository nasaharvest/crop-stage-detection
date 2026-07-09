"""
Quick smoke test — run from the repo root:
    python test_basic.py
"""

import sys
import types
from unittest.mock import patch

sys.path.insert(0, "src")

print("1. Importing crop_stage...", end=" ")
from crop_stage import (
    smooth_daily_interpolate_ndvi,
    estimate_stage_adaptive,
    run_crop_stage_from_dataframe,
)
print("OK")

print("2. Loading sample data...", end=" ")
import pandas as pd
df_all = pd.read_csv("sample_data/sample_ndvi.csv", parse_dates=["date"])
assert len(df_all) > 0, "Empty CSV"
assert "date" in df_all.columns and "NDVI" in df_all.columns
df = df_all[df_all["field_id"] == "field_001"].reset_index(drop=True)
print(f"OK ({len(df_all)} rows total, {len(df)} for field_001)")

print("3. Smoothing NDVI...", end=" ")
df_smooth = smooth_daily_interpolate_ndvi(df)
assert "NDVI_smooth" in df_smooth.columns
assert len(df_smooth) >= len(df)
print(f"OK ({len(df_smooth)} daily rows)")

print("4. Estimating stage (raw call)...", end=" ")
result = estimate_stage_adaptive(
    df_smooth["NDVI_smooth"].to_numpy(),
    dates=df_smooth["date"],
)
assert result["Stage"] in ("A", "B", "C", "D", "E", "Insufficient Data")
print(f"OK  ->  Stage {result['Stage']}: {result['Stage_description']}")
if result["Value"] is not None and result["Upper_threshold"] is not None:
    print(f"       Value={result['Value']:.3f}  Upper={result['Upper_threshold']:.3f}  Lower={result['Lower_threshold']:.3f}")
if result["Peak_date"]:
    print(f"       Peak={result['Peak_date'].date()}  Days since peak={result['Days_since_peak']}")

print("5. run_crop_stage_from_dataframe (single field)...", end=" ")
r = run_crop_stage_from_dataframe(df)
assert r["Stage"] == result["Stage"]
print("OK")

print("6. run_crop_stage_from_dataframe (multi-field via id_col)...", end=" ")
df_multi = run_crop_stage_from_dataframe(df_all, id_col="field_id")
assert len(df_multi) == df_all["field_id"].nunique()
assert "crop_stage" in df_multi.columns
print(f"OK ({len(df_multi)} field(s))")
print(df_multi[["field_id", "crop_stage", "stage_description", "peak_date", "days_since_peak"]].to_string(index=False))

print("7. run_crop_stage_from_gee forwards estimator kwargs...", end=" ")
ee_stub = types.ModuleType("ee")
ee_stub.Initialize = lambda *args, **kwargs: None
sys.modules["ee"] = ee_stub
from gee_fetch import run_crop_stage_from_gee

with patch("gee_fetch.fetch_ndvi", return_value=pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]), "NDVI": [0.2, 0.4, 0.3]})), patch("gee_fetch.smooth_daily_interpolate_ndvi", side_effect=lambda frame: frame.assign(NDVI_smooth=frame["NDVI"])), patch("gee_fetch.estimate_stage_adaptive", return_value={"Stage": "B", "Stage_description": "Greenup / rapid growth", "Value": 0.3, "Velocity": 0.0, "Last_date": None, "Peak_date": None, "Days_since_peak": None, "Upper_threshold": 0.6, "Lower_threshold": 0.3}) as mock_estimate:
    result = run_crop_stage_from_gee("dummy.geojson", lower_threshold=0.2)
assert result["Stage"] == "B"
assert mock_estimate.call_args.kwargs["lower_threshold"] == 0.2
print("OK")

print("\nAll tests passed.")
