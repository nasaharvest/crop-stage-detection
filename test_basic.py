"""
Quick smoke test — run from the repo root:
    python test_basic.py
"""

import sys
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
df = pd.read_csv("sample_data/sample_ndvi.csv", parse_dates=["date"])
assert len(df) > 0, "Empty CSV"
assert "date" in df.columns and "NDVI" in df.columns
print(f"OK ({len(df)} rows)")

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
if result["Value"] is not None:
    print(f"       Value={result['Value']:.3f}  Upper={result['Upper_threshold']:.3f}  Lower={result['Lower_threshold']:.3f}")
if result["Peak_date"]:
    print(f"       Peak={result['Peak_date'].date()}  Days since peak={result['Days_since_peak']}")

print("5. run_crop_stage_from_dataframe (single field)...", end=" ")
r = run_crop_stage_from_dataframe(df)
assert r["Stage"] == result["Stage"]
print("OK")

print("6. run_crop_stage_from_dataframe (multi-field via id_col)...", end=" ")
df_multi = run_crop_stage_from_dataframe(df, id_col="field_id")
assert len(df_multi) == df["field_id"].nunique()
assert "crop_stage" in df_multi.columns
print(f"OK ({len(df_multi)} field(s))")
print(df_multi[["field_id", "crop_stage", "stage_description", "peak_date", "days_since_peak"]].to_string(index=False))

print("\nAll tests passed.")
