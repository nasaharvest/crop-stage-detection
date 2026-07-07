"""
crop_stage.py
-------------
Crop stage estimation from NDVI time series.

Most row crops and vegetables follow a similar NDVI trajectory over a growing
season — a bell-curve-like rise to peak vegetative vigor, followed by a decline
through senescence. Duration, steepness, and width vary by crop, region, and
season, but the overall shape is consistent enough that a crop-agnostic,
curve-relative model can locate "where on the curve" a given observation falls,
without needing crop-specific calibration.

The curve is divided into five generic stages (A–E). Stage boundaries are
detected adaptively per field/season — not fixed calendar dates.

Stages
------
A  Bare soil / planting / emergence
B  Greenup / rapid growth
C  Peak maturity
D  Senescence
E  Post-harvest / residue / bare soil

Public API
----------
smooth_daily_interpolate_ndvi(df)
    Resample, interpolate, and smooth a raw NDVI DataFrame.

estimate_stage_adaptive(ndvi_series, dates=None, ...)
    Estimate current stage from a smoothed NDVI array.

run_crop_stage_from_dataframe(ndvi_df, ...)
    Convenience wrapper: smooth + estimate for one or many fields at once.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage labels (override via stage_descriptions parameter)
# ---------------------------------------------------------------------------

STAGE_DESCRIPTIONS: dict[str, str] = {
    "A": "Bare soil / planting / emergence",
    "B": "Greenup / rapid growth",
    "C": "Peak maturity",
    "D": "Senescence",
    "E": "Post-harvest / residue / bare soil",
    "Insufficient Data": "Insufficient data for reliable estimation",
}


# ---------------------------------------------------------------------------
# Whittaker smoother
# ---------------------------------------------------------------------------

def _whittaker_smooth(
    y: np.ndarray,
    lam: float = 6_000,
    differences: int = 2,
) -> np.ndarray:
    """
    Apply Whittaker smoothing to a 1-D signal.

    Solves the penalized least-squares problem
        min_z ||y - z||² + λ ||Dᵈz||²
    where D is the first-order difference operator and d = `differences`.
    Larger λ → smoother result.

    Parameters
    ----------
    y : np.ndarray   1-D array of observed values (e.g., NDVI).
    lam : float      Smoothing parameter λ (default 6 000).
    differences : int  Order of the finite-difference penalty (default 2).

    Returns
    -------
    np.ndarray  Smoothed signal with the same length as y.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("_whittaker_smooth expects a 1-D array.")
    if y.size == 0:
        raise ValueError("Input array must not be empty.")
    if differences < 1:
        raise ValueError("`differences` must be >= 1.")

    n = y.size
    E = sparse.eye(n, format="csr")
    D = E[1:, :] - E[:-1, :]
    for _ in range(differences - 1):
        D = D[1:, :] - D[:-1, :]

    A = E + lam * (D.T @ D)
    return np.asarray(spsolve(A, y), dtype=float)


# ---------------------------------------------------------------------------
# NDVI smoothing
# ---------------------------------------------------------------------------

def smooth_daily_interpolate_ndvi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample NDVI observations to daily frequency, interpolate gaps with
    PCHIP, and smooth with the Whittaker filter (HP filter as fallback).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least two columns:
        - ``date``  : datetime-like (any parseable format)
        - ``NDVI``  : float in [0, 1], cloud-free observations only

    Returns
    -------
    pd.DataFrame
        Columns ``date`` (daily, dtype datetime64), ``NDVI`` (PCHIP-interpolated daily), and ``NDVI_smooth``.

    Notes
    -----
    The input may be irregularly spaced (e.g., every 5–12 days from
    Sentinel-2 / Landsat). The function first resamples to a daily grid,
    fills gaps via monotone PCHIP interpolation, then applies Whittaker
    smoothing (λ = 6 000, 2nd differences) to suppress residual noise.
    """
    ndvi = (
        df.assign(date=pd.to_datetime(df["date"]))
        .sort_values("date")
        .set_index("date")[["NDVI"]]
        .resample("D")
        .mean()
    )
    ndvi["NDVI"] = ndvi["NDVI"].interpolate(method="pchip")

    try:
        ndvi["NDVI_smooth"] = _whittaker_smooth(ndvi["NDVI"].to_numpy())
    except Exception:
        from statsmodels.tsa.filters.hp_filter import hpfilter
        _, trend = hpfilter(ndvi["NDVI"], lamb=25_000)
        ndvi["NDVI_smooth"] = trend

    return ndvi.reset_index()


# ---------------------------------------------------------------------------
# Internal helpers for estimate_stage_adaptive
# ---------------------------------------------------------------------------

def _derive_upper_threshold(
    ndvi: np.ndarray,
    upper_percentile: float,
    min_peak_ndvi: float,
) -> float:
    return max(float(np.percentile(ndvi, upper_percentile)), min_peak_ndvi)


def _kalman_velocity(
    ndvi: np.ndarray,
    dts: np.ndarray,
    measurement_noise: float,
    process_noise_level: float,
    process_noise_velocity: float,
) -> float:
    """Constant-velocity Kalman filter; returns final velocity (NDVI/day)."""
    x = np.array([ndvi[0], 0.0])
    P = np.eye(2)
    H = np.array([[1.0, 0.0]])
    I2 = np.eye(2)

    for i in range(1, len(ndvi)):
        dt = dts[i]
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.array([[process_noise_level * dt, 0.0],
                      [0.0, process_noise_velocity * dt]])
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        innovation = float(ndvi[i]) - float(np.dot(H, x_pred).flat[0])
        S = float(np.dot(H, np.dot(P_pred, H.T)).flat[0]) + measurement_noise
        K = np.dot(P_pred, H.T) / S
        x = x_pred + K.flatten() * innovation
        P = (I2 - np.outer(K.flatten(), H.flatten())) @ P_pred

    return float(x[1])


def _find_last_peak(
    ndvi: np.ndarray,
    upper_threshold: float,
    min_peak_width: int,
) -> Optional[int]:
    """
    Return the index of the last genuine peak above upper_threshold,
    or None. A genuine peak is a contiguous run of >= min_peak_width
    observations above the threshold.
    """
    above = ndvi >= upper_threshold
    last_peak_idx = None
    i = 0
    while i < len(ndvi):
        if above[i]:
            j = i
            while j < len(ndvi) and above[j]:
                j += 1
            if (j - i) >= min_peak_width:
                last_peak_idx = i + int(np.argmax(ndvi[i:j]))
            i = j
        else:
            i += 1
    return last_peak_idx


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def estimate_stage_adaptive(
    ndvi_series,
    dates=None,
    upper_percentile: float = 90.0,
    min_peak_ndvi: float = 0.50,
    lower_threshold: float = 0.35,
    min_peak_width: int = 5,
    measurement_noise: float = 0.005,
    process_noise_level: float = 0.001,
    process_noise_velocity: float = 0.0001,
    min_observations: int = 30,
    stage_descriptions: Optional[dict[str, str]] = None,
) -> dict:
    """
    Estimate the current crop stage from an NDVI time series.

    The model is crop-agnostic: it detects where the current observation falls
    on the characteristic bell-curve NDVI trajectory shared by most row crops
    and vegetables. Thresholds adapt to the field's actual seasonal peak — no
    crop-specific calibration is needed.

    Parameters
    ----------
    ndvi_series : array-like
        1-D sequence of NDVI values ordered in time (typically the output of
        ``smooth_daily_interpolate_ndvi``).
    dates : array-like of datetime-like, optional
        Observation dates. When provided, used to compute time-varying dt for
        the Kalman filter and to populate ``Last_date`` / ``Peak_date`` in the
        result. When omitted, a uniform daily cadence (dt = 1) is assumed.
    upper_percentile : float
        Percentile of the series used as the upper (peak) threshold (default 90).
        The threshold adapts to each field's actual seasonal NDVI range.
    min_peak_ndvi : float
        Hard floor for the derived upper threshold (default 0.50). Prevents a
        low-biomass fluctuation from being mistaken for a genuine seasonal peak.
    lower_threshold : float
        Fixed lower threshold (default 0.35). NDVI below this is
        unconditionally assigned to the lower band (stage A or E).
    min_peak_width : int
        Minimum number of consecutive observations above the upper threshold
        required to confirm a genuine seasonal peak (default 5). Guards against
        noise spikes.
    measurement_noise : float
        Kalman observation noise variance R (default 0.005).
    process_noise_level : float
        Kalman process noise for the level state (default 0.001).
    process_noise_velocity : float
        Kalman process noise for the velocity state (default 0.0001).
    min_observations : int
        Minimum series length for a reliable stage estimate (default 30).
        Shorter series return stage ``"Insufficient Data"``.
    stage_descriptions : dict, optional
        Custom ``{stage_label: description}`` mapping. Overrides the built-in
        ``STAGE_DESCRIPTIONS``. Useful for translating labels or renaming stages.

    Returns
    -------
    dict with keys:
        ``Stage``             str    Stage label (A–E or "Insufficient Data")
        ``Stage_description`` str    Human-readable description
        ``Value``             float  Last NDVI value in the series
        ``Velocity``          float or None  Kalman-estimated rate of change (NDVI/day); None for stage C and "Insufficient Data"
        ``Last_date``         pd.Timestamp or None
        ``Peak_date``         pd.Timestamp or None
        ``Days_since_peak``   int or None
        ``Upper_threshold``   float or None  Adaptive upper threshold; None for "Insufficient Data"
        ``Lower_threshold``   float

    Decision logic
    --------------
    Position relative to thresholds + Kalman velocity + peak history:

    +-----------------------+-----------+----------------+-------+
    | NDVI position         | Velocity  | Peak confirmed | Stage |
    +=======================+===========+================+=======+
    | >= upper threshold    | any       | any            | C     |
    | mid band              | rising    | any            | B     |
    | mid band              | falling   | no             | B     |
    | mid band              | falling   | yes            | D     |
    | < lower threshold     | any       | no             | A     |
    | < lower threshold     | any       | yes            | E     |
    +-----------------------+-----------+----------------+-------+
    """
    ndvi = np.asarray(ndvi_series, dtype=float)
    descs = stage_descriptions or STAGE_DESCRIPTIONS

    last_date = None
    if dates is not None:
        parsed = pd.to_datetime(dates)
        last_date = pd.Timestamp(
            parsed.iloc[-1] if hasattr(parsed, "iloc") else parsed[-1]
        )

    def _result(stage, velocity=None, upper=None, peak_date=None):
        days_since_peak = (
            int((last_date - peak_date).days)
            if peak_date is not None and last_date is not None
            else None
        )
        return {
            "Stage":             stage,
            "Stage_description": descs.get(stage, stage),
            "Value":             float(ndvi[-1]) if len(ndvi) else None,
            "Velocity":          velocity,
            "Last_date":         last_date,
            "Peak_date":         peak_date,
            "Days_since_peak":   days_since_peak,
            "Upper_threshold":   upper,
            "Lower_threshold":   lower_threshold,
        }

    if len(ndvi) < min_observations:
        logger.debug(f"Series too short ({len(ndvi)} < {min_observations}).")
        return _result("Insufficient Data")

    upper_thresh = _derive_upper_threshold(ndvi, upper_percentile, min_peak_ndvi)
    current = float(ndvi[-1])

    if current >= upper_thresh:
        return _result("C", upper=upper_thresh)

    dts = (
        pd.Series(pd.to_datetime(dates)).diff().dt.days.fillna(1.0).clip(lower=1.0).to_numpy(dtype=float)
        if dates is not None
        else np.ones(len(ndvi), dtype=float)
    )

    try:
        velocity = _kalman_velocity(
            ndvi, dts, measurement_noise,
            process_noise_level, process_noise_velocity,
        )
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as e:
        logger.warning(f"Kalman failed ({e}). Falling back to raw gradient.")
        velocity = float(np.gradient(ndvi)[-1])

    last_peak_idx = _find_last_peak(ndvi, upper_thresh, min_peak_width)
    had_peak = last_peak_idx is not None

    if current >= lower_threshold:
        stage = "D" if (had_peak and velocity <= 0) else "B"
    else:
        stage = "E" if had_peak else "A"

    peak_date = None
    if had_peak and dates is not None:
        parsed_dates = pd.to_datetime(dates)
        peak_date = pd.Timestamp(
            parsed_dates.iloc[last_peak_idx]
            if hasattr(parsed_dates, "iloc")
            else parsed_dates[last_peak_idx]
        )

    return _result(stage, velocity=velocity, upper=upper_thresh, peak_date=peak_date)


# ---------------------------------------------------------------------------
# Convenience batch runner (no GEE — works on any NDVI DataFrame)
# ---------------------------------------------------------------------------

def run_crop_stage_from_dataframe(
    ndvi_df: pd.DataFrame,
    date_col: str = "date",
    ndvi_col: str = "NDVI",
    id_col: Optional[str] = None,
    max_workers: int = 8,
    **estimate_kwargs,
) -> dict | pd.DataFrame:
    """
    Smooth and estimate crop stage for one field or many fields at once.

    Parameters
    ----------
    ndvi_df : pd.DataFrame
        NDVI observations. Must contain ``date_col`` and ``ndvi_col``.
        If ``id_col`` is provided, the DataFrame may contain multiple fields.
    date_col : str   Column name for dates (default ``"date"``).
    ndvi_col : str   Column name for NDVI values (default ``"NDVI"``).
    id_col : str or None
        If None, treats the entire DataFrame as one field and returns a dict.
        If provided, groups by this column and returns a DataFrame with one
        row per field.
    max_workers : int  Thread-pool size when processing multiple fields (default 8).
    **estimate_kwargs
        Forwarded to ``estimate_stage_adaptive`` (e.g., ``lower_threshold=0.30``).

    Returns
    -------
    dict
        When ``id_col`` is None — raw result from ``estimate_stage_adaptive``.
    pd.DataFrame
        When ``id_col`` is provided — columns:
        [id_col, ``crop_stage``, ``stage_description``, ``peak_date``,
        ``days_since_peak``, ``last_date``].

    Example — single field
    ----------------------
    >>> import pandas as pd
    >>> from crop_stage import run_crop_stage_from_dataframe
    >>> df = pd.read_csv("sample_data/sample_ndvi.csv", parse_dates=["date"])
    >>> result = run_crop_stage_from_dataframe(df)
    >>> print(result["Stage"], result["Stage_description"])

    Example — multiple fields
    -------------------------
    >>> df = pd.read_csv("my_ndvi_data.csv", parse_dates=["date"])
    >>> results = run_crop_stage_from_dataframe(df, id_col="field_id")
    >>> print(results)
    """
    def _run_single(df_group, field_id=None):
        no_result = {
            id_col:             field_id,
            "crop_stage":       None,
            "stage_description": None,
            "peak_date":        None,
            "days_since_peak":  None,
            "last_date":        None,
        }
        try:
            df_std = (
                df_group
                .rename(columns={date_col: "date", ndvi_col: "NDVI"})
                [["date", "NDVI"]]
                .assign(date=lambda d: pd.to_datetime(d["date"]))
                .dropna()
            )
            if len(df_std) < 3:
                return no_result
            df_smooth = smooth_daily_interpolate_ndvi(df_std)
            result = estimate_stage_adaptive(
                df_smooth["NDVI_smooth"].to_numpy(),
                dates=df_smooth["date"],
                **estimate_kwargs,
            )
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

    # Single-field mode
    if id_col is None:
        df_std = (
            ndvi_df
            .rename(columns={date_col: "date", ndvi_col: "NDVI"})
            [["date", "NDVI"]]
            .assign(date=lambda d: pd.to_datetime(d["date"]))
            .dropna()
        )
        if len(df_std) < 3:
            return estimate_stage_adaptive(np.array([]), **estimate_kwargs)
        df_smooth = smooth_daily_interpolate_ndvi(df_std)
        return estimate_stage_adaptive(
            df_smooth["NDVI_smooth"].to_numpy(),
            dates=df_smooth["date"],
            **estimate_kwargs,
        )

    # Multi-field mode
    groups = {fid: grp for fid, grp in ndvi_df.groupby(id_col)}
    logger.info(f"Running crop stage for {len(groups)} fields")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        futures = {exc.submit(_run_single, grp, fid): fid for fid, grp in groups.items()}
        for f in tqdm(as_completed(futures), total=len(futures)):
            results.append(f.result())

    return pd.DataFrame(results)
