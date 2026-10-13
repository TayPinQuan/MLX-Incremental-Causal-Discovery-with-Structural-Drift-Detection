"""
data/loaders.py
---------------
Data loaders for Tennessee-Eastman (TE) and SKAB datasets.

Both loaders return (data, meta) where:
    data: (T, n_vars) float32 numpy array (normalised)
    meta: dict with dataset-specific metadata
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ---------------------------------------------------------------------------
# Tennessee-Eastman Process (TEP)
# ---------------------------------------------------------------------------

def load_tennessee_eastman(
    data_dir: Union[str, Path],
    fault_id: int = 0,
    split: str = "train",     # 'train' | 'test'
    normalize: bool = True,
) -> tuple[np.ndarray, dict]:
    """Load a Tennessee-Eastman fault simulation file.

    Expects files named ``d{fault_id:02d}.dat`` (training) or
    ``d{fault_id:02d}_te.dat`` (test) in *data_dir*.

    The TE dataset has 52 variables measured at 3-minute intervals.
    Fault introduction begins at sample 160 in the test set.

    Args:
        data_dir:  Directory containing the TE data files.
        fault_id:  Fault case ID (0 = normal, 1-21 = fault cases).
        split:     'train' or 'test'.
        normalize: Standardise each variable to zero mean and unit variance.

    Returns:
        data: (T, 52) float32 array.
        meta: dict with dataset info.
    """
    assert HAS_PANDAS, "pandas is required for TE loader."
    data_dir = Path(data_dir)

    if split == "train":
        fname = data_dir / f"d{fault_id:02d}.dat"
    else:
        fname = data_dir / f"d{fault_id:02d}_te.dat"

    if not fname.exists():
        raise FileNotFoundError(
            f"Tennessee-Eastman file not found: {fname}\n"
            "Download from: https://www.cs.cmu.edu/~myung/APADS/tep_data/"
        )

    df = pd.read_csv(fname, delim_whitespace=True, header=None)
    data = df.values.astype(np.float32)

    if normalize:
        mu = data.mean(axis=0, keepdims=True)
        sigma = data.std(axis=0, keepdims=True) + 1e-8
        data = (data - mu) / sigma

    meta = {
        "dataset": "Tennessee-Eastman",
        "fault_id": fault_id,
        "split": split,
        "n_vars": data.shape[1],
        "n_samples": data.shape[0],
        "fault_start": 160 if split == "test" else None,
        "sample_interval_min": 3,
    }
    return data, meta


# ---------------------------------------------------------------------------
# SKAB (Skoltech Anomaly Benchmark)
# ---------------------------------------------------------------------------

def load_skab(
    data_dir: Union[str, Path],
    file_name: Optional[str] = None,
    normalize: bool = True,
) -> tuple[np.ndarray, dict]:
    """Load a SKAB experiment file.

    SKAB files are CSVs with columns:
        datetime, Accelerometer1RMS, Accelerometer2RMS, Current,
        Pressure, Temperature, Thermocouple, Voltage, Volume Flow RateRMS,
        anomaly, changepoint

    We drop the last two columns (labels) and the datetime index.

    Args:
        data_dir:  Directory containing SKAB CSV files.
        file_name: Specific file to load. If None, loads the first CSV found.
        normalize: Standardise each variable.

    Returns:
        data: (T, 8) float32 array (8 sensor variables).
        meta: dict with dataset info.
    """
    assert HAS_PANDAS, "pandas is required for SKAB loader."
    data_dir = Path(data_dir)

    if file_name is None:
        csv_files = sorted(data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {data_dir}. "
                "Download from: https://github.com/waico/SKAB"
            )
        path = csv_files[0]
    else:
        path = data_dir / file_name

    df = pd.read_csv(path, sep=";", index_col=0, parse_dates=True)
    label_cols = [c for c in df.columns if c.lower() in ("anomaly", "changepoint")]
    anomaly_labels = None
    changepoint_labels = None
    if "anomaly" in df.columns:
        anomaly_labels = df["anomaly"].values.astype(int)
    if "changepoint" in df.columns:
        changepoint_labels = df["changepoint"].values.astype(int)

    sensor_df = df.drop(columns=label_cols, errors="ignore")
    data = sensor_df.values.astype(np.float32)

    if normalize:
        mu = data.mean(axis=0, keepdims=True)
        sigma = data.std(axis=0, keepdims=True) + 1e-8
        data = (data - mu) / sigma

    meta = {
        "dataset": "SKAB",
        "file": str(path),
        "n_vars": data.shape[1],
        "n_samples": data.shape[0],
        "columns": list(sensor_df.columns),
        "anomaly_labels": anomaly_labels,
        "changepoint_labels": changepoint_labels,
    }
    return data, meta
