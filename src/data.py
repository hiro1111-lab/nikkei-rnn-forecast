"""Data loading, windowing, splitting and scaling for time-series forecasting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def load_series(csv_path: str, column: str, encoding: str = "shift_jis") -> np.ndarray:
    """Load one numeric column from a CSV file as a 1-D float32 array.

    Column names are stripped of whitespace and BOM characters, which often
    appear in CSV files exported on Windows.
    """
    df = pd.read_csv(csv_path, encoding=encoding)
    df.columns = df.columns.str.strip().str.replace("﻿", "", regex=False)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")

    # Values such as "38,000.50" are read as strings; remove thousands separators.
    values = pd.to_numeric(
        df[column].astype(str).str.replace(",", "", regex=False), errors="coerce"
    ).dropna()
    return values.to_numpy(dtype=np.float32)


def make_windows(values: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn a 1-D series into supervised (X, y) pairs with a sliding window.

    X[i] = values[i : i + n_steps]      (shape: n_steps x 1)
    y[i] = values[i + n_steps]          (the next value)

    Returns X with shape (n_samples, n_steps, 1) and y with shape (n_samples,).
    Uses a strided view, so it runs in O(n) memory-efficient time.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if len(values) <= n_steps:
        raise ValueError(f"Series of length {len(values)} is too short for n_steps={n_steps}")

    windows = np.lib.stride_tricks.sliding_window_view(values, n_steps + 1)
    X = windows[:, :n_steps].copy()[..., np.newaxis]
    y = windows[:, n_steps].copy()
    return X, y


def chronological_split(X: np.ndarray, y: np.ndarray, train_ratio: float):
    """Split without shuffling so that the test period lies strictly after training."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    n_train = int(len(X) * train_ratio)
    return (X[:n_train], y[:n_train]), (X[n_train:], y[n_train:])


@dataclass
class StandardScaler:
    """Z-score scaler. Fit on training data only to avoid look-ahead leakage."""

    mean: float = 0.0
    std: float = 1.0

    def fit(self, values: np.ndarray) -> "StandardScaler":
        self.mean = float(np.mean(values))
        self.std = float(np.std(values)) or 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


def prepare_datasets(values: np.ndarray, n_steps: int, train_ratio: float):
    """Window, split chronologically, and scale using training statistics only.

    Returns (X_train, y_train), (X_test, y_test), scaler — all scaled.
    """
    X, y = make_windows(values, n_steps)
    (X_train, y_train), (X_test, y_test) = chronological_split(X, y, train_ratio)

    # Statistics come only from values visible in the training period.
    n_train_values = len(y_train) + n_steps
    scaler = StandardScaler().fit(values[:n_train_values])

    return (
        (scaler.transform(X_train), scaler.transform(y_train)),
        (scaler.transform(X_test), scaler.transform(y_test)),
        scaler,
    )
