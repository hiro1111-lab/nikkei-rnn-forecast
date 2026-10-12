"""Metrics, baseline, and plotting."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE and MAPE (%) in the original price scale."""
    err = y_pred - y_true
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE(%)": float(np.mean(np.abs(err / y_true)) * 100),
    }


def naive_forecast(X: np.ndarray) -> np.ndarray:
    """Persistence baseline: tomorrow's price = today's price.

    For daily stock prices this is a surprisingly strong baseline;
    a model is only useful if it beats it.
    """
    return X[:, -1, 0]


def plot_predictions(y_true, predictions: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_true, label="Actual", color="black", linewidth=2)
    for name, y_pred in predictions.items():
        ax.plot(y_pred, label=name, alpha=0.8)
    ax.set_xlabel("Test day")
    ax.set_ylabel("Closing price (JPY)")
    ax.set_title("Next-day Nikkei 225 close: actual vs. predicted (test period)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curves(histories: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(histories), figsize=(5 * len(histories), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, hist) in zip(axes, histories.items()):
        ax.plot(hist["loss"], label="train")
        if "val_loss" in hist:
            ax.plot(hist["val_loss"], label="validation")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("MSE (scaled)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
