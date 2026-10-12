"""Experiment configuration.

All hyperparameters live here so that an experiment is fully described
by a single object (and can be overridden from the command line).
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # Data
    csv_path: str = "data/nikkei_stock_average_daily_jp.csv"
    column: str = "終値"          # closing price column in the source CSV
    encoding: str = "shift_jis"

    # Windowing: use the past `n_steps` days to predict the next day
    n_steps: int = 50
    train_ratio: float = 0.8

    # Model / training
    models: tuple = ("SimpleRNN", "LSTM", "GRU")
    n_units: int = 20
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.01
    val_ratio: float = 0.1        # fraction of the training set used for validation
    seed: int = 42

    # Output
    results_dir: str = "results"
    extra: dict = field(default_factory=dict)
