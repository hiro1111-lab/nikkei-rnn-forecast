"""Compare SimpleRNN, LSTM and GRU for next-day Nikkei 225 closing price prediction.

Usage:
    python main.py --csv data/nikkei_stock_average_daily_jp.csv
    python main.py --epochs 100 --n-steps 30
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import Config
from src.data import load_series, prepare_datasets
from src.evaluate import (
    naive_forecast,
    plot_learning_curves,
    plot_predictions,
    regression_metrics,
)
from src.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=cfg.csv_path, help="path to the price CSV")
    p.add_argument("--column", default=cfg.column, help="price column name")
    p.add_argument("--encoding", default=cfg.encoding)
    p.add_argument("--n-steps", type=int, default=cfg.n_steps, help="input window length (days)")
    p.add_argument("--units", type=int, default=cfg.n_units)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--lr", type=float, default=cfg.learning_rate)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--results-dir", default=cfg.results_dir)
    a = p.parse_args()
    return Config(
        csv_path=a.csv, column=a.column, encoding=a.encoding, n_steps=a.n_steps,
        n_units=a.units, epochs=a.epochs, learning_rate=a.lr, seed=a.seed,
        results_dir=a.results_dir,
    )


def run(cfg: Config) -> pd.DataFrame:
    set_seed(cfg.seed)
    out = Path(cfg.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    values = load_series(cfg.csv_path, cfg.column, cfg.encoding)
    (X_train, y_train), (X_test, y_test), scaler = prepare_datasets(values, cfg.n_steps, cfg.train_ratio)
    print(f"Loaded {len(values)} days -> train {len(X_train)} / test {len(X_test)} windows")

    y_true = scaler.inverse_transform(y_test)
    predictions = {"Naive (persistence)": scaler.inverse_transform(naive_forecast(X_test))}
    histories = {}

    for name in cfg.models:
        print(f"\n=== {name} ===")
        model = build_model(name, cfg.n_steps, cfg.n_units, cfg.learning_rate)
        history = model.fit(
            X_train, y_train,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            validation_split=cfg.val_ratio,  # last 10% of training windows, in time order
            shuffle=True,
            verbose=0,
        )
        histories[name] = history.history
        y_pred = model.predict(X_test, verbose=0).ravel()
        predictions[name] = scaler.inverse_transform(y_pred)
        print(f"final train loss={history.history['loss'][-1]:.5f}, "
              f"val loss={history.history['val_loss'][-1]:.5f}")

    table = pd.DataFrame(
        {name: regression_metrics(y_true, y_pred) for name, y_pred in predictions.items()}
    ).T.round(2)
    print("\n=== Test metrics (original price scale) ===")
    print(table.to_string())

    table.to_csv(out / "metrics.csv")
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2, default=list)
    plot_predictions(y_true, predictions, out / "predictions.png")
    plot_learning_curves(histories, out / "learning_curves.png")
    print(f"\nSaved results to {out.resolve()}")
    return table


if __name__ == "__main__":
    run(parse_args())
