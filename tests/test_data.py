import numpy as np
import pytest

from src.data import StandardScaler, chronological_split, make_windows, prepare_datasets
from src.evaluate import naive_forecast, regression_metrics


def test_make_windows_shapes_and_values():
    values = np.arange(10, dtype=np.float32)
    X, y = make_windows(values, n_steps=3)
    # 10 values, window of 3 + 1 target -> 7 samples (the last window is included)
    assert X.shape == (7, 3, 1)
    assert y.shape == (7,)
    np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2])
    assert y[0] == 3
    np.testing.assert_array_equal(X[-1, :, 0], [6, 7, 8])
    assert y[-1] == 9


def test_make_windows_rejects_short_series():
    with pytest.raises(ValueError):
        make_windows(np.arange(3, dtype=np.float32), n_steps=3)


def test_chronological_split_keeps_order():
    X, y = make_windows(np.arange(20, dtype=np.float32), n_steps=2)
    (X_tr, y_tr), (X_te, y_te) = chronological_split(X, y, 0.8)
    assert len(X_tr) + len(X_te) == len(X)
    assert y_tr.max() < y_te.min()  # test period lies strictly after training


def test_scaler_roundtrip():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    s = StandardScaler().fit(v)
    np.testing.assert_allclose(s.inverse_transform(s.transform(v)), v)


def test_scaler_uses_training_data_only():
    # A huge jump in the test period must not affect the scaling statistics.
    # 95 windows -> 76 for training, covering values[0:81]; the jump starts at 85.
    values = np.concatenate([np.ones(85), np.full(15, 1000.0)]).astype(np.float32)
    values[:85] += np.linspace(0, 1, 85, dtype=np.float32)
    _, _, scaler = prepare_datasets(values, n_steps=5, train_ratio=0.8)
    assert scaler.mean < 2.0


def test_naive_forecast_and_metrics():
    X, y = make_windows(np.arange(10, dtype=np.float32) + 1, n_steps=3)
    pred = naive_forecast(X)
    m = regression_metrics(y, pred)
    assert m["MAE"] == pytest.approx(1.0)
    assert m["RMSE"] == pytest.approx(1.0)
