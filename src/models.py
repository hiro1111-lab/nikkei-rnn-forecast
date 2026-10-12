"""Recurrent model definitions."""
from tensorflow import keras

RNN_LAYERS = {
    "SimpleRNN": keras.layers.SimpleRNN,
    "LSTM": keras.layers.LSTM,
    "GRU": keras.layers.GRU,
}


def build_model(model_type: str, n_steps: int, n_units: int, learning_rate: float) -> keras.Model:
    """Build and compile a single-layer RNN followed by a linear output.

    Every model shares the same architecture except for the recurrent cell,
    so differences in results can be attributed to the cell type.
    """
    if model_type not in RNN_LAYERS:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(RNN_LAYERS)}")

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_steps, 1)),
            RNN_LAYERS[model_type](n_units),
            keras.layers.Dense(1),
        ],
        name=model_type,
    )
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    return model
