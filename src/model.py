"""
Model definitions for adaptive time-series forecasting.

This module provides simple neural-network models that can be used
for supervised time-series forecasting experiments.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam


def build_mlp(input_dim: int, hidden_units: int = 64, learning_rate: float = 0.001):
    """
    Build a multilayer perceptron model for one-step-ahead forecasting.

    Parameters
    ----------
    input_dim : int
        Number of lagged observations used as input features.
    hidden_units : int
        Number of hidden units in the dense layers.
    learning_rate : float
        Learning rate for the Adam optimizer.

    Returns
    -------
    tensorflow.keras.Model
        Compiled MLP forecasting model.
    """
    model = Sequential(
        [
            Dense(hidden_units, activation="relu", input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(hidden_units // 2, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model


def build_lstm(window_size: int, hidden_units: int = 64, learning_rate: float = 0.001):
    """
    Build an LSTM model for one-step-ahead forecasting.

    Parameters
    ----------
    window_size : int
        Number of time steps used as input.
    hidden_units : int
        Number of LSTM hidden units.
    learning_rate : float
        Learning rate for the Adam optimizer.

    Returns
    -------
    tensorflow.keras.Model
        Compiled LSTM forecasting model.
    """
    model = Sequential(
        [
            LSTM(hidden_units, input_shape=(window_size, 1)),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model
