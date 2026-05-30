"""
Example training pipeline for adaptive time-series forecasting.

This script demonstrates:
- loading a univariate time-series CSV file,
- creating supervised learning windows,
- training an MLP model,
- evaluating forecasting performance.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from model import build_mlp
from evaluate import regression_metrics, print_metrics


def create_windows(series, window_size: int):
    """
    Convert a univariate time series into supervised learning windows.

    Parameters
    ----------
    series : array-like
        One-dimensional time-series values.
    window_size : int
        Number of past observations used to predict the next value.

    Returns
    -------
    X : np.ndarray
        Input windows with shape (n_samples, window_size).
    y : np.ndarray
        Target values with shape (n_samples,).
    """
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])

    return np.asarray(X), np.asarray(y)


def load_series(csv_path: str, value_column: str):
    """
    Load a univariate time series from a CSV file.

    The CSV file should include at least one numeric column containing
    the target time-series values.
    """
    df = pd.read_csv(csv_path)

    if value_column not in df.columns:
        raise ValueError(
            f"Column '{value_column}' not found. Available columns: {list(df.columns)}"
        )

    values = df[value_column].astype(float).values.reshape(-1, 1)
    return values


def main():
    parser = argparse.ArgumentParser(description="Train an MLP forecasting model.")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument(
        "--value-column",
        type=str,
        default="value",
        help="Name of the numeric time-series column.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=14,
        help="Number of lagged observations used as input.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")

    args = parser.parse_args()

    values = load_series(args.csv, args.value_column)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values).reshape(-1)

    X, y = create_windows(scaled_values, args.window_size)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = build_mlp(input_dim=args.window_size)

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    y_pred = model.predict(X_test).reshape(-1)

    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)

    metrics = regression_metrics(y_test_original, y_pred_original)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
