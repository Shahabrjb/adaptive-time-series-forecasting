# Adaptive Time-Series Forecasting with Machine Learning

A compact, reproducible Python project for experimenting with **neural-network-based time-series forecasting** and classical evaluation workflows.

The repository is inspired by my published research on **adaptive learnable window-size selection** for time-series prediction. The current public implementation provides the reusable forecasting foundation — preprocessing, supervised window generation, MLP/LSTM model definitions, training, and regression evaluation — while the full adaptive-window research method remains documented separately in the published work.

## Current Public Implementation

The repository currently includes:

- CSV-based univariate time-series loading
- Min-Max scaling
- chronological supervised-window generation
- MLP forecasting model
- LSTM model definition for sequential experiments
- train/test splitting without temporal shuffling
- MAE, RMSE, MAPE, and R² evaluation utilities
- command-line training workflow

## Project Structure

```text
adaptive-time-series-forecasting/
├── src/
│   ├── train.py       # data loading, windowing, training pipeline
│   ├── model.py       # MLP and LSTM model definitions
│   └── evaluate.py    # regression metrics and reporting
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Input Data

Use a CSV file containing at least one numeric time-series column. For example:

```csv
value
101.2
102.4
101.8
103.1
```

## Example Usage

```bash
python src/train.py --csv data/example.csv --value-column value --window-size 14 --epochs 50
```

The current `train.py` workflow creates chronological supervised windows, trains the public MLP model, predicts the held-out final portion of the series, reverses the scaling transformation, and reports regression metrics.

## Available Models

`src/model.py` currently defines:

- **MLP** — dense neural network for fixed-length lag vectors
- **LSTM** — recurrent model definition for sequence-based experiments

The public command-line trainer currently uses the MLP path. The LSTM definition is available for further experiments and extension.

## Research Background

My published work, **“MLP-based Learnable Window Size for Bitcoin Price Prediction” (Applied Soft Computing, 2022)**, investigated a two-stage framework in which a first neural model learns an appropriate historical input-window size and a second neural model performs forecasting.

That research motivates a broader question that continues across my current work:

> Can machine-learning systems adapt important modelling choices to changing data conditions instead of relying on one permanently fixed setting?

This same research direction now extends into my work on adaptive anomaly detection for wireless-network optimization.

## Evaluation

The repository reports standard regression metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R²

## Related Research

- [MLP-based Learnable Window Size for Bitcoin Price Prediction](https://github.com/Shahabrjb/MLP-based-Learnable-Window-Size-for-Bitcoin-Price-Prediction)
- [Adaptive RAN Anomaly Detection — research overview](https://github.com/Shahabrjb/About-Me/blob/main/projects/adaptive-ran-anomaly-detection/README.md)
- [Research portfolio](https://github.com/Shahabrjb/About-Me)

## Author

**Shahab Rajabi**  
Machine Learning & RAN Optimization Engineer  
Research interests: adaptive ML, time-series forecasting, anomaly detection, and AI for wireless networks.
