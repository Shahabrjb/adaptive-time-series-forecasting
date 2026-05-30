# Adaptive Time-Series Forecasting with Machine Learning

This repository contains a machine learning workflow for adaptive time-series forecasting using neural-network-based models and classical baseline methods.

The project is inspired by my published research on deep learning-based forecasting, where an adaptive learnable window-size strategy was used to improve prediction performance under changing time-series conditions.

## Project Goal

The goal of this project is to demonstrate a clean and reproducible forecasting pipeline, including:

- time-series preprocessing
- supervised learning window generation
- neural-network-based forecasting
- baseline model comparison
- model evaluation using standard regression metrics
- visualization of predicted versus observed values

## Methods

The repository is designed to support experiments with:

- Multilayer Perceptron models
- LSTM-based forecasting models
- Support Vector Regression baselines
- ARIMA or statistical forecasting baselines
- error analysis and model comparison

## Relevance

This project demonstrates experience in:

- Python-based machine learning
- time-series forecasting
- neural network model design
- data preprocessing
- model evaluation
- reproducible ML workflows

## Example Usage

After installing the required packages, a forecasting experiment can be run with:

```bash
python src/train.py --csv data/example.csv --value-column value --window-size 14 --epochs 50

The script creates supervised learning windows from a univariate time series, trains an MLP forecasting model, and reports MAE, RMSE, MAPE, and R².

My Contribution

I designed this repository to demonstrate a structured machine learning workflow for time-series forecasting, including data preprocessing, supervised window generation, neural-network model definition, and regression-based evaluation. The structure is inspired by my published work on adaptive deep learning for time-series prediction.
