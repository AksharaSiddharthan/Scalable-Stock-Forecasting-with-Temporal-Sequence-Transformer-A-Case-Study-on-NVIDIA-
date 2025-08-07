# Scalable Stock Forecasting with Temporal Sequence Transformer: A Case Study on NVIDIA


## Overview

This project implements a **Transformer-based Time Series model (Temporal Sequence Transformer - TST)** to forecast **NVIDIA Corporation's (NVDA)** stock **Open** prices. The goal is to improve prediction accuracy and demonstrate the scalability and efficiency of transformer models for financial time series data.

---

## Model Details

* **Model Type**: Temporal Sequence Transformer (TST)
* **Framework**: PyTorch
* **Language**: Python
* **Input Features**: Open, High, Low, Close, Volume
* **Target**: Open price
* **Forecast Horizon**: 14 days
* **Sequence Length**: 20 days

---

## Dataset

* **Source**: [Macrotrends.net](https://www.macrotrends.net/)
* **Company**: NVIDIA Corporation (NVDA)
* **Time Period**: 1999–2024
* **Data Preprocessing**:

  * Missing values handled
  * Standardization or normalization (based on config)

---

## Project Structure

| File              | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `models.py`       | Transformer model architecture (TST)                         |
| `modules.py`      | Position encoding, attention, and other components           |
| `datasets.py`     | Defines the PyTorch Dataset                                  |
| `train.py`        | Training, evaluation, and plotting logic                     |
| `config.json`     | Hyperparameter configuration for model and training          |
| `run_results.txt` | Logged evaluation metrics for different model runs           |




---

## Key Findings

* The Transformer model effectively learned temporal dependencies in stock data.
* Achieved high R² values and low errors across splits.
* TST demonstrates better scalability and interpretability than traditional models like LSTM for this task.

---

## Development Info

* **IDE**: Visual Studio Code (VSCode)
* **Environment**: Local (CPU)
* **Dependencies**:

  * PyTorch
  * NumPy
  * Matplotlib
  * Pandas

---

## License

This project is part of an academic capstone submission and is released under the License.

---

