# 📈 Stock Price Prediction — ML Dashboard

> Predict future share prices using Machine Learning on 10,000+ historical data points.
> Supports 100 stocks, 7 algorithms, 50+ technical features.

---

## 🚀 Quick Start

### Option A — Run Script (Linux / Mac)
```bash
chmod +x run.sh
./run.sh
```

### Option B — Manual
```bash
pip install -r requirements.txt
streamlit run Home.py
```

Open your browser at **http://localhost:8501**

---

## 📁 Project Structure

```
stock_predictor/
├── Home.py                    ← Main app (single-stock prediction)
├── requirements.txt
├── run.sh                     ← One-click launcher
├── .streamlit/
│   └── config.toml            ← Dark theme config
├── pages/
│   ├── 1_Multi_Stock_Scan.py  ← Scan up to 100 stocks
│   ├── 2_Model_Comparison.py  ← Compare all 7 models
│   └── 3_Data_Explorer.py     ← Explore raw + feature data
└── utils/
    ├── data_engine.py         ← Data fetch + 50+ feature engineering
    ├── ml_engine.py           ← Training, evaluation, forecasting
    ├── charts.py              ← Plotly chart builders
    └── ui.py                  ← CSS + reusable components
```

---

## 🧠 Machine Learning Models

| Model | Type |
|---|---|
| XGBoost | Gradient-boosted trees |
| Random Forest | Bagging ensemble |
| Gradient Boosting | Sequential ensemble |
| Extra Trees | Randomized ensemble |
| Ridge Regression | Regularized linear |
| Lasso Regression | Sparse linear |
| Linear Regression | Baseline linear |

---

## 📊 Features Engineered (50+)

- **Price Lags:** t-1 to t-30 day lagged closes
- **Returns:** 1, 2, 3, 5, 7, 10, 14, 21, 30-day returns
- **SMA/EMA:** 5, 10, 20, 50, 100, 200 period moving averages
- **MACD:** Line, signal, histogram
- **RSI:** 7, 14, 21 period
- **Bollinger Bands:** Width, %B
- **ATR:** 14-period average true range
- **Stochastic:** %K, %D
- **Volume:** OBV, CMF, volume ratio
- **Candle:** Body %, H-L range, upper/lower wicks, gap
- **Calendar:** Day of week, month, quarter

---

## 📄 Pages

| Page | Description |
|---|---|
| 🏠 Home | Single stock prediction + N-day forecast |
| 📊 Multi-Stock Scan | Batch scan 100 stocks, rank by signal |
| 🆚 Model Comparison | Compare all 7 models on one stock |
| 📋 Data Explorer | OHLCV, features, correlations, distributions |

---

## ⚙️ Requirements

- Python 3.9+
- Internet connection (data from Yahoo Finance)
- ~500 MB RAM

---

## 📝 Notes

- Data is fetched with `period="max"` — most large-cap stocks have **10,000–15,000+** daily data points (since ~1980)
- The model predicts **next-day close price**
- The multi-day forecast uses **walk-forward** prediction (each predicted price feeds the next step)
- All models are retrained fresh on each run (no saved weights)
