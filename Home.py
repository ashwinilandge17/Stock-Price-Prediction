"""
Home.py  –  Main entry point
Run:  streamlit run Home.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="Stock Price Predictor — ML",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.ui import inject_css, metric_card, section_header
from utils.data_engine import (
    TOP_100_STOCKS, STOCK_NAMES, fetch_stock_data, fetch_stock_info, add_features
)
from utils.ml_engine import train_and_evaluate, forecast_future
from utils.charts import (
    candlestick_chart, prediction_chart, actual_vs_predicted_chart,
    scatter_pred_chart, feature_importance_chart, macd_chart
)

inject_css()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Stock Predictor")
    st.markdown("*ML-powered future price forecast*")
    st.markdown("---")

    st.markdown("### 🔎 Stock Selection")
    ticker = st.selectbox(
        "Choose Ticker",
        TOP_100_STOCKS,
        format_func=lambda x: f"{x} — {STOCK_NAMES.get(x, x)}",
        index=0
    )

    st.markdown("### 🤖 Model")
    from utils.ml_engine import get_models
    model_name = st.selectbox("Algorithm", list(get_models().keys()))

    st.markdown("### ⚙️ Settings")
    test_pct   = st.slider("Test Split %", 5, 25, 15)
    forecast_d = st.slider("Forecast Horizon (days)", 5, 90, 30)

    st.markdown("### 📊 Chart")
    chart_period = st.selectbox("History to display",
                                ["3 months", "6 months", "1 year",
                                 "3 years", "5 years", "All"])
    show_macd = st.checkbox("Show MACD Chart", True)
    show_fi   = st.checkbox("Show Feature Importances", True)

    st.markdown("---")
    run_btn = st.button("🚀  Run Prediction", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    **Navigation**
    - 🏠 Home (Single Stock)
    - 📊 [Multi-Stock Scan](/Multi_Stock_Scan)
    - 🆚 [Model Comparison](/Model_Comparison)
    - 📋 [Data Explorer](/Data_Explorer)
    """)
    st.caption("Data: Yahoo Finance · v1.0")

# ─────────────────────────────────────────────────────────────────────────────
# LANDING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 📈 Stock Price Prediction")
st.markdown("##### Forecast future share prices using Machine Learning on historical data")
st.markdown("---")

if not run_btn:
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("100", "Stocks Covered", "S&P 500 + NASDAQ")
    with c2: metric_card("50+", "Features Engineered", "per stock")
    with c3: metric_card("10K+", "Data Points", "per ticker (max history)")
    with c4: metric_card("7", "ML Algorithms", "XGBoost, RF, GBM …")

    st.markdown("")
    st.info("👈 Select a stock & model in the sidebar, then click **Run Prediction**.")

    section_header("How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **① Fetch Data**
        Downloads full price history from Yahoo Finance — most large-cap stocks
        have 40+ years of daily OHLCV data (10,000 + points).
        """)
    with col2:
        st.markdown("""
        **② Engineer Features**
        Computes 50+ indicators: SMA/EMA, RSI, MACD, Bollinger Bands, ATR,
        Stochastic, OBV, candle patterns, calendar features & lagged returns.
        """)
    with col3:
        st.markdown("""
        **③ Train & Forecast**
        Trains the chosen model on 85% of data, evaluates on held-out 15%,
        then generates a multi-day walk-forward future price forecast.
        """)

    section_header("Supported Algorithms")
    algos = pd.DataFrame({
        "Algorithm": ["XGBoost","Random Forest","Gradient Boosting",
                       "Extra Trees","Ridge Regression","Lasso Regression","Linear Regression"],
        "Type": ["Ensemble","Ensemble","Ensemble","Ensemble","Linear","Linear","Linear"],
        "Best For": [
            "Non-linear patterns, high accuracy",
            "Robust, low overfitting",
            "Strong sequential learner",
            "Fast, low variance",
            "Regularized baseline",
            "Sparse feature selection",
            "Interpretable baseline",
        ]
    })
    st.dataframe(algos, use_container_width=True, hide_index=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"📥 Downloading historical data for **{ticker}** …"):
    raw = fetch_stock_data(ticker)

if raw.empty:
    st.error(f"Could not fetch data for **{ticker}**. Check your internet connection.")
    st.stop()

n_pts = len(raw)

with st.spinner("ℹ️ Fetching company info …"):
    info = fetch_stock_info(ticker)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"## {info['name']} &nbsp; `{ticker}`")
col_info = st.columns([1,1,1,1,1])
tags = [
    info.get("sector","N/A"), info.get("industry","N/A"),
    info.get("country","N/A"), info.get("currency","USD")
]
for c, t in zip(col_info, tags):
    c.markdown(f'<span class="tag">{t}</span>', unsafe_allow_html=True)
st.markdown("")

# ── KPI row ──────────────────────────────────────────────────────────────────
price_now   = float(raw["Close"].iloc[-1])
price_prev  = float(raw["Close"].iloc[-2])
change      = price_now - price_prev
change_pct  = change / price_prev * 100
hi52        = float(raw["Close"].tail(252).max())
lo52        = float(raw["Close"].tail(252).min())
vol_avg20   = float(raw["Volume"].tail(20).mean())
mktcap      = info.get("mktcap", 0)

c1,c2,c3,c4,c5,c6 = st.columns(6)
clr = "green" if change >= 0 else "red"
arrow = "▲" if change >= 0 else "▼"
with c1: metric_card(f"${price_now:,.2f}", "Last Close")
with c2: metric_card(f"{arrow} {abs(change_pct):.2f}%", "Day Change",
                     f"${abs(change):,.2f}", clr)
with c3: metric_card(f"${hi52:,.2f}", "52-Week High")
with c4: metric_card(f"${lo52:,.2f}", "52-Week Low")
with c5: metric_card(f"{n_pts:,}", "Data Points")
with c6: metric_card(
    f"${mktcap/1e9:.1f}B" if mktcap > 1e9 else "N/A",
    "Market Cap"
)

st.markdown("")

# ── Feature engineering ───────────────────────────────────────────────────────
with st.spinner("⚙️ Engineering 50+ features …"):
    feat_df = add_features(raw)

# ── Training ─────────────────────────────────────────────────────────────────
with st.spinner(f"🤖 Training **{model_name}** on {len(feat_df):,} samples …"):
    res = train_and_evaluate(feat_df, model_name, test_pct / 100)

# ── Future forecast ───────────────────────────────────────────────────────────
with st.spinner(f"🔮 Generating {forecast_d}-day forecast …"):
    forecast_df = forecast_future(res, feat_df, n_days=forecast_d)

tomorrow = float(forecast_df["Forecast"].iloc[0])

# ── Prediction box ────────────────────────────────────────────────────────────
from utils.ui import prediction_box
prediction_box(tomorrow, price_now, model_name, forecast_d)

# ── Model metrics ─────────────────────────────────────────────────────────────
section_header("Model Performance")
m1,m2,m3,m4,m5,m6 = st.columns(6)
with m1: metric_card(f"${res['mae']:.2f}",   "MAE")
with m2: metric_card(f"${res['rmse']:.2f}",  "RMSE")
with m3: metric_card(f"{res['r2']:.4f}",     "R² Score",
                     "1.0 = perfect", "green" if res['r2'] > 0.9 else "yellow")
with m4: metric_card(f"{res['mape']:.2f}%",  "MAPE")
with m5: metric_card(f"{res['acc_dir']:.1f}%","Direction Accuracy")
with m6: metric_card(f"{res['train_size']:,}","Train Samples")

st.markdown("")

# ── CHART 1: Prediction + Forecast ───────────────────────────────────────────
section_header("Price Prediction & Future Forecast")

# Filter display window
period_map = {"3 months": 63, "6 months": 126, "1 year": 252,
              "3 years": 756, "5 years": 1260, "All": len(raw)}
disp_n     = period_map[chart_period]
raw_disp   = raw["Close"].tail(disp_n)

fig_pred = prediction_chart(
    raw_disp, res["test_dates"], res["y_test"], res["y_pred"],
    forecast_df=forecast_df, ticker=ticker
)
st.plotly_chart(fig_pred, use_container_width=True)

# ── Forecast table ────────────────────────────────────────────────────────────
section_header(f"{forecast_d}-Day Price Forecast Table")
rows_html = ""
for i, (dt, row) in enumerate(forecast_df.iterrows()):
    fp    = row["Forecast"]
    ref   = price_now if i == 0 else float(forecast_df["Forecast"].iloc[i-1])
    chg   = (fp - ref) / ref * 100
    clr   = "#3fb950" if chg >= 0 else "#f85149"
    arrow = "▲" if chg >= 0 else "▼"
    rows_html += f"""
    <tr>
        <td>{dt.strftime('%a, %b %d %Y')}</td>
        <td>${fp:,.2f}</td>
        <td style="color:{clr};">{arrow} {abs(chg):.2f}%</td>
        <td>${fp - price_now:+,.2f}</td>
    </tr>"""

st.markdown(f"""
<table class="forecast-table">
  <thead>
    <tr>
      <th>Date</th><th>Forecast Price</th>
      <th>Day Change</th><th>vs Current</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
""", unsafe_allow_html=True)

st.markdown("")

# ── CHART 2: Actual vs Predicted ─────────────────────────────────────────────
section_header("Actual vs Predicted — Test Set")
tab1, tab2 = st.tabs(["Line Chart", "Scatter Plot"])
with tab1:
    st.plotly_chart(
        actual_vs_predicted_chart(res["test_dates"], res["y_test"], res["y_pred"]),
        use_container_width=True
    )
with tab2:
    st.plotly_chart(
        scatter_pred_chart(res["y_test"], res["y_pred"]),
        use_container_width=True
    )

# ── CHART 3: Candlestick ─────────────────────────────────────────────────────
section_header("Candlestick Chart with Indicators")
feat_disp = feat_df.tail(disp_n)
st.plotly_chart(
    candlestick_chart(feat_disp, title=f"{ticker} — OHLCV"),
    use_container_width=True
)

# ── CHART 4: MACD ─────────────────────────────────────────────────────────────
if show_macd:
    section_header("MACD Analysis")
    st.plotly_chart(macd_chart(feat_df, n=min(disp_n, 365)),
                    use_container_width=True)

# ── CHART 5: Feature Importance ──────────────────────────────────────────────
if show_fi and res["feat_imp"] is not None:
    section_header("Feature Importances")
    st.plotly_chart(feature_importance_chart(res["feat_imp"]),
                    use_container_width=True)

# ── Raw Data ─────────────────────────────────────────────────────────────────
section_header("Raw Data")
with st.expander("Show OHLCV Data"):
    st.dataframe(raw.sort_index(ascending=False).head(500),
                 use_container_width=True)

with st.expander("Show Feature Matrix (last 100 rows)"):
    st.dataframe(feat_df.tail(100), use_container_width=True)
