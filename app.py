"""Streamlit dashboard for sentiment-driven stock movement prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from predict import predict_latest  # noqa: E402
from utils import load_config, resolve_path  # noqa: E402


st.set_page_config(page_title="Sentiment Stock Predictor", layout="wide")


@st.cache_data
def load_features(ticker: str, processed_dir: str) -> pd.DataFrame:
    """Load processed features for a ticker."""
    path = resolve_path(processed_dir) / f"{ticker}_features.csv"
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data
def load_metrics(metrics_path: str) -> pd.DataFrame:
    """Load model comparison metrics."""
    path = resolve_path(metrics_path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_importance(outputs_dir: str) -> pd.DataFrame:
    """Load feature importance table."""
    path = resolve_path(outputs_dir) / "feature_importance.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    """Render the Streamlit app."""
    config = load_config()
    tickers = config["data"]["tickers"]
    ticker = st.sidebar.selectbox("Ticker", tickers, index=tickers.index(config["data"]["default_ticker"]))
    st.sidebar.caption("Run `python src/run_pipeline.py --ticker AAPL --demo-reddit` before using the dashboard.")

    st.title("Sentiment-Driven Stock Movement Predictor")
    st.caption("DistilBERT sentiment features + historical market features")

    try:
        features = load_features(ticker, config["paths"]["processed_data_dir"])
        prediction = predict_latest(ticker, config)
        model_artifact = joblib.load(resolve_path(config["training"]["model_output"]))
    except Exception as exc:
        st.error(f"Project artifacts are missing or incomplete: {exc}")
        st.stop()

    latest = features.sort_values("date").iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close", f"${latest['Close']:.2f}")
    col2.metric("Prediction", prediction["prediction"])
    col3.metric("Probability UP", f"{prediction['probability_up']:.1%}")
    col4.metric("Model", model_artifact["model_name"])

    chart_col, sentiment_col = st.columns(2)
    with chart_col:
        st.subheader("Stock Price")
        st.line_chart(features.set_index("date")["Close"])
    with sentiment_col:
        st.subheader("Recent Reddit Sentiment")
        sentiment_columns = ["weighted_sentiment", "positive_ratio", "negative_ratio"]
        st.line_chart(features.set_index("date")[sentiment_columns].tail(90))

    st.subheader("Model Metrics")
    metrics = load_metrics(config["training"]["metrics_output"])
    if metrics.empty:
        st.info("No metrics file found yet.")
    else:
        st.dataframe(metrics.sort_values("F1", ascending=False), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Feature Importance")
        importance = load_importance(config["paths"]["outputs_dir"])
        if importance.empty:
            st.info("No feature importance file found yet.")
        else:
            st.bar_chart(importance.head(15).set_index("feature")["importance"])
    with right:
        st.subheader("Recent Feature Snapshot")
        display_columns = [
            "date",
            "daily_return",
            "volatility",
            "weighted_sentiment",
            "positive_ratio",
            "negative_ratio",
            "target",
        ]
        st.dataframe(features[display_columns].tail(10), use_container_width=True)


if __name__ == "__main__":
    main()
