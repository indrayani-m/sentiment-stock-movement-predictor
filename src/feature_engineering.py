"""Create stock, sentiment, and combined modeling features."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_directories, load_config, resolve_path, set_seed, setup_logging


logger = logging.getLogger(__name__)


STOCK_FEATURES = [
    "daily_return",
    "3_day_return",
    "7_day_return",
    "5_day_moving_average",
    "10_day_moving_average",
    "volume_change_percent",
    "volatility",
]

SENTIMENT_FEATURES = [
    "average_sentiment",
    "weighted_sentiment",
    "positive_ratio",
    "negative_ratio",
    "sentiment_momentum",
    "post_count",
    "avg_score",
    "avg_num_comments",
]


def create_stock_features(stock_data: pd.DataFrame, volatility_window: int = 7) -> pd.DataFrame:
    """Engineer financial features from OHLCV data."""
    data = stock_data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["ticker", "date"])
    grouped = data.groupby("ticker", group_keys=False)
    data["daily_return"] = grouped["Close"].pct_change()
    data["3_day_return"] = grouped["Close"].pct_change(3)
    data["7_day_return"] = grouped["Close"].pct_change(7)
    data["5_day_moving_average"] = grouped["Close"].transform(lambda x: x.rolling(5).mean())
    data["10_day_moving_average"] = grouped["Close"].transform(lambda x: x.rolling(10).mean())
    data["volume_change_percent"] = grouped["Volume"].pct_change()
    data["volatility"] = grouped["daily_return"].transform(lambda x: x.rolling(volatility_window).std())
    data[STOCK_FEATURES] = data[STOCK_FEATURES].replace([np.inf, -np.inf], np.nan)
    return data


def create_sentiment_features(daily_sentiment: pd.DataFrame) -> pd.DataFrame:
    """Rename and extend DistilBERT daily sentiment features."""
    sentiment = daily_sentiment.copy()
    sentiment["date"] = pd.to_datetime(sentiment["date"])
    sentiment = sentiment.rename(
        columns={
            "daily_average_sentiment": "average_sentiment",
            "daily_weighted_sentiment": "weighted_sentiment",
            "daily_positive_ratio": "positive_ratio",
            "daily_negative_ratio": "negative_ratio",
        }
    )
    sentiment = sentiment.sort_values(["ticker", "date"])
    sentiment["sentiment_momentum"] = sentiment.groupby("ticker")["weighted_sentiment"].diff()
    return sentiment[["ticker", "date", *SENTIMENT_FEATURES]]


def combine_features(stock_features: pd.DataFrame, sentiment_features: pd.DataFrame | None) -> pd.DataFrame:
    """Merge stock and sentiment features into one supervised dataset."""
    if sentiment_features is None or sentiment_features.empty:
        combined = stock_features.copy()
        for column in SENTIMENT_FEATURES:
            combined[column] = 0.0
    else:
        combined = stock_features.merge(sentiment_features, on=["ticker", "date"], how="left")

    combined[SENTIMENT_FEATURES] = combined[SENTIMENT_FEATURES].fillna(0.0)
    combined[STOCK_FEATURES] = combined.groupby("ticker")[STOCK_FEATURES].transform(
        lambda frame: frame.ffill().bfill()
    )
    combined = combined.dropna(subset=STOCK_FEATURES + ["target"]).reset_index(drop=True)
    return combined


def build_feature_dataset(ticker: str, config: dict) -> Path:
    """Load raw inputs, build features, and save processed modeling data."""
    stock_path = resolve_path(config["paths"]["stock_data_dir"]) / f"{ticker.upper()}_stock.csv"
    sentiment_path = resolve_path(config["paths"]["sentiment_data_dir"]) / f"{ticker.upper()}_daily_sentiment.csv"

    stock = pd.read_csv(stock_path)
    stock_features = create_stock_features(
        stock,
        volatility_window=int(config["features"]["rolling_volatility_window"]),
    )
    sentiment_features = create_sentiment_features(pd.read_csv(sentiment_path)) if sentiment_path.exists() else None
    combined = combine_features(stock_features, sentiment_features)

    output_dir = resolve_path(config["paths"]["processed_data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker.upper()}_features.csv"
    combined.to_csv(output_path, index=False)
    logger.info("Saved feature dataset to %s", output_path)
    return output_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build stock and sentiment feature table.")
    parser.add_argument("--ticker", type=str, default=None)
    args = parser.parse_args()

    setup_logging()
    config = load_config()
    set_seed(config["project"]["random_seed"])
    ensure_directories(config)
    ticker = (args.ticker or config["data"]["default_ticker"]).upper()
    build_feature_dataset(ticker, config)


if __name__ == "__main__":
    main()
