"""Download historical stock prices and create next-day direction labels."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_directories, load_config, resolve_path, set_seed, setup_logging


logger = logging.getLogger(__name__)


def download_stock_data(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    period: str = "5y",
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and add a binary target."""
    logger.info("Downloading stock data for %s", ticker)
    if start_date:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    else:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=False)

    if data.empty:
        raise RuntimeError(f"No stock data returned for ticker {ticker}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data = data.reset_index()
    data = data.rename(columns={"Date": "date"})
    data["date"] = pd.to_datetime(data["date"]).dt.date
    columns = ["date", "Open", "High", "Low", "Close", "Volume"]
    data = data[columns].copy()
    data["next_day_close"] = data["Close"].shift(-1)
    data["target"] = (data["next_day_close"] > data["Close"]).astype(int)
    data = data.dropna(subset=["next_day_close"]).reset_index(drop=True)
    data["ticker"] = ticker.upper()
    return data


def save_stock_data(data: pd.DataFrame, ticker: str, output_dir: Path) -> Path:
    """Save ticker data as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{ticker.upper()}_stock.csv"
    data.to_csv(path, index=False)
    logger.info("Saved stock data to %s", path)
    return path


def collect_all_tickers(config: dict) -> list[Path]:
    """Download all configured ticker datasets."""
    output_dir = resolve_path(config["paths"]["stock_data_dir"])
    saved_paths: list[Path] = []
    for ticker in config["data"]["tickers"]:
        data = download_stock_data(
            ticker=ticker,
            start_date=config["data"].get("start_date"),
            end_date=config["data"].get("end_date"),
            period=config["data"].get("stock_period", "5y"),
        )
        saved_paths.append(save_stock_data(data, ticker, output_dir))
    return saved_paths


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Download stock data with yfinance.")
    parser.add_argument("--ticker", type=str, default=None, help="Ticker to download. Omit for all config tickers.")
    args = parser.parse_args()

    setup_logging()
    config = load_config()
    set_seed(config["project"]["random_seed"])
    ensure_directories(config)

    if args.ticker:
        data = download_stock_data(
            args.ticker,
            config["data"].get("start_date"),
            config["data"].get("end_date"),
            config["data"].get("stock_period", "5y"),
        )
        save_stock_data(data, args.ticker, resolve_path(config["paths"]["stock_data_dir"]))
    else:
        collect_all_tickers(config)


if __name__ == "__main__":
    main()
