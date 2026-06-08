"""Run the complete project pipeline for a ticker."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_fetcher import download_stock_data, save_stock_data
from src.feature_engineering import build_feature_dataset
from src.reddit_scraper import create_demo_reddit_data, process_reddit_data, save_reddit_data, scrape_reddit_posts
from src.sentiment_analyzer import run_sentiment_pipeline
from src.train_model import run_training
from src.utils import ensure_directories, load_config, resolve_path, set_seed, setup_logging


logger = logging.getLogger(__name__)


def run_pipeline(ticker: str, demo_reddit: bool = False) -> None:
    """Execute data collection, sentiment scoring, feature engineering, and training."""
    config = load_config()
    set_seed(config["project"]["random_seed"])
    ensure_directories(config)
    ticker = ticker.upper()

    stock = download_stock_data(
        ticker,
        start_date=config["data"].get("start_date"),
        end_date=config["data"].get("end_date"),
        period=config["data"].get("stock_period", "5y"),
    )
    save_stock_data(stock, ticker, resolve_path(config["paths"]["stock_data_dir"]))

    if demo_reddit:
        reddit_raw = create_demo_reddit_data(ticker)
    else:
        reddit_raw = scrape_reddit_posts(
            ticker,
            subreddits=config["data"]["subreddits"],
            limit=int(config["data"]["reddit_limit_per_subreddit"]),
        )
    reddit_processed = process_reddit_data(reddit_raw)
    save_reddit_data(reddit_processed, ticker, resolve_path(config["paths"]["reddit_data_dir"]))

    run_sentiment_pipeline(ticker, config)
    build_feature_dataset(ticker, config)
    results = run_training(ticker, config)
    logger.info("Pipeline complete. Best rows:\n%s", results.sort_values("F1", ascending=False).head())


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the complete ML pipeline.")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--demo-reddit", action="store_true", help="Use generated Reddit posts.")
    args = parser.parse_args()

    setup_logging()
    config = load_config()

    if args.ticker:
        run_pipeline(args.ticker.upper(), demo_reddit=args.demo_reddit)
    else:
        tickers = config["data"]["tickers"]

        for ticker in tickers:
            logger.info("=" * 60)
            logger.info("Running pipeline for %s", ticker)
            logger.info("=" * 60)

            try:
                run_pipeline(ticker, demo_reddit=args.demo_reddit)
            except Exception as exc:
                logger.error("Failed for %s: %s", ticker, exc)
if __name__ == "__main__":
    main()