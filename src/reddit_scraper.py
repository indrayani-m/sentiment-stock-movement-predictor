"""Collect and clean Reddit posts using PRAW.

The scraper supports two modes:
1. Real Reddit collection when PRAW credentials are available as environment variables.
2. A small synthetic demo dataset when credentials are missing, so students can still run
   the rest of the machine learning project end to end.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import clean_text, ensure_directories, load_config, resolve_path, set_seed, setup_logging


logger = logging.getLogger(__name__)


def _build_reddit_client():
    """Create a PRAW client from environment variables."""
    import praw

    required = ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise EnvironmentError(
            "Missing Reddit credentials: "
            + ", ".join(missing)
            + ". Set them or run with --demo to create sample data."
        )
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )


def scrape_reddit_posts(ticker: str, subreddits: list[str], limit: int) -> pd.DataFrame:
    """Search Reddit posts for a ticker across configured subreddits."""
    reddit = _build_reddit_client()
    rows: list[dict] = []

    for subreddit_name in subreddits:
        logger.info("Searching r/%s for %s", subreddit_name, ticker)
        subreddit = reddit.subreddit(subreddit_name)
        try:
            for submission in subreddit.search(ticker, sort="new", limit=limit):
                rows.append(
                    {
                        "ticker": ticker.upper(),
                        "subreddit": subreddit_name,
                        "title": submission.title or "",
                        "selftext": submission.selftext or "",
                        "score": int(submission.score),
                        "num_comments": int(submission.num_comments),
                        "created_utc": float(submission.created_utc),
                    }
                )
        except Exception as exc:  # PRAW raises several API-specific exceptions.
            logger.warning("Failed to scrape r/%s for %s: %s", subreddit_name, ticker, exc)

    if not rows:
        raise RuntimeError("No Reddit posts collected. Try a larger limit or verify API access.")

    return pd.DataFrame(rows)


def create_demo_reddit_data(ticker: str, days: int = 120) -> pd.DataFrame:
    """Create reproducible sample Reddit posts for offline demos."""
    templates = [
        ("bullish breakout for {ticker}", "{ticker} earnings look strong and guidance is improving", 42, 15),
        ("concerns about {ticker}", "valuation looks stretched and traders are taking profits", 21, 9),
        ("holding {ticker} long term", "product demand and cash flow remain impressive", 35, 11),
        ("is {ticker} overbought", "short term chart looks risky after a big rally", 18, 13),
    ]
    today = datetime.now(timezone.utc).date()
    rows: list[dict] = []
    for day in range(days):
        date = today - timedelta(days=days - day)
        for index, template in enumerate(templates):
            title, body, score, comments = template
            rows.append(
                {
                    "ticker": ticker.upper(),
                    "subreddit": ["stocks", "investing", "wallstreetbets"][index % 3],
                    "title": title.format(ticker=ticker.upper()),
                    "selftext": body.format(ticker=ticker.upper()),
                    "score": score + (day % 7),
                    "num_comments": comments + (day % 5),
                    "created_utc": datetime(date.year, date.month, date.day, 14, tzinfo=timezone.utc).timestamp(),
                }
            )
    return pd.DataFrame(rows)


def process_reddit_data(data: pd.DataFrame) -> pd.DataFrame:
    """Combine and clean title/body text."""
    data = data.copy()
    data["created_datetime"] = pd.to_datetime(data["created_utc"], unit="s", utc=True)
    data["date"] = data["created_datetime"].dt.date
    data["text"] = (data["title"].fillna("") + " " + data["selftext"].fillna("")).map(clean_text)
    return data[["ticker", "date", "subreddit", "title", "selftext", "text", "score", "num_comments", "created_utc"]]


def save_reddit_data(data: pd.DataFrame, ticker: str, output_dir: Path) -> Path:
    """Save processed Reddit posts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{ticker.upper()}_reddit.csv"
    data.to_csv(path, index=False)
    logger.info("Saved Reddit data to %s", path)
    return path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Scrape Reddit data using PRAW.")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--demo", action="store_true", help="Generate demo Reddit data without API credentials.")
    args = parser.parse_args()

    setup_logging()
    config = load_config()
    set_seed(config["project"]["random_seed"])
    ensure_directories(config)
    ticker = (args.ticker or config["data"]["default_ticker"]).upper()

    if args.demo:
        raw = create_demo_reddit_data(ticker)
    else:
        raw = scrape_reddit_posts(
            ticker=ticker,
            subreddits=config["data"]["subreddits"],
            limit=config["data"]["reddit_limit_per_subreddit"],
        )
    processed = process_reddit_data(raw)
    save_reddit_data(processed, ticker, resolve_path(config["paths"]["reddit_data_dir"]))


if __name__ == "__main__":
    main()
