"""Generate DistilBERT sentiment features for Reddit posts."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from transformers import pipeline

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_directories, load_config, resolve_path, set_seed, setup_logging


logger = logging.getLogger(__name__)


def load_sentiment_pipeline(model_name: str):
    """Load a Hugging Face sentiment-classification pipeline.

    DistilBERT is a smaller, faster version of BERT. It keeps the Transformer encoder
    idea: tokens are converted into embeddings, attention layers let each token look at
    other relevant tokens, and a classification head maps the final representation to
    sentiment labels. The SST-2 checkpoint is already fine-tuned for POSITIVE/NEGATIVE
    English sentiment classification.
    """
    logger.info("Loading sentiment model: %s", model_name)
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)


def score_posts(data: pd.DataFrame, model_name: str, batch_size: int, max_length: int) -> pd.DataFrame:
    """Add sentiment label, confidence, signed score, and weighted sentiment."""
    classifier = load_sentiment_pipeline(model_name)
    texts = data["text"].fillna("").astype(str).tolist()
    results: list[dict] = []

    for index in tqdm(range(0, len(texts), batch_size), desc="Scoring Reddit posts"):
        batch = texts[index : index + batch_size]
        # Tokenizer concept: raw text is split into WordPiece token ids. Padding and
        # truncation make batches equal length so PyTorch can process them efficiently.
        predictions = classifier(batch, truncation=True, max_length=max_length)
        results.extend(predictions)

    output = data.copy()
    output["sentiment_label"] = [item["label"] for item in results]
    output["confidence"] = [float(item["score"]) for item in results]
    output["sentiment"] = output["sentiment_label"].map({"POSITIVE": 1, "NEGATIVE": -1}).fillna(0)
    output["weighted_sentiment"] = output["sentiment"] * output["confidence"]
    return output


def aggregate_daily_sentiment(scored_posts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate post-level sentiment into daily ticker-level features."""
    data = scored_posts.copy()
    data["date"] = pd.to_datetime(data["date"])
    grouped = data.groupby(["ticker", "date"])
    daily = grouped.agg(
        daily_average_sentiment=("sentiment", "mean"),
        daily_weighted_sentiment=("weighted_sentiment", "mean"),
        post_count=("text", "count"),
        avg_score=("score", "mean"),
        avg_num_comments=("num_comments", "mean"),
    ).reset_index()
    positive_counts = grouped["sentiment"].apply(lambda series: (series > 0).mean())
    negative_counts = grouped["sentiment"].apply(lambda series: (series < 0).mean())
    daily["daily_positive_ratio"] = positive_counts.to_numpy()
    daily["daily_negative_ratio"] = negative_counts.to_numpy()
    daily = daily.sort_values(["ticker", "date"])
    return daily


def plot_sentiment_trend(daily_sentiment: pd.DataFrame, ticker: str, output_dir: Path) -> Path:
    """Save a daily weighted sentiment trend chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ticker_data = daily_sentiment[daily_sentiment["ticker"] == ticker.upper()].copy()
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=ticker_data, x="date", y="daily_weighted_sentiment")
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.title(f"{ticker.upper()} Reddit DistilBERT Sentiment Trend")
    plt.xlabel("Date")
    plt.ylabel("Average Weighted Sentiment")
    plt.xticks(rotation=30)
    plt.tight_layout()
    path = output_dir / f"{ticker.upper()}_sentiment_trend.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def run_sentiment_pipeline(ticker: str, config: dict) -> tuple[Path, Path]:
    """Score processed Reddit data and save post-level and daily sentiment CSVs."""
    reddit_path = resolve_path(config["paths"]["reddit_data_dir"]) / f"{ticker.upper()}_reddit.csv"
    posts = pd.read_csv(reddit_path)
    scored = score_posts(
        posts,
        model_name=config["sentiment"]["model_name"],
        batch_size=int(config["sentiment"]["batch_size"]),
        max_length=int(config["sentiment"]["max_length"]),
    )
    daily = aggregate_daily_sentiment(scored)
    sentiment_dir = resolve_path(config["paths"]["sentiment_data_dir"])
    sentiment_dir.mkdir(parents=True, exist_ok=True)
    scored_path = sentiment_dir / f"{ticker.upper()}_reddit_scored.csv"
    daily_path = sentiment_dir / f"{ticker.upper()}_daily_sentiment.csv"
    scored.to_csv(scored_path, index=False)
    daily.to_csv(daily_path, index=False)
    plot_sentiment_trend(daily, ticker, resolve_path(config["paths"]["outputs_dir"]))
    return scored_path, daily_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Analyze Reddit sentiment with DistilBERT.")
    parser.add_argument("--ticker", type=str, default=None)
    args = parser.parse_args()

    setup_logging()
    config = load_config()
    set_seed(config["project"]["random_seed"])
    ensure_directories(config)
    ticker = (args.ticker or config["data"]["default_ticker"]).upper()
    run_sentiment_pipeline(ticker, config)


if __name__ == "__main__":
    main()
