"""Load a trained model and predict the latest stock movement direction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, resolve_path, setup_logging


def load_model(model_path: Path) -> dict[str, Any]:
    """Load the saved Joblib model artifact."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run src/train_model.py first.")
    return joblib.load(model_path)


def predict_latest(ticker: str, config: dict) -> dict[str, Any]:
    """Predict UP/DOWN for the newest feature row available."""
    model_artifact = load_model(resolve_path(config["training"]["model_output"]))
    feature_path = resolve_path(config["paths"]["processed_data_dir"]) / f"{ticker.upper()}_features.csv"
    data = pd.read_csv(feature_path).sort_values("date")
    latest = data.iloc[-1]
    features = model_artifact["feature_columns"]
    x_latest = latest[features].to_frame().T
    probability_up = float(model_artifact["pipeline"].predict_proba(x_latest)[0, 1])
    prediction = "UP" if probability_up >= 0.5 else "DOWN"
    return {
        "ticker": ticker.upper(),
        "date": str(latest["date"]),
        "prediction": prediction,
        "probability_up": probability_up,
        "model_name": model_artifact["model_name"],
        "experiment": model_artifact["experiment"],
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Predict latest stock movement.")
    parser.add_argument("--ticker", type=str, default=None)
    args = parser.parse_args()

    setup_logging()
    config = load_config()
    ticker = (args.ticker or config["data"]["default_ticker"]).upper()
    print(predict_latest(ticker, config))


if __name__ == "__main__":
    main()
