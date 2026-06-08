"""Train stock movement classifiers and compare sentiment experiments."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.feature_engineering import SENTIMENT_FEATURES, STOCK_FEATURES
from src.utils import ensure_directories, load_config, resolve_path, set_seed, setup_logging


logger = logging.getLogger(__name__)


def get_model_grid(random_seed: int) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Return model candidates and compact hyperparameter grids."""
    grids: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=random_seed),
            {"model__C": [0.1, 1.0, 5.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=random_seed, class_weight="balanced"),
            {"model__n_estimators": [100, 200], "model__max_depth": [3, 6, None]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=random_seed),
            {"model__n_estimators": [100, 150], "model__learning_rate": [0.03, 0.1]},
        ),
    }
    try:
        from xgboost import XGBClassifier

        grids["XGBoost"] = (
            XGBClassifier(
                random_state=random_seed,
                eval_metric="logloss",
                n_estimators=150,
                learning_rate=0.05,
            ),
            {"model__max_depth": [2, 4], "model__subsample": [0.8, 1.0]},
        )
    except ImportError:
        logger.info("XGBoost not installed; skipping optional XGBoost model.")
    return grids


def split_time_ordered(data: pd.DataFrame, feature_columns: list[str], test_size: float):
    """Split by time order to reduce leakage from future observations."""
    data = data.sort_values("date").reset_index(drop=True)
    split_index = int(len(data) * (1 - test_size))
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]
    return train[feature_columns], test[feature_columns], train["target"], test["target"], test


def evaluate_predictions(y_true, y_pred, y_proba) -> dict[str, float]:
    """Calculate common classification metrics."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba)
    except ValueError:
        metrics["ROC-AUC"] = 0.0
    return metrics


def train_experiment(
    data: pd.DataFrame,
    feature_columns: list[str],
    experiment_name: str,
    config: dict,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train all candidate models for one feature set and return the best artifact."""
    seed = int(config["project"]["random_seed"])
    test_size = float(config["training"]["test_size"])
    cv_folds = int(config["training"]["cv_folds"])
    scoring = config["training"]["scoring"]

    x_train, x_test, y_train, y_test, test_rows = split_time_ordered(data, feature_columns, test_size)
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        logger.warning("One split has a single class; metrics may be less informative.")

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    model_grids = get_model_grid(seed)
    min_class_count = int(y_train.value_counts().min())
    cv_splits = min(cv_folds, min_class_count)

    for model_name, (model, params) in model_grids.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
        if cv_splits >= 2:
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
            estimator = GridSearchCV(pipeline, params, cv=cv, scoring=scoring, n_jobs=-1)
            estimator.fit(x_train, y_train)
            cv_score = float(estimator.best_score_)
            fitted_model = estimator.best_estimator_
        else:
            logger.warning("Too few samples per class for CV; fitting %s with default parameters.", model_name)
            pipeline.fit(x_train, y_train)
            cv_score = 0.0
            fitted_model = pipeline

        y_pred = fitted_model.predict(x_test)
        y_proba = fitted_model.predict_proba(x_test)[:, 1] if hasattr(fitted_model, "predict_proba") else y_pred
        metrics = evaluate_predictions(y_test, y_pred, y_proba)
        row = {"Experiment": experiment_name, "Model": model_name, "CV Score": cv_score, **metrics}
        rows.append(row)
        logger.info("%s | %s | F1 %.3f", experiment_name, model_name, metrics["F1"])
        if best is None or metrics["F1"] > best["metrics"]["F1"]:
            best = {
                "experiment": experiment_name,
                "model_name": model_name,
                "pipeline": fitted_model,
                "feature_columns": feature_columns,
                "metrics": metrics,
                "y_test": y_test,
                "y_pred": y_pred,
                "y_proba": y_proba,
                "test_rows": test_rows,
                "classification_report": classification_report(y_test, y_pred, zero_division=0),
            }

    assert best is not None
    return best, pd.DataFrame(rows)


def plot_confusion_matrix(best: dict[str, Any], output_dir: Path) -> Path:
    """Save confusion matrix for the best model."""
    matrix = confusion_matrix(best["y_test"], best["y_pred"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_roc_curve(best: dict[str, Any], output_dir: Path) -> Path:
    """Save ROC curve for the best model."""
    fpr, tpr, _ = roc_curve(best["y_test"], best["y_proba"])
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {best['metrics']['ROC-AUC']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "roc_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_feature_importance(best: dict[str, Any], output_dir: Path) -> Path:
    """Save feature importance or model coefficients."""
    pipeline = best["pipeline"]
    model = pipeline.named_steps["model"]
    features = best["feature_columns"]
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = abs(model.coef_[0])
    else:
        importance = [0] * len(features)

    frame = pd.DataFrame({"feature": features, "importance": importance}).sort_values("importance", ascending=False)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=frame.head(15), x="importance", y="feature")
    plt.title("Top Feature Importance")
    plt.tight_layout()
    path = output_dir / "feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    frame.to_csv(output_dir / "feature_importance.csv", index=False)
    return path


def save_outputs(best: dict[str, Any], comparison: pd.DataFrame, config: dict) -> None:
    """Persist model, metrics, reports, and plots."""
    output_dir = resolve_path(config["paths"]["outputs_dir"])
    model_dir = resolve_path(config["paths"]["models_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_path(config["training"]["model_output"])
    joblib.dump(
        {
            "pipeline": best["pipeline"],
            "feature_columns": best["feature_columns"],
            "experiment": best["experiment"],
            "model_name": best["model_name"],
            "metrics": best["metrics"],
        },
        model_path,
    )
    comparison.to_csv(resolve_path(config["training"]["metrics_output"]), index=False)
    with (output_dir / "classification_report.txt").open("w", encoding="utf-8") as file:
        file.write(best["classification_report"])
    with (output_dir / "best_model_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "experiment": best["experiment"],
                "model_name": best["model_name"],
                "metrics": best["metrics"],
                "features": best["feature_columns"],
            },
            file,
            indent=2,
        )
    plot_confusion_matrix(best, output_dir)
    plot_roc_curve(best, output_dir)
    plot_feature_importance(best, output_dir)


def run_training(ticker: str, config: dict) -> pd.DataFrame:
    """Run stock-only and stock-plus-sentiment experiments."""
    feature_path = resolve_path(config["paths"]["processed_data_dir"]) / f"{ticker.upper()}_features.csv"
    data = pd.read_csv(feature_path)
    data["date"] = pd.to_datetime(data["date"])

    best_stock, stock_results = train_experiment(data, STOCK_FEATURES, "A: Stock features only", config)
    best_sentiment, sentiment_results = train_experiment(
        data,
        STOCK_FEATURES + SENTIMENT_FEATURES,
        "B: Stock + DistilBERT sentiment",
        config,
    )
    comparison = pd.concat([stock_results, sentiment_results], ignore_index=True)
    overall_best = best_sentiment if best_sentiment["metrics"]["F1"] >= best_stock["metrics"]["F1"] else best_stock
    save_outputs(overall_best, comparison, config)
    return comparison


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train stock movement prediction models.")
    parser.add_argument("--ticker", type=str, default=None)
    args = parser.parse_args()

    setup_logging()
    config = load_config()
    set_seed(config["project"]["random_seed"])
    ensure_directories(config)
    ticker = (args.ticker or config["data"]["default_ticker"]).upper()
    run_training(ticker, config)


if __name__ == "__main__":
    main()
