#!/usr/bin/env python3
"""
Simple active learning for AG News.
Compares uncertainty sampling (active) vs random sampling (passive)
without any data augmentation.
只使用這篇論文中的 Uncertainty Sampling
"""

import os
import sys
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))


def setup_logging(config_name):
    """Setup logging to both console and file."""
    logs_dir = f"{HOME_DIR}/experimentation/data/logs"
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{logs_dir}/experiment_log_{config_name}_{timestamp}.txt"

    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

        def close(self):
            self.log.close()

    logger = Logger(log_filename)
    sys.stdout = logger

    print(f"Logging started - Output saved to: {log_filename}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return logger


def load_and_split_data(test_size=0.2, random_state=42):
    """Load AG News and split into train/test."""
    np.random.seed(random_state)

    dataset = load_dataset("ag_news")
    df = dataset["train"].to_pandas()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print("Loaded AG News")
    print(f"Train set: {len(X_train)}")
    print(f"Test set: {len(X_test)}")
    print(f"Label distribution in train: {y_train.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


def uncertainty_sampling(model, X_unlabeled, n_samples):
    """Select most uncertain samples by entropy (multi-class)."""
    probabilities = model.predict_proba(X_unlabeled)
    eps = 1e-12
    ent = -((probabilities * np.log(probabilities + eps)).sum(axis=1))
    uncertain_idx = np.argsort(-ent)[: min(n_samples, len(X_unlabeled))]
    return X_unlabeled.iloc[uncertain_idx]


def random_sampling(X_unlabeled, n_samples, random_seed=42):
    """Select random samples."""
    return X_unlabeled.sample(min(n_samples, len(X_unlabeled)), random_state=random_seed)


def train_and_evaluate(X_train, y_train, X_val, y_val):
    """Train text classifier and evaluate on validation set."""
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    solver="lbfgs",
                    multi_class="multinomial",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_val, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_val, y_pred, average="macro", zero_division=0),
    }
    return model, metrics


def run_active_learning_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    initial_samples=300,
    batch_size=68,
    n_iterations=5,
    strategies=None,
    random_seed=42,
    initial_strategy="random",
):
    """Run active learning with uncertainty sampling."""
    print("\n" + "=" * 60)
    print("ACTIVE LEARNING EXPERIMENT")
    print("=" * 60)

    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_seed
    )

    if initial_strategy != "random":
        raise ValueError("Only 'random' is supported for initial_strategy.")

    rng = np.random.RandomState(random_seed)
    initial_indices = rng.choice(X_train_val.index, size=initial_samples, replace=False)
    X_labeled = X_train_val.loc[initial_indices]
    y_labeled = y_train_val.loc[initial_indices]
    X_unlabeled = X_train_val.drop(index=X_labeled.index)

    print(f"Initial labeled pool: {len(X_labeled)}")
    print(f"Remaining unlabeled: {len(X_unlabeled)}")

    results = []

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")

        model, metrics = train_and_evaluate(X_labeled, y_labeled, X_val, y_val)
        print(f"Validation - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        results.append(
            {
                "iteration": iteration,
                "labeled_samples": len(X_labeled),
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )

        if iteration < n_iterations and len(X_unlabeled) > 0:
            if strategies and iteration <= len(strategies):
                strategy = strategies[iteration - 1]
            else:
                strategy = "uncertainty"

            if strategy == "uncertainty":
                new_samples = uncertainty_sampling(model, X_unlabeled, batch_size)
            elif strategy == "random":
                new_samples = random_sampling(X_unlabeled, batch_size, random_seed)
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")

            print(f"Selected {len(new_samples)} samples using {strategy} sampling")
            new_labels = y_train_val.loc[new_samples.index]
            X_labeled = pd.concat([X_labeled, new_samples])
            y_labeled = pd.concat([y_labeled, new_labels])
            X_unlabeled = X_unlabeled.drop(index=new_samples.index)
            print(f"Total labeled samples: {len(X_labeled)}")
            print(f"Remaining unlabeled: {len(X_unlabeled)}")

    print("\n--- Final Test Evaluation ---")
    _, final_metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
    print(f"Test Performance - F1: {final_metrics['f1']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}")
    return results, final_metrics


def run_passive_learning_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    initial_samples=300,
    batch_size=68,
    n_iterations=5,
    random_seed=42,
):
    """Run passive learning baseline (random sampling)."""
    print("\n" + "=" * 60)
    print("PASSIVE LEARNING EXPERIMENT")
    print("=" * 60)

    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_seed
    )

    rng = np.random.RandomState(random_seed)
    initial_indices = rng.choice(X_train_val.index, size=initial_samples, replace=False)
    X_labeled = X_train_val.loc[initial_indices]
    y_labeled = y_train_val.loc[initial_indices]
    X_unlabeled = X_train_val.drop(index=initial_indices)

    print(f"Initial labeled pool: {len(X_labeled)}")
    print(f"Remaining unlabeled: {len(X_unlabeled)}")

    results = []

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        model, metrics = train_and_evaluate(X_labeled, y_labeled, X_val, y_val)
        print(f"Validation - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        results.append(
            {
                "iteration": iteration,
                "labeled_samples": len(X_labeled),
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )

        if iteration < n_iterations and len(X_unlabeled) > 0:
            new_samples = random_sampling(X_unlabeled, batch_size, random_seed)
            print(f"Selected {len(new_samples)} samples using random sampling")
            new_labels = y_train_val.loc[new_samples.index]
            X_labeled = pd.concat([X_labeled, new_samples])
            y_labeled = pd.concat([y_labeled, new_labels])
            X_unlabeled = X_unlabeled.drop(index=new_samples.index)
            print(f"Total labeled samples: {len(X_labeled)}")
            print(f"Remaining unlabeled: {len(X_unlabeled)}")

    print("\n--- Final Test Evaluation ---")
    _, final_metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
    print(f"Test Performance - F1: {final_metrics['f1']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}")
    return results, final_metrics


def plot_comparison(active_results, passive_results, config_name="", show_plots=True):
    """Plot active vs passive comparison."""
    active_df = pd.DataFrame(active_results)
    passive_df = pd.DataFrame(passive_results)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(active_df["iteration"], active_df["f1"], "o-", label="Active Learning", linewidth=2, markersize=8)
    plt.plot(passive_df["iteration"], passive_df["f1"], "s-", label="Passive Learning", linewidth=2, markersize=8)
    plt.xlabel("Iteration")
    plt.ylabel("F1 Score")
    plt.title("F1 Score Progression")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(active_df["iteration"], active_df["accuracy"], "o-", label="Active Learning", linewidth=2, markersize=8)
    plt.plot(passive_df["iteration"], passive_df["accuracy"], "s-", label="Passive Learning", linewidth=2, markersize=8)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Progression")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(active_df["iteration"], active_df["labeled_samples"], "o-", label="Active Learning", linewidth=2, markersize=8)
    plt.plot(passive_df["iteration"], passive_df["labeled_samples"], "s-", label="Passive Learning", linewidth=2, markersize=8)
    plt.xlabel("Iteration")
    plt.ylabel("Labeled Samples")
    plt.title("Labeled Samples Progression")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_filename = f"active_vs_passive_comparison_{config_name}.png" if config_name else "active_vs_passive_comparison.png"
    plt.savefig(f"{HOME_DIR}/experimentation/data/{plot_filename}", dpi=300, bbox_inches="tight")

    if show_plots:
        plt.show(block=False)
        plt.pause(0.1)
    else:
        plt.close()


def run_multiple_experiments(X_train, X_test, y_train, y_test, config, n_runs=10):
    """Run repeated experiments for significance testing."""
    print("\n" + "=" * 80)
    print(f"RUNNING {n_runs} EXPERIMENTS FOR STATISTICAL SIGNIFICANCE")
    print("=" * 80)

    all_active_results, all_passive_results = [], []
    all_active_finals, all_passive_finals = [], []

    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")
        random_seed = 42 + run

        active_results, active_final = run_active_learning_experiment(
            X_train,
            y_train,
            X_test,
            y_test,
            initial_samples=config["initial_samples"],
            batch_size=config["batch_size"],
            n_iterations=config["n_iterations"],
            strategies=config["iteration_strategies"],
            random_seed=random_seed,
            initial_strategy=config.get("initial_strategy", "random"),
        )

        passive_results, passive_final = run_passive_learning_experiment(
            X_train,
            y_train,
            X_test,
            y_test,
            initial_samples=config["initial_samples"],
            batch_size=config["batch_size"],
            n_iterations=config["n_iterations"],
            random_seed=random_seed,
        )

        all_active_results.append(active_results)
        all_passive_results.append(passive_results)
        all_active_finals.append(active_final)
        all_passive_finals.append(passive_final)

        print(f"Run {run + 1} completed - Active F1: {active_final['f1']:.4f}, Passive F1: {passive_final['f1']:.4f}")

    return all_active_results, all_passive_results, all_active_finals, all_passive_finals


def perform_statistical_tests(active_finals, passive_finals):
    """Perform statistical tests on final F1 scores."""
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 80)

    active_f1_scores = [result["f1"] for result in active_finals]
    passive_f1_scores = [result["f1"] for result in passive_finals]

    active_mean = np.mean(active_f1_scores)
    active_std = np.std(active_f1_scores)
    passive_mean = np.mean(passive_f1_scores)
    passive_std = np.std(passive_f1_scores)

    print(f"Active Learning F1: {active_mean:.4f} +/- {active_std:.4f}")
    print(f"Passive Learning F1: {passive_mean:.4f} +/- {passive_std:.4f}")
    print(f"Difference: {active_mean - passive_mean:.4f}")

    t_stat, p_value = stats.ttest_rel(active_f1_scores, passive_f1_scores)
    print("\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    w_stat, w_p_value = stats.wilcoxon(active_f1_scores, passive_f1_scores)
    print("\nWilcoxon signed-rank test:")
    print(f"  W-statistic: {w_stat:.4f}")
    print(f"  p-value: {w_p_value:.6f}")

    cohens_d = (active_mean - passive_mean) / np.sqrt((active_std**2 + passive_std**2) / 2)

    return {
        "active_mean": active_mean,
        "active_std": active_std,
        "passive_mean": passive_mean,
        "passive_std": passive_std,
        "difference": active_mean - passive_mean,
        "t_stat": t_stat,
        "p_value": p_value,
        "w_stat": w_stat,
        "w_p_value": w_p_value,
        "cohens_d": cohens_d,
    }


def plot_statistical_comparison(active_finals, passive_finals, config_name="", show_plots=True):
    """Plot distributions and summary stats for repeated runs."""
    active_f1_scores = [result["f1"] for result in active_finals]
    passive_f1_scores = [result["f1"] for result in passive_finals]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(active_f1_scores, alpha=0.7, label="Active", bins=10, color="blue")
    plt.hist(passive_f1_scores, alpha=0.7, label="Passive", bins=10, color="red")
    plt.xlabel("F1 Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of F1 Scores")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.boxplot([active_f1_scores, passive_f1_scores], labels=["Active", "Passive"])
    plt.ylabel("F1 Score")
    plt.title("Box Plot")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    runs = range(1, len(active_f1_scores) + 1)
    plt.plot(runs, active_f1_scores, "o-", label="Active", linewidth=2, markersize=5)
    plt.plot(runs, passive_f1_scores, "s-", label="Passive", linewidth=2, markersize=5)
    plt.xlabel("Run")
    plt.ylabel("F1 Score")
    plt.title("F1 by Run")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    differences = [a - p for a, p in zip(active_f1_scores, passive_f1_scores)]
    plt.hist(differences, alpha=0.7, bins=10, color="green")
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("F1 Difference (Active - Passive)")
    plt.ylabel("Frequency")
    plt.title("Difference Distribution")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_filename = f"statistical_comparison_{config_name}.png" if config_name else "statistical_comparison.png"
    plt.savefig(f"{HOME_DIR}/experimentation/data/{plot_filename}", dpi=300, bbox_inches="tight")

    if show_plots:
        plt.show(block=False)
        plt.pause(0.1)
    else:
        plt.close()


def main():
    """Main entry point for AG News active vs passive comparison."""
    show_plots = False
    statistical_testing = False
    n_runs = 10

    experiment_config = {
        "initial_samples": 300,
        "initial_strategy": "random",
        "batch_size": 68,
        "n_iterations": 11,
        "iteration_strategies": ["uncertainty"] * 11,
    }
    config_name = "agnews_uncertainty_only"

    logger = setup_logging(config_name)
    try:
        print("AG News - Active Learning vs Passive Learning")
        print("=" * 80)
        print("Sampling strategy: Uncertainty only (no augmentation)")

        X_train, X_test, y_train, y_test = load_and_split_data(test_size=0.2, random_state=42)

        if statistical_testing:
            all_active_results, all_passive_results, all_active_finals, all_passive_finals = run_multiple_experiments(
                X_train, X_test, y_train, y_test, experiment_config, n_runs=n_runs
            )
            stats_results = perform_statistical_tests(all_active_finals, all_passive_finals)
            plot_statistical_comparison(all_active_finals, all_passive_finals, config_name, show_plots=show_plots)

            active_results = all_active_results[0]
            passive_results = all_passive_results[0]
            active_final = all_active_finals[0]
            passive_final = all_passive_finals[0]

            stats_filename = f"statistical_results_{config_name}.csv"
            pd.DataFrame([stats_results]).to_csv(f"{HOME_DIR}/experimentation/data/{stats_filename}", index=False)
            print(f"\nStatistical results saved to: {stats_filename}")
        else:
            print("\nRunning single experiment...")
            active_results, active_final = run_active_learning_experiment(
                X_train,
                y_train,
                X_test,
                y_test,
                initial_samples=experiment_config["initial_samples"],
                batch_size=experiment_config["batch_size"],
                n_iterations=experiment_config["n_iterations"],
                strategies=experiment_config["iteration_strategies"],
                initial_strategy=experiment_config.get("initial_strategy", "random"),
            )
            passive_results, passive_final = run_passive_learning_experiment(
                X_train,
                y_train,
                X_test,
                y_test,
                initial_samples=experiment_config["initial_samples"],
                batch_size=experiment_config["batch_size"],
                n_iterations=experiment_config["n_iterations"],
            )

        comparison_data = [
            {
                "Run": 1,
                "Random_Seed": 42,
                "Active_F1": active_final["f1"],
                "Passive_F1": passive_final["f1"],
                "Active_Accuracy": active_final["accuracy"],
                "Passive_Accuracy": passive_final["accuracy"],
                "F1_Improvement": active_final["f1"] - passive_final["f1"],
                "Improvement_%": ((active_final["f1"] - passive_final["f1"]) / passive_final["f1"] * 100)
                if passive_final["f1"] > 0
                else 0,
            }
        ]

        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + "=" * 80)
        print("FINAL TEST SET PERFORMANCE COMPARISON")
        print("=" * 80)
        print(comparison_df.to_string(index=False))

        results_filename = f"statistical_test_data_{config_name}.csv"
        comparison_df.to_csv(f"{HOME_DIR}/experimentation/data/{results_filename}", index=False)
        plot_comparison(active_results, passive_results, config_name, show_plots=show_plots)

        print(f"\nResults saved to: {HOME_DIR}/experimentation/data/{results_filename}")
        print("\nLogging completed")
    finally:
        logger.close()
        sys.stdout = logger.terminal


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {e}")
        raise
