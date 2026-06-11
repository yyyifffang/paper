#!/usr/bin/env python3
"""
Banking77 Long-Tail Text Classification Active Learning Framework
Plateau-aware LLM-Augmented Active Learning Version

核心設計：
1. BAAI/bge-large-en-v1.5 作為固定文字特徵編碼器。
2. Banking77 人為長尾切分：Head=15 seed samples/class, Tail=2 seed samples/class。
3. 比較 Passive、Active、Generate-only、Generate-and-Verify 四種策略。
4. Macro F1 + Tail F1 plateau detection 作為主停止規則。
5. Query / generation / verification counts 僅作為 saving analysis 的 operation-count proxy。
"""

import argparse
import gc
import os
import sys
import json
import subprocess
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# Model and dataset configuration
# =============================================================================
SENTENCE_TRANSFORMER_MODEL_ID = "BAAI/bge-large-en-v1.5"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "head_tail_split_robuse")

HEAD_LABELS = list(range(0, 10))
TAIL_LABELS = list(range(10, 77))
HEAD_LABEL_COUNT = len(HEAD_LABELS)
TAIL_LABEL_COUNT = len(TAIL_LABELS)

# =============================================================================
# Head / Tail split
# =============================================================================
def configure_head_tail_split(
    split_mode: str = "fixed",
    class_split_seed: int = 0,
    n_head_classes: int = 10,
):
    """
    Configure head/tail class assignment.

    fixed:
        Head = labels 0-9, Tail = labels 10-76

    random:
        Randomly select n_head_classes labels as head classes,
        and use the remaining labels as tail classes.
    """
    global HEAD_LABELS, TAIL_LABELS, HEAD_LABEL_COUNT, TAIL_LABEL_COUNT

    all_labels = np.arange(77)

    if split_mode == "fixed":
        HEAD_LABELS = list(range(0, 10))
        TAIL_LABELS = list(range(10, 77))

    elif split_mode == "random":
        rng = np.random.RandomState(class_split_seed)
        head = sorted(rng.choice(all_labels, size=n_head_classes, replace=False).tolist())
        tail = sorted([int(x) for x in all_labels if int(x) not in set(head)])

        HEAD_LABELS = head
        TAIL_LABELS = tail

    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    HEAD_LABEL_COUNT = len(HEAD_LABELS)
    TAIL_LABEL_COUNT = len(TAIL_LABELS)

    print("\n=== Head/Tail Class Split Configuration ===")
    print(f"Split mode: {split_mode}")
    print(f"Class split seed: {class_split_seed}")
    print(f"Head labels ({len(HEAD_LABELS)}): {HEAD_LABELS}")
    print(f"Tail labels ({len(TAIL_LABELS)}): {TAIL_LABELS}")
    print("==========================================\n")

# =============================================================================
# Logging
# =============================================================================
def setup_logging(config_name: str, output_dir: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(output_dir, f"experiment_log_{config_name}_{timestamp}.txt")

    class Logger:
        def __init__(self, filename: str):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding="utf-8")

        def write(self, message: str):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

        def isatty(self):
            return self.terminal.isatty() if hasattr(self.terminal, "isatty") else False

        def __getattr__(self, name: str):
            return getattr(self.terminal, name)

        def close(self):
            self.log.close()

    logger = Logger(log_filename)
    sys.stdout = logger
    print(f"Logging started: {log_filename}")
    return logger

def _build_output_filename(prefix: str, config_name: str, run_tag: str, extension: str) -> str:
    parts = [prefix]
    if config_name:
        parts.append(config_name)
    if run_tag:
        parts.append(run_tag)
    return f"{'_'.join(parts)}.{extension}"


def _init_operation_state(initial_human_labels: int) -> Dict[str, int]:
    """
    Operation-count state for saving analysis.
    """
    return {
        "human_labels": int(initial_human_labels),
        "llm_generation_requests": 0,
        "llm_valid_generated": 0,
        "llm_verification_requests": 0,
        "synthetic_accepted": 0,
    }

# =============================================================================
# Dataset loading
# =============================================================================
def get_banking77_label_mapping() -> Dict[int, str]:
    dataset = load_dataset("PolyAI/banking77", split="train")
    return {i: name for i, name in enumerate(dataset.features["label"].names)}


def load_banking77_long_tail(random_seed: int = 42, return_texts: bool = False):
    np.random.seed(random_seed)
    print("Loading dataset: PolyAI/banking77 (Simulated Long-Tail)")
    dataset = load_dataset("PolyAI/banking77", split="train")
    df = dataset.to_pandas()[["text", "label"]].copy()

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    head_seed_size = 15
    tail_seed_size = 2
    test_size_per_class = 20

    df_seed_list, df_test_list, df_unlabeled_list = [], [], []

    for label_id in range(77):
        group = df[df["label"] == label_id].copy()

        test_idx = group.index[:test_size_per_class]
        df_test_list.append(group.loc[test_idx])
        group = group.drop(test_idx)

        seed_size = head_seed_size if label_id in HEAD_LABELS else tail_seed_size
        seed_idx = group.index[:seed_size]
        df_seed_list.append(group.loc[seed_idx])

        unlabeled_idx = group.drop(seed_idx).index
        df_unlabeled_list.append(group.loc[unlabeled_idx])

    df_seed = pd.concat(df_seed_list).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    df_test = pd.concat(df_test_list).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    df_unlabeled = pd.concat(df_unlabeled_list).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    print(f"\n=== Encoding with {SENTENCE_TRANSFORMER_MODEL_ID} ===")
    combined_text = pd.concat(
        [df_seed["text"], df_unlabeled["text"], df_test["text"]],
        ignore_index=True,
    )
    text_encoder = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_ID, device="cpu")

    X_all = text_encoder.encode(
        combined_text.tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    seed_end = len(df_seed)
    unlabeled_end = seed_end + len(df_unlabeled)

    X_seed = X_all[:seed_end]
    X_unlabeled = X_all[seed_end:unlabeled_end]
    X_test = X_all[unlabeled_end:]

    y_seed = df_seed["label"].to_numpy()
    y_unlabeled = df_unlabeled["label"].to_numpy()
    y_test = df_test["label"].to_numpy()

    print("\n=== Dataset Partition Summary ===")
    print(f"Random seed: {random_seed}")
    print(f"Head classes ({HEAD_LABEL_COUNT} classes): {HEAD_LABELS}")
    print(f"Tail classes ({TAIL_LABEL_COUNT} classes): {TAIL_LABELS}")
    print(f"Head seed samples/class: {head_seed_size}")
    print(f"Tail seed samples/class: {tail_seed_size}")
    print(f"Seed set size: {X_seed.shape[0]}")
    print(f"Unlabeled pool size: {X_unlabeled.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print("=================================\n")

    if return_texts:
        return (
            X_seed,
            y_seed,
            X_unlabeled,
            y_unlabeled,
            X_test,
            y_test,
            df_seed["text"].values,
            df_unlabeled["text"].values,
            df_test["text"].values,
            text_encoder,
        )
    return X_seed, y_seed, X_unlabeled, y_unlabeled, X_test, y_test

# =============================================================================
# ML core
# =============================================================================
def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    return_predictions: bool = False,
):
    model = LogisticRegression(max_iter=1000, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)

    report = classification_report(y_eval, y_pred, output_dict=True, zero_division=0)
    head_f1 = np.mean([report[str(i)]["f1-score"] for i in HEAD_LABELS if str(i) in report])
    tail_f1 = np.mean([report[str(i)]["f1-score"] for i in TAIL_LABELS if str(i) in report])

    metrics = {
        "accuracy": accuracy_score(y_eval, y_pred),
        "f1": f1_score(y_eval, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_eval, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y_eval, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_eval, y_pred, average="macro", zero_division=0),
        "head_f1": head_f1,
        "tail_f1": tail_f1,
    }
    return (model, metrics, y_pred) if return_predictions else (model, metrics)


def random_sampling(pool_size: int, n_samples: int, random_seed: int = 42) -> np.ndarray:
    take_n = min(n_samples, pool_size)
    rng = np.random.RandomState(random_seed)
    return rng.choice(pool_size, size=take_n, replace=False)


def uncertainty_sampling(model, X_unlabeled: np.ndarray, n_samples: int) -> np.ndarray:
    take_n = min(n_samples, X_unlabeled.shape[0])
    probs = model.predict_proba(X_unlabeled)
    entropy_scores = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    ranked_idx = np.argsort(-entropy_scores)
    return ranked_idx[:take_n]


def _append_selected(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    selected_idx: np.ndarray,
):
    selected_idx = np.asarray(selected_idx, dtype=int)
    X_labeled_next = np.vstack([X_labeled, X_pool[selected_idx]])
    y_labeled_next = np.concatenate([y_labeled, y_pool[selected_idx]])

    keep_mask = np.ones(X_pool.shape[0], dtype=bool)
    keep_mask[selected_idx] = False
    return X_labeled_next, y_labeled_next, X_pool[keep_mask], y_pool[keep_mask]


def _print_performance_summary(metrics: Dict[str, float]):
    """Use the same per-iteration metric log format for all methods."""
    print(
        f"Performance Summary | Macro F1: {metrics['f1']:.4f} | "
        f"Head Macro F1 ({HEAD_LABEL_COUNT} classes): {metrics['head_f1']:.4f} | "
        f"Tail Macro F1 ({TAIL_LABEL_COUNT} classes): {metrics['tail_f1']:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f}"
    )


def _compute_stopping_iteration_plateau(
    results: List[Dict],
    patience: int = 3,
    min_delta: float = 0.005,
    macro_metric: str = "f1",
    tail_metric: str = "tail_f1",
) -> Optional[int]:
    """
    Plateau-aware stopping based on Macro F1 and Tail F1.

    Stop when both Macro F1 and Tail F1 improve by less than min_delta
    for `patience` consecutive transitions. Negative changes are treated
    as no effective improvement. The experiment continues after logging
    the first triggered iteration.
    """
    if len(results) < patience + 1:
        return None

    recent = results[-(patience + 1):]
    macro_deltas = [
        float(recent[i][macro_metric]) - float(recent[i - 1][macro_metric])
        for i in range(1, len(recent))
    ]
    tail_deltas = [
        float(recent[i][tail_metric]) - float(recent[i - 1][tail_metric])
        for i in range(1, len(recent))
    ]

    macro_plateau = all(delta < min_delta for delta in macro_deltas)
    tail_plateau = all(delta < min_delta for delta in tail_deltas)

    if macro_plateau and tail_plateau:
        print("\n[Stopping Auto-Triggered by Plateau]")
        print(
            "  Reason: Macro F1 and Tail F1 showed limited improvement "
            f"for the last {patience} consecutive iterations."
        )
        print(f"  Recent Macro F1 deltas: {[round(x, 6) for x in macro_deltas]}")
        print(f"  Recent Tail F1 deltas: {[round(x, 6) for x in tail_deltas]}")
        print(f"  min_delta: {min_delta}, patience: {patience}")
        return int(results[-1]["iteration"])

    return None


def summarize_plateau_saving(
    results: List[Dict],
    framework_name: str,
    plateau_stop_iter: int,
    max_iterations: int,
) -> Dict:
    """
    Summarize saving metrics after the first plateau stop.

    Cost is reported as operation-count proxy, not monetary cost.
    """
    df = pd.DataFrame(results)
    if df.empty:
        raise ValueError(f"Cannot summarize empty results for {framework_name}")

    final_row = df.iloc[-1]
    stop_row = df[df["iteration"] == plateau_stop_iter]
    if stop_row.empty:
        stop_row = df.iloc[[-1]]
    stop_row = stop_row.iloc[0]

    saved_iterations = max_iterations - int(plateau_stop_iter)

    return {
        "Framework": framework_name,
        "Plateau_Stop_Iteration": int(plateau_stop_iter),
        "Saved_Iterations": int(saved_iterations),
        "Stop_Macro_F1": float(stop_row["f1"]),
        "Stop_Tail_F1": float(stop_row["tail_f1"]),
        "Final_Macro_F1": float(final_row["f1"]),
        "Final_Tail_F1": float(final_row["tail_f1"]),
        "Macro_F1_Gap_To_Final": float(final_row["f1"] - stop_row["f1"]),
        "Tail_F1_Gap_To_Final": float(final_row["tail_f1"] - stop_row["tail_f1"]),
        "Saved_Query_Labels": int(final_row["human_labels"] - stop_row["human_labels"]),
        "Saved_Generation_Requests": int(
            final_row["llm_generation_requests"] - stop_row["llm_generation_requests"]
        ),
        "Saved_Valid_Generated": int(
            final_row["llm_valid_generated"] - stop_row["llm_valid_generated"]
        ),
        "Saved_Verification_Requests": int(
            final_row["llm_verification_requests"] - stop_row["llm_verification_requests"]
        ),
        "Saved_Synthetic_Accepted": int(
            final_row["synthetic_accepted"] - stop_row["synthetic_accepted"]
        ),
    }


# =============================================================================
# Experiment runners
# =============================================================================
def _record_iteration(
    results: List[Dict],
    iteration: int,
    X_labeled: np.ndarray,
    metrics: Dict[str, float],
    cost_state: Dict[str, int],
):
    results.append(
        {
            "iteration": iteration,
            "labeled_samples": X_labeled.shape[0],
            **metrics,
            **cost_state,
        }
    )


def run_passive_learning_experiment(
    X_seed,
    y_seed,
    X_unlabeled,
    y_unlabeled,
    X_test,
    y_test,
    batch_size=40,
    n_iterations=40,
    random_seed=42,
    plateau_patience: int = 3,
    plateau_delta: float = 0.005,
):
    print(f"\n{'=' * 60}\nPASSIVE LEARNING EXPERIMENT (Random)\n{'=' * 60}")
    X_labeled, y_labeled = X_seed.copy(), y_seed.copy()
    X_pool, y_pool = X_unlabeled.copy(), y_unlabeled.copy()

    results = []
    plateau_stop_iter = n_iterations
    cost_state = _init_operation_state(initial_human_labels=X_seed.shape[0])

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        _, metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
        _print_performance_summary(metrics)

        _record_iteration(results, iteration, X_labeled, metrics, cost_state)

        if iteration > 5:
            if plateau_stop_iter == n_iterations:
                plateau_stop = _compute_stopping_iteration_plateau(
                    results,
                    patience=plateau_patience,
                    min_delta=plateau_delta,
                )
                if plateau_stop:
                    plateau_stop_iter = plateau_stop
                    print(
                        f"*** [Ghost Tracking] Plateau stopping triggered at Iteration {plateau_stop}. "
                        f"Continuing to max iterations. ***"
                    )

        if iteration < n_iterations and X_pool.shape[0] > 0:
            selected_idx = random_sampling(X_pool.shape[0], batch_size, random_seed + iteration)
            X_labeled, y_labeled, X_pool, y_pool = _append_selected(
                X_labeled, y_labeled, X_pool, y_pool, selected_idx
            )
            cost_state["human_labels"] += len(selected_idx)

    _, final_metrics, final_pred = train_and_evaluate(
        X_labeled, y_labeled, X_test, y_test, return_predictions=True
    )
    return results, final_metrics, plateau_stop_iter, final_pred


def run_active_baseline(
    X_seed,
    y_seed,
    X_unlabeled,
    y_unlabeled,
    X_test,
    y_test,
    batch_size=40,
    n_iterations=40,
    random_seed=42,
    plateau_patience: int = 3,
    plateau_delta: float = 0.005,
):
    print(f"\n{'=' * 60}\nACTIVE LEARNING BASELINE (Warmup + Entropy)\n{'=' * 60}")
    X_labeled, y_labeled = X_seed.copy(), y_seed.copy()
    X_pool, y_pool = X_unlabeled.copy(), y_unlabeled.copy()

    results = []
    plateau_stop_iter = n_iterations
    cost_state = _init_operation_state(initial_human_labels=X_seed.shape[0])

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        model, metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
        _print_performance_summary(metrics)

        _record_iteration(results, iteration, X_labeled, metrics, cost_state)

        if iteration > 5:
            if plateau_stop_iter == n_iterations:
                plateau_stop = _compute_stopping_iteration_plateau(
                    results,
                    patience=plateau_patience,
                    min_delta=plateau_delta,
                )
                if plateau_stop:
                    plateau_stop_iter = plateau_stop
                    print(
                        f"*** [Ghost Tracking] Plateau stopping triggered at Iteration {plateau_stop}. "
                        f"Continuing to max iterations. ***"
                    )

        if iteration < n_iterations and X_pool.shape[0] > 0:
            selected_idx = (
                random_sampling(X_pool.shape[0], batch_size, random_seed + iteration)
                if iteration <= 2
                else uncertainty_sampling(model, X_pool, batch_size)
            )
            X_labeled, y_labeled, X_pool, y_pool = _append_selected(
                X_labeled, y_labeled, X_pool, y_pool, selected_idx
            )
            cost_state["human_labels"] += len(selected_idx)

    _, final_metrics, final_pred = train_and_evaluate(
        X_labeled, y_labeled, X_test, y_test, return_predictions=True
    )
    return results, final_metrics, plateau_stop_iter, final_pred


def run_generator_subprocess(
    texts,
    labels,
    label_mapping,
    output_dir,
    iteration,
    log_file,
    n_variants=1,
):
    io_dir = os.path.join(output_dir, "llm_io")
    os.makedirs(io_dir, exist_ok=True)

    input_path = os.path.join(io_dir, f"generate_input_round_{iteration:02d}.json")
    output_path = os.path.join(io_dir, f"generate_output_round_{iteration:02d}.json")

    payload = {
        "texts": [str(x) for x in texts],
        "labels": [int(x) for x in labels],
        "label_mapping": {str(k): v for k, v in label_mapping.items()},
        "n_variants": int(n_variants),
    }

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "llm_generate_worker.py"),
        "--input-json",
        input_path,
        "--output-json",
        output_path,
        "--log-file",
        log_file,
    ]

    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    print(f"Running generator subprocess: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

    with open(output_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    return (
        np.asarray(result["generated_texts"], dtype=str),
        np.asarray(result["generated_labels"], dtype=int),
        np.asarray(result["source_texts"], dtype=str),
    )


def run_validator_subprocess(
    generated_texts,
    generated_labels,
    original_texts,
    label_mapping,
    output_dir,
    iteration,
    log_file,
):
    io_dir = os.path.join(output_dir, "llm_io")
    os.makedirs(io_dir, exist_ok=True)

    input_path = os.path.join(io_dir, f"validate_input_round_{iteration:02d}.json")
    output_path = os.path.join(io_dir, f"validate_output_round_{iteration:02d}.json")

    payload = {
        "generated_texts": [str(x) for x in generated_texts],
        "generated_labels": [int(x) for x in generated_labels],
        "original_texts": [str(x) for x in original_texts],
        "label_mapping": {str(k): v for k, v in label_mapping.items()},
    }

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "llm_validate_worker.py"),
        "--input-json",
        input_path,
        "--output-json",
        output_path,
        "--log-file",
        log_file,
    ]

    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    print(f"Running validator subprocess: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

    with open(output_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    return (
        np.asarray(result["valid_texts"], dtype=str),
        np.asarray(result["valid_labels"], dtype=int),
        float(result["accept_rate"]),
    )


def run_proposed_framework(
    X_seed,
    y_seed,
    X_pool,
    y_pool,
    X_test,
    y_test,
    pool_texts,
    text_encoder,
    label_mapping,
    batch_size=40,
    n_iterations=40,
    random_seed=42,
    use_validator=True,
    plateau_patience: int = 3,
    plateau_delta: float = 0.005,
):
    variant_name = "Generate-and-Verify" if use_validator else "Generate-only"
    variant_slug = "generate_verify" if use_validator else "generate_only"
    print(f"\n{'=' * 60}\nPROPOSED FRAMEWORK ({variant_name})\n{'=' * 60}")

    X_labeled, y_labeled = X_seed.copy(), y_seed.copy()
    X_pool, y_pool, pool_texts = X_pool.copy(), y_pool.copy(), np.asarray(pool_texts).copy()

    results = []
    plateau_stop_iter = n_iterations
    cost_state = _init_operation_state(initial_human_labels=X_seed.shape[0])

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        log_file = os.path.join(DATA_DIR, f"{variant_slug}_round_{iteration:02d}_augmentation_log.txt")
        model, metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)

        _record_iteration(results, iteration, X_labeled, metrics, cost_state)

        _print_performance_summary(metrics)

        if iteration > 5:
            if plateau_stop_iter == n_iterations:
                plateau_stop = _compute_stopping_iteration_plateau(
                    results,
                    patience=plateau_patience,
                    min_delta=plateau_delta,
                )
                if plateau_stop:
                    plateau_stop_iter = plateau_stop
                    print(
                        f"*** [Ghost Tracking] Plateau stopping triggered at Iteration {plateau_stop}. "
                        f"Continuing to max iterations. ***"
                    )

        if iteration < n_iterations and X_pool.shape[0] > 0:
            selected_idx = (
                random_sampling(X_pool.shape[0], batch_size, random_seed + iteration)
                if iteration <= 2
                else uncertainty_sampling(model, X_pool, batch_size)
            )
            sel_texts, sel_labels = pool_texts[selected_idx], y_pool[selected_idx]

            gen_texts, gen_labels, gen_src = run_generator_subprocess(
                texts=sel_texts,
                labels=sel_labels,
                label_mapping=label_mapping,
                output_dir=DATA_DIR,
                iteration=iteration,
                log_file=log_file,
                n_variants=1,
            )

            cost_state["llm_generation_requests"] += len(sel_texts)

            valid_idx = [
                i
                for i, (aug_text, src_text) in enumerate(zip(gen_texts, gen_src))
                if str(aug_text).strip() and str(aug_text).strip() != str(src_text).strip()
            ]
            cost_state["llm_valid_generated"] += len(valid_idx)

            if valid_idx:
                if use_validator:
                    cost_state["llm_verification_requests"] += len(valid_idx)
                    val_texts, val_labels, ag_rate = run_validator_subprocess(
                        generated_texts=gen_texts[valid_idx],
                        generated_labels=gen_labels[valid_idx],
                        original_texts=gen_src[valid_idx],
                        label_mapping=label_mapping,
                        output_dir=DATA_DIR,
                        iteration=iteration,
                        log_file=log_file,
                    )
                    print(f"Accepted: {len(val_texts)}/{len(valid_idx)} (Rate: {ag_rate:.2f})")
                else:
                    val_texts = gen_texts[valid_idx]
                    val_labels = gen_labels[valid_idx]
                    print(f"Accepted without validator: {len(val_texts)}/{len(valid_idx)} (Rate: 1.00)")

                cost_state["synthetic_accepted"] += len(val_texts)

                if len(val_texts) > 0:
                    X_val = text_encoder.encode(
                        val_texts.tolist(),
                        batch_size=64,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    ).astype(np.float32)

                    X_labeled = np.vstack([X_labeled, X_val])
                    y_labeled = np.concatenate([y_labeled, val_labels])

                    print(f"Synthetic added: {len(val_texts)}")
                    print(f"Labeled size after synthetic: {X_labeled.shape[0]}")

            X_labeled = np.vstack([X_labeled, X_pool[selected_idx]])
            y_labeled = np.concatenate([y_labeled, sel_labels])
            cost_state["human_labels"] += len(selected_idx)

            keep_mask = np.ones(X_pool.shape[0], dtype=bool)
            keep_mask[selected_idx] = False
            X_pool, y_pool, pool_texts = X_pool[keep_mask], y_pool[keep_mask], pool_texts[keep_mask]

    _, final_metrics, final_pred = train_and_evaluate(
        X_labeled, y_labeled, X_test, y_test, return_predictions=True
    )
    return results, final_metrics, plateau_stop_iter, final_pred


# =============================================================================
# Plotting
# =============================================================================
def plot_metric_curves(
    method_results: Dict[str, List[Dict]],
    metric: str,
    ylabel: str,
    title: str,
    output_dir: str,
    run_tag: str,
    stop_iters: Optional[Dict[str, int]] = None,
):
    """
    Plot metric trajectories for all methods.

    This function is used for Macro F1 and Tail F1 curves only.
    It also marks the plateau stop iteration for each method.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    markers = {
        "Passive": "o--",
        "Active": "s-",
        "Generate-only": "^-",
        "Generate-and-Verify": "D-",
    }

    for name, results in method_results.items():
        df = pd.DataFrame(results)

        if df.empty:
            print(f"[Warning] Empty results for {name}; skip plotting {metric}.")
            continue

        if metric not in df.columns:
            print(f"[Warning] Metric '{metric}' not found for {name}; skip plotting.")
            continue

        ax.plot(
            df["iteration"],
            df[metric],
            markers.get(name, "-"),
            label=name,
            linewidth=2,
            markersize=5,
            alpha=0.9,
        )

    if stop_iters:
        for name, stop_iter in stop_iters.items():
            if stop_iter is None:
                continue

            try:
                stop_iter = int(stop_iter)
            except (TypeError, ValueError):
                continue

            
            method_df = pd.DataFrame(method_results.get(name, []))
            if method_df.empty:
                continue
            
            max_iter = int(method_df["iteration"].max())

            # 如果 stop_iter 等於最後一輪，代表沒有提前觸發 plateau，避免誤畫成停止點
            if stop_iter == max_iter:
                continue

            ax.axvline(
                x=stop_iter,
                linestyle="--",
                linewidth=1.2,
                alpha=0.55,
                label=f"{name} plateau stop",
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0.0, top=1.05)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{metric}_curve_{run_tag}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {metric} curve: {output_path}")
    return output_path


def plot_head_tail_comparison(final_metrics: Dict[str, Dict[str, float]], output_dir: str, run_tag: str):
    methods = list(final_metrics.keys())
    head_f1_scores = [final_metrics[m].get("head_f1", np.nan) for m in methods]
    tail_f1_scores = [final_metrics[m].get("tail_f1", np.nan) for m in methods]

    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    bars1 = ax.bar(x - width / 2, head_f1_scores, width, label="Head Classes")
    bars2 = ax.bar(x + width / 2, tail_f1_scores, width, label="Tail Classes")

    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Performance Comparison: Head vs. Tail Classes")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_ylim(0, 1.05)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                "nan" if np.isnan(height) else f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, 0 if np.isnan(height) else height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"head_tail_f1_comparison_{run_tag}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Head/Tail comparison: {output_path}")
    return output_path


def plot_confusion_matrix_comparison(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_dir: str,
    run_tag: str,
    label_mapping: Optional[Dict[int, str]] = None,
):
    y_true = np.asarray(y_true)
    all_preds = [np.asarray(pred) for pred in predictions.values()]
    n_classes = int(np.max(np.concatenate([y_true, *all_preds]))) + 1
    labels = np.arange(n_classes)

    display_labels = (
        [label_mapping.get(int(label), f"Class {label}") for label in labels]
        if label_mapping
        else [str(label) for label in labels]
    )

    n_methods = len(predictions)
    fig, axes = plt.subplots(1, n_methods, figsize=(7 * n_methods, 7), dpi=150, constrained_layout=True)
    if n_methods == 1:
        axes = [axes]

    last_image = None
    for ax, (name, pred) in zip(axes, predictions.items()):
        matrix = confusion_matrix(y_true, pred, labels=labels)
        row_sums = matrix.sum(axis=1, keepdims=True)
        cm = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

        last_image = ax.imshow(cm, interpolation="nearest", vmin=0.0, vmax=1.0)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(labels)
        ax.set_yticks(labels)
        ax.set_xticklabels(display_labels, fontsize=7, rotation=90)
        ax.set_yticklabels(display_labels, fontsize=7)

    fig.colorbar(last_image, ax=list(axes), fraction=0.025, pad=0.02, label="Row-normalized rate")
    output_path = os.path.join(output_dir, f"confusion_matrix_comparison_{run_tag}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix comparison: {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run plateau-aware LLM-augmented active learning experiments on Banking77")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data split and sampling")
    parser.add_argument("--batch-size", type=int, default=40, help="Samples selected per iteration")
    parser.add_argument("--n-iterations", type=int, default=40, help="Number of active learning iterations")
    parser.add_argument("--plateau-delta", type=float, default=0.005, help="Minimum F1 improvement threshold for plateau detection")
    parser.add_argument("--plateau-patience", type=int, default=3, help="Number of consecutive low-improvement iterations for plateau detection")
    parser.add_argument("--split-mode", type=str, default="fixed", choices=["fixed", "random"], help="Head/tail class assignment mode")
    parser.add_argument("--class-split-seed", type=int, default=0, help="Random seed for selecting head classes when split-mode=random")
    args = parser.parse_args()

    current_seed = args.seed
    print(f"\n>>> INITIALIZING EXPERIMENT WITH RANDOM SEED: {current_seed} <<<\n")

    global DATA_DIR
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(DATA_DIR, run_tag)
    os.makedirs(run_output_dir, exist_ok=True)
    DATA_DIR = run_output_dir

    config_name = (
        f"Banking77_{args.split_mode}"
        f"_classsplit{args.class_split_seed}"
        f"_seed{current_seed}"
    )
    logger = setup_logging(config_name, run_output_dir)

    try:
        configure_head_tail_split(
            split_mode=args.split_mode,
            class_split_seed=args.class_split_seed,
            n_head_classes=10,
        )

        split_config_path = os.path.join(DATA_DIR, "head_tail_split_config.json")
        with open(split_config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split_mode": args.split_mode,
                    "class_split_seed": args.class_split_seed,
                    "head_labels": HEAD_LABELS,
                    "tail_labels": TAIL_LABELS,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved split config: {split_config_path}")

        (
            X_seed,
            y_seed,
            X_un,
            y_un,
            X_test,
            y_test,
            _,
            pool_texts,
            _,
            text_encoder,
        ) = load_banking77_long_tail(random_seed=current_seed, return_texts=True)

        mapping = get_banking77_label_mapping()

        passive_results, passive_final, passive_plateau_stop, passive_pred = run_passive_learning_experiment(
            X_seed,
            y_seed,
            X_un,
            y_un,
            X_test,
            y_test,
            batch_size=args.batch_size,
            n_iterations=args.n_iterations,
            random_seed=current_seed,
            plateau_patience=args.plateau_patience,
            plateau_delta=args.plateau_delta,
        )

        active_results, active_final, active_plateau_stop, active_pred = run_active_baseline(
            X_seed,
            y_seed,
            X_un,
            y_un,
            X_test,
            y_test,
            batch_size=args.batch_size,
            n_iterations=args.n_iterations,
            random_seed=current_seed,
            plateau_patience=args.plateau_patience,
            plateau_delta=args.plateau_delta,
        )

        prop_no_verify_res, prop_no_verify_fin, prop_no_verify_plateau_stop, prop_no_verify_pred = run_proposed_framework(
            X_seed,
            y_seed,
            X_un,
            y_un,
            X_test,
            y_test,
            pool_texts,
            text_encoder,
            mapping,
            batch_size=args.batch_size,
            n_iterations=args.n_iterations,
            random_seed=current_seed,
            use_validator=False,
            plateau_patience=args.plateau_patience,
            plateau_delta=args.plateau_delta,
        )

        prop_verify_res, prop_verify_fin, prop_verify_plateau_stop, prop_verify_pred = run_proposed_framework(
            X_seed,
            y_seed,
            X_un,
            y_un,
            X_test,
            y_test,
            pool_texts,
            text_encoder,
            mapping,
            batch_size=args.batch_size,
            n_iterations=args.n_iterations,
            random_seed=current_seed,
            use_validator=True,
            plateau_patience=args.plateau_patience,
            plateau_delta=args.plateau_delta,
        )

        method_results = {
            "Passive": passive_results,
            "Active": active_results,
            "Generate-only": prop_no_verify_res,
            "Generate-and-Verify": prop_verify_res,
        }

        plateau_stop_iters = {
            "Passive": passive_plateau_stop,
            "Active": active_plateau_stop,
            "Generate-only": prop_no_verify_plateau_stop,
            "Generate-and-Verify": prop_verify_plateau_stop,
        }

        final_metrics = {
            "Passive": passive_final,
            "Active": active_final,
            "Generate-only": prop_no_verify_fin,
            "Generate-and-Verify": prop_verify_fin,
        }

        plot_metric_curves(
            method_results,
            metric="f1",
            ylabel="Macro F1 Score",
            title="Macro F1 Trajectory and Plateau-aware Stopping",
            output_dir=DATA_DIR,
            run_tag=run_tag,
            stop_iters=plateau_stop_iters,
        )
        plot_metric_curves(
            method_results,
            metric="tail_f1",
            ylabel="Tail Macro F1 Score",
            title="Tail F1 Trajectory and Plateau-aware Stopping",
            output_dir=DATA_DIR,
            run_tag=run_tag,
            stop_iters=plateau_stop_iters,
        )
        plot_head_tail_comparison(final_metrics, output_dir=DATA_DIR, run_tag=run_tag)
        plot_confusion_matrix_comparison(
            y_test,
            predictions={
                "Active": active_pred,
                "Generate-only": prop_no_verify_pred,
                "Generate-and-Verify": prop_verify_pred,
            },
            output_dir=DATA_DIR,
            run_tag=run_tag,
            label_mapping=mapping,
        )

        plateau_df = pd.DataFrame(
            [
                summarize_plateau_saving(passive_results, "Passive", passive_plateau_stop, args.n_iterations),
                summarize_plateau_saving(active_results, "Active", active_plateau_stop, args.n_iterations),
                summarize_plateau_saving(prop_no_verify_res, "Generate-only", prop_no_verify_plateau_stop, args.n_iterations),
                summarize_plateau_saving(prop_verify_res, "Generate-and-Verify", prop_verify_plateau_stop, args.n_iterations),
            ]
        )
        plateau_path = os.path.join(DATA_DIR, f"plateau_saving_table_{run_tag}.csv")
        plateau_df.to_csv(plateau_path, index=False)

        print("\n" + "=" * 80)
        print("PLATEAU-AWARE SAVING SUMMARY")
        print("=" * 80)
        print(plateau_df.to_string(index=False))
        print(f"\nSaved plateau saving table: {plateau_path}")

        summary_df = pd.DataFrame(
            [
                {
                    "Framework": framework,
                    "Macro_F1": metrics["f1"],
                    "Weighted_F1": metrics["weighted_f1"],
                    "Accuracy": metrics["accuracy"],
                    "Head_F1": metrics["head_f1"],
                    "Tail_F1": metrics["tail_f1"],
                    "Plateau_Stop_Iteration": plateau_stop_iters[framework],
                }
                for framework, metrics in final_metrics.items()
            ]
        )
        summary_filename = _build_output_filename("metrics_table", "Banking77_Refactored", run_tag, "csv")
        summary_path = os.path.join(DATA_DIR, summary_filename)
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 80)
        print("FRAMEWORK COMPARISON SUMMARY")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print(f"\nSaved: {summary_path}")

        print("\n=== Experimental Run Completed ===")
        print(f"Final Generate-and-Verify Macro F1: {prop_verify_fin['f1']:.4f}")
        print(f"Final Generate-and-Verify Head Macro F1: {prop_verify_fin['head_f1']:.4f}")
        print(f"Final Generate-and-Verify Tail Macro F1: {prop_verify_fin['tail_f1']:.4f}")

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if logger:
            logger.close()
            sys.stdout = logger.terminal


if __name__ == "__main__":
    main()
