#!/usr/bin/env python3
"""
Banking77 Long-Tail Text Classification Active Learning Framework
Cost-Aware Generate-only / Generate-and-Verify Refactored Version

核心設計：
1. BAAI/bge-large-en-v1.5 作為固定文字特徵編碼器。
2. Banking77 人為長尾切分：Head=15 seed samples/class, Tail=2 seed samples/class。
3. 比較 Passive、Active、Generate-only、Generate-and-Verify 四種策略。
4. Cost-aware utility 將 human labeling、LLM generation、verification、synthetic processing 拆開計算。
5. Sliding-window t-test 作為統計式 early-stopping heuristic，不中斷實驗，只記錄首次觸發點。
"""

import argparse
import gc
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from scipy import stats
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# Model and dataset configuration
# =============================================================================
LLM_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
VALIDATOR_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SENTENCE_TRANSFORMER_MODEL_ID = "BAAI/bge-large-en-v1.5"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Ghost_Tracking_BAAI")

HEAD_LABELS = list(range(0, 10))
TAIL_LABELS = list(range(10, 77))
HEAD_LABEL_COUNT = len(HEAD_LABELS)
TAIL_LABEL_COUNT = len(TAIL_LABELS)


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


def _to_text_array(values: Iterable) -> np.ndarray:
    return np.asarray(["" if value is None else str(value) for value in values])


def _build_output_filename(prefix: str, config_name: str, run_tag: str, extension: str) -> str:
    parts = [prefix]
    if config_name:
        parts.append(config_name)
    if run_tag:
        parts.append(run_tag)
    return f"{'_'.join(parts)}.{extension}"


# =============================================================================
# Cost-aware utility
# =============================================================================
@dataclass
class CostConfig:
    """
    Relative cost coefficients.
    """

    human_label_unit_cost: float = 1.0
    llm_generate_unit_cost: float = 0.03
    llm_verify_unit_cost: float = 0.03
    synthetic_processing_unit_cost: float = 0.005
    tail_weight_beta: float = 0.35
    lambda_base: float = 0.20
    lambda_growth_alpha: float = 2.0
    cost_budget: float = 1.0


def _init_cost_state(initial_human_labels: int) -> Dict[str, int]:
    """
    Cost state accumulates over iterations.

    human_labels:
        人工標註資料數；預設包含 initial seed set。
    llm_generation_requests:
        呼叫 generator 的次數。即使格式錯誤或無有效 variation，也算 generation cost。
    llm_valid_generated:
        成功解析且非原文的生成樣本數。
    llm_verification_requests:
        呼叫 validator 的次數。
    synthetic_accepted:
        最後進入訓練集的 synthetic samples 數。
    """
    return {
        "human_labels": int(initial_human_labels),
        "llm_generation_requests": 0,
        "llm_valid_generated": 0,
        "llm_verification_requests": 0,
        "synthetic_accepted": 0,
    }


def _compute_cost_aware_utility(
    metrics: Dict[str, float],
    cost_state: Dict[str, int],
    iteration: int,
    max_iter: int,
    cost_config: CostConfig,
) -> Dict[str, float]:
    macro_f1 = float(metrics.get("f1", 0.0))
    tail_f1 = float(metrics.get("tail_f1", macro_f1))
    beta = cost_config.tail_weight_beta

    performance_score = ((1.0 - beta) * macro_f1) + (beta * tail_f1)

    human_cost = cost_config.human_label_unit_cost * cost_state["human_labels"]
    generation_cost = cost_config.llm_generate_unit_cost * cost_state["llm_generation_requests"]
    verification_cost = cost_config.llm_verify_unit_cost * cost_state["llm_verification_requests"]

    total_cost = human_cost + generation_cost + verification_cost
    normalized_cost = total_cost / max(cost_config.cost_budget, 1e-8)

    lambda_t = cost_config.lambda_base * (
        1.0 + cost_config.lambda_growth_alpha * (iteration / max_iter)
    )

    utility = performance_score - (lambda_t * normalized_cost)

    return {
        "utility": utility,
        "performance_score": performance_score,
        "lambda_t": lambda_t,
        "normalized_cost": normalized_cost,
        "total_cost": total_cost,
        "human_cost": human_cost,
        "generation_cost": generation_cost,
        "verification_cost": verification_cost,
        **cost_state,
    }


def summarize_roi(results: List[Dict], framework_name: str) -> Dict:
    df = pd.DataFrame(results)
    if df.empty:
        raise ValueError(f"Cannot summarize empty results for {framework_name}")

    max_utility_idx = df["utility"].idxmax()
    max_utility_row = df.loc[max_utility_idx]
    final_row = df.iloc[-1]

    return {
        "Framework": framework_name,
        "Max_Utility": max_utility_row["utility"],
        "Max_Utility_Iteration": int(max_utility_row["iteration"]),
        "Final_Utility": final_row["utility"],
        "Final_Performance_Score": final_row["performance_score"],
        "Final_Normalized_Cost": final_row["normalized_cost"],
        "Final_Total_Cost": final_row["total_cost"],
        "Final_Human_Labels": int(final_row["human_labels"]),
        "Final_LLM_Generation_Requests": int(final_row["llm_generation_requests"]),
        "Final_LLM_Valid_Generated": int(final_row["llm_valid_generated"]),
        "Final_LLM_Verification_Requests": int(final_row["llm_verification_requests"]),
        "Final_Synthetic_Accepted": int(final_row["synthetic_accepted"]),
    }


# =============================================================================
# Model manager
# =============================================================================
class ModelContextManager:
    """Coordinate generator and validator models under limited VRAM."""

    def __init__(self):
        self.generator_tokenizer = None
        self.generator_model = None
        self.validator_tokenizer = None
        self.validator_model = None
        self.generator_on_gpu = False
        self.validator_on_gpu = False

    def _get_quant_config(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    def load_generator(self):
        if self.generator_tokenizer is None or self.generator_model is None:
            torch.cuda.empty_cache()
            print(f"\nLoading generator model: {LLM_MODEL_ID}")
            self.generator_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
            self.generator_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_ID,
                quantization_config=self._get_quant_config(),
                device_map="cuda:0",
            )
            self.generator_on_gpu = True
        elif not self.generator_on_gpu:
            print("Moving generator back to GPU")
            self.generator_model.to("cuda:0")
            self.generator_on_gpu = True
            torch.cuda.empty_cache()
        return self.generator_tokenizer, self.generator_model

    def load_validator(self):
        if self.validator_tokenizer is None or self.validator_model is None:
            torch.cuda.empty_cache()
            print(f"\nLoading validator model: {VALIDATOR_MODEL_ID}")
            self.validator_tokenizer = AutoTokenizer.from_pretrained(VALIDATOR_MODEL_ID)
            self.validator_model = AutoModelForCausalLM.from_pretrained(
                VALIDATOR_MODEL_ID,
                quantization_config=self._get_quant_config(),
                device_map="cuda:0",
            )
            self.validator_on_gpu = True
        elif not self.validator_on_gpu:
            print("Moving validator back to GPU")
            self.validator_model.to("cuda:0")
            self.validator_on_gpu = True
            torch.cuda.empty_cache()
        return self.validator_tokenizer, self.validator_model

    def offload_all_to_cpu(self):
        """
        Speed-first offload. If the local bitsandbytes/transformers stack does not allow
        `.to("cpu")` on quantized models, fall back to unloading the models.
        """
        try:
            if self.generator_model is not None and self.generator_on_gpu:
                self.generator_model.to("cpu")
                self.generator_on_gpu = False
            if self.validator_model is not None and self.validator_on_gpu:
                self.validator_model.to("cpu")
                self.validator_on_gpu = False
        except Exception as exc:
            print(f"Warning: CPU offload failed: {exc}. Unloading LLM models instead.")
            self.unload_all()
            return

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_all(self):
        if self.generator_model is not None:
            del self.generator_model
        if self.generator_tokenizer is not None:
            del self.generator_tokenizer
        if self.validator_model is not None:
            del self.validator_model
        if self.validator_tokenizer is not None:
            del self.validator_tokenizer

        self.generator_model = None
        self.generator_tokenizer = None
        self.validator_model = None
        self.validator_tokenizer = None
        self.generator_on_gpu = False
        self.validator_on_gpu = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_MODEL_MANAGER = ModelContextManager()


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
    text_encoder = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_ID)
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
    print(f"Head classes ({HEAD_LABELS[0]}-{HEAD_LABELS[-1]}): {HEAD_LABEL_COUNT} classes x {head_seed_size} seed samples")
    print(f"Tail classes ({TAIL_LABELS[0]}-{TAIL_LABELS[-1]}): {TAIL_LABEL_COUNT} classes x {tail_seed_size} seed samples")
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
# LLM generation and validation
# =============================================================================
def _generate_chat_response(tokenizer, model, messages: List[Dict[str, str]], max_new_tokens: int = 256) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()


def generate_variants_llama3(
    texts: Iterable[str],
    labels: Iterable[int],
    label_mapping: Dict[int, str],
    n_variants: int = 1,
    output_txt_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokenizer, model = _MODEL_MANAGER.load_generator()
    augmented_texts, augmented_labels, source_texts, records = [], [], [], []

    for text, label in zip(_to_text_array(texts), np.asarray(labels)):
        label_name = label_mapping.get(int(label), f"Intent_{label}")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise banking customer service text augmentation assistant. "
                    f"Your task is to generate variations strictly belonging to: '{label_name}'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Target Intent Category: {label_name}\n\n"
                    "Generate variations for the following customer service text.\n"
                    f"The variation MUST strictly preserve the exact semantic meaning of '{label_name}'.\n"
                    f"Output exactly {n_variants} variation(s), each strictly enclosed within "
                    "<variation></variation> tags.\n"
                    "Do NOT include titles, numbers, markdown, or introductory text.\n\n"
                    f"Original text:\n{text}"
                ),
            },
        ]
        response_text = _generate_chat_response(tokenizer, model, messages)
        matches = re.findall(r"<variation>(.*?)</variation>", response_text, re.DOTALL | re.IGNORECASE)
        variations = [m.strip() for m in matches if m.strip()]

        if not variations:
            continue

        for var in variations[:n_variants]:
            augmented_texts.append(var)
            augmented_labels.append(label)
            source_texts.append(text)
            records.append({"label_name": label_name, "original": text, "generated": var})

    if output_txt_path:
        with open(output_txt_path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(f"GEN | [{r['label_name']}] | Orig: {r['original']} -> Aug: {r['generated']}\n")

    _MODEL_MANAGER.offload_all_to_cpu()
    return (
        np.asarray(augmented_texts, dtype=str),
        np.asarray(augmented_labels),
        np.asarray(source_texts, dtype=str),
    )


def validate_with_qwen25(
    generated_texts: Iterable[str],
    generated_labels: Iterable[int],
    label_mapping: Dict[int, str],
    original_texts: Iterable[str],
    output_txt_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    generated_texts = _to_text_array(generated_texts)
    generated_labels = np.asarray(generated_labels)
    original_texts = _to_text_array(original_texts)

    if len(generated_texts) == 0:
        return np.array([], dtype=str), np.array([], dtype=generated_labels.dtype), 0.0

    tokenizer, model = _MODEL_MANAGER.load_validator()
    valid_texts, valid_labels, accepted = [], [], 0

    for idx, (text, label) in enumerate(zip(generated_texts, generated_labels)):
        label_name = label_mapping.get(int(label), f"Label {label}")
        orig_text = str(original_texts[idx])

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict data quality auditor for banking datasets. "
                    f"Verify whether the augmented text precisely matches the intent: '{label_name}'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Target Intent: {label_name}\n\n"
                    "Compare the Augmented Text against the Target Intent.\n"
                    "1. Does the Augmented Text accurately reflect the specific scenario?\n"
                    "2. Is it free of conversational filler?\n\n"
                    f"Original Text: {orig_text}\n"
                    f"Augmented Text: {text}\n\n"
                    "State reasoning in <reasoning> tags, then final decision strictly in "
                    "<decision>YES</decision> or <decision>NO</decision>."
                ),
            },
        ]
        resp = _generate_chat_response(tokenizer, model, messages)
        match_dec = re.search(r"<decision>(YES|NO)</decision>", resp, re.IGNORECASE)
        decision = match_dec.group(1).upper() if match_dec else "UNKNOWN"
        is_accepted = decision == "YES"

        if output_txt_path:
            with open(output_txt_path, "a", encoding="utf-8") as f:
                f.write(f"VAL | [{label_name}] | {decision} | Text: {text}\n")

        if is_accepted:
            valid_texts.append(text)
            valid_labels.append(label)
            accepted += 1

    _MODEL_MANAGER.offload_all_to_cpu()
    return np.asarray(valid_texts, dtype=str), np.asarray(valid_labels), accepted / len(generated_texts)


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


def _compute_stopping_iteration_ttest(
    results: List[Dict],
    window_size: int = 4,
    p_value_threshold: float = 0.05,
    max_samples: Optional[int] = None,
) -> Optional[int]:
    """
    Sliding-window t-test heuristic.

    注意：iteration-wise F1 不是完全獨立樣本，因此論文中應稱為
    statistical stopping heuristic，而不是嚴格統計證明。
    """
    if not results:
        return None

    if max_samples and results[-1].get("labeled_samples", 0) >= max_samples:
        return int(results[-1]["iteration"])

    if len(results) >= window_size * 2:
        current_window = [res["f1"] for res in results[-window_size:]]
        previous_window = [res["f1"] for res in results[-(window_size * 2) : -window_size]]
        _, p_value = stats.ttest_ind(current_window, previous_window, alternative="greater")

        if np.isnan(p_value):
            print("\n[Stopping Auto-Triggered by T-Test]")
            print("  Reason: Zero variance detected. Performance has flatlined.")
            return int(results[-1]["iteration"])

        if p_value >= p_value_threshold:
            print("\n[Stopping Auto-Triggered by T-Test]")
            print("  Reason: No statistically significant improvement.")
            print(f"  P-value: {p_value:.4f} >= {p_value_threshold}")
            return int(results[-1]["iteration"])

    return None

def _compute_stopping_iteration_roi(
    results,
    patience=5,
    min_delta=0.002,
    metric="utility",
):
    """
    Cost-aware ROI stopping heuristic.

    若 utility 連續 patience 輪沒有明顯突破歷史最佳值，
    表示額外資料帶來的 ROI 已經進入 diminishing returns。
    """
    if len(results) < patience + 1:
        return None

    values = [float(res[metric]) for res in results]

    best_before_recent = max(values[:-patience])
    recent_values = values[-patience:]

    no_meaningful_gain = all(
        value <= best_before_recent + min_delta
        for value in recent_values
    )

    if no_meaningful_gain:
        print("\n[Stopping Auto-Triggered by ROI]")
        print(f"  Reason: No meaningful {metric} improvement in the last {patience} iterations.")
        print(f"  Best previous {metric}: {best_before_recent:.6f}")
        print(f"  Recent {metric}: {[round(v, 6) for v in recent_values]}")
        print(f"  min_delta: {min_delta}")
        return int(results[-1]["iteration"])

    return None


# =============================================================================
# Experiment runners
# =============================================================================
def _record_iteration(
    results: List[Dict],
    iteration: int,
    X_labeled: np.ndarray,
    metrics: Dict[str, float],
    cost_state: Dict[str, int],
    cost_config: CostConfig,
    n_iterations: int,
):
    utility_info = _compute_cost_aware_utility(
        metrics=metrics,
        cost_state=cost_state,
        iteration=iteration,
        max_iter=n_iterations,
        cost_config=cost_config,
    )

    results.append(
        {
            "iteration": iteration,
            "labeled_samples": X_labeled.shape[0],
            **metrics,
            **utility_info,
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
    cost_config: Optional[CostConfig] = None,
):
    print(f"\n{'=' * 60}\nPASSIVE LEARNING EXPERIMENT (Random)\n{'=' * 60}")
    X_labeled, y_labeled = X_seed.copy(), y_seed.copy()
    X_pool, y_pool = X_unlabeled.copy(), y_unlabeled.copy()

    max_human_budget = X_seed.shape[0] + batch_size * (n_iterations - 1)
    cost_config = cost_config or CostConfig(cost_budget=max_human_budget)
    if cost_config.cost_budget == 1.0:
        cost_config.cost_budget = max_human_budget
    cost_state = _init_cost_state(initial_human_labels=X_seed.shape[0])

    results = []
    stat_stop_iter = n_iterations
    roi_stop_iter = n_iterations    

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        _, metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
        print(f"Test - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        _record_iteration(results, iteration, X_labeled, metrics, cost_state, cost_config, n_iterations)

        if iteration > 5:
            if stat_stop_iter == n_iterations:
                stat_stop = _compute_stopping_iteration_ttest(
                    results,
                    window_size=4,
                    p_value_threshold=0.05,
                )
                if stat_stop:
                    stat_stop_iter = stat_stop
                    print(
                        f"*** [Ghost Tracking] Statistical stopping triggered at Iteration {stat_stop}. "
                        f"Continuing to max iterations. ***"
                    )

            if roi_stop_iter == n_iterations:
                roi_stop = _compute_stopping_iteration_roi(
                    results,
                    patience=5,
                    min_delta=0.002,
                    metric="utility",
                )
                if roi_stop:
                    roi_stop_iter = roi_stop
                    print(
                        f"*** [Ghost Tracking] ROI stopping triggered at Iteration {roi_stop}. "
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
    return results, final_metrics, stat_stop_iter, roi_stop_iter, final_pred


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
    cost_config: Optional[CostConfig] = None,
):
    print(f"\n{'=' * 60}\nACTIVE LEARNING BASELINE (Warmup + Entropy)\n{'=' * 60}")
    X_labeled, y_labeled = X_seed.copy(), y_seed.copy()
    X_pool, y_pool = X_unlabeled.copy(), y_unlabeled.copy()

    max_human_budget = X_seed.shape[0] + batch_size * (n_iterations - 1)
    cost_config = cost_config or CostConfig(cost_budget=max_human_budget)
    if cost_config.cost_budget == 1.0:
        cost_config.cost_budget = max_human_budget
    cost_state = _init_cost_state(initial_human_labels=X_seed.shape[0])

    results = []
    stat_stop_iter = n_iterations
    roi_stop_iter = n_iterations 

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        model, metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
        print(f"Test - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        _record_iteration(results, iteration, X_labeled, metrics, cost_state, cost_config, n_iterations)

        if iteration > 5:
            if stat_stop_iter == n_iterations:
                stat_stop = _compute_stopping_iteration_ttest(
                    results,
                    window_size=4,
                    p_value_threshold=0.05,
                )
                if stat_stop:
                    stat_stop_iter = stat_stop
                    print(
                        f"*** [Ghost Tracking] Statistical stopping triggered at Iteration {stat_stop}. "
                        f"Continuing to max iterations. ***"
                    )

            if roi_stop_iter == n_iterations:
                roi_stop = _compute_stopping_iteration_roi(
                    results,
                    patience=5,
                    min_delta=0.002,
                    metric="utility",
                )
                if roi_stop:
                    roi_stop_iter = roi_stop
                    print(
                        f"*** [Ghost Tracking] ROI stopping triggered at Iteration {roi_stop}. "
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
    return results, final_metrics, stat_stop_iter, roi_stop_iter, final_pred


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
    cost_config: Optional[CostConfig] = None,
):
    variant_name = "Generate-and-Verify" if use_validator else "Generate-only"
    variant_slug = "generate_verify" if use_validator else "generate_only"
    print(f"\n{'=' * 60}\nPROPOSED FRAMEWORK ({variant_name})\n{'=' * 60}")

    X_labeled, y_labeled = X_seed.copy(), y_seed.copy()
    X_pool, y_pool, pool_texts = X_pool.copy(), y_pool.copy(), np.asarray(pool_texts).copy()

    max_human_budget = X_seed.shape[0] + batch_size * (n_iterations - 1)
    cost_config = cost_config or CostConfig(cost_budget=max_human_budget)
    if cost_config.cost_budget == 1.0:
        cost_config.cost_budget = max_human_budget
    cost_state = _init_cost_state(initial_human_labels=X_seed.shape[0])

    results = []
    stat_stop_iter = n_iterations
    roi_stop_iter = n_iterations 

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        log_file = os.path.join(DATA_DIR, f"{variant_slug}_round_{iteration:02d}_augmentation_log.txt")
        model, metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)

        _record_iteration(results, iteration, X_labeled, metrics, cost_state, cost_config, n_iterations)

        print(
            f"Performance Summary | Macro F1: {metrics['f1']:.4f} | "
            f"Head Macro F1 ({HEAD_LABELS[0]}-{HEAD_LABELS[-1]}): {metrics['head_f1']:.4f} | "
            f"Tail Macro F1 ({TAIL_LABELS[0]}-{TAIL_LABELS[-1]}): {metrics['tail_f1']:.4f}"
        )

        if iteration > 5:
            if stat_stop_iter == n_iterations:
                stat_stop = _compute_stopping_iteration_ttest(
                    results,
                    window_size=4,
                    p_value_threshold=0.05,
                )
                if stat_stop:
                    stat_stop_iter = stat_stop
                    print(
                        f"*** [Ghost Tracking] Statistical stopping triggered at Iteration {stat_stop}. "
                        f"Continuing to max iterations. ***"
                    )

            if roi_stop_iter == n_iterations:
                roi_stop = _compute_stopping_iteration_roi(
                    results,
                    patience=5,
                    min_delta=0.002,
                    metric="utility",
                )
                if roi_stop:
                    roi_stop_iter = roi_stop
                    print(
                        f"*** [Ghost Tracking] ROI stopping triggered at Iteration {roi_stop}. "
                        f"Continuing to max iterations. ***"
                    )

        if iteration < n_iterations and X_pool.shape[0] > 0:
            selected_idx = (
                random_sampling(X_pool.shape[0], batch_size, random_seed + iteration)
                if iteration <= 2
                else uncertainty_sampling(model, X_pool, batch_size)
            )
            sel_texts, sel_labels = pool_texts[selected_idx], y_pool[selected_idx]

            gen_texts, gen_labels, gen_src = generate_variants_llama3(
                sel_texts,
                sel_labels,
                label_mapping,
                output_txt_path=log_file,
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
                    val_texts, val_labels, ag_rate = validate_with_qwen25(
                        gen_texts[valid_idx],
                        gen_labels[valid_idx],
                        label_mapping,
                        gen_src[valid_idx],
                        output_txt_path=log_file,
                    )
                    print(f"Accepted: {len(val_texts)}/{len(valid_idx)} (Rate: {ag_rate:.2f})")
                else:
                    val_texts = gen_texts[valid_idx]
                    val_labels = gen_labels[valid_idx]
                    print(f"Accepted without validator: {len(val_texts)}/{len(valid_idx)} (Rate: 1.00)")

                cost_state["synthetic_accepted"] += len(val_texts)

                if len(val_texts) > 0:
                    _MODEL_MANAGER.offload_all_to_cpu()
                    X_val = text_encoder.encode(
                        val_texts.tolist(),
                        batch_size=64,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    ).astype(np.float32)
                    X_labeled = np.vstack([X_labeled, X_val])
                    y_labeled = np.concatenate([y_labeled, val_labels])

            X_labeled = np.vstack([X_labeled, X_pool[selected_idx]])
            y_labeled = np.concatenate([y_labeled, sel_labels])
            cost_state["human_labels"] += len(selected_idx)

            keep_mask = np.ones(X_pool.shape[0], dtype=bool)
            keep_mask[selected_idx] = False
            X_pool, y_pool, pool_texts = X_pool[keep_mask], y_pool[keep_mask], pool_texts[keep_mask]

    _, final_metrics, final_pred = train_and_evaluate(
        X_labeled, y_labeled, X_test, y_test, return_predictions=True
    )
    return results, final_metrics, stat_stop_iter, roi_stop_iter, final_pred


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
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    markers = {
        "Passive": "o--",
        "Active": "s-",
        "Generate-only": "^-",
        "Generate-and-Verify": "D-",
    }

    for name, results in method_results.items():
        df = pd.DataFrame(results)
        if df.empty or metric not in df.columns:
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
            if stop_iter and name in ["Generate-only", "Generate-and-Verify"]:
                ax.axvline(
                    x=stop_iter,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label=f"{name} stop heuristic",
                )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
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
    parser = argparse.ArgumentParser(description="Run cost-aware active learning experiments on Banking77")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data split and sampling")
    parser.add_argument("--batch-size", type=int, default=40, help="Samples selected per iteration")
    parser.add_argument("--n-iterations", type=int, default=40, help="Number of active learning iterations")
    parser.add_argument("--lambda-base", type=float, default=0.20, help="Base lambda for utility penalty")
    parser.add_argument("--lambda-growth-alpha", type=float, default=2.0, help="Growth rate of lambda over iterations")
    parser.add_argument("--tail-weight-beta", type=float, default=0.35, help="Weight for tail F1 in performance score")
    parser.add_argument("--llm-cost", type=float, default=0.03, help="Relative cost of one local LLM generation request")
    parser.add_argument("--verify-cost", type=float, default=0.03, help="Relative cost of one validation request")
    parser.add_argument("--synthetic-processing-cost", type=float, default=0.005, help="Relative cost of keeping one synthetic sample")
    args = parser.parse_args()

    current_seed = args.seed
    print(f"\n>>> INITIALIZING EXPERIMENT WITH RANDOM SEED: {current_seed} <<<\n")

    global DATA_DIR
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(DATA_DIR, run_tag)
    os.makedirs(run_output_dir, exist_ok=True)
    DATA_DIR = run_output_dir

    logger = setup_logging("Banking77_Refactored", run_output_dir)

    try:
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

        max_human_budget = X_seed.shape[0] + args.batch_size * (args.n_iterations - 1)
        base_cost_config = CostConfig(
            human_label_unit_cost=1.0,
            llm_generate_unit_cost=args.llm_cost,
            llm_verify_unit_cost=args.verify_cost,
            tail_weight_beta=args.tail_weight_beta,
            lambda_base=args.lambda_base,
            lambda_growth_alpha=args.lambda_growth_alpha,
            cost_budget=max_human_budget,
        )

        print("\n=== Cost Configuration ===")
        print(base_cost_config)
        print("==========================\n")

        passive_results, passive_final, passive_stat_stop, passive_roi_stop, passive_pred = run_passive_learning_experiment(
            X_seed,
            y_seed,
            X_un,
            y_un,
            X_test,
            y_test,
            batch_size=args.batch_size,
            n_iterations=args.n_iterations,
            random_seed=current_seed,
            cost_config=base_cost_config,
        )

        active_results, active_final, active_stat_stop, active_roi_stop, active_pred = run_active_baseline(
            X_seed,
            y_seed,
            X_un,
            y_un,
            X_test,
            y_test,
            batch_size=args.batch_size,
            n_iterations=args.n_iterations,
            random_seed=current_seed,
            cost_config=base_cost_config,
        )

        prop_no_verify_res, prop_no_verify_fin, prop_no_verify_stat_stop, prop_no_verify_roi_stop, prop_no_verify_pred = run_proposed_framework(
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
            cost_config=base_cost_config,
        )

        prop_verify_res, prop_verify_fin, prop_verify_stat_stop, prop_verify_roi_stop, prop_verify_pred = run_proposed_framework(
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
            cost_config=base_cost_config,
        )

        method_results = {
            "Passive": passive_results,
            "Active": active_results,
            "Generate-only": prop_no_verify_res,
            "Generate-and-Verify": prop_verify_res,
        }
        stat_stop_iters = {
            "Passive": passive_stat_stop,
            "Active": active_stat_stop,
            "Generate-only": prop_no_verify_stat_stop,
            "Generate-and-Verify": prop_verify_stat_stop,
        }

        roi_stop_iters = {
            "Passive": passive_roi_stop,
            "Active": active_roi_stop,
            "Generate-only": prop_no_verify_roi_stop,
            "Generate-and-Verify": prop_verify_roi_stop,
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
            title="Performance Trajectory and Statistical Stopping Heuristic",
            output_dir=DATA_DIR,
            run_tag=run_tag,
            stop_iters=stat_stop_iters,
        )
        plot_metric_curves(
            method_results,
            metric="utility",
            ylabel="Cost-aware Utility",
            title="Budget Utility and ROI Trajectory",
            output_dir=DATA_DIR,
            run_tag=run_tag,
            stop_iters=roi_stop_iters,
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

        roi_df = pd.DataFrame(
            [
                summarize_roi(passive_results, "Passive"),
                summarize_roi(active_results, "Active"),
                summarize_roi(prop_no_verify_res, "Generate-only"),
                summarize_roi(prop_verify_res, "Generate-and-Verify"),
            ]
        )
        roi_path = os.path.join(DATA_DIR, f"roi_table_{run_tag}.csv")
        roi_df.to_csv(roi_path, index=False)

        print("\n" + "=" * 80)
        print("ROI SUMMARY")
        print("=" * 80)
        print(roi_df.to_string(index=False))
        print(f"\nSaved ROI table: {roi_path}")

        summary_df = pd.DataFrame(
            [
                {
                    "Framework": framework,
                    "Macro_F1": metrics["f1"],
                    "Weighted_F1": metrics["weighted_f1"],
                    "Accuracy": metrics["accuracy"],
                    "Head_F1": metrics["head_f1"],
                    "Tail_F1": metrics["tail_f1"],
                    "Stat_Stop_Iteration": stat_stop_iters[framework],
                    "ROI_Stop_Iteration": roi_stop_iters[framework],
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
        _MODEL_MANAGER.offload_all_to_cpu()
        gc.collect()
        if logger:
            logger.close()
            sys.stdout = logger.terminal


if __name__ == "__main__":
    main()
