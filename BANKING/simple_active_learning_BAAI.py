#!/usr/bin/env python3
"""
Banking77 長尾文本分類主動學習框架 (嚴謹重構版)

核心特徵：
1. BGE-Large 高維度特徵編碼。
2. 人為構造長尾分佈 (Simulated Imbalance: Head=15, Tail=2)。
3. 嚴格的 Qwen 決策收束 (XML Tag-only)。
4. ModelContextManager 解決 VRAM 碎片化與 OOM。
5. 動態停止機制邏輯修正。
"""

import os
import sys
import gc
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from scipy import stats
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings_filter = True
if warnings_filter:
    import warnings
    warnings.filterwarnings("ignore")

try:
    from data_augmentation_logger import DataAugmentationLogger
    LOGGING_AVAILABLE = True
except ImportError:
    print("Warning: DataAugmentationLogger not found. Logging disabled.")
    LOGGING_AVAILABLE = False


# ============================================================================
# 模型配置與上下文管理器 (解決 VRAM 碎片化)
# ============================================================================
LLM_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
VALIDATOR_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SENTENCE_TRANSFORMER_MODEL_ID = "BAAI/bge-large-en-v1.5"

class ModelContextManager:
    """協調多模型在有限 24GB VRAM 下的載入與卸載"""
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
                LLM_MODEL_ID, quantization_config=self._get_quant_config(), device_map="cuda:0"
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
                VALIDATOR_MODEL_ID, quantization_config=self._get_quant_config(), device_map="cuda:0"
            )
            self.validator_on_gpu = True
        elif not self.validator_on_gpu:
            print("Moving validator back to GPU")
            self.validator_model.to("cuda:0")
            self.validator_on_gpu = True
            torch.cuda.empty_cache()
        return self.validator_tokenizer, self.validator_model
    
    def offload_all_to_cpu(self):
        if self.generator_model is not None and self.generator_on_gpu:
            self.generator_model.to("cpu")
            self.generator_on_gpu = False
        if self.validator_model is not None and self.validator_on_gpu:
            self.validator_model.to("cpu")
            self.validator_on_gpu = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

_MODEL_MANAGER = ModelContextManager()

# ============================================================================
# 全域路徑與日誌系統
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "BBAI") 

HEAD_LABELS = list(range(0, 10))
TAIL_LABELS = list(range(10, 77))
HEAD_LABEL_COUNT = len(HEAD_LABELS)
TAIL_LABEL_COUNT = len(TAIL_LABELS)

def setup_logging(config_name, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(output_dir, f"experiment_log_{config_name}_{timestamp}.txt")
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
        def isatty(self):
            if hasattr(self.terminal, "isatty"):
                return self.terminal.isatty()
            return False
        def __getattr__(self, name):
            return getattr(self.terminal, name)
        def close(self):
            self.log.close()
    logger = Logger(log_filename)
    sys.stdout = logger
    print(f"Logging started: {log_filename}")
    return logger

def _to_text_array(values):
    return np.asarray(["" if value is None else str(value) for value in values])

# ============================================================================
# 資料集與長尾特徵處理
# ============================================================================
def get_banking77_label_mapping():
    dataset = load_dataset("PolyAI/banking77", split="train")
    return {i: name for i, name in enumerate(dataset.features['label'].names)}

def load_banking77_long_tail(random_seed=42, return_texts=False):
    """
    構造嚴謹的 Simulated Imbalance (人為長尾)。
    Head classes: 0-9 with 15 initial seed samples each.
    Tail classes: 10-76 with 2 initial seed samples each.
    Test: 每類 20 筆確保評估平衡。
    """
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
        
        # 1. 抽取平衡測試集
        test_idx = group.index[:test_size_per_class]
        df_test_list.append(group.loc[test_idx])
        group = group.drop(test_idx)
        
        # 2. 構造不平衡種子集
        seed_size = head_seed_size if label_id in HEAD_LABELS else tail_seed_size
        seed_idx = group.index[:seed_size]
        df_seed_list.append(group.loc[seed_idx])
        
        # 3. 剩餘進入未標註池
        unlabeled_idx = group.drop(seed_idx).index
        df_unlabeled_list.append(group.loc[unlabeled_idx])

    df_seed = pd.concat(df_seed_list).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    df_test = pd.concat(df_test_list).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    df_unlabeled = pd.concat(df_unlabeled_list).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    print(f"\n=== Encoding with {SENTENCE_TRANSFORMER_MODEL_ID} ===")
    combined_text = pd.concat([df_seed["text"], df_unlabeled["text"], df_test["text"]], ignore_index=True)
    text_encoder = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_ID)
    X_all = text_encoder.encode(
        combined_text.tolist(), batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True
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
    print(f"Head classes ({HEAD_LABELS[0]}-{HEAD_LABELS[-1]}): {HEAD_LABEL_COUNT} classes x {head_seed_size} seed samples")
    print(f"Tail classes ({TAIL_LABELS[0]}-{TAIL_LABELS[-1]}): {TAIL_LABEL_COUNT} classes x {tail_seed_size} seed samples")
    print(f"Seed set size: {X_seed.shape[0]}")
    print(f"Unlabeled pool size: {X_unlabeled.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print("=================================\n")

    if return_texts:
        return X_seed, y_seed, X_unlabeled, y_unlabeled, X_test, y_test, df_seed["text"].values, df_unlabeled["text"].values, df_test["text"].values, text_encoder
    return X_seed, y_seed, X_unlabeled, y_unlabeled, X_test, y_test

# ============================================================================
# LLM 擴增與驗證邏輯
# ============================================================================
def _generate_chat_response(tokenizer, model, messages, max_new_tokens=256):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        # 強制設定 do_sample=False (等同於 do_sample=0)，並移除 temperature 與 top_p
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def generate_variants_llama3(texts, labels, label_mapping, n_variants=1, output_txt_path=None):
    tokenizer, model = _MODEL_MANAGER.load_generator()
    augmented_texts, augmented_labels, source_texts, records = [], [], [], []

    for text, label in zip(_to_text_array(texts), np.asarray(labels)):
        label_name = label_mapping.get(int(label), f"Intent_{label}")
        messages = [
            {"role": "system", "content": f"You are a precise banking customer service text augmentation assistant. Your core task is to generate variations that strictly belong to the intent category: '{label_name}'."},
            {"role": "user", "content": f"Target Intent Category: {label_name}\n\nGenerate variations for the following customer service text.\nThe variation MUST strictly preserve the exact semantic meaning of the target intent '{label_name}'.\nPlease output exactly {n_variants} variation(s) strictly enclosed within <variation></variation> tags.\nDo NOT include any titles, numbers, markdown, or introductory text.\n\nOriginal text:\n{text}"}
        ]
        response_text = _generate_chat_response(tokenizer, model, messages)
        matches = re.findall(r'<variation>(.*?)</variation>', response_text, re.DOTALL | re.IGNORECASE)
        variations = [m.strip() for m in matches if m.strip()]

        if not variations:
            continue # 拒絕使用原文充數

        for var in variations[:n_variants]:
            augmented_texts.append(var); augmented_labels.append(label); source_texts.append(text)
            records.append({"label_name": label_name, "original": text, "generated": var})

    if output_txt_path:
        with open(output_txt_path, "a", encoding="utf-8") as f:
            for r in records: f.write(f"GEN | [{r['label_name']}] | Orig: {r['original']} -> Aug: {r['generated']}\n")

    _MODEL_MANAGER.offload_all_to_cpu()
    return np.asarray(augmented_texts, dtype=str), np.asarray(augmented_labels), np.asarray(source_texts, dtype=str)

def validate_with_qwen25(generated_texts, generated_labels, label_mapping, original_texts, output_txt_path=None):
    if len(generated_texts) == 0:
        return np.array([]), np.array([]), 0.0

    tokenizer, model = _MODEL_MANAGER.load_validator()
    valid_texts, valid_labels, accepted = [], [], 0

    for idx, (text, label) in enumerate(zip(_to_text_array(generated_texts), np.asarray(generated_labels))):
        label_name = label_mapping.get(int(label), f"Label {label}")
        orig_text = str(original_texts[idx])
        
        messages = [
            {"role": "system", "content": f"You are a strict data quality auditor for banking datasets. Your task is to verify if an augmented text precisely matches the target customer service intent: '{label_name}'."},
            {"role": "user", "content": f"Target Intent: {label_name}\n\nCompare the 'Augmented Text' against the 'Target Intent'.\n1. Does the Augmented Text accurately reflect the specific scenario defined by the intent?\n2. Is it free of conversational filler?\n\nOriginal Text: {orig_text}\nAugmented Text: {text}\n\nState reasoning in <reasoning> tags, then final decision strictly in <decision>YES</decision> or <decision>NO</decision>."}
        ]
        resp = _generate_chat_response(tokenizer, model, messages)
        
        # 嚴格的 XML 解析，捨棄有漏洞的字串比對
        match_dec = re.search(r'<decision>(YES|NO)</decision>', resp, re.IGNORECASE)
        decision = match_dec.group(1).upper() if match_dec else "UNKNOWN"
        is_accepted = (decision == "YES")

        if output_txt_path:
            with open(output_txt_path, "a", encoding="utf-8") as f:
                f.write(f"VAL | [{label_name}] | {decision} | Text: {text}\n")

        if is_accepted:
            valid_texts.append(text)
            valid_labels.append(label)
            accepted += 1

    _MODEL_MANAGER.offload_all_to_cpu()
    return np.asarray(valid_texts, dtype=str), np.asarray(valid_labels), (accepted / len(generated_texts))

# ============================================================================
# 機器學習核心與動態指標
# ============================================================================
def train_and_evaluate(X_train, y_train, X_eval, y_eval, return_predictions=False):
    model = LogisticRegression(max_iter=1000)
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
        "head_f1": head_f1, "tail_f1": tail_f1
    }
    return (model, metrics, y_pred) if return_predictions else (model, metrics)


def _append_selected(X_labeled, y_labeled, X_pool, y_pool, selected_idx):
    selected_idx = np.asarray(selected_idx, dtype=int)

    X_new = X_pool[selected_idx]
    y_new = y_pool[selected_idx]

    X_labeled_next = np.vstack([X_labeled, X_new])
    y_labeled_next = np.concatenate([y_labeled, y_new])

    keep_mask = np.ones(X_pool.shape[0], dtype=bool)
    keep_mask[selected_idx] = False

    X_pool_next = X_pool[keep_mask]
    y_pool_next = y_pool[keep_mask]

    return X_labeled_next, y_labeled_next, X_pool_next, y_pool_next


def random_sampling(pool_size, n_samples, random_seed=42):
    take_n = min(n_samples, pool_size)
    rng = np.random.RandomState(random_seed)
    return rng.choice(pool_size, size=take_n, replace=False)

def _build_output_filename(prefix, config_name, run_tag, extension):
    parts = [prefix]
    if config_name:
        parts.append(config_name)
    if run_tag:
        parts.append(run_tag)
    return f"{'_'.join(parts)}.{extension}"


def uncertainty_sampling(model, X_unlabeled, n_samples):
    take_n = min(n_samples, X_unlabeled.shape[0])
    probs = model.predict_proba(X_unlabeled)
    entropy_scores = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    ranked_idx = np.argsort(-entropy_scores)
    return ranked_idx[:take_n]


def run_passive_learning_experiment(X_seed, y_seed, X_unlabeled, y_unlabeled, X_test, y_test, batch_size=40, n_iterations=40, random_seed=42):
    print(f"\n{'=' * 60}\nPASSIVE LEARNING EXPERIMENT (Random)\n{'=' * 60}")
    X_labeled, y_labeled = X_seed.copy(), y_seed.copy()
    X_pool, y_pool = X_unlabeled.copy(), y_unlabeled.copy()

    results = []
    final_stop_iter = n_iterations

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        _, metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
        print(f"Test - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        utility, lambda_t = _compute_utility(metrics["f1"], X_labeled.shape[0], iteration, n_iterations)
        results.append({
            "iteration": iteration, "labeled_samples": X_labeled.shape[0], **metrics,
            "utility": utility, "lambda_t": lambda_t
        })

        if iteration > 5:
            stop_iter = _compute_stopping_iteration_ttest(results, window_size=4, p_value_threshold=0.05)
            if stop_iter:
                final_stop_iter = iteration
                break

        if iteration < n_iterations and X_pool.shape[0] > 0:
            selected_idx = random_sampling(X_pool.shape[0], batch_size, random_seed + iteration)
            X_labeled, y_labeled, X_pool, y_pool = _append_selected(X_labeled, y_labeled, X_pool, y_pool, selected_idx)

    _, final_metrics, final_pred = train_and_evaluate(X_labeled, y_labeled, X_test, y_test, return_predictions=True)
    return results, final_metrics, final_stop_iter, final_pred


def run_active_baseline(X_seed, y_seed, X_unlabeled, y_unlabeled, X_test, y_test, batch_size=40, n_iterations=40, random_seed=42):
    print(f"\n{'=' * 60}\nACTIVE LEARNING BASELINE (Warmup + Entropy)\n{'=' * 60}")
    X_labeled, y_labeled = X_seed.copy(), y_seed.copy()
    X_pool, y_pool = X_unlabeled.copy(), y_unlabeled.copy()

    results = []
    final_stop_iter = n_iterations

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        model, metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
        print(f"Test - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        utility, lambda_t = _compute_utility(metrics["f1"], X_labeled.shape[0], iteration, n_iterations)
        results.append({
            "iteration": iteration, "labeled_samples": X_labeled.shape[0], **metrics,
            "utility": utility, "lambda_t": lambda_t
        })

        if iteration > 5:
            stop_iter = _compute_stopping_iteration_ttest(results, window_size=4, p_value_threshold=0.05)
            if stop_iter:
                final_stop_iter = iteration
                break

        if iteration < n_iterations and X_pool.shape[0] > 0:
            selected_idx = random_sampling(X_pool.shape[0], batch_size, random_seed + iteration) if iteration <= 2 else uncertainty_sampling(model, X_pool, batch_size)
            X_labeled, y_labeled, X_pool, y_pool = _append_selected(X_labeled, y_labeled, X_pool, y_pool, selected_idx)

    _, final_metrics, final_pred = train_and_evaluate(X_labeled, y_labeled, X_test, y_test, return_predictions=True)
    return results, final_metrics, final_stop_iter, final_pred


def plot_utility_curve(passive_results, active_results, proposed_results, stopping_iters, config_name=""):
    passive_df = pd.DataFrame(passive_results)
    active_df = pd.DataFrame(active_results)
    proposed_df = pd.DataFrame(proposed_results)

    plt.figure(figsize=(9, 6))

    plt.plot(passive_df["iteration"], passive_df["utility"], "o--", label="Passive", linewidth=2, markersize=7)
    plt.plot(active_df["iteration"], active_df["utility"], "s-", label="Active", linewidth=2, markersize=7)
    plt.plot(proposed_df["iteration"], proposed_df["utility"], "^-", label="Proposed", linewidth=2, markersize=7)

    proposed_stopping_iteration = stopping_iters.get("proposed", 0) if isinstance(stopping_iters, dict) else 0
    if proposed_stopping_iteration > 0:
        plt.axvline(
            x=proposed_stopping_iteration,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Auto-stopping Point",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Utility")
    plt.title("Utility Curve Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(DATA_DIR, "utility_curve_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_head_tail_comparison(passive_final, active_final, proposed_final, config_name="", run_tag="", show_plots=True):
    methods = ["Passive (Random)", "Active (Entropy)", "Proposed (LLM+Qwen)"]
    head_f1_scores = [passive_final.get("head_f1", np.nan), active_final.get("head_f1", np.nan), proposed_final.get("head_f1", np.nan)]
    tail_f1_scores = [passive_final.get("tail_f1", np.nan), active_final.get("tail_f1", np.nan), proposed_final.get("tail_f1", np.nan)]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    rects1 = ax.bar(x - width / 2, head_f1_scores, width, label="Head Classes (Top 10)", color="#4F81BD", edgecolor="black")
    rects2 = ax.bar(x + width / 2, tail_f1_scores, width, label="Tail Classes (Bottom 10)", color="#C0504D", edgecolor="black")

    ax.set_ylabel("Macro F1 Score", fontsize=12, fontweight="bold")
    ax.set_title("Performance Comparison: Head vs. Tail Classes at Final Iteration", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_ylim(0, 1.05)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            label_text = "nan" if np.isnan(height) else f"{height:.3f}"
            ax.annotate(
                label_text,
                xy=(rect.get_x() + rect.get_width() / 2, 0 if np.isnan(height) else height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()

    plot_filename = _build_output_filename("head_tail_f1_comparison", config_name, run_tag, "png")
    output_path = os.path.join(DATA_DIR, plot_filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show_plots:
        plt.show(block=False)
        plt.pause(0.1)
    else:
        plt.close(fig)

    return output_path


def plot_confusion_matrix_comparison(y_true, active_pred, proposed_pred, config_name="", run_tag="", show_plots=True, label_mapping=None):
    y_true = np.asarray(y_true)
    active_pred = np.asarray(active_pred)
    proposed_pred = np.asarray(proposed_pred)

    n_classes = int(np.max(np.concatenate([y_true, active_pred, proposed_pred]))) + 1
    labels = np.arange(n_classes)

    if label_mapping is not None:
        display_labels = [label_mapping.get(int(label), f"Class {label}") for label in labels]
    else:
        display_labels = [str(label) for label in labels]

    def _normalize_confusion(true_labels, pred_labels):
        matrix = confusion_matrix(true_labels, pred_labels, labels=labels)
        row_sums = matrix.sum(axis=1, keepdims=True)
        return np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

    active_cm = _normalize_confusion(y_true, active_pred)
    proposed_cm = _normalize_confusion(y_true, proposed_pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150, constrained_layout=True)
    cm_data = [
        (axes[0], active_cm, "Active Learning (Warmup+Entropy)", "Blues"),
        (axes[1], proposed_cm, "Proposed Framework (LLM+Qwen)", "Greens"),
    ]

    last_image = None
    for ax, cm, title, cmap in cm_data:
        last_image = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(labels)
        ax.set_yticks(labels)
        ax.set_xticklabels(display_labels, fontsize=8, rotation=90)
        ax.set_yticklabels(display_labels, fontsize=8)

    fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Row-normalized rate")

    plot_filename = _build_output_filename("confusion_matrix_active_vs_proposed", config_name, run_tag, "png")
    output_path = os.path.join(DATA_DIR, plot_filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show_plots:
        plt.show(block=False)
        plt.pause(0.1)
    else:
        plt.close(fig)

    return output_path

def _compute_utility(f1, labeled_samples, iteration, max_iter, lambda_base=0.00005, alpha=2.0):
    lambda_t = lambda_base * (1.0 + alpha * (iteration / max_iter))
    return f1 - (lambda_t * labeled_samples), lambda_t

def _compute_stopping_iteration_ttest(results, window_size=4, p_value_threshold=0.05, max_samples=None):
    """
    基於 T 檢定的自適應提早停止機制
    """
    if not results:
        return None

    if max_samples and results[-1].get("labeled_samples", 0) >= max_samples:
        return results[-1]["iteration"]

    # 至少需要 2 個 window_size 的歷史資料才能進行檢定
    if len(results) >= window_size * 2:
        # 取出最近一組 (Current Window) 與前一組 (Previous Window) 的 F1 分數
        current_window = [res["f1"] for res in results[-window_size:]]
        previous_window = [res["f1"] for res in results[-(window_size * 2):-window_size]]

        # 執行單尾獨立樣本 T 檢定 (測試 current 是否「顯著大於」 previous)
        t_stat, p_value = stats.ttest_ind(current_window, previous_window, alternative='greater')
        
        # 邊界條件攔截：若變異數為 0，p_value 會回傳 NaN。這代表效能已呈現絕對的水平停滯 (Perfect Flatline)
        if np.isnan(p_value):
            print(f"\n[Stopping Auto-Triggered by T-Test]")
            print(f"  Reason: Zero variance detected. Performance has perfectly flatlined.")
            return results[-1]["iteration"]
        
        # 統計顯著性檢驗
        if p_value >= p_value_threshold:
            print(f"\n[Stopping Auto-Triggered by T-Test]")
            print(f"  Reason: No statistically significant improvement.")
            print(f"  P-value: {p_value:.4f} >= {p_value_threshold}")
            return results[-1]["iteration"]

    return None

def uncertainty_sampling(model, X_unlabeled, n_samples):
    probs = model.predict_proba(X_unlabeled)
    entropy_scores = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    return np.argsort(-entropy_scores)[:min(n_samples, X_unlabeled.shape[0])]


def plot_f1_curve_with_ttest(passive_results, active_results, proposed_results, ttest_stop_iter, output_dir, run_tag):
    """
    圖表 1：基於統計收斂的 F1 成長曲線與 T 檢定停止點
    學術論述：證明模型特徵空間已被充分探索，後續增益僅為隨機雜訊。
    """
    passive_df = pd.DataFrame(passive_results)
    active_df = pd.DataFrame(active_results)
    proposed_df = pd.DataFrame(proposed_results)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    if not passive_df.empty:
        ax.plot(passive_df["iteration"], passive_df["f1"], "o--", color="gray", label="Passive (Random)", linewidth=2, markersize=6, alpha=0.7)
    if not active_df.empty:
        ax.plot(active_df["iteration"], active_df["f1"], "s-", color="#4F81BD", label="Active (Entropy)", linewidth=2, markersize=6)
    if not proposed_df.empty:
        ax.plot(proposed_df["iteration"], proposed_df["f1"], "^-", color="#00A300", label="Proposed (Active + LLM)", linewidth=2, markersize=8)

    # 標示 T 檢定的統計收斂停止點
    if ttest_stop_iter and not proposed_df.empty and ttest_stop_iter <= proposed_df["iteration"].max():
        ax.axvline(x=ttest_stop_iter, color="red", linestyle="--", linewidth=2.5, label="Statistical Convergence (T-Test p>=0.05)")
        y_min, y_max = ax.get_ylim()
        ax.text(ttest_stop_iter + 0.3, y_max - (y_max - y_min) * 0.1, "Statistical Stop", color="red", fontsize=10, fontweight="bold")

    ax.set_xlabel("Iteration (Data Accumulation)", fontsize=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("Performance Trajectory and Statistical Convergence", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"f1_curve_ttest_{run_tag}.png")
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved T-Test F1 Curve: {output_path}")


def plot_utility_curve_with_roi(passive_results, active_results, proposed_results, output_dir, run_tag):
    """
    圖表 2：基於經濟收斂的預算效用曲線與最大 ROI 拐點
    學術論述：找出標註成本與效能增益的黃金交叉點 (Point of Diminishing Returns)。
    """
    passive_df = pd.DataFrame(passive_results)
    active_df = pd.DataFrame(active_results)
    proposed_df = pd.DataFrame(proposed_results)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    if not passive_df.empty:
        ax.plot(passive_df["iteration"], passive_df["utility"], "o--", color="gray", label="Passive", linewidth=2, alpha=0.7)
    if not active_df.empty:
        ax.plot(active_df["iteration"], active_df["utility"], "s-", color="#4F81BD", label="Active", linewidth=2)
    if not proposed_df.empty:
        ax.plot(proposed_df["iteration"], proposed_df["utility"], "^-", color="#00A300", label="Proposed", linewidth=2)

    # 尋找 Proposed 框架的最大效用值 (Max ROI)
    if not proposed_df.empty and "utility" in proposed_df.columns:
        max_utility_idx = proposed_df["utility"].idxmax()
        max_utility_iter = proposed_df.loc[max_utility_idx, "iteration"]
        max_utility_val = proposed_df.loc[max_utility_idx, "utility"]

        ax.axvline(x=max_utility_iter, color="blue", linestyle=":", linewidth=2.5, label="Max ROI Point (Peak Utility)")
        ax.plot(max_utility_iter, max_utility_val, "r*", markersize=15)

        y_min, y_max = ax.get_ylim()
        ax.text(max_utility_iter + 0.3, y_min + (y_max - y_min) * 0.1, "Max ROI Point", color="blue", fontsize=10, fontweight="bold")

    ax.set_xlabel("Iteration (Accumulated Cost)", fontsize=12)
    ax.set_ylabel("Utility Score (F1 - Budget Penalty)", fontsize=12)
    ax.set_title("Budget Utility and Point of Diminishing Returns", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"utility_roi_curve_{run_tag}.png")
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved Utility ROI Curve: {output_path}")

def plot_head_tail_comparison(passive_final, active_final, proposed_final, config_name="", run_tag="", show_plots=True):
    methods = ["Passive (Random)", "Active (Entropy)", "Proposed (LLM+Qwen)"]
    head_f1_scores = [passive_final.get("head_f1", np.nan), active_final.get("head_f1", np.nan), proposed_final.get("head_f1", np.nan)]
    tail_f1_scores = [passive_final.get("tail_f1", np.nan), active_final.get("tail_f1", np.nan), proposed_final.get("tail_f1", np.nan)]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    rects1 = ax.bar(x - width / 2, head_f1_scores, width, label="Head Classes", color="#4F81BD", edgecolor="black")
    rects2 = ax.bar(x + width / 2, tail_f1_scores, width, label="Tail Classes", color="#C0504D", edgecolor="black")

    ax.set_ylabel("Macro F1 Score", fontsize=12, fontweight="bold")
    ax.set_title("Performance Comparison: Head vs. Tail Classes", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    output_path = os.path.join(DATA_DIR, f"head_tail_f1_comparison_{run_tag}.png")
    plt.savefig(output_path)
    plt.close(fig)

def plot_confusion_matrix_comparison(y_true, active_pred, proposed_pred, config_name="", run_tag="", show_plots=True, label_mapping=None):
    y_true, active_pred, proposed_pred = np.asarray(y_true), np.asarray(active_pred), np.asarray(proposed_pred)
    n_classes = int(np.max(np.concatenate([y_true, active_pred, proposed_pred]))) + 1
    labels = np.arange(n_classes)

    display_labels = [label_mapping.get(int(l), f"Class {l}") for l in labels] if label_mapping else [str(l) for l in labels]

    def _normalize_confusion(true_labels, pred_labels):
        matrix = confusion_matrix(true_labels, pred_labels, labels=labels)
        row_sums = matrix.sum(axis=1, keepdims=True)
        return np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

    active_cm = _normalize_confusion(y_true, active_pred)
    proposed_cm = _normalize_confusion(y_true, proposed_pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150, constrained_layout=True)
    cm_data = [
        (axes[0], active_cm, "Active Learning (Warmup+Entropy)", "Blues"),
        (axes[1], proposed_cm, "Proposed Framework (LLM+Qwen)", "Greens"),
    ]

    for ax, cm, title, cmap in cm_data:
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(labels)
        ax.set_yticks(labels)
        ax.set_xticklabels(display_labels, fontsize=8, rotation=90)
        ax.set_yticklabels(display_labels, fontsize=8)

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Row-normalized rate")

    output_path = os.path.join(DATA_DIR, f"confusion_matrix_active_vs_proposed_{run_tag}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ============================================================================
# 執行框架
# ============================================================================
def run_proposed_framework(X_seed, y_seed, X_pool, y_pool, X_test, y_test, pool_texts, text_encoder, label_mapping, batch_size=40, n_iterations=40, random_seed=42):
    print(f"\n{'='*60}\nPROPOSED FRAMEWORK (Active + LLM + Qwen)\n{'='*60}")
    X_labeled, y_labeled = X_seed.copy(), y_seed.copy()
    
    results = []
    final_stop_iter = n_iterations

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        log_file = os.path.join(DATA_DIR, f"round_{iteration:02d}_augmentation_log.txt")
        model, metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
        
        utility, lambda_t = _compute_utility(metrics["f1"], X_labeled.shape[0], iteration, n_iterations)
        results.append({"iteration": iteration, "labeled_samples": X_labeled.shape[0], **metrics, "utility": utility, "lambda_t": lambda_t})
        
        print(
            f"Performance Summary | Macro F1: {metrics['f1']:.4f} | "
            f"Head Macro F1 ({HEAD_LABELS[0]}-{HEAD_LABELS[-1]}): {metrics['head_f1']:.4f} | "
            f"Tail Macro F1 ({TAIL_LABELS[0]}-{TAIL_LABELS[-1]}): {metrics['tail_f1']:.4f}"
        )

        if iteration > 5:
            stop_iter = _compute_stopping_iteration_ttest(results, window_size=4, p_value_threshold=0.05)
            if stop_iter:
                final_stop_iter = iteration
                break

        if iteration < n_iterations and X_pool.shape[0] > 0:
            selected_idx = random_sampling(X_pool.shape[0], batch_size, random_seed + iteration) if iteration <= 2 else uncertainty_sampling(model, X_pool, batch_size)
            
            sel_texts, sel_labels = pool_texts[selected_idx], y_pool[selected_idx]

            gen_texts, gen_labels, gen_src = generate_variants_llama3(sel_texts, sel_labels, label_mapping, output_txt_path=log_file)
            
            valid_idx = [i for i, (a, o) in enumerate(zip(gen_texts, gen_src)) if a.strip() and a != o]
            
            if valid_idx:
                val_texts, val_labels, ag_rate = validate_with_qwen25(gen_texts[valid_idx], gen_labels[valid_idx], label_mapping, gen_src[valid_idx], output_txt_path=log_file)
                print(f"Accepted: {len(val_texts)}/{len(valid_idx)} (Rate: {ag_rate:.2f})")
                
                if len(val_texts) > 0:
                    _MODEL_MANAGER.offload_all_to_cpu()
                    X_val = text_encoder.encode(val_texts.tolist(), batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
                    X_labeled = np.vstack([X_labeled, X_val])
                    y_labeled = np.concatenate([y_labeled, val_labels])

            X_labeled = np.vstack([X_labeled, X_pool[selected_idx]])
            y_labeled = np.concatenate([y_labeled, sel_labels])

            mask = np.ones(X_pool.shape[0], dtype=bool)
            mask[selected_idx] = False
            X_pool, y_pool, pool_texts = X_pool[mask], y_pool[mask], pool_texts[mask]

    _, final_metrics, final_pred = train_and_evaluate(X_labeled, y_labeled, X_test, y_test, return_predictions=True)
    return results, final_metrics, final_stop_iter, final_pred

def main():
    global DATA_DIR
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(DATA_DIR, run_tag)
    os.makedirs(run_output_dir, exist_ok=True)
    DATA_DIR = run_output_dir

    logger = setup_logging("Banking77_Refactored", run_output_dir)

    try:
        X_seed, y_seed, X_un, y_un, X_test, y_test, _, pool_texts, _, text_encoder = load_banking77_long_tail(return_texts=True)
        mapping = get_banking77_label_mapping()

        passive_results, passive_final, passive_stop, passive_pred = run_passive_learning_experiment(
            X_seed,
            y_seed,
            X_un,
            y_un,
            X_test,
            y_test,
            batch_size=40,
            n_iterations=40,
            random_seed=42,
        )

        active_results, active_final, active_stop, active_pred = run_active_baseline(
            X_seed,
            y_seed,
            X_un,
            y_un,
            X_test,
            y_test,
            batch_size=40,
            n_iterations=40,
            random_seed=42,
        )

        prop_res, prop_fin, prop_stop, proposed_pred = run_proposed_framework(
            X_seed,
            y_seed,
            X_un,
            y_un,
            X_test,
            y_test,
            pool_texts,
            text_encoder,
            mapping,
            batch_size=40,
            n_iterations=40,
        )

        # 繪製圖表 1: T 檢定統計收斂圖
        plot_f1_curve_with_ttest(
            passive_results=passive_results,
            active_results=active_results,
            proposed_results=prop_res,
            ttest_stop_iter=prop_stop,
            output_dir=DATA_DIR,
            run_tag=run_tag
        )

        # 繪製圖表 2: ROI 最大化效用圖
        plot_utility_curve_with_roi(
            passive_results=passive_results,
            active_results=active_results,
            proposed_results=prop_res,
            output_dir=DATA_DIR,
            run_tag=run_tag
        )

        # 繪製圖表 3: 長尾效應對比
        plot_head_tail_comparison(
            passive_final,
            active_final,
            prop_fin,
            config_name="Banking77_Refactored",
            run_tag=run_tag,
            show_plots=False,
        )

        # 繪製圖表 4: 混淆矩陣
        plot_confusion_matrix_comparison(
            y_test,
            active_pred,
            proposed_pred,
            config_name="Banking77_Refactored",
            run_tag=run_tag,
            show_plots=False,
            label_mapping=mapping,
        )

        summary_df = pd.DataFrame(
            [
                {"Framework": "Passive", "Macro_F1": passive_final["f1"], "Weighted_F1": passive_final["weighted_f1"], "Accuracy": passive_final["accuracy"], "Head_F1": passive_final["head_f1"], "Tail_F1": passive_final["tail_f1"]},
                {"Framework": "Active", "Macro_F1": active_final["f1"], "Weighted_F1": active_final["weighted_f1"], "Accuracy": active_final["accuracy"], "Head_F1": active_final["head_f1"], "Tail_F1": active_final["tail_f1"]},
                {"Framework": "Proposed", "Macro_F1": prop_fin["f1"], "Weighted_F1": prop_fin["weighted_f1"], "Accuracy": prop_fin["accuracy"], "Head_F1": prop_fin["head_f1"], "Tail_F1": prop_fin["tail_f1"]},
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
        print(f"Final Macro F1: {prop_fin['f1']:.4f}")
        print(f"Final Head Macro F1 ({HEAD_LABELS[0]}-{HEAD_LABELS[-1]}): {prop_fin['head_f1']:.4f}")
        print(f"Final Tail Macro F1 ({TAIL_LABELS[0]}-{TAIL_LABELS[-1]}): {prop_fin['tail_f1']:.4f}")
        
    finally:
        _MODEL_MANAGER.offload_all_to_cpu()
        gc.collect()
        if logger: logger.close(); sys.stdout = logger.terminal

if __name__ == "__main__":
    main()