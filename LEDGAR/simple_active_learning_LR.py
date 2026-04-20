#!/usr/bin/env python3
"""
LEDGAR 多類別文本分類的主動學習比較腳本

重點：
1) Passive Learning: 每輪隨機抽樣
2) Active Learning: 每輪以 Entropy 不確定性抽樣
3) 保留多次實驗統計檢定與比較繪圖架構
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
from scipy import sparse, stats
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings_filter = True
if warnings_filter:
    import warnings

    warnings.filterwarnings("ignore")


# 純本地模型設定。
LLM_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
VALIDATOR_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

_LOCAL_LLM_TOKENIZER = None
_LOCAL_LLM_MODEL = None
_LOCAL_VALIDATOR_TOKENIZER = None
_LOCAL_VALIDATOR_MODEL = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(DATA_DIR, "logs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def setup_logging(config_name):
    """將輸出同時寫入終端與 log 檔。"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIR, f"experiment_log_{config_name}_{timestamp}.txt")

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

    print(f"Logging started: {log_filename}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return logger


def _build_output_filename(prefix, config_name, run_tag, extension):
    """組合不會互相覆蓋的輸出檔名。"""
    parts = [prefix]
    if config_name:
        parts.append(config_name)
    if run_tag:
        parts.append(run_tag)
    return f"{'_'.join(parts)}.{extension}"


def _to_text_array(values):
    """將輸入安全轉成純字串陣列，避免 TF-IDF 在字串陣列處理時出錯。"""
    return np.asarray(["" if value is None else str(value) for value in values])


def _load_local_generator():
    """Lazy-load 本地 generator，只在需要時載入。"""
    global _LOCAL_LLM_TOKENIZER, _LOCAL_LLM_MODEL

    if _LOCAL_LLM_TOKENIZER is None or _LOCAL_LLM_MODEL is None:
        if not torch.cuda.is_available():
            raise RuntimeError("需要 CUDA GPU 才能載入 4-bit 量化模型。")

        # 載入前先清空顯存，確保拿到最乾淨的記憶體。
        torch.cuda.empty_cache()

        # 使用 bitsandbytes 4-bit 量化，降低顯存占用並避免 CPU offload。
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        print(f"Loading local generator model: {LLM_MODEL_ID}")
        _LOCAL_LLM_TOKENIZER = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        _LOCAL_LLM_MODEL = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            quantization_config=quantization_config,
            device_map="cuda:0",
        )





def _load_local_validator():
    """Lazy-load 本地 validator，只在需要時載入。"""
    global _LOCAL_VALIDATOR_TOKENIZER, _LOCAL_VALIDATOR_MODEL

    if _LOCAL_VALIDATOR_TOKENIZER is None or _LOCAL_VALIDATOR_MODEL is None:
        if not torch.cuda.is_available():
            raise RuntimeError("需要 CUDA GPU 才能載入 4-bit 量化模型。")

        # 載入前先清空顯存，確保拿到最乾淨的記憶體。
        torch.cuda.empty_cache()

        # 使用 bitsandbytes 4-bit 量化，強制模型固定在 RTX 4090（cuda:0）。
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        print(f"Loading local validator model: {VALIDATOR_MODEL_ID}")
        _LOCAL_VALIDATOR_TOKENIZER = AutoTokenizer.from_pretrained(VALIDATOR_MODEL_ID)
        _LOCAL_VALIDATOR_MODEL = AutoModelForCausalLM.from_pretrained(
            VALIDATOR_MODEL_ID,
            quantization_config=quantization_config,
            device_map="cuda:0",
        )





def _generate_chat_response(tokenizer, model, messages, max_new_tokens=256, do_sample=True):
    """通用 chat generate helper，回傳新生成的純文字。"""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.7 if do_sample else None,
            top_p=0.95 if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = int(inputs["input_ids"].shape[1])
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def get_ledgar_label_mapping():
    """
    取得 LEDGAR 資料集的標籤對應。
    基於官方 LEDGAR 資料集的 11 個法律條款分類。
    """
    label_mapping = {
        0: "Parties",
        1: "Agreement Date",
        2: "Effective Date",
        3: "Expiration Date",
        4: "Termination Clause",
        5: "Definition",
        6: "Term",
        7: "Liability",
        8: "Binding Effect",
        9: "Counterparts",
        10: "Confidentiality",
        11: "Indemnification",
    }
    return label_mapping


def load_ledgar_tfidf(random_seed=42, return_texts=False):
    """
    載入 LEDGAR 並轉成 TF-IDF 稀疏矩陣。

    切分策略（先打亂再切）：
    - Test: 1000
    - Unlabeled Pool: 5000
    - Seed: 300
    """

    np.random.seed(random_seed)

    # 依需求優先嘗試這個資料集 ID；若 Hub 端不存在則 fallback 到可用來源。
    try:
        print("Loading dataset: lexlual/LEDGAR (train split)")
        dataset = load_dataset("lexlual/LEDGAR", split="train")
    except Exception as exc:
        print(f"Primary source failed: {exc}")
        print("Fallback to: lex_glue (ledgar)")
        dataset = load_dataset("lex_glue", "ledgar", split="train")

    df = dataset.to_pandas()[["text", "label"]].copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    # 打亂並重設索引，確保切分可重現。
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    n_test = 1000
    n_unlabeled = 5000
    n_seed = 300
    required = n_test + n_unlabeled + n_seed
    if len(df) < required:
        raise ValueError(f"Dataset too small, need {required}, got {len(df)}")

    df_test = df.iloc[:n_test].copy()
    df_unlabeled = df.iloc[n_test : n_test + n_unlabeled].copy()
    df_seed = df.iloc[n_test + n_unlabeled : required].copy()

    # 使用同一個向量器在整個子集合上建立詞彙空間，避免後續 sparse 維度不一致。
    combined_text = pd.concat(
        [df_seed["text"], df_unlabeled["text"], df_test["text"]],
        ignore_index=True,
    )
    vectorizer = TfidfVectorizer(max_features=10000)
    X_all = vectorizer.fit_transform(combined_text)

    seed_end = len(df_seed)
    unlabeled_end = seed_end + len(df_unlabeled)

    X_seed = X_all[:seed_end]
    X_unlabeled = X_all[seed_end:unlabeled_end]
    X_test = X_all[unlabeled_end:]

    y_seed = df_seed["label"].to_numpy()
    y_unlabeled = df_unlabeled["label"].to_numpy()
    y_test = df_test["label"].to_numpy()

    seed_texts = df_seed["text"].astype(str).to_numpy()
    unlabeled_texts = df_unlabeled["text"].astype(str).to_numpy()
    test_texts = df_test["text"].astype(str).to_numpy()

    print("Split summary:")
    print(f"  Seed: {X_seed.shape[0]}")
    print(f"  Unlabeled: {X_unlabeled.shape[0]}")
    print(f"  Test: {X_test.shape[0]}")
    print(f"  Classes in seed/test: {len(np.unique(np.concatenate([y_seed, y_test])))}")

    if return_texts:
        return (
            X_seed,
            y_seed,
            X_unlabeled,
            y_unlabeled,
            X_test,
            y_test,
            seed_texts,
            unlabeled_texts,
            test_texts,
            vectorizer,
        )

    return X_seed, y_seed, X_unlabeled, y_unlabeled, X_test, y_test


def generate_variants_llama3(texts, labels, n_variants=3):
    """使用本地 Llama 3 產生擴增資料。模型常駐 VRAM，無需釋放。"""
    texts = _to_text_array(texts)
    labels = np.asarray(labels)

    _load_local_generator()

    augmented_texts = []
    augmented_labels = []

    for text, label in zip(texts, labels):
        messages = [
            {
                "role": "system",
                "content": "You are a legal text augmentation assistant.",
            },
            {
                "role": "user",
                "content": (
                    "Generate legal text variations while preserving label semantics.\n"
                    f"Please output exactly {n_variants} variations separated by the delimiter '|||'. "
                    "Do NOT include any titles, numbers, or introductory text like 'Here are the variations'. "
                    f"Original text:\n{text}"
                ),
            },
        ]

        response_text = _generate_chat_response(
            _LOCAL_LLM_TOKENIZER,
            _LOCAL_LLM_MODEL,
            messages,
            max_new_tokens=384,
            do_sample=True,
        )

        # 解析 LLM 輸出
        raw_variations = response_text.split("|||") 
        variations = [v.strip() for v in raw_variations if v.strip()]

        # 防呆機制：如果模型沒有用 ||| 分隔，或是切出來的數量不對
        if len(variations) == 0:
            variations = [text]  # 至少保留原文

        for var in variations[:n_variants]:
            augmented_texts.append(var)
            augmented_labels.append(label)

    return np.asarray(augmented_texts, dtype=str), np.asarray(augmented_labels)


def _parse_validator_response(text):
    """解析 XML 格式決策：從 <decision>YES</decision> 或 <decision>NO</decision> 中提取決策。"""
    text_upper = str(text).upper()
    if "<DECISION>YES</DECISION>" in text_upper:
        return True
    elif "<DECISION>NO</DECISION>" in text_upper:
        return False
    # Fallback: 相容舊格式
    return "YES" in text_upper


def validate_with_qwen25(generated_texts, generated_labels, label_mapping=None):
    """使用本地 Qwen2.5 驗證擴增資料並回傳 agreement rate。模型常駐 VRAM，無需釋放。"""
    generated_texts = _to_text_array(generated_texts)
    generated_labels = np.asarray(generated_labels)

    if len(generated_texts) == 0:
        return np.asarray([], dtype=str), np.asarray([], dtype=generated_labels.dtype), 0.0

    _load_local_validator()

    valid_texts = []
    valid_labels = []
    accepted = 0

    for text, label in zip(generated_texts, generated_labels):
        # 取得標籤名稱
        label_name = label_mapping.get(int(label), f"Label {label}") if label_mapping else f"Label {label}"
        
        messages = [
            {
                "role": "system",
                "content": "You are a strict expert legal counsel. Your job is to filter out bad data augmentation.",
            },
            {
                "role": "user",
                "content": (
                    f"Evaluate if the following augmented text accurately preserves the legal intent of a '{label_name}' clause. "
                    "It must also be a professional legal contract clause without conversational fillers.\n\n"
                    "First, provide a one-sentence reasoning.\n"
                    "Then, output your final decision as exactly <decision>YES</decision> or <decision>NO</decision>.\n\n"
                    f"Augmented Text: {text}"
                ),
            },
        ]
        response_text = _generate_chat_response(
            _LOCAL_VALIDATOR_TOKENIZER,
            _LOCAL_VALIDATOR_MODEL,
            messages,
            max_new_tokens=256,
            do_sample=False,
        )

        if _parse_validator_response(response_text):
            valid_texts.append(text)
            valid_labels.append(label)
            accepted += 1

    agreement_rate = accepted / max(len(generated_texts), 1)
    return np.asarray(valid_texts, dtype=str), np.asarray(valid_labels), agreement_rate


def random_sampling(pool_size, n_samples, random_seed=42):
    """被動學習：從未標註池中隨機取樣索引。"""
    take_n = min(n_samples, pool_size)
    rng = np.random.RandomState(random_seed)
    return rng.choice(pool_size, size=take_n, replace=False)


def uncertainty_sampling(model, X_unlabeled, n_samples):
    """主動學習：以 Shannon Entropy 選取最高不確定性樣本。"""
    take_n = min(n_samples, X_unlabeled.shape[0])
    probs = model.predict_proba(X_unlabeled)
    entropy_scores = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    ranked_idx = np.argsort(-entropy_scores)
    return ranked_idx[:take_n]


def train_and_evaluate(X_train, y_train, X_eval, y_eval):
    """訓練多類別 LR 並回傳 weighted metrics。"""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)

    metrics = {
        "accuracy": accuracy_score(y_eval, y_pred),
        "precision": precision_score(y_eval, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_eval, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_eval, y_pred, average="weighted", zero_division=0),
    }
    return model, metrics


def _append_selected(X_labeled, y_labeled, X_pool, y_pool, selected_idx):
    """將選到的樣本從 pool 移到 labeled，並回傳更新後矩陣與標籤。"""
    selected_idx = np.asarray(selected_idx, dtype=int)

    X_new = X_pool[selected_idx]
    y_new = y_pool[selected_idx]

    # 稀疏矩陣請使用 vstack，避免型別與維度錯誤。
    X_labeled_next = sparse.vstack([X_labeled, X_new], format="csr")
    y_labeled_next = np.concatenate([y_labeled, y_new])

    keep_mask = np.ones(X_pool.shape[0], dtype=bool)
    keep_mask[selected_idx] = False

    X_pool_next = X_pool[keep_mask]
    y_pool_next = y_pool[keep_mask]

    return X_labeled_next, y_labeled_next, X_pool_next, y_pool_next


def _get_sampling_indices(model, X_pool, n_samples, iteration, warmup_iters=2, random_seed=42):
    """根據 iteration 決定使用 warmup random 或 entropy active sampling。"""
    if iteration <= warmup_iters:
        return random_sampling(X_pool.shape[0], n_samples, random_seed + iteration)
    return uncertainty_sampling(model, X_pool, n_samples)


def _compute_utility(f1_score_value, labeled_samples, lambda_penalty=0.0003):
    """依論文定義計算效用值。"""
    return f1_score_value - (lambda_penalty * labeled_samples)


def _compute_stopping_iteration(results, patience=3, epsilon=0.003, max_samples=None):
    """
    根據 F1 marginal gain 與 budget constraint 決定自動停止迭代點。
    
    Hard Constraint: 若達到 max_samples 預算上限，立即停止。
    Soft Signal: 若 ΔF1 < epsilon 連續 patience 次迭代，建議停止。
    
    Args:
        results: Iteration 結果列表，每項含 "f1" 與 "labeled_samples"
        patience: 多少次低 ΔF1 后建議停止
        epsilon: F1 marginal gain 閾值
        max_samples: 預算上限（None 表示無限制）
    
    Returns:
        建議停止的 iteration number，或 None（允許繼續）
    """
    if not results:
        return None
    
    # Hard Constraint: Check max_samples budget on last item
    if max_samples is not None and results[-1].get("labeled_samples", 0) >= max_samples:
        print(f"[Stopping] Reached max_samples budget: {results[-1]['labeled_samples']} >= {max_samples}")
        return results[-1]["iteration"]
    
    # Soft Signal: Scan from index 1 for consecutive low ΔF1 gains (skip warmup at index 0)
    if len(results) >= patience + 1:  # Need at least patience+1 iterations to check patience streaks
        consecutive_low_gain = 0
        for i in range(1, len(results)):
            prev_f1 = results[i - 1]["f1"]
            curr_f1 = results[i]["f1"]
            delta_f1 = curr_f1 - prev_f1
            
            if delta_f1 < epsilon:
                consecutive_low_gain += 1
                if consecutive_low_gain >= patience:
                    print(f"[Stopping] Low ΔF1 for {patience} consecutive iterations (threshold: {epsilon})")
                    return results[i]["iteration"]
            else:
                consecutive_low_gain = 0  # Reset streak
    
    # No stopping condition met
    return None


def run_active_learning_experiment(
    X_seed,
    y_seed,
    X_unlabeled,
    y_unlabeled,
    X_test,
    y_test,
    initial_samples=300,
    batch_size=40,
    n_iterations=25,
    random_seed=42,
):
    """固定使用 Entropy 的 Active Learning 實驗。"""
    print(f"\n{'=' * 60}")
    print("ACTIVE LEARNING EXPERIMENT (Entropy)")
    print(f"{'=' * 60}")

    # LEDGAR 已預先切好 seed/pool/test，這裡直接使用 seed 作初始標註集合。
    if X_seed.shape[0] < initial_samples:
        raise ValueError("initial_samples is larger than seed size")

    X_labeled = X_seed[:initial_samples]
    y_labeled = y_seed[:initial_samples]

    # 這裡建立可變 pool，避免更動到外部傳入資料。
    X_pool = X_unlabeled.copy()
    y_pool = y_unlabeled.copy()

    print(f"Initial labeled pool: {X_labeled.shape[0]} samples")
    print(f"Remaining unlabeled: {X_pool.shape[0]} samples")

    results = []

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")

        model, metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)

        print(f"Test - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        results.append(
            {
                "iteration": iteration,
                "labeled_samples": X_labeled.shape[0],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )

        if iteration < n_iterations and X_pool.shape[0] > 0:
            selected_idx = uncertainty_sampling(model, X_pool, batch_size)
            print(f"Selected {len(selected_idx)} samples using entropy uncertainty sampling")

            X_labeled, y_labeled, X_pool, y_pool = _append_selected(
                X_labeled,
                y_labeled,
                X_pool,
                y_pool,
                selected_idx,
            )

            print(f"Total labeled samples: {X_labeled.shape[0]}")
            print(f"Remaining unlabeled: {X_pool.shape[0]}")

    _, final_metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
    print(f"\nFinal Test - F1: {final_metrics['f1']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}")

    return results, final_metrics


def run_active_baseline(
    X_seed,
    y_seed,
    X_unlabeled,
    y_unlabeled,
    X_test,
    y_test,
    initial_samples=300,
    batch_size=40,
    n_iterations=25,
    random_seed=42,
    lambda_penalty=0.0005,
    max_samples=None,
):
    """Active baseline：前兩輪 warmup random，之後切換 entropy uncertainty sampling。"""
    print(f"\n{'=' * 60}")
    print("ACTIVE LEARNING BASELINE (Warmup + Entropy)")
    print(f"{'=' * 60}")

    if X_seed.shape[0] < initial_samples:
        raise ValueError("initial_samples is larger than seed size")

    X_labeled = X_seed[:initial_samples]
    y_labeled = y_seed[:initial_samples]
    X_pool = X_unlabeled.copy()
    y_pool = y_unlabeled.copy()

    print(f"Initial labeled pool: {X_labeled.shape[0]} samples")
    print(f"Remaining unlabeled: {X_pool.shape[0]} samples")

    results = []
    final_stop_iter = n_iterations

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")

        model, metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
        print(f"Test - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        utility = _compute_utility(metrics["f1"], X_labeled.shape[0], lambda_penalty)

        results.append(
            {
                "iteration": iteration,
                "labeled_samples": X_labeled.shape[0],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "utility": utility,
            }
        )

        # Immediate stopping check (skip warmup phase)
        if iteration > 5:
            suggested_stop = _compute_stopping_iteration(
                results,
                patience=3,
                epsilon=0.003,
                max_samples=max_samples,
            )
            if suggested_stop is not None:
                final_stop_iter = iteration
                break

        if iteration < n_iterations and X_pool.shape[0] > 0:
            selected_idx = _get_sampling_indices(
                model,
                X_pool,
                batch_size,
                iteration,
                warmup_iters=2,
                random_seed=random_seed,
            )
            print(
                f"Selected {len(selected_idx)} samples using "
                f"{'random' if iteration <= 2 else 'entropy'} sampling"
            )

            X_labeled, y_labeled, X_pool, y_pool = _append_selected(
                X_labeled,
                y_labeled,
                X_pool,
                y_pool,
                selected_idx,
            )

            print(f"Total labeled samples: {X_labeled.shape[0]}")
            print(f"Remaining unlabeled: {X_pool.shape[0]}")

    _, final_metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
    print(f"\nFinal Test - F1: {final_metrics['f1']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}")

    return results, final_metrics, final_stop_iter


def run_passive_baseline(
    X_seed,
    y_seed,
    X_unlabeled,
    y_unlabeled,
    X_test,
    y_test,
    initial_samples=300,
    batch_size=40,
    n_iterations=25,
    random_seed=42,
    lambda_penalty=0.0005,
    max_samples=None,
):
    """Passive baseline：固定 100% random sampling。"""
    print(f"\n{'=' * 60}")
    print("PASSIVE LEARNING BASELINE (Random)")
    print(f"{'=' * 60}")

    if X_seed.shape[0] < initial_samples:
        raise ValueError("initial_samples is larger than seed size")

    X_labeled = X_seed[:initial_samples]
    y_labeled = y_seed[:initial_samples]
    X_pool = X_unlabeled.copy()
    y_pool = y_unlabeled.copy()

    print(f"Initial labeled pool: {X_labeled.shape[0]} samples")
    print(f"Remaining unlabeled: {X_pool.shape[0]} samples")

    results = []
    final_stop_iter = n_iterations

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")

        _, metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
        print(f"Test - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        utility = _compute_utility(metrics["f1"], X_labeled.shape[0], lambda_penalty)

        results.append(
            {
                "iteration": iteration,
                "labeled_samples": X_labeled.shape[0],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "utility": utility,
            }
        )

        # Immediate stopping check (skip warmup phase)
        if iteration > 5:
            suggested_stop = _compute_stopping_iteration(
                results,
                patience=3,
                epsilon=0.003,
                max_samples=max_samples,
            )
            if suggested_stop is not None:
                final_stop_iter = iteration
                break

        if iteration < n_iterations and X_pool.shape[0] > 0:
            selected_idx = random_sampling(X_pool.shape[0], batch_size, random_seed + iteration)
            print(f"Selected {len(selected_idx)} samples using random sampling")

            X_labeled, y_labeled, X_pool, y_pool = _append_selected(
                X_labeled,
                y_labeled,
                X_pool,
                y_pool,
                selected_idx,
            )

            print(f"Total labeled samples: {X_labeled.shape[0]}")
            print(f"Remaining unlabeled: {X_pool.shape[0]}")

    _, final_metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
    print(f"\nFinal Test - F1: {final_metrics['f1']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}")

    return results, final_metrics, final_stop_iter


def run_proposed_framework(
    X_seed,
    y_seed,
    X_unlabeled,
    y_unlabeled,
    X_test,
    y_test,
    seed_texts,
    unlabeled_texts,
    vectorizer,
    initial_samples=300,
    batch_size=40,
    n_iterations=25,
    random_seed=42,
    lambda_penalty=0.0005,
    max_samples=None,
    label_mapping=None,
):
    """Proposed Framework：Warmup + Entropy + LLM 擴增 + Qwen 驗證。"""
    print(f"\n{'=' * 60}")
    print("PROPOSED FRAMEWORK (Warmup + Entropy + LLM + Qwen)")
    print(f"{'=' * 60}")

    if X_seed.shape[0] < initial_samples:
        raise ValueError("initial_samples is larger than seed size")

    X_labeled = X_seed[:initial_samples]
    y_labeled = y_seed[:initial_samples]
    labeled_texts = _to_text_array(seed_texts[:initial_samples])

    X_pool = X_unlabeled.copy()
    y_pool = y_unlabeled.copy()
    pool_texts = _to_text_array(unlabeled_texts)

    print(f"Initial labeled pool: {X_labeled.shape[0]} samples")
    print(f"Remaining unlabeled: {X_pool.shape[0]} samples")

    results = []
    final_stop_iter = n_iterations

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")

        model, metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
        print(f"Test - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        utility = _compute_utility(metrics["f1"], X_labeled.shape[0], lambda_penalty)

        results.append(
            {
                "iteration": iteration,
                "labeled_samples": X_labeled.shape[0],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "utility": utility,
            }
        )

        # Immediate stopping check (skip warmup phase)
        if iteration > 5:
            suggested_stop = _compute_stopping_iteration(
                results,
                patience=3,
                epsilon=0.003,
                max_samples=max_samples,
            )
            if suggested_stop is not None:
                final_stop_iter = iteration
                break

        if iteration < n_iterations and X_pool.shape[0] > 0:
            selected_idx = _get_sampling_indices(
                model,
                X_pool,
                batch_size,
                iteration,
                warmup_iters=2,
                random_seed=random_seed,
            )
            print(
                f"Selected {len(selected_idx)} samples using "
                f"{'random' if iteration <= 2 else 'entropy'} sampling"
            )

            selected_texts = pool_texts[selected_idx]
            selected_labels = y_pool[selected_idx]

            generated_texts, generated_labels = generate_variants_llama3(
                selected_texts,
                selected_labels,
                n_variants=3,
            )

            # 抽樣信任機制：每輪最多抽 40 筆
            validator_sample_size = min(
                40,
                max(1, int(np.floor(len(generated_texts) * 0.2))),
            )
            validator_rng = np.random.RandomState(random_seed + iteration)
            validator_indices = validator_rng.choice(
                len(generated_texts),
                size=validator_sample_size,
                replace=False,
            )

            validated_texts, validated_labels, agreement_rate = validate_with_qwen25(
                generated_texts[validator_indices],
                generated_labels[validator_indices],
                label_mapping=label_mapping,
            )

            print(f"Agreement Rate: {agreement_rate:.4f}")
            print(f"Validator samples: {validator_sample_size}/{len(generated_texts)}")

            if agreement_rate >= 0.75:
                print("品質達標，全數信任")
                trusted_texts = generated_texts
                trusted_labels = generated_labels
            else:
                print("品質不佳，僅保留合格樣本")
                trusted_texts = validated_texts
                trusted_labels = validated_labels

            X_selected = X_pool[selected_idx]
            X_validated = (
                vectorizer.transform(_to_text_array(trusted_texts))
                if len(trusted_texts) > 0
                else sparse.csr_matrix((0, X_pool.shape[1]))
            )

            X_labeled = sparse.vstack([X_labeled, X_selected, X_validated], format="csr")
            y_labeled = np.concatenate([y_labeled, selected_labels, trusted_labels])
            labeled_texts = np.concatenate([labeled_texts, selected_texts, trusted_texts])

            keep_mask = np.ones(X_pool.shape[0], dtype=bool)
            keep_mask[selected_idx] = False
            X_pool = X_pool[keep_mask]
            y_pool = y_pool[keep_mask]
            pool_texts = pool_texts[keep_mask]

            print(f"Total labeled samples: {X_labeled.shape[0]}")
            print(f"Remaining unlabeled: {X_pool.shape[0]}")

    _, final_metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
    print(f"\nFinal Test - F1: {final_metrics['f1']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}")

    return results, final_metrics, final_stop_iter


def run_passive_learning_experiment(
    X_seed,
    y_seed,
    X_unlabeled,
    y_unlabeled,
    X_test,
    y_test,
    initial_samples=300,
    batch_size=40,
    n_iterations=25,
    random_seed=42,
):
    """固定使用隨機抽樣的 Passive Learning 實驗。"""
    print(f"\n{'=' * 60}")
    print("PASSIVE LEARNING EXPERIMENT (Random)")
    print(f"{'=' * 60}")

    if X_seed.shape[0] < initial_samples:
        raise ValueError("initial_samples is larger than seed size")

    X_labeled = X_seed[:initial_samples]
    y_labeled = y_seed[:initial_samples]

    X_pool = X_unlabeled.copy()
    y_pool = y_unlabeled.copy()

    print(f"Initial labeled pool: {X_labeled.shape[0]} samples")
    print(f"Remaining unlabeled: {X_pool.shape[0]} samples")

    results = []

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")

        _, metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)

        print(f"Test - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        results.append(
            {
                "iteration": iteration,
                "labeled_samples": X_labeled.shape[0],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )

        if iteration < n_iterations and X_pool.shape[0] > 0:
            selected_idx = random_sampling(X_pool.shape[0], batch_size, random_seed + iteration)
            print(f"Selected {len(selected_idx)} samples using random sampling")

            X_labeled, y_labeled, X_pool, y_pool = _append_selected(
                X_labeled,
                y_labeled,
                X_pool,
                y_pool,
                selected_idx,
            )

            print(f"Total labeled samples: {X_labeled.shape[0]}")
            print(f"Remaining unlabeled: {X_pool.shape[0]}")

    _, final_metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
    print(f"\nFinal Test - F1: {final_metrics['f1']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}")

    return results, final_metrics


def plot_comparison(passive_results, active_results, proposed_results=None, config_name="", run_tag="", show_plots=True):
    """比較 Passive、Active、Proposed 三條學習曲線。"""
    passive_df = pd.DataFrame(passive_results)
    active_df = pd.DataFrame(active_results)
    proposed_df = pd.DataFrame(proposed_results) if proposed_results is not None else None

    plt.figure(figsize=(9, 6))

    plt.plot(passive_df["iteration"], passive_df["f1"], "o--", label="Passive (100% Random)", linewidth=2, markersize=7)
    plt.plot(active_df["iteration"], active_df["f1"], "s-", label="Active (Warmup + Entropy)", linewidth=2, markersize=7)
    if proposed_df is not None and len(proposed_df) > 0:
        plt.plot(
            proposed_df["iteration"],
            proposed_df["f1"],
            "^-",
            label="Proposed (Warmup + Entropy + LLM + Llama Validator)",
            linewidth=2,
            markersize=7,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Weighted F1")
    plt.title("Passive vs Active vs Proposed")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_filename = _build_output_filename("comparison_passive_active_proposed", config_name, run_tag, "png")
    output_path = os.path.join(DATA_DIR, plot_filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show_plots:
        plt.show(block=False)
        plt.pause(0.1)
    else:
        plt.close()


def plot_utility_curve(passive_results, active_results, proposed_results, stopping_iters, config_name=""):
    """繪製三條 Utility 曲線，並標示 Proposed 的自動停止點。"""
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
            label="Auto-stopping Point (Patience=2)",
        )
        y_min, y_max = plt.ylim()
        plt.text(
            proposed_stopping_iteration + 0.1,
            y_max - (y_max - y_min) * 0.08,
            "Auto-stopping Point (Patience=2)",
            color="red",
            fontsize=9,
            rotation=90,
            va="top",
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


def run_multiple_experiments(
    X_seed,
    y_seed,
    X_unlabeled,
    y_unlabeled,
    X_test,
    y_test,
    config,
    n_runs=3,
):
    """保留原架構：多次重複實驗以進行統計檢定。"""
    print(f"\n{'=' * 80}")
    print(f"RUNNING {n_runs} EXPERIMENTS FOR STATISTICAL SIGNIFICANCE")
    print(f"{'=' * 80}")

    all_active_results = []
    all_passive_results = []
    all_active_finals = []
    all_passive_finals = []

    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")
        random_seed = 42 + run

        active_results, active_final = run_active_learning_experiment(
            X_seed,
            y_seed,
            X_unlabeled,
            y_unlabeled,
            X_test,
            y_test,
            initial_samples=config["initial_samples"],
            batch_size=config["batch_size"],
            n_iterations=config["n_iterations"],
            random_seed=random_seed,
        )

        passive_results, passive_final = run_passive_learning_experiment(
            X_seed,
            y_seed,
            X_unlabeled,
            y_unlabeled,
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

        print(
            f"Run {run + 1} completed - "
            f"Active F1: {active_final['f1']:.4f}, "
            f"Passive F1: {passive_final['f1']:.4f}"
        )

    return all_active_results, all_passive_results, all_active_finals, all_passive_finals


def perform_statistical_tests(active_finals, passive_finals):
    """針對多次 run 的最終 F1 進行統計檢定。"""
    print(f"\n{'=' * 80}")
    print("STATISTICAL SIGNIFICANCE TESTING")
    print(f"{'=' * 80}")

    active_f1_scores = np.array([result["f1"] for result in active_finals])
    passive_f1_scores = np.array([result["f1"] for result in passive_finals])

    active_mean = np.mean(active_f1_scores)
    active_std = np.std(active_f1_scores)
    passive_mean = np.mean(passive_f1_scores)
    passive_std = np.std(passive_f1_scores)

    print(f"Active Learning F1: {active_mean:.4f} +- {active_std:.4f}")
    print(f"Passive Learning F1: {passive_mean:.4f} +- {passive_std:.4f}")
    print(f"Difference: {active_mean - passive_mean:.4f}")

    t_stat, p_value = stats.ttest_rel(active_f1_scores, passive_f1_scores)

    try:
        w_stat, w_p_value = stats.wilcoxon(active_f1_scores, passive_f1_scores)
    except Exception:
        w_stat, w_p_value = np.nan, np.nan

    pooled_std = np.sqrt((active_std**2 + passive_std**2) / 2) if (active_std > 0 or passive_std > 0) else np.nan
    cohens_d = (active_mean - passive_mean) / pooled_std if pooled_std and not np.isnan(pooled_std) else np.nan

    print("\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    print("\nWilcoxon signed-rank test:")
    print(f"  W-statistic: {w_stat}")
    print(f"  p-value: {w_p_value}")

    print(f"\nEffect size (Cohen's d): {cohens_d}")

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
    """保留統計比較圖架構（分佈、箱型、逐 run 對比）。"""
    active_f1_scores = [result["f1"] for result in active_finals]
    passive_f1_scores = [result["f1"] for result in passive_finals]

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.hist(active_f1_scores, alpha=0.7, label="Active", bins=8, color="blue")
    plt.hist(passive_f1_scores, alpha=0.7, label="Passive", bins=8, color="red")
    plt.xlabel("Weighted F1")
    plt.ylabel("Frequency")
    plt.title("Distribution of F1 Scores")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    plt.boxplot([active_f1_scores, passive_f1_scores], labels=["Active", "Passive"])
    plt.ylabel("Weighted F1")
    plt.title("Box Plot Comparison")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    runs = range(1, len(active_f1_scores) + 1)
    plt.plot(runs, active_f1_scores, "s-", label="Active", linewidth=2)
    plt.plot(runs, passive_f1_scores, "o--", label="Passive", linewidth=2)
    plt.xlabel("Run")
    plt.ylabel("Weighted F1")
    plt.title("F1 Score by Run")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 4)
    diffs = np.array(active_f1_scores) - np.array(passive_f1_scores)
    plt.hist(diffs, alpha=0.75, bins=8, color="green")
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("F1 Difference (Active - Passive)")
    plt.ylabel("Frequency")
    plt.title("Difference Distribution")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    active_mean, active_std = np.mean(active_f1_scores), np.std(active_f1_scores)
    passive_mean, passive_std = np.mean(passive_f1_scores), np.std(passive_f1_scores)
    plt.errorbar(["Active", "Passive"], [active_mean, passive_mean], yerr=[active_std, passive_std], fmt="o", capsize=5)
    plt.ylabel("Weighted F1")
    plt.title("Means with Std")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    pooled_std = np.sqrt((active_std**2 + passive_std**2) / 2) if (active_std > 0 or passive_std > 0) else np.nan
    cohens_d = (active_mean - passive_mean) / pooled_std if pooled_std and not np.isnan(pooled_std) else 0
    plt.bar(["Effect Size"], [cohens_d], color="orange", alpha=0.8)
    plt.axhline(y=0, color="black", linestyle="-")
    plt.ylabel("Cohen's d")
    plt.title(f"Effect Size: {cohens_d:.3f}")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_filename = f"statistical_comparison_{config_name}.png" if config_name else "statistical_comparison.png"
    output_path = os.path.join(DATA_DIR, plot_filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show_plots:
        plt.show(block=False)
        plt.pause(0.1)
    else:
        plt.close()


def main():
    """主程式：執行 Passive、Active、Proposed 三條框架比較。"""

    show_plots = False
    config_name = "ledgar_warmup_llm_framework"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 依需求更新設定
    experiment_config = {
        "initial_samples": 300,
        "batch_size": 40,
        "n_iterations": 40,
    }

    logger = setup_logging(config_name)

    try:
        (
            X_seed,
            y_seed,
            X_unlabeled,
            y_unlabeled,
            X_test,
            y_test,
            seed_texts,
            unlabeled_texts,
            _test_texts,
            vectorizer,
        ) = load_ledgar_tfidf(random_seed=42, return_texts=True)

        # 取得 LEDGAR 標籤映射
        label_mapping = get_ledgar_label_mapping()

        passive_results, passive_final, passive_stopping_iteration = run_passive_baseline(
            X_seed,
            y_seed,
            X_unlabeled,
            y_unlabeled,
            X_test,
            y_test,
            initial_samples=experiment_config["initial_samples"],
            batch_size=experiment_config["batch_size"],
            n_iterations=experiment_config["n_iterations"],
            random_seed=42,
        )

        active_results, active_final, active_stopping_iteration = run_active_baseline(
            X_seed,
            y_seed,
            X_unlabeled,
            y_unlabeled,
            X_test,
            y_test,
            initial_samples=experiment_config["initial_samples"],
            batch_size=experiment_config["batch_size"],
            n_iterations=experiment_config["n_iterations"],
            random_seed=42,
        )

        proposed_results, proposed_final, proposed_stopping_iteration = run_proposed_framework(
            X_seed,
            y_seed,
            X_unlabeled,
            y_unlabeled,
            X_test,
            y_test,
            seed_texts,
            unlabeled_texts,
            vectorizer,
            initial_samples=experiment_config["initial_samples"],
            batch_size=experiment_config["batch_size"],
            n_iterations=experiment_config["n_iterations"],
            random_seed=42,
            label_mapping=label_mapping,
        )

        plot_comparison(
            passive_results,
            active_results,
            proposed_results,
            config_name=config_name,
            run_tag=run_tag,
            show_plots=show_plots,
        )

        plot_utility_curve(
            passive_results,
            active_results,
            proposed_results,
            {
                "passive": passive_stopping_iteration,
                "active": active_stopping_iteration,
                "proposed": proposed_stopping_iteration,
            },
            config_name=config_name,
        )

        summary_df = pd.DataFrame(
            [
                {"Framework": "Passive", "Final_F1": passive_final["f1"], "Final_Accuracy": passive_final["accuracy"]},
                {"Framework": "Active", "Final_F1": active_final["f1"], "Final_Accuracy": active_final["accuracy"]},
                {"Framework": "Proposed", "Final_F1": proposed_final["f1"], "Final_Accuracy": proposed_final["accuracy"]},
            ]
        )

        summary_filename = _build_output_filename("framework_summary", config_name, run_tag, "csv")
        summary_path = os.path.join(DATA_DIR, summary_filename)
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 80)
        print("FRAMEWORK COMPARISON SUMMARY")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print(f"\nSaved: {summary_path}")

    finally:
        # 釋放本地大模型顯存，避免反覆執行時累積。
        global _LOCAL_LLM_TOKENIZER, _LOCAL_LLM_MODEL
        global _LOCAL_VALIDATOR_TOKENIZER, _LOCAL_VALIDATOR_MODEL
        _LOCAL_LLM_TOKENIZER = None
        _LOCAL_VALIDATOR_TOKENIZER = None
        _LOCAL_LLM_MODEL = None
        _LOCAL_VALIDATOR_MODEL = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.close()
        sys.stdout = logger.terminal


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {e}")
