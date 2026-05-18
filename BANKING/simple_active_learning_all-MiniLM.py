#!/usr/bin/env python3
"""
Banking77 多類別文本分類的主動學習比較腳本 (含日誌系統)

重點修正：
1) 修正 Utility Function 權重數量級。
2) 修復提早停止 (Early Stopping) 的 O(N^2) 邏輯錯誤。
3) 拔除無效參數 initial_samples。
4) 新增 LLM 生成退化文本的過濾機制，避免資料重複污染。
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
from sentence_transformers import SentenceTransformer
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


# Local Model
LLM_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
VALIDATOR_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

_LOCAL_LLM_TOKENIZER = None
_LOCAL_LLM_MODEL = None
_LOCAL_VALIDATOR_TOKENIZER = None
_LOCAL_VALIDATOR_MODEL = None

SENTENCE_TRANSFORMER_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "all-MiniLM")
LOG_DIR = os.path.join(DATA_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def setup_logging(config_name):
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
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    return logger


def _build_output_filename(prefix, config_name, run_tag, extension):
    parts = [prefix]
    if config_name:
        parts.append(config_name)
    if run_tag:
        parts.append(run_tag)
    return f"{'_'.join(parts)}.{extension}"


def _to_text_array(values):
    return np.asarray(["" if value is None else str(value) for value in values])


def _stack_features(X_top, X_bottom):
    # 移除相容 Sparse Matrix 的冗餘判斷
    return np.vstack([X_top, X_bottom])


def _load_local_generator():
    global _LOCAL_LLM_TOKENIZER, _LOCAL_LLM_MODEL
    if _LOCAL_LLM_TOKENIZER is None or _LOCAL_LLM_MODEL is None:
        if not torch.cuda.is_available():
            raise RuntimeError("需要 CUDA GPU 才能載入 4-bit 量化模型。")

        torch.cuda.empty_cache()
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
    global _LOCAL_VALIDATOR_TOKENIZER, _LOCAL_VALIDATOR_MODEL
    if _LOCAL_VALIDATOR_TOKENIZER is None or _LOCAL_VALIDATOR_MODEL is None:
        if not torch.cuda.is_available():
            raise RuntimeError("需要 CUDA GPU 才能載入 4-bit 量化模型。")

        torch.cuda.empty_cache()
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


def _load_banking77_intent_names():
    from datasets import load_dataset
    dataset = load_dataset("PolyAI/banking77", split="train")
    return dataset.features['label'].names


def get_banking77_label_mapping():
    label_names = _load_banking77_intent_names()
    return {i: name for i, name in enumerate(label_names)}


def load_banking77_sentence_transformer(random_seed=42, return_texts=False):
    np.random.seed(random_seed)
    print("Loading dataset: PolyAI/banking77 (train split)")
    dataset = load_dataset("PolyAI/banking77", split="train")
    df = dataset.to_pandas()[["text", "label"]].copy()

    label_mapping = get_banking77_label_mapping()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    seed_per_class = 10 
    test_per_class = 50 

    # 1. 依據標籤進行分層抽樣：對每個 group 使用 min(requested, group_size)
    def _safe_sample(group, n, rs):
        take = min(len(group), n)
        if take == 0:
            return group.iloc[0:0]
        return group.sample(n=take, random_state=rs)

    df_seed = df.groupby("label", group_keys=False).apply(lambda g: _safe_sample(g, seed_per_class, random_seed))
    # 2. 使用 df_seed 的原始索引從總表中剔除，確保 df_remaining 的類別數量精確無誤
    df_remaining = df.drop(df_seed.index)
    # 3. 剔除完成後，df_seed 才可以安全地打亂順序並重設索引，供後續實驗使用
    df_seed = df_seed.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    # 4. 從正確的 df_remaining 中抽取測試集
    df_test = df_remaining.groupby("label", group_keys=False).apply(
        lambda g: _safe_sample(g, test_per_class, random_seed)
    )
    # 5. 先剔除測試集索引，再重設 unlabeled 的索引
    df_unlabeled = df_remaining.drop(df_test.index).reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    combined_text = pd.concat(
        [df_seed["text"], df_unlabeled["text"], df_test["text"]],
        ignore_index=True,
    )
    text_encoder = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_ID)
    X_all = text_encoder.encode(
        combined_text.astype(str).tolist(),
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
            X_seed, y_seed, X_unlabeled, y_unlabeled, X_test, y_test,
            seed_texts, unlabeled_texts, test_texts, text_encoder,
        )
    return X_seed, y_seed, X_unlabeled, y_unlabeled, X_test, y_test


def generate_variants_llama3(texts, labels, label_mapping, n_variants=1, output_txt_path=None):
    texts = _to_text_array(texts)
    labels = np.asarray(labels)

    _load_local_generator()

    augmented_texts = []
    augmented_labels = []
    source_texts = []
    generation_records = []

    for text, label in zip(texts, labels):
        label_name = label_mapping.get(int(label), f"Intent_{label}")
        
        messages = [
            {
                "role": "system", 
                "content": f"You are a precise banking customer service text augmentation assistant. Your core task is to generate variations that strictly belong to the intent category: '{label_name}'."
            },
            {
                "role": "user",
                "content": (
                    f"Target Intent Category: {label_name}\n\n"
                    "Generate variations for the following customer service text.\n"
                    f"The variation MUST strictly preserve the exact semantic meaning of the target intent '{label_name}'.\n"
                    f"Please output exactly {n_variants} variation(s) strictly enclosed within <variation></variation> tags.\n"
                    "Do NOT include any titles, numbers, markdown, or introductory text.\n"
                    "Each variation must be wrapped with separate <variation>TEXT</variation> tags.\n\n"
                    f"Original text:\n{text}"
                ),
            },
        ]

        response_text = _generate_chat_response(
            _LOCAL_LLM_TOKENIZER,
            _LOCAL_LLM_MODEL,
            messages,
            max_new_tokens=384,
            do_sample=False,
        )

        variations = _extract_variations(response_text, n_variants=n_variants)

        if len(variations) == 0:
            variations = [text] 

        for var in variations[:n_variants]:
            augmented_texts.append(var)
            augmented_labels.append(label)
            source_texts.append(text)
            
            generation_records.append({
                "original": text,
                "generated": var,
                "label": label,
                "label_name": label_name # 日誌新增標籤名
            })

    if output_txt_path is not None:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\nLLAMA3 GENERATION RESULTS\n" + "=" * 80 + "\n\n")
            for i, record in enumerate(generation_records, 1):
                f.write(f"Sample {i}:\n  Target Intent: {record['label_name']}\n  Original Text: {record['original']}\n  Generated Text: {record['generated']}\n  Label: {record['label']}\n" + "-" * 80 + "\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        np.asarray(augmented_texts, dtype=str),
        np.asarray(augmented_labels),
        np.asarray(source_texts, dtype=str),
    )


def _extract_variations(response_text: str, n_variants: int = 1) -> list:
    matches = re.findall(r'<variation>(.*?)</variation>', response_text, re.DOTALL | re.IGNORECASE)
    return [m.strip() for m in matches if m.strip()]


def _extract_reasoning(response_text: str) -> str:
    match = re.search(r'<reasoning>(.*?)</reasoning>', response_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else response_text[:200]


def _extract_decision(response_text: str) -> str:
    match = re.search(r'<decision>(YES|NO)</decision>', response_text, re.IGNORECASE)
    return match.group(1).upper() if match else "UNKNOWN"


def _parse_validator_response(text):
    text_upper = str(text).upper()
    if "<DECISION>YES</DECISION>" in text_upper:
        return True
    elif "<DECISION>NO</DECISION>" in text_upper:
        return False
    return "YES" in text_upper


def validate_with_qwen25(
    generated_texts,
    generated_labels,
    label_mapping=None,
    logger=None,
    iteration=0,
    original_texts=None,
    output_txt_path=None,
):
    generated_texts = _to_text_array(generated_texts)
    generated_labels = np.asarray(generated_labels)

    if len(generated_texts) == 0:
        return np.asarray([], dtype=str), np.asarray([], dtype=generated_labels.dtype), 0.0

    _load_local_validator()

    valid_texts = []
    valid_labels = []
    accepted = 0
    validation_records = []

    for idx, (text, label) in enumerate(zip(generated_texts, generated_labels)):
        label_name = label_mapping.get(int(label), f"Label {label}") if label_mapping else f"Label {label}"
        original_text = str(original_texts[idx]) if original_texts is not None and idx < len(original_texts) else "[N/A]"
        
        # 將判斷基準錨定在 Target Intent 上
        messages = [
            {
                "role": "system", 
                "content": f"You are a strict data quality auditor for banking datasets. Your task is to verify if an augmented text precisely matches the target customer service intent: '{label_name}'."
            },
            {
                "role": "user",
                "content": (
                    f"Target Intent: {label_name}\n\n"
                    "Compare the 'Augmented Text' against the 'Target Intent' and the 'Original Text'.\n"
                    f"1. Does the Augmented Text accurately reflect the specific scenario defined by the intent '{label_name}'?\n"
                    "2. Is the Augmented Text a clean customer query free of ANY conversational filler like 'Here is the variation'?\n\n"
                    f"Original Text: {original_text}\n"
                    f"Augmented Text: {text}\n\n"
                    "First, briefly state your reasoning (max 2 sentences).\n"
                    "Then, on a new line, output your final decision strictly enclosed in XML tags: <decision>YES</decision> or <decision>NO</decision>."
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

        reasoning = _extract_reasoning(response_text)
        decision = _extract_decision(response_text)
        is_accepted = _parse_validator_response(response_text)

        if logger is not None and LOGGING_AVAILABLE:
            logger.log_augmentation_result(
                iteration=iteration, original_text=original_text, label=int(label), label_name=label_name,
                augmented_text=text, qwen_reasoning=reasoning, qwen_decision=decision, status="Accepted" if is_accepted else "Rejected",
            )

        validation_records.append({
            "original": original_text, "augmented": text, "label": label, "label_name": label_name,
            "reasoning": reasoning, "decision": decision, "accepted": is_accepted,
        })

        if is_accepted:
            valid_texts.append(text)
            valid_labels.append(label)
            accepted += 1

    if output_txt_path is not None:
        with open(output_txt_path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\nQWEN VALIDATION RESULTS\n" + "=" * 80 + "\n")
            f.write(f"Iteration: {iteration}\nTotal Samples: {len(validation_records)}\nAccepted: {accepted}/{len(validation_records)}\nAgreement Rate: {accepted / max(len(validation_records), 1):.4f}\n" + "-" * 80 + "\n\n")
            for i, record in enumerate(validation_records, 1):
                f.write(f"Sample {i}:\n  Label: {record['label']} ({record['label_name']})\n  Original: {record['original']}\n  Augmented: {record['augmented']}\n  Qwen Reasoning: {record['reasoning']}\n  Qwen Decision: {record['decision']}\n  Status: {'✓ ACCEPTED' if record['accepted'] else '✗ REJECTED'}\n" + "-" * 80 + "\n")

    agreement_rate = accepted / max(len(generated_texts), 1)

    # 釋放 GPU KV Cache，避免 VRAM 洩漏
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.asarray(valid_texts, dtype=str), np.asarray(valid_labels), agreement_rate


def random_sampling(pool_size, n_samples, random_seed=42):
    take_n = min(n_samples, pool_size)
    rng = np.random.RandomState(random_seed)
    return rng.choice(pool_size, size=take_n, replace=False)


def uncertainty_sampling(model, X_unlabeled, n_samples):
    take_n = min(n_samples, X_unlabeled.shape[0])
    probs = model.predict_proba(X_unlabeled)
    entropy_scores = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    ranked_idx = np.argsort(-entropy_scores)
    return ranked_idx[:take_n]


def train_and_evaluate(X_train, y_train, X_eval, y_eval, return_predictions=False):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)
    head_f1, tail_f1 = _compute_head_tail_f1(y_eval, y_pred)

    metrics = {
        "accuracy": accuracy_score(y_eval, y_pred),
        "precision": precision_score(y_eval, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_eval, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_eval, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_eval, y_pred, average="weighted", zero_division=0),
        "head_f1": head_f1,
        "tail_f1": tail_f1,
    }
    if return_predictions:
        return model, metrics, y_pred
    return model, metrics


def _append_selected(X_labeled, y_labeled, X_pool, y_pool, selected_idx):
    selected_idx = np.asarray(selected_idx, dtype=int)
    X_new = X_pool[selected_idx]
    y_new = y_pool[selected_idx]

    X_labeled_next = _stack_features(X_labeled, X_new)
    y_labeled_next = np.concatenate([y_labeled, y_new])

    keep_mask = np.ones(X_pool.shape[0], dtype=bool)
    keep_mask[selected_idx] = False

    X_pool_next = X_pool[keep_mask]
    y_pool_next = y_pool[keep_mask]
    return X_labeled_next, y_labeled_next, X_pool_next, y_pool_next


def _get_sampling_indices(model, X_pool, n_samples, iteration, warmup_iters=0, random_seed=42):
    if iteration <= warmup_iters:
        return random_sampling(X_pool.shape[0], n_samples, random_seed + iteration)
    return uncertainty_sampling(model, X_pool, n_samples)


def _compute_utility(f1_score_value, labeled_samples, iteration, max_iterations, lambda_base=0.00005, alpha=2.0):
    """
    預算感知效用函數 (Adaptive Lambda)
    
    Args:
        alpha: 緊迫性增長係數。alpha=2.0 代表到最後一輪時，懲罰力道會是初期的 3 倍。
    """
    # 計算動態 lambda
    lambda_t = lambda_base * (1.0 + alpha * (iteration / max_iterations))
    utility = f1_score_value - (lambda_t * labeled_samples)
    
    return utility, lambda_t


def _compute_head_tail_f1(y_true, y_pred, n_head=10, n_tail=10):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    head_classes = [str(i) for i in range(n_head)]
    tail_classes = [str(i) for i in range(n_head, n_head + n_tail)]

    head_f1_scores = [report[c]["f1-score"] for c in head_classes if c in report]
    tail_f1_scores = [report[c]["f1-score"] for c in tail_classes if c in report]

    head_f1 = float(np.mean(head_f1_scores)) if head_f1_scores else np.nan
    tail_f1 = float(np.mean(tail_f1_scores)) if tail_f1_scores else np.nan
    return head_f1, tail_f1


def _compute_stopping_iteration(
    results, 
    batch_size,
    base_patience=3, 
    min_patience=1, 
    max_patience=6, 
    std_window=4, 
    std_threshold_high=0.008,
    std_threshold_low=0.002,
    max_samples=None,
    cost_scaling_factor=0.2
):
    """
    穩定性感知與預算感知提早停止機制 (Dynamic Patience + Marginal Utility)
    """
    if not results:
        return None
    
    # Hard Constraint: 預算上限
    if max_samples is not None and results[-1].get("labeled_samples", 0) >= max_samples:
        print(f"[Stopping] Reached max_samples budget: {results[-1]['labeled_samples']} >= {max_samples}")
        return results[-1]["iteration"]
    
    # 確保有足夠的歷史資料計算波動度與 Patience
    required_history = max(max_patience, std_window) + 1
    if len(results) >= required_history:
        
        # 1. 穩定性感知 (Dynamic Patience) ------------------------
        # 萃取最近 std_window 輪的 F1 計算母體標準差
        recent_f1s = [res["f1"] for res in results[-std_window:]]
        current_std = float(np.std(recent_f1s))
        
        if current_std > std_threshold_high:
            current_patience = max_patience
            state_msg = "Volatile (High Variance)"
        elif current_std < std_threshold_low:
            current_patience = min_patience
            state_msg = "Stable (Low Variance)"
        else:
            current_patience = base_patience
            state_msg = "Normal"
            
        # 2. 預算感知判定 (Adaptive Thresholding) -----------------
        # 評估最近 current_patience 次的增益
        eval_window = results[-(current_patience + 1):]
        
        for i in range(1, len(eval_window)):
            delta_f1 = eval_window[i]["f1"] - eval_window[i - 1]["f1"]
            
            # 從結果中取出當下的動態 lambda (需在主迴圈中存入 results)
            lambda_t = eval_window[i].get("lambda_t", 0.00005)
            
            # 動態閾值：邊際增益必須大於邊際成本 (lambda_t * batch_size)
            # 若 alpha 很大，後期的 dynamic_epsilon 會非常嚴苛
            dynamic_epsilon = lambda_t * batch_size * cost_scaling_factor
            
            if delta_f1 >= dynamic_epsilon:
                # 只要觀測期內有任何一次增益打破動態成本閾值，就重置停止條件
                return None 
        
        # 若執行到此，代表連續 current_patience 次的增益都未能超過動態成本
        print(f"\n[Stopping Auto-Triggered]")
        print(f"  Reason: Marginal F1 gain < Marginal Cost for {current_patience} consecutive rounds.")
        print(f"  Stability State: {state_msg} (Std: {current_std:.5f})")
        print(f"  Final Cost Threshold (Epsilon): {dynamic_epsilon:.5f}")
        return results[-1]["iteration"]
        
    return None


def run_active_baseline(
    X_seed, y_seed, X_unlabeled, y_unlabeled, X_test, y_test,
    batch_size=40, n_iterations=25, random_seed=42, lambda_penalty=0.00005, max_samples=None, return_final_predictions=False,
):
    print(f"\n{'=' * 60}\nACTIVE LEARNING BASELINE (Warmup + Entropy)\n{'=' * 60}")
    X_labeled, y_labeled = X_seed.copy(), y_seed.copy()
    X_pool, y_pool = X_unlabeled.copy(), y_unlabeled.copy()

    results = []
    final_stop_iter = n_iterations

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        model, metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
        print(f"Test - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        # 1. 更新 Utility 呼叫方式 (傳入 iteration 與 n_iterations)
        utility, current_lambda = _compute_utility(
            metrics["f1"], 
            X_labeled.shape[0], 
            iteration=iteration, 
            max_iterations=n_iterations, 
            lambda_base=lambda_penalty, 
            alpha=2.0  # 你可以調整緊迫性係數
        )

        # 2. 將 lambda_t 存入 results 供 stopping 函數讀取
        results.append(
            {
                "iteration": iteration,
                "labeled_samples": X_labeled.shape[0],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "utility": utility,
                "lambda_t": current_lambda, # 新增此項
            }
        )

        # 3. 更新提早停止判斷邏輯
        if iteration > 5:
            suggested_stop = _compute_stopping_iteration(
                results,
                batch_size=batch_size,          # 傳入 batch_size 計算邊際成本
                base_patience=4,
                min_patience=2,
                max_patience=8,
                std_window=5,
                std_threshold_high=0.001,       # 可依據實驗觀測調整
                std_threshold_low=0.15,
                max_samples=max_samples,
            )
            if suggested_stop is not None:
                final_stop_iter = iteration
                break

        if iteration < n_iterations and X_pool.shape[0] > 0:
            selected_idx = _get_sampling_indices(model, X_pool, batch_size, iteration, warmup_iters=2, random_seed=random_seed)
            X_labeled, y_labeled, X_pool, y_pool = _append_selected(X_labeled, y_labeled, X_pool, y_pool, selected_idx)

    _, final_metrics, final_y_pred = train_and_evaluate(X_labeled, y_labeled, X_test, y_test, return_predictions=True)
    
    if return_final_predictions:
        return results, final_metrics, final_stop_iter, final_y_pred
    return results, final_metrics, final_stop_iter


def run_proposed_framework(
    X_seed, y_seed, X_unlabeled, y_unlabeled, X_test, y_test, seed_texts, unlabeled_texts, text_encoder,
    batch_size=40, n_iterations=25, random_seed=42, lambda_penalty=0.00005, max_samples=None, label_mapping=None,
    enable_logging=True, return_final_predictions=False,
):
    print(f"\n{'=' * 60}\nPROPOSED FRAMEWORK (Warmup + Entropy + LLM + Qwen)\n{'=' * 60}")
    X_labeled, y_labeled = X_seed.copy(), y_seed.copy()
    labeled_texts = _to_text_array(seed_texts)
    X_pool, y_pool = X_unlabeled.copy(), y_unlabeled.copy()
    pool_texts = _to_text_array(unlabeled_texts)

    txt_log_dir = os.path.join(DATA_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(txt_log_dir, exist_ok=True)
    
    results = []
    final_stop_iter = n_iterations
    logger = None
    if enable_logging and LOGGING_AVAILABLE:
        logger = DataAugmentationLogger(output_dir=DATA_DIR, log_name=f"augmentation_seed{random_seed}")

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        round_txt_file = os.path.join(txt_log_dir, f"round_{iteration:02d}_llama_qwen.txt")
        model, metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
        
        # 1. 更新 Utility 呼叫方式 (傳入 iteration 與 n_iterations)
        utility, current_lambda = _compute_utility(
            metrics["f1"], 
            X_labeled.shape[0], 
            iteration=iteration, 
            max_iterations=n_iterations, 
            lambda_base=lambda_penalty, 
            alpha=2.0  # 你可以調整緊迫性係數
        )

        # 2. 將 lambda_t 存入 results 供 stopping 函數讀取
        results.append(
            {
                "iteration": iteration,
                "labeled_samples": X_labeled.shape[0],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "utility": utility,
                "lambda_t": current_lambda, # 新增此項
            }
        )

        # 3. 更新提早停止判斷邏輯
        if iteration > 5:
            suggested_stop = _compute_stopping_iteration(
                results,
                batch_size=batch_size,          # 傳入 batch_size 計算邊際成本
                base_patience=3,
                min_patience=1,
                max_patience=6,
                std_window=4,
                std_threshold_high=0.008,       # 可依據實驗觀測調整
                std_threshold_low=0.002,
                max_samples=max_samples,
            )
            if suggested_stop is not None:
                final_stop_iter = iteration
                break

        if iteration < n_iterations and X_pool.shape[0] > 0:
            selected_idx = _get_sampling_indices(model, X_pool, batch_size, iteration, warmup_iters=2, random_seed=random_seed)
            selected_texts, selected_labels = pool_texts[selected_idx], y_pool[selected_idx]

            generated_texts, generated_labels, generated_source_texts = generate_variants_llama3(
                selected_texts, 
                selected_labels, 
                label_mapping=label_mapping,
                n_variants=1, 
                output_txt_path=round_txt_file,
            )

            # 修正：過濾退化樣本，根除重複污染
            valid_gen_indices = [
                i for i, (aug, orig) in enumerate(zip(generated_texts, generated_source_texts))
                if aug.strip() != orig.strip() and aug.strip() != ""
            ]
            
            if len(valid_gen_indices) > 0:
                filtered_texts = generated_texts[valid_gen_indices]
                filtered_labels = generated_labels[valid_gen_indices]
                filtered_sources = generated_source_texts[valid_gen_indices]

                validated_texts, validated_labels, agreement_rate = validate_with_qwen25(
                    filtered_texts, filtered_labels, label_mapping=label_mapping, logger=logger,
                    iteration=iteration, original_texts=filtered_sources, output_txt_path=round_txt_file,
                )
                print(f"Agreement Rate: {agreement_rate:.4f} | Accepted: {len(validated_texts)}/{len(filtered_texts)}")
            else:
                validated_texts, validated_labels = [], []
                print("Warning: Llama3 generated 0 valid variants in this batch.")

            trusted_texts = validated_texts
            trusted_labels = validated_labels
            X_selected = X_pool[selected_idx]

            if len(trusted_texts) > 0:
                X_validated = text_encoder.encode(
                    _to_text_array(trusted_texts).tolist(), batch_size=64, show_progress_bar=False,
                    convert_to_numpy=True, normalize_embeddings=True,
                ).astype(np.float32)
                X_labeled = _stack_features(X_labeled, X_validated)
                y_labeled = np.concatenate([y_labeled, trusted_labels])
                labeled_texts = np.concatenate([labeled_texts, trusted_texts])
                
            X_labeled = _stack_features(X_labeled, X_selected)
            y_labeled = np.concatenate([y_labeled, selected_labels])
            labeled_texts = np.concatenate([labeled_texts, selected_texts])

            keep_mask = np.ones(X_pool.shape[0], dtype=bool)
            keep_mask[selected_idx] = False
            X_pool, y_pool, pool_texts = X_pool[keep_mask], y_pool[keep_mask], pool_texts[keep_mask]

    _, final_metrics, final_y_pred = train_and_evaluate(X_labeled, y_labeled, X_test, y_test, return_predictions=True)
    
    if enable_logging and LOGGING_AVAILABLE and logger is not None:
        logger.export_to_excel()
        logger.print_statistics()

    if return_final_predictions:
        return results, final_metrics, final_stop_iter, final_y_pred
    return results, final_metrics, final_stop_iter


def run_passive_learning_experiment(
    X_seed, y_seed, X_unlabeled, y_unlabeled, X_test, y_test,
    batch_size=40, n_iterations=25, random_seed=42, lambda_penalty=0.00005
):
    print(f"\n{'=' * 60}\nPASSIVE LEARNING EXPERIMENT (Random)\n{'=' * 60}")
    X_labeled, y_labeled = X_seed.copy(), y_seed.copy()
    X_pool, y_pool = X_unlabeled.copy(), y_unlabeled.copy()

    print(f"Initial labeled pool: {X_labeled.shape[0]} samples")
    print(f"Remaining unlabeled: {X_pool.shape[0]} samples")

    results = []
    final_stop_iter = n_iterations

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        _, metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
        print(f"Test - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        utility, _ = _compute_utility(
            metrics["f1"], 
            X_labeled.shape[0], 
            iteration=iteration, 
            max_iterations=n_iterations, 
            lambda_base=lambda_penalty
        )
        results.append({
            "iteration": iteration, "labeled_samples": X_labeled.shape[0], "f1": metrics["f1"],
            "accuracy": metrics["accuracy"], "precision": metrics["precision"], "recall": metrics["recall"], "utility": utility,
        })

        if iteration < n_iterations and X_pool.shape[0] > 0:
            selected_idx = random_sampling(X_pool.shape[0], batch_size, random_seed + iteration)
            print(f"Selected {len(selected_idx)} samples using random sampling")
            X_labeled, y_labeled, X_pool, y_pool = _append_selected(X_labeled, y_labeled, X_pool, y_pool, selected_idx)
            print(f"Total labeled samples: {X_labeled.shape[0]}")
            print(f"Remaining unlabeled: {X_pool.shape[0]}")

    _, final_metrics = train_and_evaluate(X_labeled, y_labeled, X_test, y_test)
    print(f"\nFinal Test - F1: {final_metrics['f1']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}")
    return results, final_metrics, final_stop_iter


# (繪圖函式保留不變，唯 plot_utility_curve 的註解依據實作狀態微調，省略篇幅，實際執行請確保其存在)
def plot_macro_f1_curve(passive_results, active_results, proposed_results, config_name="", run_tag="", show_plots=True):
    passive_df, active_df, proposed_df = pd.DataFrame(passive_results), pd.DataFrame(active_results), pd.DataFrame(proposed_results)
    plt.figure(figsize=(9, 6))
    plt.plot(passive_df["iteration"], passive_df["f1"], "o--", label="Passive (Random)", linewidth=2, markersize=7)
    plt.plot(active_df["iteration"], active_df["f1"], "s-", label="Active (Entropy)", linewidth=2, markersize=7)
    plt.plot(proposed_df["iteration"], proposed_df["f1"], "^-", label="Proposed (LLM+Qwen)", linewidth=2, markersize=7)
    plt.xlabel("Iteration"), plt.ylabel("Macro F1"), plt.title("Macro F1 Curve Comparison")
    plt.legend(), plt.grid(True, alpha=0.3), plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, _build_output_filename("macro_f1_curve", config_name, run_tag, "png")), dpi=300, bbox_inches="tight")
    if show_plots: plt.show(block=False), plt.pause(0.1)
    else: plt.close()

def plot_utility_curve(passive_results, active_results, proposed_results, stopping_iters, config_name=""):
    passive_df, active_df, proposed_df = pd.DataFrame(passive_results), pd.DataFrame(active_results), pd.DataFrame(proposed_results)
    plt.figure(figsize=(9, 6))
    plt.plot(passive_df["iteration"], passive_df["utility"], "o--", label="Passive", linewidth=2, markersize=7)
    plt.plot(active_df["iteration"], active_df["utility"], "s-", label="Active", linewidth=2, markersize=7)
    plt.plot(proposed_df["iteration"], proposed_df["utility"], "^-", label="Proposed", linewidth=2, markersize=7)
    proposed_stopping_iteration = stopping_iters.get("proposed", 0) if isinstance(stopping_iters, dict) else 0
    if proposed_stopping_iteration > 0:
        plt.axvline(x=proposed_stopping_iteration, color="red", linestyle="--", linewidth=2, label="Auto-stopping Point")
    plt.xlabel("Iteration"), plt.ylabel("Utility"), plt.title("Utility Curve Comparison")
    plt.legend(), plt.grid(True, alpha=0.3), plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "utility_curve_comparison.png"), dpi=300, bbox_inches="tight")
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
    ax.set_ylabel("Macro F1 Score"), ax.set_title("Performance Comparison: Head vs. Tail Classes at Final Iteration")
    ax.set_xticks(x), ax.set_xticklabels(methods), ax.legend(), ax.grid(axis="y", linestyle="--", alpha=0.7), ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, _build_output_filename("head_tail_f1_comparison", config_name, run_tag, "png")), dpi=300, bbox_inches="tight")
    if show_plots: plt.show(block=False), plt.pause(0.1)
    else: plt.close(fig)

def plot_confusion_matrix_comparison(y_true, active_pred, proposed_pred, config_name="", run_tag="", show_plots=True, label_mapping=None):
    y_true, active_pred, proposed_pred = np.asarray(y_true), np.asarray(active_pred), np.asarray(proposed_pred)
    labels = np.arange(int(np.max(np.concatenate([y_true, active_pred, proposed_pred]))) + 1)
    display_labels = [label_mapping.get(int(l), f"Class {l}") for l in labels] if label_mapping else [str(l) for l in labels]
    def _norm_cm(yt, yp):
        matrix = confusion_matrix(yt, yp, labels=labels)
        rs = matrix.sum(axis=1, keepdims=True)
        return np.divide(matrix, rs, out=np.zeros_like(matrix, dtype=float), where=rs != 0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150, constrained_layout=True)
    cm_data = [(axes[0], _norm_cm(y_true, active_pred), "Active Learning (Warmup+Entropy)", "Blues"),
               (axes[1], _norm_cm(y_true, proposed_pred), "Proposed Framework (LLM+Qwen)", "Greens")]
    last_img = None
    for ax, cm, title, cmap in cm_data:
        last_img = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title), ax.set_xlabel("Predicted label"), ax.set_ylabel("True label")
        ax.set_xticks(labels), ax.set_yticks(labels)
        ax.set_xticklabels(display_labels, fontsize=8, rotation=90), ax.set_yticklabels(display_labels, fontsize=8)
    fig.colorbar(last_img, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Row-normalized rate")
    plt.savefig(os.path.join(DATA_DIR, _build_output_filename("confusion_matrix_active_vs_proposed", config_name, run_tag, "png")), dpi=300, bbox_inches="tight")
    if show_plots: plt.show(block=False), plt.pause(0.1)
    else: plt.close(fig)

def main():
    global DATA_DIR, LOG_DIR
    show_plots = False
    config_name = "banking77_warmup_llm_framework_with_logging"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 建立以日期時間為名稱的 run 資料夾，並將 DATA_DIR 指向該資料夾
    run_output_dir = os.path.join(DATA_DIR, run_tag)
    os.makedirs(run_output_dir, exist_ok=True)

    # 重新導向全域 DATA_DIR 與 LOG_DIR，以確保所有輸出集中到 run 資料夾
    DATA_DIR = run_output_dir
    LOG_DIR = os.path.join(DATA_DIR, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    experiment_config = {
        "batch_size": 40,
        "n_iterations": 40, 
    }

    logger = setup_logging(config_name)

    try:
        (
            X_seed, y_seed, X_unlabeled, y_unlabeled, X_test, y_test,
            seed_texts, unlabeled_texts, _test_texts, text_encoder,
        ) = load_banking77_sentence_transformer(random_seed=42, return_texts=True)

        label_mapping = get_banking77_label_mapping()

        passive_results, passive_final, passive_stopping_iteration = run_passive_learning_experiment(
            X_seed, y_seed, X_unlabeled, y_unlabeled, X_test, y_test,
            batch_size=experiment_config["batch_size"], n_iterations=experiment_config["n_iterations"], random_seed=42,
        )

        active_results, active_final, active_stopping_iteration, active_final_pred = run_active_baseline(
            X_seed, y_seed, X_unlabeled, y_unlabeled, X_test, y_test,
            batch_size=experiment_config["batch_size"], n_iterations=experiment_config["n_iterations"], random_seed=42, return_final_predictions=True,
        )

        proposed_results, proposed_final, proposed_stopping_iteration, proposed_final_pred = run_proposed_framework(
            X_seed, y_seed, X_unlabeled, y_unlabeled, X_test, y_test, seed_texts, unlabeled_texts, text_encoder,
            batch_size=experiment_config["batch_size"], n_iterations=experiment_config["n_iterations"], random_seed=42, label_mapping=label_mapping, enable_logging=True, return_final_predictions=True,
        )

        plot_macro_f1_curve(passive_results, active_results, proposed_results, config_name=config_name, run_tag=run_tag, show_plots=show_plots)
        plot_utility_curve(passive_results, active_results, proposed_results, {"passive": passive_stopping_iteration, "active": active_stopping_iteration, "proposed": proposed_stopping_iteration}, config_name=config_name)
        plot_head_tail_comparison(passive_final, active_final, proposed_final, config_name=config_name, run_tag=run_tag, show_plots=show_plots)
        plot_confusion_matrix_comparison(y_test, active_final_pred, proposed_final_pred, config_name=config_name, run_tag=run_tag, show_plots=show_plots, label_mapping=label_mapping)

        summary_df = pd.DataFrame([
            {"Framework": "Passive", "Macro_F1": passive_final["f1"], "Weighted_F1": passive_final["weighted_f1"], "Accuracy": passive_final["accuracy"], "Head_F1": passive_final["head_f1"], "Tail_F1": passive_final["tail_f1"]},
            {"Framework": "Active", "Macro_F1": active_final["f1"], "Weighted_F1": active_final["weighted_f1"], "Accuracy": active_final["accuracy"], "Head_F1": active_final["head_f1"], "Tail_F1": active_final["tail_f1"]},
            {"Framework": "Proposed", "Macro_F1": proposed_final["f1"], "Weighted_F1": proposed_final["weighted_f1"], "Accuracy": proposed_final["accuracy"], "Head_F1": proposed_final["head_f1"], "Tail_F1": proposed_final["tail_f1"]},
        ])
        
        summary_path = os.path.join(DATA_DIR, _build_output_filename("metrics_table", config_name, run_tag, "csv"))
        summary_df.to_csv(summary_path, index=False)
        print("\n" + "=" * 80 + "\nFRAMEWORK COMPARISON SUMMARY\n" + "=" * 80)
        print(summary_df.to_string(index=False))

    finally:
        global _LOCAL_LLM_TOKENIZER, _LOCAL_LLM_MODEL, _LOCAL_VALIDATOR_TOKENIZER, _LOCAL_VALIDATOR_MODEL
        _LOCAL_LLM_TOKENIZER = _LOCAL_VALIDATOR_TOKENIZER = _LOCAL_LLM_MODEL = _LOCAL_VALIDATOR_MODEL = None
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