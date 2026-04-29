#!/usr/bin/env python3
"""
LEDGAR 多類別文本分類的主動學習比較腳本 (含日誌系統)

重點：
1) Passive Learning: 每輪隨機抽樣
2) Active Learning: 每輪以 Entropy 不確定性抽樣
3) Proposed Framework: 結合 LLM 擴增 + Qwen 驗證 + 日誌追蹤
4) 完整日誌系統：追蹤每筆擴增決策，支援 CSV/Excel 匯出
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

# 嘗試導入日誌系統
try:
    from data_augmentation_logger import DataAugmentationLogger
    LOGGING_AVAILABLE = True
except ImportError:
    print("Warning: DataAugmentationLogger not found. Logging disabled.")
    LOGGING_AVAILABLE = False


# 純本地模型設定。
LLM_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
VALIDATOR_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

_LOCAL_LLM_TOKENIZER = None
_LOCAL_LLM_MODEL = None
_LOCAL_VALIDATOR_TOKENIZER = None
_LOCAL_VALIDATOR_MODEL = None

SENTENCE_TRANSFORMER_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "all-MiniLM")
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
    """組合不會互相覆蓋的輸出檔名。"""
    parts = [prefix]
    if config_name:
        parts.append(config_name)
    if run_tag:
        parts.append(run_tag)
    return f"{'_'.join(parts)}.{extension}"


def _to_text_array(values):
    """將輸入安全轉成純字串陣列。"""
    return np.asarray(["" if value is None else str(value) for value in values])


def _stack_features(X_top, X_bottom):
    """同時相容 dense 與 sparse 特徵拼接。"""
    if sparse.issparse(X_top) or sparse.issparse(X_bottom):
        return sparse.vstack([X_top, X_bottom], format="csr")
    return np.vstack([X_top, X_bottom])


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


def _build_head_to_tail_mapping(dataset, n_head=10, n_tail=10, min_samples=150):
    """
    建立 Head-to-Tail (熱門與次冷門) 的標籤對應表。
    """
    label_names = dataset.features['label'].names
    df_temp = dataset.to_pandas()[["label"]]
    
    # 計算每個類別的資料量
    counts = df_temp['label'].value_counts()
    
    # 過濾掉極端冷門 (少於 min_samples 筆) 的類別，避免後續 train/test 切分時報錯
    safe_counts = counts[counts >= min_samples]
    
    # 抓取 Top-N (熱門) 與 Bottom-N (次冷門)
    head_ids = safe_counts.head(n_head).index.tolist()
    tail_ids = safe_counts.tail(n_tail).index.tolist()
    
    selected_old_ids = head_ids + tail_ids
    
    label_mapping = {}      # 給 Qwen 驗證用的 {新ID: "標籤名稱"}
    old_to_new_id = {}      # 給過濾與轉換資料用的 {原始ID: 新ID}
    
    for new_id, old_id in enumerate(selected_old_ids):
        label_mapping[new_id] = label_names[old_id]
        old_to_new_id[old_id] = new_id
        
    return label_mapping, old_to_new_id, safe_counts


def get_ledgar_label_mapping(n_head=10, n_tail=10):
    """
    取得 LEDGAR 資料集的標籤對應。
    改為動態取得 Head-to-Tail 的合約條款分類。
    """
    from datasets import load_dataset
    try:
        dataset = load_dataset("lexlual/LEDGAR", split="train")
    except:
        dataset = load_dataset("lex_glue", "ledgar", split="train")
        
    label_mapping, _, _ = _build_head_to_tail_mapping(dataset, n_head=n_head, n_tail=n_tail)
    return label_mapping


def load_ledgar_sentence_transformer(random_seed=42, return_texts=False, n_head=10, n_tail=10):
    """
    載入 LEDGAR 並轉成 SentenceTransformer embedding。
    (已套用 Head-to-Tail 長尾不平衡實驗設計)
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

    # ========================================================
    # 套用 Head-to-Tail 策略過濾並重映射 ID
    # ========================================================
    label_mapping, old_to_new_id, counts = _build_head_to_tail_mapping(dataset, n_head=n_head, n_tail=n_tail)
    
    # 1. 篩選：只保留選定的 20 個類別
    df = df[df["label"].isin(old_to_new_id.keys())].copy()
    
    # 2. 重映射：把原始 ID 轉換為連續的 0 ~ 19
    df["label"] = df["label"].map(old_to_new_id)
    # ========================================================

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    # 印出你到底挑了哪些類別與樣本數，可以直接放進論文的 Dataset Table 裡
    print(f"\n=== Selected {n_head} Head & {n_tail} Tail Classes for PoC ===")
    for old_id, new_id in old_to_new_id.items():
        text_label = label_mapping[new_id]
        sample_count = counts[old_id]
        category_type = "Head (大宗條款)" if new_id < n_head else "Tail (冷門條款)"
        print(f"  ID {new_id:2d} -> {text_label:<25} | {category_type} | 總樣本數: {sample_count}")
    print("==============================================================\n")

    # 打亂並重設索引，確保切分可重現。
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # ========================================================
    # 【冷啟動切分邏輯】：K-shot 分層抽樣
    # ========================================================
    seed_per_class = 10  # 每個類別只給 10 筆 (冷啟動)
    n_test = 2000        # 測試集保持 2000 筆，確保冷門類別也有足夠考題
    n_unlabeled = 5000   # 未標註池保持 5000 筆，控制 LLM 運算時間

    # 1. 強制分層抽樣：每個類別精準抽出 seed_per_class 筆當作 Seed Set
    df_seed = df.groupby("label").sample(n=seed_per_class, random_state=random_seed)
    
    # 2. 把抽去當 Seed 的資料從母體中剔除
    df_remaining = df.drop(df_seed.index)
    
    # 3. 剩下的資料徹底打亂，準備切分 Test 和 Unlabeled
    df_remaining = df_remaining.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    # 檢查剩餘資料夠不夠
    if len(df_remaining) < (n_test + n_unlabeled):
        raise ValueError(f"剩餘資料不足！需要 {n_test + n_unlabeled} 筆，但只剩 {len(df_remaining)} 筆。")

    # 4. 依序切出 Test 和 Unlabeled
    df_test = df_remaining.iloc[:n_test].copy()
    df_unlabeled = df_remaining.iloc[n_test : n_test + n_unlabeled].copy()

    # 使用同一個編碼器在整個子集合上建立 embedding，避免後續維度不一致。
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
            X_seed,
            y_seed,
            X_unlabeled,
            y_unlabeled,
            X_test,
            y_test,
            seed_texts,
            unlabeled_texts,
            test_texts,
            text_encoder,
        )

    return X_seed, y_seed, X_unlabeled, y_unlabeled, X_test, y_test


def generate_variants_llama3(texts, labels, n_variants=1):
    """使用本地 Llama 3 產生擴增資料。模型常駐 VRAM，無需釋放。"""
    texts = _to_text_array(texts)
    labels = np.asarray(labels)

    _load_local_generator()

    augmented_texts = []
    augmented_labels = []
    source_texts = []

    for text, label in zip(texts, labels):
        messages = [
            {
                "role": "system",
                "content": "You are a precise legal text augmentation assistant.",
            },
            {
                "role": "user",
                "content": (
                    "Generate legal text variations while preserving exact legal semantics.\n"
                    f"Please output exactly {n_variants} variations separated by the delimiter '|||'. "
                    "Do NOT include any titles, numbers, markdown, or introductory text like 'Here are the variations'.\n\n"
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
            source_texts.append(text)

    return (
        np.asarray(augmented_texts, dtype=str),
        np.asarray(augmented_labels),
        np.asarray(source_texts, dtype=str),
    )


def _extract_reasoning(response_text: str) -> str:
    """從 Qwen 回應中提取推理過程 (XML 格式)。"""
    match = re.search(
        r'<reasoning>(.*?)</reasoning>',
        response_text,
        re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else response_text[:200]


def _extract_decision(response_text: str) -> str:
    """從 Qwen 回應中提取決策 (XML 格式)。"""
    match = re.search(
        r'<decision>(YES|NO)</decision>',
        response_text,
        re.IGNORECASE
    )
    return match.group(1).upper() if match else "UNKNOWN"


def _parse_validator_response(text):
    """解析 XML 格式決策：從 <decision>YES</decision> 或 <decision>NO</decision> 中提取決策。"""
    text_upper = str(text).upper()
    if "<DECISION>YES</DECISION>" in text_upper:
        return True
    elif "<DECISION>NO</DECISION>" in text_upper:
        return False
    # Fallback: 相容舊格式
    return "YES" in text_upper


def validate_with_qwen25(
    generated_texts,
    generated_labels,
    label_mapping=None,
    logger=None,
    iteration=0,
    original_texts=None,
):
    """
    使用本地 Qwen2.5 驗證擴增資料並回傳 agreement rate。
    支援可選的日誌記錄。
    """
    generated_texts = _to_text_array(generated_texts)
    generated_labels = np.asarray(generated_labels)

    if len(generated_texts) == 0:
        return np.asarray([], dtype=str), np.asarray([], dtype=generated_labels.dtype), 0.0

    _load_local_validator()

    valid_texts = []
    valid_labels = []
    accepted = 0

    for idx, (text, label) in enumerate(zip(generated_texts, generated_labels)):
        # 取得標籤名稱
        label_name = label_mapping.get(int(label), f"Label {label}") if label_mapping else f"Label {label}"
        if original_texts is not None and idx < len(original_texts):
            original_text = str(original_texts[idx])
        else:
            original_text = "[N/A]"
        
        messages = [
            {
                "role": "system",
                "content": "You are a senior legal counsel auditing data augmentation. You are tolerant of paraphrasing as long as the core legal obligations, rights, and liabilities remain unchanged.",
            },
            {
                "role": "user",
                "content": (
                    "Compare the 'Augmented Text' to the 'Original Text'.\n"
                    "1. Does the Augmented Text strictly preserve the legal intent of the Original Text (accepting synonyms and rephrasing)?\n"
                    "2. Is the Augmented Text a clean legal clause free of ANY conversational filler like 'Here is the variation' or 'Sure'?\n\n"
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

        # 提取推理和決策
        reasoning = _extract_reasoning(response_text)
        decision = _extract_decision(response_text)

        # 記錄到日誌
        if logger is not None and LOGGING_AVAILABLE:
            logger.log_augmentation_result(
                iteration=iteration,
                original_text=original_text,
                label=int(label),
                label_name=label_name,
                augmented_text=text,
                qwen_reasoning=reasoning,
                qwen_decision=decision,
                status="Accepted" if decision.upper() == "YES" else "Rejected",
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


def train_and_evaluate(X_train, y_train, X_eval, y_eval, return_predictions=False):
    """訓練多類別 LR 並回傳 metrics (用 Macro 凸顯長尾效應)"""
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
    """將選到的樣本從 pool 移到 labeled，並回傳更新後矩陣與標籤。"""
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
    """根據 iteration 決定使用 warmup random 或 entropy active sampling。"""
    if iteration <= warmup_iters:
        return random_sampling(X_pool.shape[0], n_samples, random_seed + iteration)
    return uncertainty_sampling(model, X_pool, n_samples)


def _compute_utility(f1_score_value, labeled_samples, lambda_penalty=0.0003):
    """依論文定義計算效用值。"""
    return f1_score_value - (lambda_penalty * labeled_samples)


def _compute_head_tail_f1(y_true, y_pred, n_head=10, n_tail=10):
    """從 classification report 萃取 Head / Tail 類別的 Macro F1。"""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    head_classes = [str(i) for i in range(n_head)]
    tail_classes = [str(i) for i in range(n_head, n_head + n_tail)]

    head_f1_scores = [report[c]["f1-score"] for c in head_classes if c in report]
    tail_f1_scores = [report[c]["f1-score"] for c in tail_classes if c in report]

    head_f1 = float(np.mean(head_f1_scores)) if head_f1_scores else np.nan
    tail_f1 = float(np.mean(tail_f1_scores)) if tail_f1_scores else np.nan

    return head_f1, tail_f1


def _compute_stopping_iteration(results, patience=3, epsilon=0.003, max_samples=None):
    """
    根據 F1 marginal gain 與 budget constraint 決定自動停止迭代點。
    
    Hard Constraint: 若達到 max_samples 預算上限，立即停止。
    Soft Signal: 若 ΔF1 < epsilon 連續 patience 次迭代，建議停止。
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
    return_final_predictions=False,
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
                patience=10,
                epsilon=0.0005,
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
                warmup_iters=0,
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

    _, final_metrics, final_y_pred = train_and_evaluate(
        X_labeled,
        y_labeled,
        X_test,
        y_test,
        return_predictions=True,
    )
    print(f"\nFinal Test - F1: {final_metrics['f1']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}")

    if return_final_predictions:
        return results, final_metrics, final_stop_iter, final_y_pred
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
    text_encoder,
    initial_samples=300,
    batch_size=40,
    n_iterations=25,
    random_seed=42,
    lambda_penalty=0.0005,
    max_samples=None,
    label_mapping=None,
    enable_logging=True,
    return_final_predictions=False,
):
    """
    Proposed Framework：Warmup + Entropy + LLM 擴增 + Qwen 驗證。
    支援完整日誌系統。
    """
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

    # 初始化日誌
    logger = None
    if enable_logging and LOGGING_AVAILABLE:
        logger = DataAugmentationLogger(
            output_dir=DATA_DIR,
            log_name=f"augmentation_seed{random_seed}"
        )
        print(f"Augmentation logging enabled: {logger.csv_path}")

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
                patience=10,
                epsilon=0.0005,
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
                warmup_iters=0,
                random_seed=random_seed,
            )
            print(
                f"Selected {len(selected_idx)} samples using "
                f"{'random' if iteration <= 0 else 'entropy'} sampling"
            )

            selected_texts = pool_texts[selected_idx]
            selected_labels = y_pool[selected_idx]

            # 1. 降低擴增倍率：只生成 1 個高質量變體，避免 Dense 空間特徵重疊
            generated_texts, generated_labels, generated_source_texts = generate_variants_llama3(
                selected_texts,
                selected_labels,
                n_variants=1,
            )

            # 2. 100% 嚴格驗證：最多 40 筆，讓 Qwen 全數驗證
            print(f"Sending all {len(generated_texts)} augmented samples to Validator...")
            validated_texts, validated_labels, agreement_rate = validate_with_qwen25(
                generated_texts,
                generated_labels,
                label_mapping=label_mapping,
                logger=logger,
                iteration=iteration,
                original_texts=generated_source_texts,
            )

            print(f"Agreement Rate: {agreement_rate:.4f}")
            print(f"Validator accepted: {len(validated_texts)}/{len(generated_texts)}")

            # 3. 只加入被 Qwen 認可的乾淨資料
            trusted_texts = validated_texts
            trusted_labels = validated_labels

            X_selected = X_pool[selected_idx]
            X_validated = (
                text_encoder.encode(
                    _to_text_array(trusted_texts).tolist(),
                    batch_size=64,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype(np.float32)
                if len(trusted_texts) > 0
                else np.empty((0, X_pool.shape[1]), dtype=np.float32)
            )

            X_labeled = _stack_features(X_labeled, X_selected)
            X_labeled = _stack_features(X_labeled, X_validated)
            y_labeled = np.concatenate([y_labeled, selected_labels, trusted_labels])
            labeled_texts = np.concatenate([labeled_texts, selected_texts, trusted_texts])

            keep_mask = np.ones(X_pool.shape[0], dtype=bool)
            keep_mask[selected_idx] = False
            X_pool = X_pool[keep_mask]
            y_pool = y_pool[keep_mask]
            pool_texts = pool_texts[keep_mask]

            print(f"Total labeled samples: {X_labeled.shape[0]}")
            print(f"Remaining unlabeled: {X_pool.shape[0]}")

    _, final_metrics, final_y_pred = train_and_evaluate(
        X_labeled,
        y_labeled,
        X_test,
        y_test,
        return_predictions=True,
    )
    print(f"\nFinal Test - F1: {final_metrics['f1']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}")

    # 匯出日誌
    if logger is not None and LOGGING_AVAILABLE:
        logger.export_to_excel()
        logger.print_statistics()
        print(f"\nAugmentation logs saved to: {DATA_DIR}")

    if return_final_predictions:
        return results, final_metrics, final_stop_iter, final_y_pred
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
    lambda_penalty=0.0005
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
    plt.ylabel("Macro F1")
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


def plot_macro_f1_curve(passive_results, active_results, proposed_results, config_name="", run_tag="", show_plots=True):
    """繪製三種方法的 Macro F1 折線圖。"""
    passive_df = pd.DataFrame(passive_results)
    active_df = pd.DataFrame(active_results)
    proposed_df = pd.DataFrame(proposed_results)

    plt.figure(figsize=(9, 6))

    plt.plot(passive_df["iteration"], passive_df["f1"], "o--", label="Passive (Random)", linewidth=2, markersize=7)
    plt.plot(active_df["iteration"], active_df["f1"], "s-", label="Active (Entropy)", linewidth=2, markersize=7)
    plt.plot(proposed_df["iteration"], proposed_df["f1"], "^-", label="Proposed (LLM+Qwen)", linewidth=2, markersize=7)

    plt.xlabel("Iteration")
    plt.ylabel("Macro F1")
    plt.title("Macro F1 Curve Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_filename = _build_output_filename("macro_f1_curve", config_name, run_tag, "png")
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


def plot_head_tail_comparison(passive_final, active_final, proposed_final, config_name="", run_tag="", show_plots=True):
    """比較三種框架在 Head / Tail 類別上的最終 F1。"""
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
    """比較 Active Learning 與 Proposed 的 confusion matrix。"""
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
        # 調整字體大小為 8，確保 20 個條款名稱不會互相重疊
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


def plot_statistical_comparison(active_finals, passive_finals, config_name="", show_plots=True):
    """保留統計比較圖架構（分佈、箱型、逐 run 對比）。"""
    active_f1_scores = [result["f1"] for result in active_finals]
    passive_f1_scores = [result["f1"] for result in passive_finals]

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.hist(active_f1_scores, alpha=0.7, label="Active", bins=8, color="blue")
    plt.hist(passive_f1_scores, alpha=0.7, label="Passive", bins=8, color="red")
    plt.xlabel("Macro F1")
    plt.ylabel("Frequency")
    plt.title("Distribution of F1 Scores")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    plt.boxplot([active_f1_scores, passive_f1_scores], labels=["Active", "Passive"])
    plt.ylabel("Macro F1")
    plt.title("Box Plot Comparison")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 3)
    runs = range(1, len(active_f1_scores) + 1)
    plt.plot(runs, active_f1_scores, "s-", label="Active", linewidth=2)
    plt.plot(runs, passive_f1_scores, "o--", label="Passive", linewidth=2)
    plt.xlabel("Run")
    plt.ylabel("Macro F1")
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
    """主程式：執行 Passive、Active、Proposed 三條框架比較（含日誌系統）。"""

    show_plots = False
    config_name = "ledgar_warmup_llm_framework_with_logging"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 依需求更新設定
    experiment_config = {
        "initial_samples": 200,
        "batch_size": 40,
        "n_iterations":40, 
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
            text_encoder,
        ) = load_ledgar_sentence_transformer(random_seed=42, return_texts=True)

        # 取得 LEDGAR 標籤映射
        label_mapping = get_ledgar_label_mapping()

        passive_results, passive_final, passive_stopping_iteration = run_passive_learning_experiment(
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

        active_results, active_final, active_stopping_iteration, active_final_pred = run_active_baseline(
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
            return_final_predictions=True,
        )

        proposed_results, proposed_final, proposed_stopping_iteration, proposed_final_pred = run_proposed_framework(
            X_seed,
            y_seed,
            X_unlabeled,
            y_unlabeled,
            X_test,
            y_test,
            seed_texts,
            unlabeled_texts,
            text_encoder,
            initial_samples=experiment_config["initial_samples"],
            batch_size=experiment_config["batch_size"],
            n_iterations=experiment_config["n_iterations"],
            random_seed=42,
            label_mapping=label_mapping,
            enable_logging=True,
            return_final_predictions=True,
        )

        plot_macro_f1_curve(
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

        plot_head_tail_comparison(
            passive_final,
            active_final,
            proposed_final,
            config_name=config_name,
            run_tag=run_tag,
            show_plots=show_plots,
        )

        plot_confusion_matrix_comparison(
            y_test,
            active_final_pred,
            proposed_final_pred,
            config_name=config_name,
            run_tag=run_tag,
            show_plots=show_plots,
            label_mapping=label_mapping
        )

        summary_df = pd.DataFrame(
            [
                {"Framework": "Passive", "Macro_F1": passive_final["f1"], "Weighted_F1": passive_final["weighted_f1"], "Accuracy": passive_final["accuracy"], "Head_F1": passive_final["head_f1"], "Tail_F1": passive_final["tail_f1"]},
                {"Framework": "Active", "Macro_F1": active_final["f1"], "Weighted_F1": active_final["weighted_f1"], "Accuracy": active_final["accuracy"], "Head_F1": active_final["head_f1"], "Tail_F1": active_final["tail_f1"]},
                {"Framework": "Proposed", "Macro_F1": proposed_final["f1"], "Weighted_F1": proposed_final["weighted_f1"], "Accuracy": proposed_final["accuracy"], "Head_F1": proposed_final["head_f1"], "Tail_F1": proposed_final["tail_f1"]},
            ]
        )

        summary_filename = _build_output_filename("metrics_table", config_name, run_tag, "csv")
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
