#!/usr/bin/env python3
"""
整合示例：展示如何在 Data Augmentation Pipeline 中使用日誌系統

這個檔案示範如何：
1. 在 validate_with_qwen25 中捕捉 Qwen 的推理過程
2. 使用 DataAugmentationLogger 記錄每筆結果
3. 定期匯出報告並進行採樣審查
"""

import re
import numpy as np
from typing import Tuple
from data_augmentation_logger import DataAugmentationLogger


def extract_reasoning_and_decision(response_text: str) -> Tuple[str, str]:
    """
    從 Qwen 的回應中提取推理過程和決策。
    
    預期格式：
    <reasoning>...</reasoning>
    <decision>YES</decision> 或 <decision>NO</decision>
    
    Args:
        response_text: Qwen 模型的完整回應
        
    Returns:
        (reasoning, decision) 元組
    """
    # 提取 reasoning
    reasoning_match = re.search(
        r"<reasoning>(.*?)</reasoning>",
        response_text,
        re.DOTALL | re.IGNORECASE
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else response_text[:200]
    
    # 提取 decision
    decision_match = re.search(
        r"<decision>(YES|NO)</decision>",
        response_text,
        re.IGNORECASE
    )
    decision = decision_match.group(1).upper() if decision_match else "UNKNOWN"
    
    return reasoning, decision


def validate_with_qwen25_logged(
    generated_texts,
    generated_labels,
    label_mapping=None,
    logger: DataAugmentationLogger = None,
    iteration: int = 0,
    original_texts=None,  # 用於追蹤原始文本
):
    """
    改進版本的 validate_with_qwen25，整合日誌功能。
    
    這個函數示範如何在驗證過程中記錄詳細資訊。
    
    Args:
        generated_texts: 生成的文本列表
        generated_labels: 對應的標籤
        label_mapping: 標籤名稱映射
        logger: DataAugmentationLogger 實例
        iteration: 當前迭代號
        original_texts: 原始文本（用於日誌記錄）
        
    Returns:
        (valid_texts, valid_labels, agreement_rate)
    """
    from data_augmentation_logger import DataAugmentationLogger
    
    # 如果沒有提供 logger，就建立一個
    if logger is None:
        logger = DataAugmentationLogger(output_dir="data")
    
    generated_texts = np.asarray(
        ["" if v is None else str(v) for v in generated_texts]
    )
    generated_labels = np.asarray(generated_labels)
    
    if len(generated_texts) == 0:
        return np.asarray([], dtype=str), np.asarray([], dtype=generated_labels.dtype), 0.0
    
    # 注意：這裡假設你已經有 _load_local_validator() 和 _generate_chat_response()
    # 這是示範性代碼，實際使用時需要導入這些函數
    
    valid_texts = []
    valid_labels = []
    accepted = 0
    
    for idx, (text, label) in enumerate(zip(generated_texts, generated_labels)):
        # 取得標籤名稱
        label_name = label_mapping.get(int(label), f"Label {label}") if label_mapping else f"Label {label}"
        
        # 建構 Qwen 提示（改進版，要求 XML 格式的推理和決策）
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
                    "First, provide your reasoning wrapped in <reasoning>...</reasoning> tags.\n"
                    "Then, output your final decision as exactly <decision>YES</decision> or <decision>NO</decision>.\n\n"
                    f"Augmented Text: {text}"
                ),
            },
        ]
        
        # 假設 _generate_chat_response 已定義
        # response_text = _generate_chat_response(
        #     _LOCAL_VALIDATOR_TOKENIZER,
        #     _LOCAL_VALIDATOR_MODEL,
        #     messages,
        #     max_new_tokens=256,
        #     do_sample=False,
        # )
        
        # 示範用的假回應
        response_text = "<reasoning>The text maintains proper legal terminology.</reasoning>\n<decision>YES</decision>"
        
        # 提取推理和決策
        reasoning, decision = extract_reasoning_and_decision(response_text)
        
        # 記錄到日誌
        original_text = original_texts[idx] if original_texts is not None else "[Original text not provided]"
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
        
        # 更新統計
        if decision.upper() == "YES":
            valid_texts.append(text)
            valid_labels.append(label)
            accepted += 1
    
    agreement_rate = accepted / max(len(generated_texts), 1)
    return np.asarray(valid_texts, dtype=str), np.asarray(valid_labels), agreement_rate


def main_example():
    """使用範例：展示如何整合日誌系統。"""
    
    # 1. 建立日誌記錄器
    logger = DataAugmentationLogger(output_dir="data", log_name="ledgar_augmentation_example")
    
    # 2. 定義標籤映射
    label_mapping = {
        0: "Parties",
        1: "Agreement Date",
        2: "Effective Date",
        3: "Termination Clause",
        4: "Liability",
    }
    
    # 3. 模擬擴增資料（實際使用時來自 Llama-3）
    original_texts = [
        "The parties to this agreement are Company A and Company B.",
        "This agreement becomes effective on January 1, 2024.",
    ]
    
    generated_texts = [
        "Company A and Company B are the parties to this legal agreement.",
        "The effective date of this agreement is set to January 1, 2024.",
    ]
    
    generated_labels = [0, 1]
    
    # 4. 使用帶有日誌的驗證函數
    valid_texts, valid_labels, agreement_rate = validate_with_qwen25_logged(
        generated_texts=generated_texts,
        generated_labels=generated_labels,
        label_mapping=label_mapping,
        logger=logger,
        iteration=1,
        original_texts=original_texts,
    )
    
    print(f"Agreement Rate: {agreement_rate:.2%}")
    
    # 5. 定期匯出報告
    logger.print_statistics()
    logger.export_to_excel()
    
    print(f"\n✓ Augmentation log saved to: {logger.csv_path}")
    print(f"✓ Excel report saved to: {logger.excel_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("Data Augmentation Logger Integration Example")
    print("=" * 80)
    print()
    
    main_example()
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("""
1. 在你的 simple_active_learning_LR.py 中，導入日誌系統：
   from data_augmentation_logger import DataAugmentationLogger

2. 在 run_proposed_framework() 中初始化日誌：
   logger = DataAugmentationLogger(output_dir=DATA_DIR)

3. 在每次驗證後呼叫 logger.log_augmentation_result()

4. 完成實驗後，使用採樣審查工具：
   python sample_augmentation_results.py data/augmentation_log_YYYYMMDD_HHMMSS.csv \\
       --yes-size 50 --no-size 50 --output-dir data/reviews

5. 人工審查生成的 Excel 檔案，確認邏輯是否符合預期

6. 若邏輯正確，放心讓程式自動跑完剩下的資料
    """)
    print("=" * 80)
