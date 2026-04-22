#!/usr/bin/env python3
"""
Data Augmentation 日誌記錄系統
用來追蹤每次擴增與驗證的結果，方便後續分析與人工審核。
"""

import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
import json

import pandas as pd


class DataAugmentationLogger:
    """
    記錄 Data Augmentation 和 Validation 的詳細過程。
    支援 CSV 和 Excel 格式的匯出。
    """

    def __init__(self, output_dir: str = "data", log_name: Optional[str] = None):
        """
        初始化日誌記錄器。

        Args:
            output_dir: 輸出目錄
            log_name: 日誌檔名前綴（預設為時間戳）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if log_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"augmentation_log_{timestamp}"
        
        self.log_name = log_name
        self.csv_path = self.output_dir / f"{log_name}.csv"
        self.excel_path = self.output_dir / f"{log_name}.xlsx"
        self.json_path = self.output_dir / f"{log_name}.jsonl"

        self.records = []
        self._init_csv()

    def _init_csv(self):
        """初始化 CSV 檔案並寫入標題列。"""
        fieldnames = [
            "iteration",
            "original_text",
            "label",
            "label_name",
            "augmented_text",
            "qwen_reasoning",
            "qwen_decision",
            "status",
            "timestamp",
        ]
        
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log_augmentation_result(
        self,
        iteration: int,
        original_text: str,
        label: int,
        label_name: str,
        augmented_text: str,
        qwen_reasoning: str,
        qwen_decision: str,  # "YES" or "NO"
        status: str = None,  # "Accepted", "Rejected", etc.
    ):
        """
        記錄單筆擴增和驗證結果。

        Args:
            iteration: 迭代次數
            original_text: 原始文本
            label: 標籤（數字）
            label_name: 標籤名稱（文字）
            augmented_text: 擴增後的文本
            qwen_reasoning: Qwen 的推理過程
            qwen_decision: Qwen 的決策（YES/NO）
            status: 狀態標記（Accepted/Rejected）
        """
        if status is None:
            status = "Accepted" if qwen_decision.upper() == "YES" else "Rejected"

        record = {
            "iteration": iteration,
            "original_text": original_text,
            "label": label,
            "label_name": label_name,
            "augmented_text": augmented_text,
            "qwen_reasoning": qwen_reasoning,
            "qwen_decision": qwen_decision,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }

        self.records.append(record)
        self._write_to_csv(record)
        self._write_to_jsonl(record)

    def _write_to_csv(self, record: Dict):
        """將單筆記錄寫入 CSV 檔案（追加）。"""
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "iteration",
                    "original_text",
                    "label",
                    "label_name",
                    "augmented_text",
                    "qwen_reasoning",
                    "qwen_decision",
                    "status",
                    "timestamp",
                ],
            )
            writer.writerow(record)

    def _write_to_jsonl(self, record: Dict):
        """將單筆記錄寫入 JSONL 檔案（追加）。"""
        with open(self.json_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def export_to_excel(self):
        """將所有記錄匯出為 Excel 檔案。"""
        if not self.records:
            print("No records to export.")
            return
        
        df = pd.DataFrame(self.records)
        try:
            df.to_excel(self.excel_path, index=False, engine="openpyxl")
            print(f"✓ Exported to Excel: {self.excel_path}")
        except ModuleNotFoundError:
            csv_fallback_path = self.excel_path.with_suffix(".csv")
            df.to_csv(csv_fallback_path, index=False, encoding="utf-8")
            print(f"✓ openpyxl not installed; exported CSV instead: {csv_fallback_path}")

    def export_to_csv(self):
        """已經在即時寫入 CSV，此方法用於生成最終副本。"""
        if not self.records:
            print("No records to export.")
            return
        
        df = pd.DataFrame(self.records)
        df.to_csv(self.csv_path, index=False, encoding="utf-8")
        print(f"✓ Exported to CSV: {self.csv_path}")

    def get_statistics(self) -> Dict:
        """取得日誌統計資訊。"""
        if not self.records:
            return {"total": 0, "accepted": 0, "rejected": 0, "acceptance_rate": 0.0}
        
        df = pd.DataFrame(self.records)
        total = len(df)
        accepted = len(df[df["status"] == "Accepted"])
        rejected = len(df[df["status"] == "Rejected"])
        acceptance_rate = accepted / total if total > 0 else 0.0

        return {
            "total": total,
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": acceptance_rate,
            "by_label": df.groupby("label_name")["status"].value_counts().to_dict(),
        }

    def print_statistics(self):
        """列印統計資訊。"""
        stats = self.get_statistics()
        print("\n" + "=" * 60)
        print("AUGMENTATION LOG STATISTICS")
        print("=" * 60)
        print(f"Total Records: {stats['total']}")
        print(f"Accepted: {stats['accepted']} ({stats['acceptance_rate']:.2%})")
        print(f"Rejected: {stats['rejected']} ({1 - stats['acceptance_rate']:.2%})")
        print("\nBy Label:")
        if stats['by_label']:
            for label, count in sorted(stats['by_label'].items()):
                print(f"  {label}: {count}")
        print("=" * 60 + "\n")

    def load_csv(self, csv_path: str):
        """載入既存 CSV 檔案。"""
        df = pd.read_csv(csv_path)
        self.records = df.to_dict("records")
        print(f"Loaded {len(self.records)} records from {csv_path}")


def get_logger_instance(output_dir: str = "data") -> DataAugmentationLogger:
    """取得日誌記錄器實例。"""
    return DataAugmentationLogger(output_dir=output_dir)


if __name__ == "__main__":
    # 簡單的示例用法
    logger = DataAugmentationLogger(output_dir="data", log_name="test_log")
    
    # 模擬記錄
    logger.log_augmentation_result(
        iteration=1,
        original_text="This is a sample legal text.",
        label=0,
        label_name="Parties",
        augmented_text="The following legal text represents the parties involved.",
        qwen_reasoning="The augmented text maintains the legal intent and structure.",
        qwen_decision="YES",
    )

    logger.log_augmentation_result(
        iteration=1,
        original_text="Another legal clause here.",
        label=1,
        label_name="Agreement Date",
        augmented_text="The parties agree on the date mentioned herein as per discussion.",
        qwen_reasoning="The augmented text has conversational fillers.",
        qwen_decision="NO",
    )

    logger.export_to_excel()
    logger.print_statistics()
