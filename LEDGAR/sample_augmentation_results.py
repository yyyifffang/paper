#!/usr/bin/env python3
"""
Data Augmentation 採樣審查工具
從日誌中隨機抽取 YES 和 NO 的樣本，方便人工審核。
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd


class AugmentationSampleReviewer:
    """
    從日誌中採樣和審查擴增結果。
    """

    def __init__(self, csv_path: str, random_seed: int = 42):
        """
        初始化審查工具。

        Args:
            csv_path: 日誌 CSV 檔案路徑
            random_seed: 隨機種子（確保可重現）
        """
        self.csv_path = csv_path
        self.random_seed = random_seed
        random.seed(random_seed)

        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} records from {csv_path}")

    def sample_by_decision(
        self, 
        decision: str, 
        sample_size: int = 50
    ) -> pd.DataFrame:
        """
        按決策類型採樣。

        Args:
            decision: "YES" 或 "NO"
            sample_size: 採樣數量

        Returns:
            採樣後的 DataFrame
        """
        filtered = self.df[self.df["qwen_decision"].str.upper() == decision.upper()]
        
        if len(filtered) < sample_size:
            print(
                f"⚠ Only {len(filtered)} {decision} records available, "
                f"but requested {sample_size}. Using all available records."
            )
            sample_size = len(filtered)
        
        sampled = filtered.sample(n=sample_size, random_state=self.random_seed)
        return sampled

    def generate_review_report(
        self, 
        yes_sample_size: int = 50,
        no_sample_size: int = 50,
        output_dir: str = "data"
    ) -> Tuple[str, str]:
        """
        生成人工審查報告（分別輸出 YES 和 NO 的樣本）。

        Args:
            yes_sample_size: YES 樣本數量
            no_sample_size: NO 樣本數量
            output_dir: 輸出目錄

        Returns:
            (yes_report_path, no_report_path)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # 採樣 YES 和 NO
        yes_samples = self.sample_by_decision("YES", yes_sample_size)
        no_samples = self.sample_by_decision("NO", no_sample_size)

        # 輸出 YES 報告，若缺少 openpyxl 則自動改存 CSV
        yes_report_path = output_path / f"review_report_YES_{timestamp}.xlsx"
        try:
            yes_samples.to_excel(yes_report_path, index=False, engine="openpyxl")
            print(f"✓ Exported YES samples ({len(yes_samples)} records): {yes_report_path}")
        except ModuleNotFoundError:
            yes_report_path = output_path / f"review_report_YES_{timestamp}.csv"
            yes_samples.to_csv(yes_report_path, index=False, encoding="utf-8")
            print(f"✓ Exported YES samples as CSV ({len(yes_samples)} records): {yes_report_path}")

        # 輸出 NO 報告，若缺少 openpyxl 則自動改存 CSV
        no_report_path = output_path / f"review_report_NO_{timestamp}.xlsx"
        try:
            no_samples.to_excel(no_report_path, index=False, engine="openpyxl")
            print(f"✓ Exported NO samples ({len(no_samples)} records): {no_report_path}")
        except ModuleNotFoundError:
            no_report_path = output_path / f"review_report_NO_{timestamp}.csv"
            no_samples.to_csv(no_report_path, index=False, encoding="utf-8")
            print(f"✓ Exported NO samples as CSV ({len(no_samples)} records): {no_report_path}")

        return str(yes_report_path), str(no_report_path)

    def print_random_samples(
        self, 
        decision: str, 
        sample_size: int = 5,
        max_text_length: int = 200
    ):
        """
        在終端列印隨機樣本供快速預覽。

        Args:
            decision: "YES" 或 "NO"
            sample_size: 要列印的樣本數
            max_text_length: 文本截斷長度
        """
        sampled = self.sample_by_decision(decision, sample_size)

        print(f"\n{'=' * 80}")
        print(f"SAMPLE PREVIEW: {decision} DECISIONS ({len(sampled)} records)")
        print(f"{'=' * 80}\n")

        for idx, row in sampled.iterrows():
            original = row["original_text"][:max_text_length] + "..." if len(
                row["original_text"]
            ) > max_text_length else row["original_text"]
            
            augmented = row["augmented_text"][:max_text_length] + "..." if len(
                row["augmented_text"]
            ) > max_text_length else row["augmented_text"]
            
            print(f"Record #{idx + 1}")
            print(f"  Label: {row['label_name']}")
            print(f"  Original: {original}")
            print(f"  Augmented: {augmented}")
            print(f"  Reasoning: {row['qwen_reasoning'][:150]}...")
            print(f"  Decision: {row['qwen_decision']}")
            print()

    def get_statistics_by_decision(self) -> dict:
        """取得按決策類型分類的統計。"""
        stats = {
            "YES": len(self.df[self.df["qwen_decision"].str.upper() == "YES"]),
            "NO": len(self.df[self.df["qwen_decision"].str.upper() == "NO"]),
        }
        total = stats["YES"] + stats["NO"]
        stats["acceptance_rate"] = stats["YES"] / total if total > 0 else 0
        stats["total"] = total
        return stats

    def get_statistics_by_label(self) -> dict:
        """取得按標籤分類的統計。"""
        return (
            self.df.groupby("label_name")["qwen_decision"]
            .value_counts()
            .to_dict()
        )

    def print_full_statistics(self):
        """列印完整統計資訊。"""
        stats_decision = self.get_statistics_by_decision()
        stats_label = self.get_statistics_by_label()

        print("\n" + "=" * 80)
        print("AUGMENTATION REVIEW STATISTICS")
        print("=" * 80)
        print(f"Total Records: {stats_decision['total']}")
        print(f"YES Decisions: {stats_decision['YES']} ({stats_decision['acceptance_rate']:.2%})")
        print(f"NO Decisions: {stats_decision['NO']} ({1 - stats_decision['acceptance_rate']:.2%})")
        
        print("\nBreakdown by Label:")
        for label in sorted(self.df["label_name"].unique()):
            label_data = self.df[self.df["label_name"] == label]
            yes_count = len(label_data[label_data["qwen_decision"].str.upper() == "YES"])
            no_count = len(label_data[label_data["qwen_decision"].str.upper() == "NO"])
            total = len(label_data)
            yes_rate = yes_count / total if total > 0 else 0
            print(
                f"  {label}: {total} records | YES: {yes_count} ({yes_rate:.2%}) | NO: {no_count}"
            )
        print("=" * 80 + "\n")


def main():
    """
    主程序：使用方式
    
    Usage:
        python sample_augmentation_results.py <csv_path> [--yes-size 50] [--no-size 50]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Augmentation Result Sampling & Review Tool"
    )
    parser.add_argument("csv_path", help="Path to augmentation log CSV file")
    parser.add_argument(
        "--yes-size",
        type=int,
        default=50,
        help="Number of YES samples to extract (default: 50)",
    )
    parser.add_argument(
        "--no-size",
        type=int,
        default=50,
        help="Number of NO samples to extract (default: 50)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print N random samples preview in terminal (default: 0, skip preview)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for review reports (default: data)",
    )

    args = parser.parse_args()

    # 初始化審查工具
    reviewer = AugmentationSampleReviewer(args.csv_path)

    # 列印統計資訊
    reviewer.print_full_statistics()

    # 如果指定了預覽，先列印終端預覽
    if args.preview > 0:
        print(f"\n→ Terminal Preview (first {args.preview} samples of each type)\n")
        reviewer.print_random_samples("YES", min(args.preview, 5))
        reviewer.print_random_samples("NO", min(args.preview, 5))
        print("\n✓ Preview complete. Check the Excel files for full details.\n")

    # 生成人工審查報告
    yes_report, no_report = reviewer.generate_review_report(
        yes_sample_size=args.yes_size,
        no_sample_size=args.no_size,
        output_dir=args.output_dir,
    )

    print(f"\n✓ Review reports generated successfully!")
    print(f"  YES samples: {yes_report}")
    print(f"  NO samples: {no_report}")


if __name__ == "__main__":
    main()
