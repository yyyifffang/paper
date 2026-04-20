# Data Augmentation 日誌系統 - 使用指南

這套工具用來追蹤和審查 Data Augmentation 過程中的所有細節，方便進行品質控制。

## 📦 文件清單

| 檔案 | 用途 |
|------|------|
| `data_augmentation_logger.py` | 日誌記錄系統（CSV/Excel 匯出） |
| `sample_augmentation_results.py` | 採樣審查工具 |
| `augmentation_logging_example.py` | 整合示例和使用說明 |

## 🚀 快速開始

### 1. 使用日誌系統記錄結果

```python
from data_augmentation_logger import DataAugmentationLogger

# 建立日誌記錄器
logger = DataAugmentationLogger(output_dir="data")

# 記錄單筆擴增結果
logger.log_augmentation_result(
    iteration=1,
    original_text="The parties to this agreement are...",
    label=0,
    label_name="Parties",
    augmented_text="Company A and Company B are parties to...",
    qwen_reasoning="The augmented text preserves legal intent.",
    qwen_decision="YES",
    status="Accepted"
)

# 定期檢查統計
logger.print_statistics()

# 完成後匯出
logger.export_to_excel()
```

### 2. 從日誌中採樣審查結果

```bash
# 生成 50 筆 YES 和 50 筆 NO 的樣本報告
python sample_augmentation_results.py data/augmentation_log_YYYYMMDD_HHMMSS.csv \
    --yes-size 50 --no-size 50 --output-dir data/reviews

# 帶有終端預覽
python sample_augmentation_results.py data/augmentation_log_YYYYMMDD_HHMMSS.csv \
    --yes-size 50 --no-size 50 --preview 5
```

## 📋 日誌欄位說明

生成的 CSV/Excel 文件包含以下欄位：

| 欄位 | 說明 |
|------|------|
| `iteration` | 擴增的迭代次數 |
| `original_text` | 原始的 LEDGAR 句子 |
| `label` | 標籤編號（0-11） |
| `label_name` | 標籤名稱（如 "Parties"） |
| `augmented_text` | Llama-3 生成的句子 |
| `qwen_reasoning` | Qwen 的推理過程 |
| `qwen_decision` | YES 或 NO |
| `status` | Accepted 或 Rejected |
| `timestamp` | 記錄時間戳 |

## 📊 統計資訊示例

日誌系統會自動計算：

```
AUGMENTATION LOG STATISTICS
==================================================
Total Records: 1500
Accepted: 1245 (83%)
Rejected: 255 (17%)

By Label:
  Parties: 150
  Agreement Date: 180
  Effective Date: 175
  Termination Clause: 190
  Liability: 155
==================================================
```

## 🔍 人工審查流程

### 建議的審查步驟：

1. **執行擴增過程** → 生成 augmentation_log_*.csv

2. **提取樣本**
   ```bash
   python sample_augmentation_results.py data/augmentation_log_*.csv \
       --yes-size 50 --no-size 50 --output-dir data/reviews
   ```

3. **審查生成的 Excel 檔案**
   - `review_report_YES_*.xlsx` - 檢查 YES 決策的邏輯是否正確
   - `review_report_NO_*.xlsx` - 檢查 NO 決策的邏輯是否正確

4. **驗證決策邏輯**
   - 抽樣的 100 筆資料（50 YES + 50 NO）
   - 檢查 Qwen 的推理是否與你的期望一致
   - 確認品質控制標準得到滿足

5. **品質確認後**
   - 如果邏輯符合預期 → 放心執行完整擴增
   - 如果發現問題 → 調整 prompt 或參數，重新執行

## 💡 實務建議

### 採樣規模

- **50 + 50 = 100 筆** 是合理的人工審查規模
- 對於 10,000+ 筆擴增資料，100 筆樣本代表約 1% 的覆蓋率
- 足以發現 10% 以上的系統性錯誤（基於統計抽樣理論）

### 品質信心度

| 採樣量 | 涵蓋率 | 信心度 |
|--------|--------|--------|
| 50 + 50 | ~1% | 80% (95% CI) |
| 100 + 100 | ~2% | 90% (95% CI) |
| 200 + 200 | ~4% | 95% (95% CI) |

### 檢查清單

審查時使用以下清單：

- [ ] **法律術語** - 使用了正確的法律詞彙嗎？
- [ ] **句構完整** - 句子邏輯完整且通順嗎？
- [ ] **無對話內容** - 避免了會話內容嗎？
- [ ] **條款關聯性** - 保留了原始條款的法律意圖嗎？
- [ ] **標籤一致性** - 擴增文本與標籤相符嗎？

## 🔧 進階用法

### 載入既存日誌

```python
from data_augmentation_logger import DataAugmentationLogger

logger = DataAugmentationLogger(output_dir="data")
logger.load_csv("data/augmentation_log_old.csv")

# 檢查統計
stats = logger.get_statistics()
print(f"總筆數: {stats['total']}")
print(f"接受率: {stats['acceptance_rate']:.2%}")
```

### 從命令行查看統計

```bash
python sample_augmentation_results.py data/augmentation_log_*.csv \
    --output-dir data/reports
```

會輸出完整的統計信息。

## 📝 與 simple_active_learning_LR.py 的整合

在你的主程式中整合日誌系統：

```python
from data_augmentation_logger import DataAugmentationLogger

def run_proposed_framework(...):
    # 初始化日誌
    logger = DataAugmentationLogger(output_dir=DATA_DIR)
    
    for iteration in range(1, n_iterations + 1):
        # ... 現有代碼 ...
        
        # 在驗證後記錄結果
        for text, label, reasoning, decision in validation_results:
            logger.log_augmentation_result(
                iteration=iteration,
                original_text=original_text,
                label=label,
                label_name=label_mapping[label],
                augmented_text=text,
                qwen_reasoning=reasoning,
                qwen_decision=decision,
            )
    
    # 完成後匯出
    logger.export_to_excel()
    logger.print_statistics()
```

## 🐛 常見問題

### Q: 如何處理記錄過程中出現的錯誤？

A: 日誌系統會持續寫入 CSV 檔案，即使出現錯誤也不會丟失數據。每筆記錄在生成時都會立即寫入。

### Q: 能否修改已記錄的內容？

A: 建議使用 pandas 編輯 CSV 檔案：
```python
import pandas as pd

df = pd.read_csv("data/augmentation_log_*.csv")
# 修改...
df.to_csv("data/augmentation_log_updated.csv", index=False)
```

### Q: 如何合併多個日誌檔案？

A: 使用 pandas 合併：
```python
import pandas as pd
import glob

dfs = [pd.read_csv(f) for f in glob.glob("data/augmentation_log_*.csv")]
merged = pd.concat(dfs, ignore_index=True)
merged.to_csv("data/augmentation_log_merged.csv", index=False)
```

## 📧 支援

如有問題或建議，請參考主程式中的相關函數定義。

---

**建立日期**: 2024  
**版本**: 1.0  
**相容性**: Python 3.8+，需要 pandas, openpyxl
