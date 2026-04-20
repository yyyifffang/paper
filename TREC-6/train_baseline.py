import torch
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch.nn.functional as F

from data_utils import prepare_datasets

def compute_metrics(pred):
    """計算評估指標：Accuracy 與 Macro-F1"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'macro_f1': f1}

def calculate_and_save_entropy(trainer, dataset, output_file="val_entropy.csv"):
    """計算驗證集的預測熵並保存到 CSV 文件"""
    print(f"開始計算entropy，目標資料集大小：{len(dataset)}...")

    # 取得模型預測 logits
    predictions = trainer.predict(dataset)
    logits = predictions.predictions
    true_labels = predictions.label_ids

    # 將 logits 轉換為機率分佈 (Softmax)
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()

    #計算entropy (以2為底，單位bit)
    #預測越不確定的樣本，entropy越高；預測越確定的樣本，entropy越低。
    entropies = entropy(probs, axis=-1, base=2)

    #將結果整理成 DataFrame 並保存為 CSV
    #這裡紀錄真實標籤、預測標籤、預測機率以及entropy值
    df_results = pd.DataFrame({
        'true_label': true_labels,
        'pred_label': probs.argmax(axis=-1),
        'entropy': entropies,
        'prob_class_0': probs[:, 0],
        'prob_class_1': probs[:, 1],
        'prob_class_2': probs[:, 2],
        'prob_class_3': probs[:, 3],
        'prob_class_4': probs[:, 4],
        'prob_class_5': probs[:, 5]
    })

    df_results.to_csv(output_file, index=False)
    print(f"Entropy計算完成並保存到 {output_file}")
    return df_results

def main():
    # 1. 準備資料集
    train_ds, val_ds, test_ds = prepare_datasets()

    # 2. 初始化模型
    # TREC-6 是6分類問題，所以 num_labels=6
    model_name = 'bert-base-uncased'
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)

    # 3. 設定訓練參數
    training_args = TrainingArguments(
        output_dir='./results_baseline',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        seed=42,
        logging_steps=10,
    )

    # 4. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # 5. 開始訓練 (僅使用40筆seed data)
    print("\n=== 開始訓練 Baseline 模型 ===")
    trainer.train()

    # 6. 最終測試 (在 2000 筆 test data 上評估真實效能)
    print("\n=== 評估 Test Set 效能 ===")
    test_results = trainer.evaluate(test_ds)
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Test Macro-F1: {test_results['eval_macro_f1']:.4f}")

    # 7. 計算 Validation Set 的 Entropy 並儲存
    # 這些數值將決定我們在後續「主動學習」時，要挑選哪些不確定性高的資料交給 LLM 處理
    print("\n=== 計算 Validation Set Entropy ===")
    calculate_and_save_entropy(trainer, val_ds, output_file="val_entropy.csv")

if __name__ == "__main__":
    main()
