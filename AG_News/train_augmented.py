import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from data_utils import partition_data

def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    fl_metric = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = fl_metric.compute(predictions=predictions, references=labels, average="macro")

    return {"accuracy": acc["accuracy"], "f1_macro": f1["f1"]}

def main():
    print("1. 準備合併訓練資料...")
    # 取得與 baseline 相同的 seed, validation, test 切分
    dataset = load_dataset("sh0416/ag_news")
    df = dataset['train'].to_pandas()
    df['label'] = df['label'] - 1
    df['text'] = df['title'] + ' - ' + df['description']
    df_clean = df[['text', 'label']]

    df_seed, df_val, df_test = partition_data(df_clean, random_seed=42)

    # 讀取 LLM 擴增資料
    df_aug = pd.read_csv("augmented_samples.csv")

    # 合併 seed 資料與 LLM 擴增資料
    df_combined_train = pd.concat([df_seed, df_aug[['text', 'label']]], ignore_index=True)

    print(f" => 原始 Seed 資料: {len(df_seed)} 筆")
    print(f" => LLM 擴增資料: {len(df_aug)} 筆")
    print(f" => 最終合併訓練集: {len(df_combined_train)} 筆\n")

    # 轉換為 Hugging Face Dataset
    train_dataset = Dataset.from_pandas(df_combined_train)
    val_dataset = Dataset.from_pandas(df_val)
    test_dataset = Dataset.from_pandas(df_test)

    print("2. 載入 Tokenizer 並處理文字...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    traian_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)

    print("3. 載入全新的 BERT 模型準備訓練...")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

    training_args = TrainingArguments(
        output_dir="./results_augmented",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="./logs_aug",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=traian_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics
    )

    print("4. 開始進行擴增資料微調 (Fine-tuning)...")
    trainer.train()

    print("\n5. 在 Test Set 上進行最終評估...")
    test_results = trainer.evaluate(test_tokenized)

    print("\n" + "="*40)
    print("🏆 擴增後模型 (Augmented Model) 最終測試成績：")
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Test Macro-F1: {test_results['eval_f1_macro']:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()