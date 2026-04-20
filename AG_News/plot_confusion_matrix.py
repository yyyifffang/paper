import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers.trainer_utils import get_last_checkpoint
from data_utils import partition_data

def main():
    print("1. 載入 Test Set 測試資料...")
    dataset = load_dataset("sh0416/ag_news")
    df = dataset['train'].to_pandas()
    df['label'] = df['label'] - 1
    df['text'] = df['title'] + ' - ' + df['description']
    df_clean = df[['text', 'label']]
    
    # 嚴格固定 seed，確保測試集跟之前一模一樣
    _, _, df_test = partition_data(df_clean, random_seed=42)
    test_dataset = Dataset.from_pandas(df_test)

    print("2. 載入 Tokenizer 並處理文字...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)

    print("3. 自動尋找剛剛訓練好的最佳模型權重...")
    # Trainer 預設會把權重存在這個資料夾下的 checkpoint 裡
    checkpoint = get_last_checkpoint("./results_augmented")
    if checkpoint is None:
        print("❌ 找不到訓練好的模型權重！請確認 ./results_augmented 資料夾是否存在。")
        return
    print(f"✅ 成功找到權重: {checkpoint}")
    
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=4)
    trainer = Trainer(model=model)

    print("4. 對 Test Set (2000筆) 進行預測...")
    predictions_output = trainer.predict(test_tokenized)
    y_pred = np.argmax(predictions_output.predictions, axis=-1)
    y_true = predictions_output.label_ids

    print("5. 繪製並儲存混淆矩陣...")
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    cm = confusion_matrix(y_true, y_pred)
    
    # 繪製視覺化熱力圖
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label (模型預測)')
    plt.ylabel('True Label (真實標籤)')
    plt.title('Confusion Matrix - Augmented BERT Model')
    
    output_file = "confusion_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # 在終端機印出文字版矩陣
    print(f"\n✅ 視覺化圖片已成功儲存為: {output_file}")
    print("\n" + "="*50)
    print("📊 文字版混淆矩陣 (Row: 真實類別, Column: 預測類別)")
    print("="*50)
    df_cm = pd.DataFrame(cm, index=[f"True: {l}" for l in labels], columns=[f"Pred: {l}" for l in labels])
    print(df_cm)
    print("="*50)

if __name__ == "__main__":
    main()