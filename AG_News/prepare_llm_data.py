import pandas as pd
from datasets import load_dataset
from data_utils import partition_data

def merge_text_and_extract_hard_samples(k=40):
    print("正在還原原始文字資料...")
    # 1. 重新用相同的 Random Seed 取得原始文字
    dataset = load_dataset("sh0416/ag_news")
    df = dataset['train'].to_pandas()
    df['label'] = df['label'] - 1  # 調整標籤從 0 開始
    df['text'] = df['title'] + ' - ' + df['description']  # 合併title與description
    df_clean = df[['text', 'label']] # 保留需要的欄位

    # 取得與 Baseline 相同的 Validation Set
    _, df_val, _ = partition_data(df_clean, random_seed=42)

    # 2. 讀取之前計算的 Entropy 紀錄
    df_entropy = pd.read_csv("val_entropy.csv")

    # 3. 將原始文字對齊合併
    df_entropy['text'] = df_val['text']

    # 4. 根據 Entropy 值排序，選擇 Top-K 
    df_hard = df_entropy.sort_values(by='entropy', ascending=False).head(k)

    # 5. 儲存並準備交給LLM
    output_file = "hard_samples_with_text.csv"
    df_hard.to_csv(output_file, index=False)

    print(f"\n成功將文字合併並儲存至 {output_file}！")
    print("以下是 Top-2 最困難樣本的真實面貌：")
    for i , row in df_hard.head(2).iterrows():
        print(f"\n[真實標籤: {row['true_label']} | 預測: {row['pred_label']}] Entropy: {row['entropy']:.4f}")
        print(f"文本內容: {row['text']}")

if __name__ == "__main__":
    merge_text_and_extract_hard_samples(k=40)
    
