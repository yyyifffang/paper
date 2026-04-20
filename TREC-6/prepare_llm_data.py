import pandas as pd
from datasets import load_dataset
from data_utils import partition_data

def merge_text_and_extract_hard_samples(k=40):
    print("正在還原原始文字資料...")
    # 1. 重新用相同的 Random Seed 取得原始文字
    dataset = load_dataset("trec", trust_remote_code=True)
    df = dataset['train'].to_pandas()
    df['label'] = df['coarse_label']  
    df_clean = df[['text', 'label']]

    # 取得與 Baseline 相同的 Validation Set
    _, df_val, _ = partition_data(df_clean, random_seed=42)

    # 2. 讀取之前計算的 Entropy 紀錄
    df_entropy = pd.read_csv("val_entropy.csv")

    # 3. 將原始文字對齊合併
    df_entropy['text'] = df_val['text']

    # 4. 根據 Entropy 值排序，選擇 Top-K 
    df_hard = df_entropy.sort_values(by='entropy', ascending=False).head(k)

    # 5. 儲存並準備交給LLM
    output_file = "hard_samples_trec6.csv"
    df_hard.to_csv(output_file, index=False)

    print(f"\n成功將文字合併並儲存至 {output_file}！")

if __name__ == "__main__":
    merge_text_and_extract_hard_samples(k=40)
    
