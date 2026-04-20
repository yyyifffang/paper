import pandas as pd

def select_top_k_uncertain_samples(csv_file="val_entropy.csv", k=40):
    # 1. 讀取 Entropy 紀錄
    df = pd.read_csv(csv_file)

    # 2. 根據 Entropy 值排序，選擇前 k 個不確定的樣本
    df_sorted = df.sort_values(by='entropy', ascending=False)

    # 3. 取出前 k 個樣本
    top_k_samples = df_sorted.head(k)

    print(f"已挑選出 {k} 筆最困難的樣本。")
    print("前 5 筆最高 Entropy 分數 (請確認分數是否大於 1.0 以上)：")
    print(top_k_samples[['true_label', 'pred_label', 'entropy']].head())

    # 4. 儲存並準備交給LLM
    output_file = "hard_samples_for_llm.csv"
    top_k_samples.to_csv(output_file, index=False)
    print(f"\n檔案已儲存至：{output_file}")

    return top_k_samples

if __name__ == "__main__":
    select_top_k_uncertain_samples(k=40)