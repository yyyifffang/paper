import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("1. 讀取實驗數據...")
    try:
        df_active = pd.read_csv("data/active_learning_history.csv")
        df_random = pd.read_csv("data/baseline_random_history.csv")
    except FileNotFoundError:
        print("❌ 找不到 CSV 檔案！請確認你已經把它們放在 data/ 資料夾下。")
        return

    print("2. 繪製圖表...")
    # 設定圖表大小與子圖表 (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- 圖表 1: Accuracy vs Cost ---
    ax1.plot(df_active['cost'], df_active['accuracy'], marker='o', color='red', linewidth=2, label='Active Learning (Entropy)')
    ax1.plot(df_random['cost'], df_random['accuracy'], marker='s', color='blue', linewidth=2, linestyle='--', label='Random Baseline')
    ax1.set_title("Model Accuracy vs. Cost", fontsize=14)
    ax1.set_xlabel("Cumulative Cost (Generated Samples)", fontsize=12)
    ax1.set_ylabel("Validation Accuracy", fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(fontsize=11)

    # --- 圖表 2: Utility vs Cost ---
    ax2.plot(df_active['cost'], df_active['utility'], marker='o', color='red', linewidth=2, label='Active Learning (Entropy)')
    ax2.plot(df_random['cost'], df_random['utility'], marker='s', color='blue', linewidth=2, linestyle='--', label='Random Baseline')
    
    # 標示最高點
    max_u_random = df_random.loc[df_random['utility'].idxmax()]
    ax2.annotate('Optimal Stopping Point', 
                 xy=(max_u_random['cost'], max_u_random['utility']),
                 xytext=(max_u_random['cost']-150, max_u_random['utility']+0.02),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10)

    ax2.set_title("Utility Function (U) vs. Cost", fontsize=14)
    ax2.set_xlabel("Cumulative Cost (Generated Samples)", fontsize=12)
    ax2.set_ylabel("Utility Score (U = Accuracy - λ*Cost)", fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(fontsize=11)

    # 儲存圖片
    plt.tight_layout()
    output_path = "plots/experiment_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ 雙曲線對比圖已成功儲存至: {output_path}")

if __name__ == "__main__":
    main()