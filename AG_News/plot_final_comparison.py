import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("1. 讀取三組實驗數據...")
    try:
        df_active = pd.read_csv("data/active_learning_history.csv")
        df_random = pd.read_csv("data/baseline_random_history.csv")
        df_hybrid = pd.read_csv("data/hybrid_learning_history.csv")
    except FileNotFoundError:
        print("❌ 找不到 CSV 檔案！請確認三份檔案都存在於 data/ 資料夾下。")
        return

    print("2. 繪製終極對比圖表...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- 圖表 1: Accuracy vs Cost ---
    ax1.plot(df_active['cost'], df_active['accuracy'], marker='o', color='red', linewidth=2, label='Pure Active (Entropy Only)')
    ax1.plot(df_random['cost'], df_random['accuracy'], marker='s', color='blue', linewidth=2, linestyle='--', label='Pure Random Baseline')
    ax1.plot(df_hybrid['cost'], df_hybrid['accuracy'], marker='*', color='green', markersize=12, linewidth=3, label='Hybrid Strategy (Proposed)')
    
    ax1.set_title("Model Accuracy vs. Cost", fontsize=15, fontweight='bold')
    ax1.set_xlabel("Cumulative Cost (Generated Samples)", fontsize=12)
    ax1.set_ylabel("Validation Accuracy", fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(fontsize=11, loc='lower right')

    # --- 圖表 2: Utility vs Cost ---
    ax2.plot(df_active['cost'], df_active['utility'], marker='o', color='red', linewidth=2, label='Pure Active')
    ax2.plot(df_random['cost'], df_random['utility'], marker='s', color='blue', linewidth=2, linestyle='--', label='Pure Random')
    ax2.plot(df_hybrid['cost'], df_hybrid['utility'], marker='*', color='green', markersize=12, linewidth=3, label='Hybrid Strategy')
    
    # 標示 Hybrid 的最佳停止點
    max_u_hybrid = df_hybrid.loc[df_hybrid['utility'].idxmax()]
    ax2.annotate('Optimal Early Stopping\n(Hybrid)', 
                 xy=(max_u_hybrid['cost'], max_u_hybrid['utility']),
                 xytext=(max_u_hybrid['cost']-120, max_u_hybrid['utility']+0.03),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, fontweight='bold', color='green')

    ax2.set_title("Utility Function (U) vs. Cost", fontsize=15, fontweight='bold')
    ax2.set_xlabel("Cumulative Cost (Generated Samples)", fontsize=12)
    ax2.set_ylabel("Utility Score (U = Accuracy - λ*Cost)", fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(fontsize=11, loc='lower right')

    # 儲存圖片
    plt.tight_layout()
    output_path = "plots/final_hybrid_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ 終極對比圖已成功儲存至: {output_path}")

if __name__ == "__main__":
    main()