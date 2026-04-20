import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("載入多個Seed的歷程數據...")
    df = pd.read_csv("data/hybrid_learning_history_all_seeds_lambda0.0005.csv")

    # --- 🛡️ 軍師的數據修復魔法：處理 Early Stopping 造成的斷層 ---
    # 找出實驗中走到最遠的 cost 節點 (例如 600)
    all_costs = sorted(df['cost'].unique())
    padded_data = []
    LAMBDA_PENALTY = 0.0005 # 確保與主程式設定一致

    for seed in df['seed'].unique():
        # 取出該 seed 的所有歷史紀錄並排序
        seed_df = df[df['seed'] == seed].sort_values('cost')
        last_acc = seed_df['accuracy'].iloc[-1]
        last_cost = seed_df['cost'].iloc[-1]

        for cost in all_costs:
            if cost <= last_cost:
                # 該 Seed 在這個 Cost 還有在跑，直接取用真實數據
                row = seed_df[seed_df['cost'] == cost].to_dict('records')[0]
            else:
                # 該 Seed 已經觸發 Early Stopping 煞車了！
                # 【反事實模擬】：假設它沒煞車，繼續盲目擴增。
                # Accuracy 保持平盤，但 Utility 會因為持續扣錢 (Cost) 而下降
                simulated_utility = last_acc - (LAMBDA_PENALTY * cost)
                row = {
                    'iter': seed_df['iter'].max() + 1, 
                    'accuracy': last_acc, 
                    'cost': cost, 
                    'utility': simulated_utility,
                    'seed': seed
                }
            padded_data.append(row)

    # 用補齊後的完美數據來畫圖
    df_padded = pd.DataFrame(padded_data)
    # -----------------------------------------------------------

    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial']  

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. accuracy vs. cumulative cost
    # 注意：這裡改用 df_padded
    sns.lineplot(
        data=df_padded, x='cost', y='accuracy',
        marker='o', ax=ax1, errorbar='sd',
        color='#1f77b4', label='Hybrid Strategy (Mean ± SD)'
    )
    ax1.set_title("Model Accuracy vs. Cumulative Cost", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Cumulative Cost ( Generated Samples )", fontsize=12)
    ax1.set_ylabel("Validation Accuracy", fontsize=12)
    ax1.legend()

    # 2. utility function vs. cumulative cost
    # 注意：這裡改用 df_padded
    sns.lineplot(
        data=df_padded, x='cost', y='utility',
        marker='s', ax=ax2, errorbar='sd',
        color='#2ca02c', label='Utility (U = Acc - λ*Cost)'
    )
    ax2.set_title("Utility Function vs. Cumulative Cost", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Cumulative Cost ( Generated Samples )", fontsize=12)
    ax2.set_ylabel("Utility Score (U)", fontsize=12)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("plots/robustness_check_plot_lambda0.0005_fixed.png", dpi=300)
    print("Plot saved to plots/robustness_check_plot_lambda0.0005_fixed.png")

if __name__ == "__main__":
    main()