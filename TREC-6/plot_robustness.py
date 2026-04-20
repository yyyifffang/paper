import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("載入多個Seed的歷程數據...")
    df = pd.read_csv("data/hybrid_learning_history_all_seeds_lambda0.0005.csv")

    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial']  

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    #1. accuracy vs. cumulative cost
    # errorbar='sd' 會自動計算多個 seed 的標準差並畫出陰影
    sns.lineplot(
        data=df, x='cost', y='accuracy',
        marker='o', ax=ax1, errorbar='sd',
        color='#1f77b4', label='Hybrid Strategy (Mean ± SD)'
    )
    ax1.set_title("Model Accuracy vs. Cumulative Cost", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Cumulative Cost ( Generated Samples )", fontsize=12)
    ax1.set_ylabel("Validation Accuracy", fontsize=12)
    ax1.legend()

    #2. utility function vs. cumulative cost
    sns.lineplot(
        data=df, x='cost', y='utility',
        marker='s', ax=ax2, errorbar='sd',
        color='#2ca02c', label='Utility (U = Acc - λ*Cost)'
    )
    ax2.set_title("Utility Function vs. Cumulative Cost", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Cumulative Cost ( Generated Samples )", fontsize=12)
    ax2.set_ylabel("Utility Score (U)", fontsize=12)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("plots/robustness_check_plot_lambda0.0005.png", dpi=300)
    print("Plot saved to plots/robustness_check_plot_lambda0.0005.png")

if __name__ == "__main__":
    main()  