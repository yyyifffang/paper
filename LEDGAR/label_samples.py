import matplotlib.pyplot as plt
import re
from pathlib import Path

# 1. 讀取並解析 Log 數據
def parse_log(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Log file not found: {file_path}")

    with file_path.open('r', encoding='utf-8') as f:
        content = f.read()

    # 以明確標題切分，避免被分隔線數量影響而錯位
    group_sections = {
        "Group 1 (Baseline)": ("ACTIVE LEARNING EXPERIMENT (Entropy)", "Group 2 (Unfiltered):"),
        "Group 2 (Unfiltered)": ("Group 2 (Unfiltered):", "Group 3 (Filtered/Proposed):"),
        "Group 3 (Filtered/Proposed)": ("Group 3 (Filtered/Proposed):", "ABLATION STUDY SUMMARY"),
    }

    data = {}

    for group_name, (start_marker, end_marker) in group_sections.items():
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker, start_idx if start_idx != -1 else 0)

        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            data[group_name] = ([], [])
            continue

        group_content = content[start_idx:end_idx]
        f1_scores = []
        sample_counts = []
        
        # 初始狀態：所有組別起始於 300 樣本
        # 根據 Log：Iteration 1 的 F1 是在該次採樣「前」的結果
        # 所以 Iteration 1 F1 對應到的是前一次的 Total Labeled (起始為 300)
        
        # 提取所有 F1 和樣本數
        f1_matches = re.findall(r"Test - F1: ([\d\.]+)", group_content)
        sample_matches = re.findall(r"Total labeled samples: (\d+)", group_content)
        
        # 邏輯對齊：
        # Iteration 1 的 F1 (0.2826) 對應 300 樣本
        # Iteration 2 的 F1 對應 Iteration 1 結束後的樣本數
        current_samples = 300
        for j in range(len(f1_matches)):
            f1_scores.append(float(f1_matches[j]))
            sample_counts.append(current_samples)
            if j < len(sample_matches):
                current_samples = int(sample_matches[j])
        
        data[group_name] = (sample_counts, f1_scores)
    
    return data

# 2. 繪圖
def plot_learning_curve(data):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use('seaborn-v0_8-paper')
    
    # 配色方案：強調藍色，淡化其他
    # 藍: #1f77b4 (強調), 紅: #d62728 (次要), 灰: #95a5a6 (背景)
    colors = ['#95a5a6', '#c0392b', '#1f77b4'] 
    markers = ['o', 's', '^']
    
    for i, (name, (x, y)) in enumerate(data.items()):
        # Group 3 (i=2) 加粗且完全不透明，其餘變細且透明
        lw = 2.2 if i == 2 else 1.2
        alpha = 1.0 if i == 2 else 0.5
        
        plt.plot(x, y, label=name, color=colors[i], marker=markers[i], 
                 markersize=4, linewidth=lw, alpha=alpha, markevery=3)

    plt.title('Sample Efficiency on LEDGAR (98 Classes)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Total Labeled Samples (Initial + Augmented)', fontsize=11)
    plt.ylabel('Test F1 Score', fontsize=11)
    
    # 1. 調整 Baseline Ceiling：放在左側邊界上方
    plt.axhline(y=0.6715, color='#7f8c8d', linestyle='--', linewidth=1, alpha=0.6)
    plt.text(50, 0.675, 'Baseline (0.67)', ha='left', color='#7f8c8d', fontsize=9, fontweight='bold')

    # 2. 精簡 ΔF1 標註：使用垂直 Bracket 或簡潔箭頭
    plt.annotate('', xy=(2000, 0.632), xytext=(2000, 0.562),
                 arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=1.5))
    plt.text(2100, 0.597, r'$\Delta F1 \approx 0.07$', color='#27ae60', fontsize=10, fontweight='bold')

    # 3. 改進 Cross-over Point：使用帶弧度的引線到下方空白處
    plt.plot(3570, 0.689, marker='o', markersize=8, markerfacecolor='none', markeredgecolor='black', mew=1.5)
    plt.annotate('Surpasses Baseline\nw/ fewer samples', 
                 xy=(3570, 0.689), xytext=(4500, 0.52),
                 arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-0.15", color='black'),
                 fontsize=9, color='black', ha='center')

    # 4. 圖例移至右下角，並增加外框透明度
    plt.legend(frameon=True, loc='lower right', fontsize=9, framealpha=0.9)

    # 調整範圍與細節
    plt.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.xlim(0, 6600)
    plt.ylim(0.25, 0.78)
    plt.tight_layout()
    
    plt.savefig('ledgar_final_plot.png', dpi=300)

# 執行
if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    default_log = base_dir / 'data' / 'logs' / 'experiment_log_ledgar_ablation_study_20260412_153416.txt'

    log_data = parse_log(default_log)
    plot_learning_curve(log_data)