import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Log files
#    若你在自己的電腦執行，請把這三個檔案放在同一資料夾，
#    或改成你自己的完整路徑。
# ============================================================
LOG_FILES = [
    "experiment_log_Banking77_Refactored_20260601_080143.txt",
    "experiment_log_Banking77_Refactored_20260601_121519.txt",
    "experiment_log_Banking77_Refactored_20260601_142533.txt",
]


# ============================================================
# 2. Method mapping
# ============================================================
METHOD_TITLE_MAP = {
    "PASSIVE LEARNING EXPERIMENT": "Passive",
    "ACTIVE LEARNING BASELINE": "Active",
    "PROPOSED FRAMEWORK (Generate-only)": "Generate-only",
    "PROPOSED FRAMEWORK (Generate-and-Verify)": "Generate-and-Verify",
}

METHOD_ORDER = [
    "Passive",
    "Active",
    "Generate-only",
    "Generate-and-Verify",
]


# ============================================================
# 3. Parse one log file
# ============================================================
def parse_log_file(log_path: str | Path) -> pd.DataFrame:
    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Cannot find log file: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    # Parse random seed
    seed_match = re.search(r"Random seed:\s*(\d+)", text)
    seed = int(seed_match.group(1)) if seed_match else None

    current_method = None
    current_iteration = None
    records = []

    iter_pattern = re.compile(r"--- Iteration\s+(\d+)\s+---")
    perf_pattern = re.compile(
        r"Performance Summary\s*\|\s*"
        r"Macro F1:\s*([0-9.]+)\s*\|\s*"
        r"Head Macro F1 \(0-9\):\s*([0-9.]+)\s*\|\s*"
        r"Tail Macro F1 \(10-76\):\s*([0-9.]+)\s*\|\s*"
        r"Accuracy:\s*([0-9.]+)"
    )

    for line in lines:
        # Detect method section
        for title_key, method_name in METHOD_TITLE_MAP.items():
            if title_key in line:
                current_method = method_name
                current_iteration = None
                break

        # Detect iteration
        iter_match = iter_pattern.search(line)
        if iter_match:
            current_iteration = int(iter_match.group(1))
            continue

        # Detect performance summary
        perf_match = perf_pattern.search(line)
        if perf_match and current_method is not None and current_iteration is not None:
            macro_f1 = float(perf_match.group(1))
            head_f1 = float(perf_match.group(2))
            tail_f1 = float(perf_match.group(3))
            accuracy = float(perf_match.group(4))

            records.append(
                {
                    "seed": seed,
                    "method": current_method,
                    "iteration": current_iteration,
                    "macro_f1": macro_f1,
                    "head_f1": head_f1,
                    "tail_f1": tail_f1,
                    "accuracy": accuracy,
                    "log_file": log_path.name,
                }
            )

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(f"No performance records parsed from: {log_path}")

    return df


# ============================================================
# 4. Parse all logs
# ============================================================
all_runs = []

for log_file in LOG_FILES:
    df_one = parse_log_file(log_file)
    all_runs.append(df_one)

df = pd.concat(all_runs, ignore_index=True)

# Save parsed raw curve data for checking
df.to_csv("parsed_learning_curve_by_seed.csv", index=False)

print("Parsed records:")
print(df.groupby(["seed", "method"])["iteration"].agg(["min", "max", "count"]))


# ============================================================
# 5. Aggregate mean and std over seeds
# ============================================================
summary = (
    df.groupby(["method", "iteration"], as_index=False)
    .agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        tail_f1_mean=("tail_f1", "mean"),
        tail_f1_std=("tail_f1", "std"),
        head_f1_mean=("head_f1", "mean"),
        head_f1_std=("head_f1", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
    )
)

summary.to_csv("learning_curve_mean_std.csv", index=False)


# ============================================================
# 6. Plot helper
# ============================================================
def plot_learning_curve(
    summary_df: pd.DataFrame,
    metric_mean_col: str,
    metric_std_col: str,
    ylabel: str,
    output_prefix: str,
):
    fig, ax = plt.subplots(figsize=(8.5, 5))

    line_styles = {
        "Passive": "-",
        "Active": "--",
        "Generate-only": "-.",
        "Generate-and-Verify": ":",
    }

    markers = {
        "Passive": "o",
        "Active": "s",
        "Generate-only": "^",
        "Generate-and-Verify": "D",
    }

    for method in METHOD_ORDER:
        sub = summary_df[summary_df["method"] == method].sort_values("iteration")

        if sub.empty:
            print(f"Warning: no data for method: {method}")
            continue

        x = sub["iteration"].to_numpy()
        y = sub[metric_mean_col].to_numpy()
        y_std = sub[metric_std_col].fillna(0).to_numpy()

        ax.plot(
            x,
            y,
            label=method,
            linestyle=line_styles.get(method, "-"),
            marker=markers.get(method, "o"),
            markevery=5,
            linewidth=1.8,
            markersize=4,
        )

        ax.fill_between(
            x,
            y - y_std,
            y + y_std,
            alpha=0.15,
        )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Learning curve 從初期開始，建議保留 0.0 起點
    ax.set_xlim(1, 40)
    ax.set_ylim(0.0, 0.95)

    ax.set_xticks(np.arange(1, 41, 5))

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=9,
    )

    plt.tight_layout()

    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")

    plt.close(fig)


# ============================================================
# 7. Generate figures
# ============================================================
plot_learning_curve(
    summary_df=summary,
    metric_mean_col="macro_f1_mean",
    metric_std_col="macro_f1_std",
    ylabel="Macro F1",
    output_prefix="macro_f1_learning_curve",
)

plot_learning_curve(
    summary_df=summary,
    metric_mean_col="tail_f1_mean",
    metric_std_col="tail_f1_std",
    ylabel="Tail F1",
    output_prefix="tail_f1_learning_curve",
)

print("Saved figures:")
print("- macro_f1_learning_curve.png")
print("- macro_f1_learning_curve.pdf")
print("- tail_f1_learning_curve.png")
print("- tail_f1_learning_curve.pdf")
print()
print("Saved parsed data:")
print("- parsed_learning_curve_by_seed.csv")
print("- learning_curve_mean_std.csv")