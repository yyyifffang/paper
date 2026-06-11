import numpy as np
import matplotlib.pyplot as plt

# =========================
# Data: mean ± std over 3 random seeds
# =========================
methods = [
    "Passive",
    "Active",
    "Generate-only",
    "Generate-and-\nVerify"
]

metrics = ["Head F1", "Tail F1"]

means = np.array([
    [0.8175, 0.7271],  # Passive
    [0.8790, 0.8139],  # Active
    [0.9316, 0.8717],  # Generate-only
    [0.9320, 0.8750],  # Generate-and-Verify
])

stds = np.array([
    [0.0193, 0.0085],
    [0.0274, 0.0190],
    [0.0169, 0.0088],
    [0.0171, 0.0096],
])

# =========================
# Plot settings
# =========================
x = np.arange(len(methods))
bar_width = 0.32

fig, ax = plt.subplots(figsize=(8.5, 5))

hatches = ["", "//"]

for i, metric in enumerate(metrics):
    offset = (i - 0.5) * bar_width

    bars = ax.bar(
        x + offset,
        means[:, i],
        bar_width,
        yerr=stds[:, i],
        capsize=4,
        label=metric,
        edgecolor="black",
        linewidth=0.8
    )

    # Add hatch patterns for black-and-white readability
    for bar in bars:
        bar.set_hatch(hatches[i])

# =========================
# Axes and labels
# =========================
ax.set_xlabel("Method", fontsize=12)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)

ax.set_ylim(0.70, 0.97)

ax.legend(loc="lower right", frameon=True, fontsize=10)

ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()

# =========================
# Save figures
# =========================
plt.savefig("head_tail_f1_grouped_bar.png", dpi=300, bbox_inches="tight")
plt.savefig("head_tail_f1_grouped_bar.pdf", bbox_inches="tight")

plt.show()