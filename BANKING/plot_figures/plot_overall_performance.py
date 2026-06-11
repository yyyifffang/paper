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

metrics = ["Macro F1", "Tail F1", "Accuracy"]

means = np.array([
    [0.7388, 0.7271, 0.7721],  # Passive
    [0.8223, 0.8139, 0.8348],  # Active
    [0.8795, 0.8717, 0.8803],  # Generate-only
    [0.8824, 0.8750, 0.8831],  # Generate-and-Verify
])

stds = np.array([
    [0.0049, 0.0085, 0.0017],
    [0.0184, 0.0190, 0.0178],
    [0.0098, 0.0088, 0.0099],
    [0.0104, 0.0096, 0.0101],
])

# =========================
# Plot settings
# =========================
x = np.arange(len(methods))
bar_width = 0.23

fig, ax = plt.subplots(figsize=(9, 5))

hatches = ["", "//", "\\\\"]

for i, metric in enumerate(metrics):
    offset = (i - 1) * bar_width

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

    # Add hatch patterns for better black-and-white readability
    for bar in bars:
        bar.set_hatch(hatches[i])

# =========================
# Axes and labels
# =========================
ax.set_ylabel("Score", fontsize=12)
ax.set_xlabel("Method", fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)

# Recommended range for clearer comparison
ax.set_ylim(0.70, 0.90)

ax.legend(
    loc="lower right",
    frameon=True,
    fontsize=10
)

ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
ax.set_axisbelow(True)

# Make layout clean
plt.tight_layout()

# =========================
# Save figures
# =========================
plt.savefig("overall_performance_grouped_bar.png", dpi=300, bbox_inches="tight")
plt.savefig("overall_performance_grouped_bar.pdf", bbox_inches="tight")

plt.show()