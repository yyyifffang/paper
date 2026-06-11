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

means = np.array([38.00, 40.00, 32.33, 37.00])
stds = np.array([3.46, 0.00, 1.15, 5.20])

max_iteration = 40

# =========================
# Plot
# =========================
x = np.arange(len(methods))

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    x,
    means,
    yerr=stds,
    capsize=5,
    width=0.55,
    edgecolor="black",
    linewidth=0.8
)

# Add hatch patterns for black-and-white readability
hatches = ["", "//", "\\\\", ".."]
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

# Reference line for maximum iteration
ax.axhline(
    y=max_iteration,
    linestyle="--",
    linewidth=1.2,
    label=f"Max iteration = {max_iteration}"
)

# Add value labels
for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(
        i,
        mean + std + 0.6,
        f"{mean:.2f} ± {std:.2f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

# =========================
# Axes settings
# =========================
ax.set_xlabel("Method", fontsize=12)
ax.set_ylabel("Plateau Stop Iteration", fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)

ax.set_ylim(0, 45)
ax.set_yticks(np.arange(0, 46, 5))

ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
ax.set_axisbelow(True)

ax.legend(loc="lower right", frameon=True, fontsize=10)

plt.tight_layout()

# =========================
# Save figures
# =========================
plt.savefig("plateau_stop_iteration_bar.png", dpi=300, bbox_inches="tight")
plt.savefig("plateau_stop_iteration_bar.pdf", bbox_inches="tight")

plt.show()