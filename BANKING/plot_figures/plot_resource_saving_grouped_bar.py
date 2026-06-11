import numpy as np
import matplotlib.pyplot as plt

# =========================
# Experimental setting
# =========================
max_iteration = 40
batch_size = 40

methods = [
    "Passive",
    "Active",
    "Generate-only",
    "Generate-and-\nVerify"
]

# Plateau stop iteration: mean over 3 random seeds
plateau_stop_mean = np.array([38.00, 40.00, 32.33, 37.00])

# =========================
# Compute saved operations
# =========================
saved_iterations = max_iteration - plateau_stop_mean

saved_query_labels = saved_iterations * batch_size

# Passive and Active do not use LLM generation
saved_generation_requests = np.array([
    0.00,
    0.00,
    saved_iterations[2] * batch_size,
    saved_iterations[3] * batch_size
])

# Only Generate-and-Verify uses verifier
saved_verification_requests = np.array([
    0.00,
    0.00,
    0.00,
    saved_iterations[3] * batch_size
])

# Data for grouped bar chart
metrics = [
    "Saved Query Labels",
    "Saved Generation Requests",
    "Saved Verification Requests"
]

values = np.array([
    saved_query_labels,
    saved_generation_requests,
    saved_verification_requests
]).T

# =========================
# Plot
# =========================
x = np.arange(len(methods))
bar_width = 0.24

fig, ax = plt.subplots(figsize=(9, 5))

hatches = ["", "//", "\\\\"]

for i, metric in enumerate(metrics):
    offset = (i - 1) * bar_width

    bars = ax.bar(
        x + offset,
        values[:, i],
        bar_width,
        label=metric,
        edgecolor="black",
        linewidth=0.8
    )

    # Hatch patterns for black-and-white readability
    for bar in bars:
        bar.set_hatch(hatches[i])

# =========================
# Add value labels
# =========================
for i in range(len(methods)):
    for j in range(len(metrics)):
        value = values[i, j]
        if value > 0:
            ax.text(
                x[i] + (j - 1) * bar_width,
                value + 8,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

# =========================
# Axes settings
# =========================
ax.set_xlabel("Method", fontsize=12)
ax.set_ylabel("Saved Operation Count (Proxy)", fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)

ax.set_ylim(0, max(values.flatten()) + 80)

ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
ax.set_axisbelow(True)

ax.legend(
    loc="upper left",
    frameon=True,
    fontsize=9
)

plt.tight_layout()

# =========================
# Save figures
# =========================
plt.savefig("resource_saving_grouped_bar.png", dpi=300, bbox_inches="tight")
plt.savefig("resource_saving_grouped_bar.pdf", bbox_inches="tight")

plt.show()

# =========================
# Print table values for checking
# =========================
print("Resource saving estimates")
print("-" * 80)
print(f"{'Method':<25}{'Saved Iter.':>12}{'Query':>12}{'Generation':>15}{'Verification':>15}")

for i, method in enumerate(methods):
    method_name = method.replace("\n", " ")
    print(
        f"{method_name:<25}"
        f"{saved_iterations[i]:>12.2f}"
        f"{saved_query_labels[i]:>12.2f}"
        f"{saved_generation_requests[i]:>15.2f}"
        f"{saved_verification_requests[i]:>15.2f}"
    )