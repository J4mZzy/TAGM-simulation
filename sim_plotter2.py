import numpy as np
import matplotlib.pyplot as plt

# Probability values (update/communication probabilities)
probs = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

# Iteration counts from the table for different methods
iters_GD  = np.array([666, 740, 828, 953, 1107, 1351, 1696, 2272, 3553, 8486, 24309])
iters_HB  = np.array([309, 347, 390, 449, 526, 651, 818, 1083, 1715, 4154, 11539])
iters_NAG = np.array([218, 248, 280, 322, 379, 475, 591, 768, 1249, 3018, 8425])
iters_GM  = np.array([168, 194, 222, 256, 302, 377, 478, 614, 1013, 2424, 6722])

# Use GM as the reference
def compute_reduction(reference, method):
    return 100 * (reference - method) / reference

# Now compute how much *worse* HB and NAG are compared to GM
reduction_HB  = compute_reduction(iters_HB, iters_GM)   # How much faster GM is vs HB
reduction_NAG = compute_reduction(iters_NAG, iters_GM)  # How much faster GM is vs NAG
reduction_GD  = compute_reduction(iters_GD, iters_GM)   # How much faster GM is vs GD

# Positions for evenly spaced x-axis ticks
x_positions = np.arange(len(probs))  # 0, 1, 2, ..., 10

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'text.usetex': True,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

fig, ax = plt.subplots(figsize=(9, 6))

# Plot using evenly spaced positions instead of raw probabilities
ax.plot(x_positions, reduction_GD, marker='o', linestyle='-', color='b', label='GD', linewidth=3, markersize=10)     # Blue
ax.plot(x_positions, reduction_HB, marker='s', linestyle='-', color='r', label='HB', linewidth=3, markersize=10)     # Red
ax.plot(x_positions, reduction_NAG, marker='^', linestyle='-', color='g', label='NAG', linewidth=3, markersize=10)   # Green

# Set ticks to the evenly spaced positions but label them with the original probabilities
ax.set_xticks(x_positions)
ax.set_xticklabels([f"{p:.2f}" for p in probs])

# Axis labels
ax.set_xlabel('Update/Communication Probability')
ax.set_ylabel('Iteration Reduction (\%)')
# Make labels and ticks bigger
ax.set_xlabel('Update/Communication Probability', fontsize=18)
ax.set_ylabel('Iteration Reduction vs GM (\%)', fontsize=18)
ax.tick_params(axis='both', labelsize=18)  # This sets both x and y ticks
# Reverse x-axis order to show probabilities decreasing left to right
ax.invert_xaxis()

# Grid style for better readability
ax.grid(True, linestyle='--', alpha=0.6)

# Legend
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.8), fontsize=16)

plt.tight_layout()
# plt.savefig("Alg_comparison_reduction.pdf", format='pdf', bbox_inches='tight')
plt.show()
