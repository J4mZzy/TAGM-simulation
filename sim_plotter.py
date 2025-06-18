import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

## Helper funciton for plotting
def log_format_func(x, pos):
    exponent = int(np.log10(x))
    return r'$10^{{{}}}$'.format(exponent)

############################### Load data #######################################
cost_diffGD_lst_loaded = np.load("cost_diff_GD_prob=10.npy")
cost_diffHB_lst_loaded = np.load("cost_diff_HB_prob=10.npy")
cost_diffNAG_lst_loaded = np.load("cost_diff_NAG_prob=10.npy")
cost_diffGM_lst_loaded = np.load("cost_diff_GM_prob=10.npy")
#################################################################################

## Time vector (iterations)
dt = 0.1
tf = 600 
tvec = np.arange(0, tf + dt, dt)

# Set global font to Times New Roman and enable LaTeX
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering

# Color theme based on pp
pp = 1  # Change to 2, 3, or 4 for other color schemes
colors = {1: 'b', 2: 'r', 3: 'g', 4: 'm'}
c = colors.get(pp, 'k')

plt.figure(figsize=(8, 5))

# Plot 
line1, = plt.semilogy(np.arange(len(tvec)), cost_diffGD_lst_loaded, f'{c}-', linewidth=2, label='GD')
line2, = plt.semilogy(np.arange(len(tvec)), cost_diffHB_lst_loaded, f'{c}--', linewidth=2, label='HB')
line3, = plt.semilogy(np.arange(len(tvec)), cost_diffNAG_lst_loaded, f'{c}-.', linewidth=2, label='NAG')
line4, = plt.semilogy(np.arange(len(tvec)), cost_diffGM_lst_loaded, f'{c}-o', linewidth=3, markersize=2, label='GM')

# # Optional: probability markers
# p1, = plt.semilogy([], [], 'b*', markersize=8, markerfacecolor='b', linewidth=2, label=f'{probs[0]:.1f}')
# p2, = plt.semilogy([], [], 'r*', markersize=8, markerfacecolor='r', linewidth=2, label=f'{probs[1]:.1f}')
# p3, = plt.semilogy([], [], 'g*', markersize=8, markerfacecolor='g', linewidth=2, label=f'{probs[2]:.1f}')

# Labels and formatting
plt.xlabel(r'Iterations, $k$', fontsize=14)
plt.ylabel(r'$\|f(x) - f(x^\star)\|$', fontsize=14)
plt.xlim([0, len(tvec)+100])
plt.grid(True, which='both')

# Log scale formatting
ax = plt.gca()
ax.yaxis.set_major_locator(LogLocator(base=10.0))
ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f'$10^{{{int(np.log10(val))}}}$' if val != 0 else '0'))

# Legends
legend1 = plt.legend([line1, line2, line3, line4], ['GD', 'HB', 'NAG', 'GM'], loc='upper right', fontsize=12, title='Algorithm')
# legend2 = plt.legend([p1, p2, p3], [f'{probs[0]:.1f}', f'{probs[1]:.1f}', f'{probs[2]:.1f}'], loc='upper right', fontsize=12, title='Probability')

# Add the main legend back
plt.gca().add_artist(legend1)
plt.tight_layout()
plt.savefig("Alg_comparison.png", dpi=300)
plt.show()