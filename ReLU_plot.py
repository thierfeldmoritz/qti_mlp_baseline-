# imports
import os.path
from os import makedirs

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import text

#-------------------------------------------------------------------------------------
# plot settings:
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 12

matplotlib.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=BIGGER_SIZE)
matplotlib.rc('legend', markerscale=MEDIUM_SIZE)


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

x = np.linspace(-5, 5, 1000)
y = relu(x)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)

fig, ax = plt.subplots(figsize=(8.5, 5.25))

ax.plot(x, y_sigmoid, color='steelblue', label='Sigmoid', linewidth=4)
ax.plot(x, y_tanh, color='indianred', label='tanh', linewidth=4)
ax.plot(x, y, color='mediumaquamarine', label='ReLU', linewidth=4)

# Add a box around the figure

ax.set_ylim(-2,4)
ax.set_xlim(-3,3)

# Add arrows to x-axis and y-axis
ax.spines['left'].set_position('zero')
ax.spines['left'].set_linewidth(1.5)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_position('zero')
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['bottom'].set_color('black')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.arrow(0, -2, 0, 6.05, length_includes_head=True, head_width=0.1, head_length=0.4, color='black')
ax.arrow(-3, 0, 6.05, 0, length_includes_head=True, head_width=0.1, head_length=0.4, color='black')

ax.set_xlabel('x', ha='right', x=1, labelpad=-10)
ax.set_ylabel(r'$\varphi(x)$', ha='right', y=1, labelpad=-10)  # Adjust the labelpad parameter to move the label closer to the y-axis

# Remove ticks
ax.set_xticks([1])
ax.set_yticks([1])

# Label the origin
# ax.text(0, 0, '0', ha='right', va='bottom', position=(-0.1, 0.1), fontsize=MEDIUM_SIZE)
# ax.text(0, 0, '0', ha='left', va='top', position=(0.1, -0.1), fontsize=MEDIUM_SIZE)

ax.set_aspect('equal')

# Move the legend to the upper right corner
ax.legend(loc='upper right', bbox_to_anchor=(0.4, .85))
# ax.legend(loc='upper right')


fig.patch.set_edgecolor('black')
fig.patch.set_linewidth(1.5)

# ax.grid(False)  # Enable grid

plt.show()

#fig.savefig(os.path.join('/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/MATex/Figures/loss_curves', 'ActivationFunctions' + '.pdf'), format='pdf', dpi=1200, bbox_inches='tight')
