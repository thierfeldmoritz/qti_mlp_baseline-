# imports
import os.path
from os import makedirs

import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import text
#-------------------------------------------------------------------------------------
# plot settings:
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

matplotlib.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=BIGGER_SIZE)
matplotlib.rc('legend', markerscale=MEDIUM_SIZE)

def plot_csv_files(lr_file, train_loss_file, val_loss_file, out_dir=None):
    # Read CSV files into numpy arrays
    lr_data = pd.read_csv(lr_file).values
    train_loss_data = pd.read_csv(train_loss_file).values
    val_loss_data = pd.read_csv(val_loss_file).values

    # Create index array
    index = np.arange(len(lr_data))

    # Plot values vs. index
    fig, ax1 = plt.subplots(figsize=(8.5, 5.25))
    ax2 = ax1.twinx()
    
    # ax2.zorder = 1; ax1.zorder=1

    # Plot the data
    ax2.plot(index, lr_data, color='steelblue', label='Learning Rate', linewidth=3.3, zorder=1)
    ax1.plot(index, val_loss_data, color='indianred', label='Validation Loss', linewidth=3.3, zorder=2)
    ax1.plot(index, train_loss_data, color='mediumaquamarine', label='Training Loss', linewidth=3.3, zorder=2)

    # Set labels and legends
    ax2.set_ylabel('Learning Rate')
    ax1.set_xlabel('Epoch Index')
    ax1.set_ylabel('Loss [a.u.]')

    # Set y tick labels in exponent notation
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    # Set y-limits for loss axis
    train_loss_range = np.max(train_loss_data) - np.min(train_loss_data)
    ax1.set_ylim(np.min(train_loss_data) - .3*train_loss_range, np.max(train_loss_data) + .3*train_loss_range)

    # Legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles2 + handles1
    labels = labels2 + labels1
    ax1.legend(handles, labels, loc='best', frameon = False)

    # Show the plot
    # plt.show()

    #---------------------------------------------------------
    # save figure:
    if out_dir is None:
        out_dir = '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/MATex/Figures/loss_curves'
        if not os.path.exists(out_dir):
            makedirs(out_dir)
    fig.savefig(os.path.join(out_dir, 'loss_curves' + '.pdf'), format='pdf', dpi=1200, bbox_inches='tight')


plot_csv_files(\
'/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Code/QTI_ML/LR.txt',\
'/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Code/QTI_ML/tloss.txt',\
'/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Code/QTI_ML/vloss.txt')

