#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# create nice boxplot for figure 3
#-------------------------------------------------------------------------------------

# imports
import os.path
from os import makedirs

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import text
# plt.rcParams["font.family"] = "Calibri"

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

# colors = ['darkred', 'indianred', 'salmon', 'goldenrod', 'y', 'lightgreen', 'mediumaquamarine', 'lightsteelblue', 'steelblue', 'midnightblue', 'indigo']
# colors2 = ['lightgreen', 'mediumaquamarine', 'skyblue', 'steelblue', 'mediumblue' ,'midnightblue', 'indigo']
# colors = ['darkred', 'indianred', 'salmon', 'tan', 'darkkhaki', 'lightgreen', 'mediumaquamarine', 'c', 'lightsteelblue', 'steelblue', 'midnightblue']
# colors2 = ['lightgreen', 'mediumaquamarine', 'c', 'cadetblue', 'steelblue', 'slateblue', 'blue', 'darkslateblue','navy', 'indigo']

#-------------------------------------------------------------------------------------

def SSIM_comp_plot(SSIM_path_1, SSIM_path_2, invar_keys = ['MD','FA', 'uFA', 'C_c', 'C_MD'], out_dir = None, fig = None, ax = None, annotate = True, save=True, mean_vs_med=False):
    if fig is None and ax is None:
        fig = plt.figure(figsize =(8.5, 5))
        ax = fig.add_subplot(111)
    elif ax is None and fig is not None:
        ax = fig.add_subplot(111)
    else:
        raise Exception('Unexpected input of Matplotlib containers.')

    if invar_keys == ['MD','FA', 'uFA', 'C_c', 'C_MD']:
        ticklabels = [r'$\mathrm{MD}$',r'$\mathrm{FA}$', r'$\mathrm{\mu FA}$', r'$\mathrm{C_c}$', r'$\mathrm{C_{MD}}$']
        positions_1 = [0.5 + 4*i for i in range(len(invar_keys))]  ; positions_2 = [1.5 + pos for pos in positions_1]
    else:
        ticklabels = None
        positions_1 = None

    container_1 = sio.loadmat(SSIM_path_1)
    container_2 = sio.loadmat(SSIM_path_2)

    SSIM_1 = container_1['SSIM']
    SSIM_2 = container_2['SSIM']
    SSIM_1 = np.transpose(SSIM_1)
    SSIM_2 = np.transpose(SSIM_2)

    if np.shape(SSIM_1)[0] == 1 and np.shape(SSIM_2)[0] == 1:
        boxplot = False
    else:
        boxplot = True


    if boxplot:
       #---------------------------------------------------------
        # plot customization:    
        boxprops_1 = dict(linestyle='-', linewidth=1, facecolor='steelblue', edgecolor='steelblue')
        flierprops_1 = dict(marker='o', markerfacecolor='none',
                        markeredgecolor='steelblue', markersize=6) # markersize=12
        capprops_1 = dict(color='steelblue')
        whiskerpros_1 =  dict(color='steelblue')

        boxprops_2 = dict(linestyle='-', linewidth=1, facecolor='mediumaquamarine', edgecolor='mediumaquamarine')
        flierprops_2 = dict(marker='o', markerfacecolor='none',
                        markeredgecolor='mediumaquamarine', markersize=6) # markersize=12
        capprops_2 = dict(color='mediumaquamarine')
        whiskerpros_2 =  dict(color='mediumaquamarine')

        medianprops = dict(linestyle='-', linewidth=1, color='indianred')
        meanpointprops = dict(marker='^', markeredgecolor='none',
                            markerfacecolor='gold', markersize=6)
        meanlineprops = dict(linestyle='-', linewidth=.75, color='gold')
        #---------------------------------------------------------
        bp_1 = ax.boxplot(SSIM_1, positions=positions_1, patch_artist = True, showmeans=True, meanline=False, widths=1.2,\
                        boxprops=boxprops_1, flierprops=flierprops_1, medianprops=medianprops,\
                        meanprops=meanpointprops, capprops=capprops_1, whiskerprops=whiskerpros_1)
        bp_2 = ax.boxplot(SSIM_2, positions=positions_2, patch_artist = True, showmeans=True, meanline=False, widths=1.2,\
                        boxprops=boxprops_2, flierprops=flierprops_2, medianprops=medianprops,\
                        meanprops=meanpointprops, capprops=capprops_2, whiskerprops=whiskerpros_2)
          
        ax.xaxis.set_ticks([0.75 + pos for pos in positions_1], ticklabels, )
        ax.xaxis.set_tick_params(length = 0)
        ax.set_xlim(-1,19)
        ax.set_ylim(0.8975,1)
        legend = ax.legend([bp_1["boxes"][0], bp_2["boxes"][0], bp_1['medians'][0], bp_1['means'][0]], ['NLLS', 'MLP', 'median', 'mean'], loc='lower left', fontsize = MEDIUM_SIZE*.9, frameon = False)
        legend.legendHandles[-1].set_markersize(8)
        legend.legendHandles[-2].set_linewidth(1.5)
    #---------------------------------------------------------
    # annotate median to boxplot, if desired:
        if annotate:
            caps_1 = bp_1['caps']; caps_2 = bp_2['caps']
            caps_1 = caps_1[::2]; caps_2 = caps_2[::2]
            if not mean_vs_med:
                mean_vs_med_str = '_med'
                for line_1, line_2, median_1, median_2 in zip(caps_1, caps_2, bp_1['medians'], bp_2['medians']):
                    # get position data for median line
                    x_med1, y_med1 = median_1.get_xydata()[1] # top of median line
                    y1 = line_1.get_xydata()[0,1] # bottom of cap line
                    # overlay median value
                    text(x_med1+0.031, y1-0.0035, '%.3f' % y_med1, fontsize = MEDIUM_SIZE,
                        horizontalalignment='right', verticalalignment='top') # draw above, right
                    # get position data for median line
                    x_med2, y_med2 = median_2.get_xydata()[1]
                    y2 = line_2.get_xydata()[0,1] # bottom of cap line
                    # overlay median value
                    text(x_med2+0.031, y2-0.0035, '%.3f' % y_med2, fontsize = MEDIUM_SIZE,
                        horizontalalignment='right', verticalalignment='top') # draw above, right
            else:
                mean_vs_med_str = '_mean'
                for line_1, line_2, mean_1, mean_2 in zip(caps_1, caps_2, bp_1['means'], bp_2['means']):
                    # get position data for mean line
                    x_mean1, y_mean1 = mean_1.get_xydata()[0] # top of mean line
                    y1 = line_1.get_xydata()[0,1] # bottom of cap line
                    # overlay mean value
                    text(x_mean1+0.6, y1-0.0035, '%.3f' % y_mean1, fontsize = MEDIUM_SIZE,
                        horizontalalignment='right', verticalalignment='top') # draw above, right
                    # get position data for mean line
                    x_mean2, y_mean2 = mean_2.get_xydata()[0]
                    y2 = line_2.get_xydata()[0,1] # bottom of cap line
                    # overlay mean value
                    text(x_mean2+0.6, y2-0.0035, '%.3f' % y_mean2, fontsize = MEDIUM_SIZE,
                        horizontalalignment='right', verticalalignment='top') # draw above, right            
    else:
        mean_vs_med_str = ''
        bar_1 = ax.bar(x=positions_1, height=SSIM_1[0], color='steelblue', label = 'NLLS')
        bar_2 = ax.bar(x=positions_2, height=SSIM_2[0], color='mediumaquamarine', label = 'MLP')
        if annotate:
            ax.bar_label(bar_1, label_type='edge', fmt='%.3f')           
            ax.bar_label(bar_2, label_type='edge', fmt='%.3f')
        ax.xaxis.set_ticks([0.75 + pos for pos in positions_1], ticklabels)
        ax.xaxis.set_tick_params(length = 0)
        ax.set_ylim(0.9,1)
        ax.legend(loc='upper right', fontsize = MEDIUM_SIZE*.9, frameon = False)
    #---------------------------------------------------------
    # plot options:
    # plt.title(title_str)
    plt.ylabel('SSIM')
    #---------------------------------------------------------
    # save figure:                   
    if save:
        if out_dir is None:
            out_dir = os.path.split(SSIM_path_1)[0] 
            if not os.path.exists(out_dir):
                makedirs(out_dir)      
        plt.savefig(os.path.join(out_dir, 'SSIM_comp_plot' + mean_vs_med_str  + '.pdf'), format='pdf', dpi=1200, bbox_inches='tight')
    return fig, ax