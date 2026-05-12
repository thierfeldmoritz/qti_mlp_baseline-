#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# base classes for performance evaluation using standard (image processing) metrics
#-------------------------------------------------------------------------------------
# imports
import torch
from torchmetrics.functional.image import structural_similarity_index_measure as SSIM

import os.path
from os import makedirs
from datetime import datetime

import nibabel as nib
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider
from matplotlib.pyplot import text
import cmasher as cmr
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
matplotlib.rc('ytick', labelsize=9)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=BIGGER_SIZE)
matplotlib.rc('legend', markerscale=MEDIUM_SIZE)

cmap = cmr.eclipse

# colors = ['darkred', 'indianred', 'salmon', 'goldenrod', 'y', 'lightgreen', 'mediumaquamarine', 'lightsteelblue', 'steelblue', 'midnightblue', 'indigo']
# colors2 = ['lightgreen', 'mediumaquamarine', 'skyblue', 'steelblue', 'mediumblue' ,'midnightblue', 'indigo']
# colors = ['darkred', 'indianred', 'salmon', 'tan', 'darkkhaki', 'lightgreen', 'mediumaquamarine', 'c', 'lightsteelblue', 'steelblue', 'midnightblue']
# colors2 = ['lightgreen', 'mediumaquamarine', 'c', 'cadetblue', 'steelblue', 'slateblue', 'blue', 'darkslateblue','navy', 'indigo']

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# QTI_Fit_Dataset class
#-------------------------------------------------------------------------------------
# prepare QTI data and fit results for regression
class QTI_Fit_Dataset():
    #-------------------------------------------------------------------------------------
    # Constructor:
    def __init__(self, pred_path, target_path, mask_path = None, invar_keys = ['FA', 'uFA', 'C_c', 'C_MD'],\
                 out_path = None, out_dir = None, slice_ind = None, brain_wise = True, slice_wise = False):
        # output path:
        #---------------------------------------------------------
        if out_dir is None:
            out_dir = 'Performance_eval_'
        elif not isinstance(out_dir, str):
            raise Exception('Invalid input for out_dir.')
        if out_path is None:
            self.out_path = os.path.join(os.path.split(os.path.commonpath(target_path))[0], out_dir + datetime.now().strftime('%Y%m%d'))
        else:
            self.out_path = os.path.join(out_path, out_dir)
        if not os.path.exists(self.out_path):
            makedirs(self.out_path)

        #--------------------------------------------------------
        # read target (dps.mat), affines for writing output:
        self.invar_keys = invar_keys
        self.y_pred, self.affines_pred = read_mat(pred_path, self.invar_keys)
        self.y_target, self.affines_target = read_mat(target_path, self.invar_keys)
        #--------------------------------------------------------
        # read mask (.nii), if not present, fit all voxels
        if mask_path is not None:
            self.mask = read_mask_nii(mask_path)
        else:
            self.mask = self.create_dummy_mask()

        #--------------------------------------------------------
        # set averaging mode:
        self.brain_wise = brain_wise
        self.slice_wise = slice_wise
        self.set_voxel_wise()

        #---------------------------------------------------------
        # initialize metrics:
        self.diff = None
        self.abs_diff = None
        self.rel_diff = None

        self.nRMSE = None
        self.pSNR = None
        self.SSIM = None

        #--------------------------------------------------------
        # absorb bottom/top slices into mask, if specified:
        if slice_ind == None:
            self.slice_ind = [[0, self.X.size()[1]] for _ in range(self.X.size()[0])]
        else:
            self.slice_ind = slice_ind
        self.sel_slices()

    #-------------------------------------------------------------------------------------
    # Default methods:
    def set_brain_wise(self, value):
        if type(value) != bool:
            raise Exception("Input has to be of type 'bool'.")
        else:
            self.brain_wise = value
            self.set_voxel_wise()
    
    def set_slice_wise(self, value):
        if type(value) != bool:
            raise Exception("Input has to be of type 'bool'.")
        else:
            self.slice_wise = value
            self.set_voxel_wise()

    def set_voxel_wise(self):
        if self.brain_wise and self.slice_wise:
            raise Exception("Choose for either brain- or slice-wise.")
        elif self.brain_wise or self.slice_wise:
            self.voxel_wise = False
        elif not (self.brain_wise or self.slice_wise):
            self.voxel_wise = True

    #-------------------------------------------------------------------------------------
    # Methods/data preparation:
    #-------------------------------------------------------------------------------------
    # masking:
    def create_dummy_mask(self):
        return torch.ones(self.X.size()[:-1], dtype=torch.bool)
    
    # unlike QTI_Dataset, expand mask to full dimension since data not necessarily flattened
    def expand_mask(self):
        self.mask = self.mask[...,None].expand(-1,-1,-1,-1, self.y_pred.size()[-1])

    # edge slice exclusion
    # adapted to allow flexible exclusion of lower/upper slices per brain
    # ind[0]: 1st included slice, ind[1]: last included slice (0-based)
    def sel_slices(self):
        for i, ind in enumerate(self.slice_ind):
            self.mask[i, :ind[0], ...] = False
            self.mask[i, (ind[1] + 1) :, ...] = False

    # mask undesirable values: thus far, one mask for all metrics
    #------------------------------------------------------------
    # method to remove nan values (absorb into mask)
    def mask_nan_vals(self):
        mask = torch.logical_and(~torch.isnan(self.y_pred), ~torch.isnan(self.y_target))
        self.mask = torch.logical_and(self.mask, mask)

    # method to remove zero values
    # only if no normalization is used, since it introduces neg. values inevitably
    def mask_zero_vals(self):
        mask = self.y_target != 0
        self.mask = torch.logical_and(self.mask, mask)

    # method (obsolete) to mask very small values
    # only if no normalization is used, since it introduces neg. values inevitably
    def mask_near_zero_vals(self, eps=1e-4):
        mask = self.y_target > eps
        self.mask = torch.logical_and(self.mask, mask)

    #-------------------------------------------------------------------------------------
    # Methods/metric calculation:
    #-------------------------------------------------------------------------------------
    # voxel-wise difference maps/metrics:
    def compute_diff(self):
        self.diff = torch.subtract(self.y_pred,self.y_target)    

    def compute_abs_diff(self):
        if self.diff is None:
            self.compute_diff()
        self.abs_diff = torch.abs(self.diff)   

    def compute_rel_diff(self):
        if self.abs_diff is None:
            self.compute_abs_diff()
        self.rel_diff = torch.divide(self.abs_diff, torch.abs(self.y_target))

    #------------------------------------------------------------
    # save: 
    def save_diff_2_mat(self, diff='', sep_files = False, sep_invars = False, out_dir = None):
        if out_dir is not None:
            out_pn = os.path.join(self.out_path, out_dir)
            if not os.path.exists(out_pn):
                makedirs(out_pn)
        else:
            out_pn = self.out_path
        #---------------------------------------------------------
        # prepare options and inputs:
        if diff == '':
            if self.abs_diff is None:
                self.compute_diff()
            data = self.diff
        elif diff == 'abs':
            if self.abs_diff is None:
                self.compute_abs_diff()
            data = self.abs_diff
        elif diff == 'rel':
            if self.rel_diff is None:
                self.compute_rel_diff()
        else:
            raise Exception('Invalid option for diff.')
        #---------------------------------------------------------
        if sep_files:
            for file_ind in range(self.abs_diff.size(0)):
                if sep_invars:
                    for invar_ind, invar in enumerate(self.invar_keys):
                        sio.savemat(os.path.join(out_pn, diff + 'diff_' + 'f{}_'.format(file_ind+1) + invar + '.mat'),\
                            {'abs_diff': self.abs_diff[file_ind, ..., invar_ind].detach().cpu().numpy()})
                else:
                    sio.savemat(os.path.join(out_pn, diff + 'diff_' + 'f{}_'.format(file_ind+1) + '.mat'),\
                        {'abs_diff': self.abs_diff[file_ind, ...].detach().cpu().numpy()})
        else:
            sio.savemat(os.path.join(out_pn, diff + 'diff' + '.mat'),\
                {'abs_diff': self.abs_diff.detach().cpu().numpy()})

    #-------------------------------------------------------------------------------------
    # nRMSE with averaging over 3 different sets of dims:
    def compute_nRMSE(self):
        #------------------------------------------------------------
        # compute squared diff/vox, apply mask:
        ASE = torch.square(self.abs_diff) # eig egal ob abs oder nicht
        ASE[~self.mask] = 0
        # apply mask to target data:
        y = self.y_target.clone().detach()
        y[~self.mask] = 0
        #------------------------------------------------------------
        # different modes of averaging:
        if self.slice_wise:
            # Summe über (x,y), nRMSE output hat dim (file, z, invar)
            dim=(2,3)
        elif self.brain_wise:
            # Summe über (z,x,y), nRMSE output hat dim (file, invar)
            dim=(1,2,3)
        elif self.voxel_wise:
            # Summe über (file,z,x,y), nRMSE output hat dim (invar)
            dim=(0,1,2,3)
        #------------------------------------------------------------
        # compute root mean squared error (div. by no. of contributing voxels in mask):
        n = torch.count_nonzero(self.mask,  dim=dim)
        RMSE = torch.sqrt(torch.divide(torch.sum(ASE, dim=dim), n)) # das war falschrum!!!
        #------------------------------------------------------------
        # compute mean across resp. dims for normalization:
        mean_y_target = torch.divide(torch.sum(y, dim=dim), n)
        #------------------------------------------------------------
        # nRMSE:
        self.nRMSE = torch.divide(RMSE, mean_y_target)
        #------------------------------------------------------------
        # clean up
        del ASE, y, n, RMSE, mean_y_target

    #------------------------------------------------------------
    # save: 
    def save_nRMSE(self, out_dir = None):
            if out_dir is not None:
                out_pn = os.path.join(self.out_path, out_dir)
                if not os.path.exists(out_pn):
                    makedirs(out_pn)
            else:
                out_pn = self.out_path
            if self.slice_wise:
                identifier = 'sw'
                data = torch.flatten(self.nRMSE, end_dim=1)
                # fix for excluded slices (n = 0 everywhere) and slice_wise:    
                data = data[~torch.isnan(torch.sum(data,dim=1)), :].detach().cpu().numpy()
            elif self.brain_wise:
                identifier = 'bw'
                data = self.nRMSE.detach().cpu().numpy()
            elif self.voxel_wise:
                identifier = 'vw'
                data = self.nRMSE.detach().cpu().numpy()
            sio.savemat(os.path.join(out_pn,'nRMSE_' + identifier + '.mat'),\
            {'nRMSE': data})

    #-------------------------------------------------------------------------------------
    # pSNR with averaging over 3 different sets of dims:
    def compute_pSNR(self):
        #------------------------------------------------------------
        # compute squared diff/vox, apply mask:
        ASE = torch.square(self.abs_diff)
        ASE[~self.mask] = 0
        # apply mask to target data:
        y = self.y_target.clone().detach()
        y[~self.mask] = 0
        #------------------------------------------------------------
        # different modes of averaging:
        if self.slice_wise:
            dim=(2,3)
        elif self.brain_wise:
            dim=(1,2,3)
        elif self.voxel_wise:
            dim=(0,1,2,3)
        #------------------------------------------------------------
        # compute mean squared error (div. by no. of contributing voxels in mask):
        SE = torch.sum(ASE, dim=dim)
        n = torch.count_nonzero(self.mask,  dim=dim)
        #------------------------------------------------------------
        # compute max across resp. dims for normalization:
        max_y_target_sq = torch.square(torch.amax(y, dim=dim))
        #------------------------------------------------------------
        # pSNR:
        self.pSNR = 10 * torch.log10(torch.multiply(n,torch.divide(max_y_target_sq,SE)))
        # #------------------------------------------------------------
        # # fix for excluded slices (MSE = 0 everywhere) and slice_wise: (moved to save & plot)
        # if self.slice_wise:
        #     self.pSNR= self.pSNR[:,torch.sum(MSE, dim=(2)) != 0, ...]
        #------------------------------------------------------------
        # clean up
        del ASE, y, SE, n, max_y_target_sq

    #------------------------------------------------------------
    # save: 
    def save_pSNR(self, out_dir = None):
            if out_dir is not None:
                out_pn = os.path.join(self.out_path, out_dir)
                if not os.path.exists(out_pn):
                    makedirs(out_pn)
            else:
                out_pn = self.out_path
            if self.slice_wise:
                identifier = 'sw'
                data = torch.flatten(self.pSNR, end_dim=1)
                # fix for excluded slices (MSE = 0 everywhere) and slice_wise:
                data = data[~torch.isnan(torch.sum(data,dim=1)), :].detach().cpu().numpy()
            elif self.brain_wise:
                identifier = 'bw'
                data = self.pSNR.detach().cpu().numpy()
            elif self.voxel_wise:
                identifier = 'vw'
                data = self.pSNR.detach().cpu().numpy()
            sio.savemat(os.path.join(out_pn, 'pSNR_' + identifier + '.mat'),\
            {'pSNR': data})

    def compute_SSIM(self, kernel_size=11, save = False):
        # perhaps include option for slice vs. brain-wise
        # flatten along z and P dim
        # how does SSIM handle NaNs? better apply mask here
        SSIMs = []
        data_pred = self.y_pred.clone().detach(); data_target = self.y_target.clone().detach()
        data_pred[~self.mask] = 0; data_target[~self.mask] = 0
        # if self.slice_wise:
        #     for invar_ind in range(len(self.invar_keys)):
        #         SSIMs.append(SSIM(torch.flatten(self.y_pred, end_dim=1)[..., invar_ind],\
        #                         torch.flatten(self.y_target, end_dim=1)[..., invar_ind],\
        #                         reduction='none'))
        # else
        for invar_ind in range(len(self.invar_keys)):
            SSIMs.append(SSIM(data_pred[..., invar_ind], data_target[..., invar_ind],\
                         reduction='none', kernel_size=kernel_size))
               
        self.SSIM = torch.stack(SSIMs, dim=0)
        del SSIMs, data_pred, data_target

    #------------------------------------------------------------
    # save: 
    def save_SSIM(self, out_dir = None):
            if out_dir is not None:
                out_pn = os.path.join(self.out_path, out_dir)
                if not os.path.exists(out_pn):
                    makedirs(out_pn)
            else:
                out_pn = self.out_path
            if self.slice_wise:
                identifier = 'sw'
            elif self.brain_wise:
                identifier = 'bw'
            elif self.voxel_wise:
                identifier = 'vw'
            data = self.SSIM.detach().cpu().numpy() # uff... lieber gleich als Tensor? fix this

            sio.savemat(os.path.join(out_pn, 'SSIM_' + identifier + '.mat'),\
            {'SSIM': data})

    #-------------------------------------------------------------------------------------
    # create and save maps for voxel-wise difference measures with sliders for file, slice
    def plot_diff(self, diff='', file_ind = 0, slice_ind=None, invar_key = None, save = False, cmap = cmap,\
                    vmax = None, vmin = None, out_dir = None, skip_plot = False):
        # reference of sliders must be handed thru/returned..
        #---------------------------------------------------------
        # prepare options and inputs:
        if diff == '':
            if self.abs_diff is None:
                self.compute_diff()
            data = self.diff
        elif diff == 'abs':
            if self.abs_diff is None:
                self.compute_abs_diff()
            data = self.abs_diff
        elif diff == 'rel':
            if self.rel_diff is None:
                self.compute_rel_diff()
            data = self.rel_diff
        else:
            raise('Invalid option for diff.')
        #---------------------------------------------------------
        # save figures:
        if save:
            if out_dir is not None:
                out_pn = os.path.join(self.out_path, out_dir)
                if not os.path.exists(out_pn):
                    makedirs(out_pn)
            else:
                out_pn = self.out_path  
        #---------------------------------------------------------
        # for some reason, physical x,y are interchanged here..
        data = torch.flip(torch.permute(data, dims=(0, 1, 3, 2, 4)), dims=[2])
        #---------------------------------------------------------  
        if slice_ind is None:
            slice_ind = self.abs_diff.size(1)//2
        if invar_key is None:
            invar_key = self.invar_keys
        elif ~isinstance(invar_key, list):
            invar_key = [invar_key]
        #---------------------------------------------------------
        # plot:
        # for now, list of slider objects created, such that each
        # figure can be toggled individually, to sync up, only create one set of sliders
        #---------------------------------------------------------
        # loop over invars, all relevant objects placed into lists:  
        figs = []; axs = []; lines = []; sldrs_file = []; sldrs_slice = []; 
        for ind, key in enumerate(invar_key):
            try:
                invar_ind = self.invar_keys.index(key)
            except:
                print('Invar_key unknown.')
            #---------------------------------------------------------
            # save figures:
            # option to pass window and store in filename here:
            if save:
                if vmax and vmin is not None:
                    w_str = ('w_{:.2f}_{:.2f}_'.format(vmax, vmin)).replace('.','_')
                else:
                    w_str = ''
                # imsave and pdf do not seem to work
                # cm = 1/2.54
                fig = plt.figure(figsize =(1.5, 1.5))
                ax = fig.add_subplot(111)
                line = ax.imshow(data[file_ind, slice_ind, ..., invar_ind].detach().cpu().numpy(), cmap = cmap, vmax = vmax , vmin = vmin)
                ax.axis('off')
                plt.savefig(os.path.join(out_pn, diff + 'diff_' + 'f{}_s{}_'.format(file_ind, slice_ind) + w_str + key + '.pdf'),\
                            format='pdf', dpi=1200, bbox_inches='tight', transparent=True, pad_inches=0)
                # fig.colorbar
                del fig, ax, line

                # plt.imsave(os.path.join(out_pn, diff + 'diff_' + 'f{}_s{}_'.format(file_ind, slice_ind) + w_str + key + '.png'),\
                #             data[file_ind, slice_ind, ..., invar_ind].detach().cpu().numpy(), format='png', dpi=1200,\
                #                 vmax = vmax , vmin = vmin, cmap = 'inferno')
            if skip_plot:
                return None, None, None, None
            else:
                figs.append(plt.figure(figsize =(10, 7)))
                axs.append(figs[ind].add_subplot(111))
                lines.append(\
                    axs[ind].imshow(data[file_ind, slice_ind, ..., invar_ind], cmap = cmap,\
                            vmax = vmax , vmin = vmin))
                axs[ind].axis('off')
                axs[ind].set_title(diff + ' diff ' + key)
                figs[ind].colorbar(lines[ind])

                #---------------------------------------------------------
                # sliders:    
                ax_file = figs[ind].add_axes([0.05, 0.15, 0.0225, 0.63])
                ax_slice = figs[ind].add_axes([0.15, 0.15, 0.0225, 0.63])
                sldrs_file.append(Slider(ax_file, key + ' file index', 0, self.abs_diff.size(0)-1, valinit=file_ind, orientation='vertical', valfmt='%d'))
                sldrs_slice.append(Slider(ax_slice, ' slice index', 0, self.abs_diff.size(1)-1, valinit=slice_ind, orientation='vertical', valfmt='%d'))

                def update(val):
                    for n, (line, fig) in enumerate(zip(lines, figs)):
                        line.set_data(data[int(sldrs_file[n].val), int(sldrs_slice[n].val), ..., n])
                        fig.canvas.draw_idle()

                sldrs_file[ind].on_changed(update)
                sldrs_slice[ind].on_changed(update)

            # reference must be returned, otherwise slider unresponsive
                # plt.savefig(os.path.join(out_pn, identifier  + '.pdf'), format='pdf', dpi=1200, bbox_inches='tight')

        return figs, axs, sldrs_file, sldrs_slice


    #-------------------------------------------------------------------------------------
    # write maps for voxel-wise difference measures to .nii
    def save_diff_2_nii(self, diff='', out_dir = None):
        if out_dir is not None:
            out_pn = os.path.join(self.out_path, out_dir)
            if not os.path.exists(out_pn):
                makedirs(out_pn)
        else:
            out_pn = self.out_path
        #---------------------------------------------------------
        # prepare options and inputs:
        if diff == '':
            if self.abs_diff is None:
                self.compute_diff()
            data = self.diff
        elif diff == 'abs':
            if self.abs_diff is None:
                self.compute_abs_diff()
            data = self.abs_diff
        elif diff == 'rel':
            if self.rel_diff is None:
                self.compute_rel_diff()
            data = self.rel_diff
        else:
            raise('Invalid option for diff.')
        #---------------------------------------------------------
        data = torch.permute(data, (0, 2, 3, 1, 4)).detach().cpu().numpy()
        #---------------------------------------------------------
        for file_ind in range(np.shape(data)[0]):
            affine = self.affines_target[file_ind]
            for invar_ind, invar in enumerate(self.invar_keys):
                output_img = nib.Nifti1Image(data[file_ind, ..., invar_ind], affine)
                nib.save(output_img, os.path.join(out_pn, diff + 'diff_' + 'f{}_'.format(file_ind) + invar + '.nii'))
        del data


    #-------------------------------------------------------------------------------------
    # Plotting and saving of figures:
    #-------------------------------------------------------------------------------------
    # box- and bar plots for nRMSE
    def nRMSE_boxplot(self, save = False, out_dir = None, fig = None, ax = None, annotate = True):
        if fig is None and ax is None:
            fig = plt.figure(figsize =(7.5, 5.25))
            ax = fig.add_subplot(111)
        elif ax is None and fig is not None:
            ax = fig.add_subplot(111)
        else:
            raise Exception('Unexpected input of Matplotlib containers.')
        
        if self.invar_keys == ['MD','FA', 'uFA', 'C_c', 'C_MD']:
            ticklabels = [r'$\mathrm{MD}$',r'$\mathrm{FA}$', r'$\mathrm{\mu FA}$', r'$\mathrm{C_c}$', r'$\mathrm{C_{MD}}$']
        #---------------------------------------------------------
        # different modes of averaging:      
        if self.brain_wise:
            # first dim in boxplot is label index, second is data
            data = torch.permute(self.nRMSE, (1,0))
            title_str = 'Brain-wise nRMSE'; identifier = 'nRMSE_bw'
        elif self.slice_wise:
            data = torch.permute(torch.flatten(self.nRMSE, end_dim=1), (1,0))
            # fix for excluded slices (n = 0, 1/n = nan) and slice_wise:
            data = data[:,~torch.isnan(torch.sum(data,dim=0))]
            title_str = 'Slice-wise nRMSE'; identifier = 'nRMSE_sw'
        if not self.voxel_wise:
            #---------------------------------------------------------
            # plot customization:    
            boxprops = dict(linestyle='-', linewidth=1, facecolor='steelblue', edgecolor='steelblue')
            flierprops = dict(marker='o', markerfacecolor='none',
                            markeredgecolor='steelblue', markersize=6) # markersize=12
            capprops = dict(color='steelblue')
            whiskerpros =  dict(color='steelblue')
            medianprops = dict(linestyle='-', linewidth=.75, color='r')
            meanpointprops = dict(marker='^', markeredgecolor='none',
                                markerfacecolor='gold', markersize=6)
            meanlineprops = dict(linestyle='-', linewidth=.75, color='gold')
            #---------------------------------------------------------
            bp = ax.boxplot(data, patch_artist = True, showmeans=True, meanline=False,\
                            boxprops=boxprops, flierprops=flierprops, medianprops=medianprops,\
                            meanprops=meanpointprops, capprops=capprops, whiskerprops=whiskerpros)
            ax.set_xticklabels(ticklabels)
            ax.set_xlim(0.3,5.8)
        #---------------------------------------------------------
        # annotate median to boxplot, if desired:
            if annotate:
                for line in bp['medians']:
                    # get position data for median line
                    x, y = line.get_xydata()[1] # top of median line
                    # overlay median value
                    text(x+.37, y+.006, '%.3f' % y, fontsize = 9,
                        horizontalalignment='right', verticalalignment='top') # draw above, right            
        else:
            bar = ax.bar(self.invar_keys, self.nRMSE, color='steelblue')
            if annotate:
                ax.bar_label(bar, label_type='edge', fmt='%.3f')           
            title_str = 'Voxel-wise nRMSE'; identifier = 'nRMSE_vw'
        #---------------------------------------------------------
        # plot options:
        # plt.title(title_str)
        plt.ylabel('nRMSE')
        #---------------------------------------------------------
        # save figure:                   
        if save:
            if out_dir is not None:
                out_pn = os.path.join(self.out_path, out_dir)
                if not os.path.exists(out_pn):
                    makedirs(out_pn)
            else:
                out_pn = self.out_path          
            plt.savefig(os.path.join(out_pn, identifier  + '.pdf'), format='pdf', dpi=1200, bbox_inches='tight')
        return fig, ax

    #-------------------------------------------------------------------------------------
    # box- and bar plots for pSNR
    def pSNR_boxplot(self, save = False, out_dir = None, fig = None, ax = None, annotate = True):
        if fig is None and ax is None:
            fig = plt.figure(figsize =(10, 7))
            ax = fig.add_subplot(111)
        elif ax is None and fig is not None:
            ax = fig.add_subplot(122)
        else:
            raise('Unexpected input of Matplotlib containers.')
        #---------------------------------------------------------
        # different modes of averaging:
        if self.brain_wise:
            # first dim in boxplot is label index, second is data
            data = torch.permute(self.pSNR, (1,0))
            title_str = 'Brain-wise pSNR'; identifier = 'pSNR_bw'
        elif self.slice_wise:
            data = torch.permute(torch.flatten(self.pSNR, end_dim=1), (1,0))
            # fix for excluded slices (MSE = 0 everywhere) and slice_wise:
            data = data[:,~torch.isnan(torch.sum(data,dim=0))]
            title_str = 'Slice-wise pSNR'; identifier = 'pSNR_sw'
        if not self.voxel_wise:
            bp = ax.boxplot(data, patch_artist = True)
            ax.set_xticklabels(self.invar_keys)
        #---------------------------------------------------------
        # annotate median to boxplot, if desired:
            if annotate:
                for line in bp['medians']:
                    # get position data for median line
                    x, y = line.get_xydata()[1] # top of median line
                    # overlay median value
                    text(x+0.23, y, '%.1f' % y, fontsize = 7.0,
                        horizontalalignment='right', verticalalignment='top') # draw above, right
        else:
            bar = ax.bar(self.invar_keys, self.pSNR, label = self.invar_keys)
            if annotate:
                ax.bar_label(bar, label_type='edge', fmt='%.2f')
            title_str = 'Voxel-wise pSNR'; identifier = 'pSNR_vw'
        #---------------------------------------------------------
        # plot options:
        plt.title(title_str)
        #---------------------------------------------------------
        # save figure:
        if save:
            if out_dir is not None:
                out_pn = os.path.join(self.out_path, out_dir)
                if not os.path.exists(out_pn):
                    makedirs(out_pn)
            else:
                out_pn = self.out_path          
            plt.savefig(os.path.join(out_pn, identifier  + '.pdf'), format='pdf', dpi=1200, bbox_inches='tight')
        return fig, ax
    
    #-------------------------------------------------------------------------------------
    # box- and bar plots for SSIM
    def SSIM_boxplot(self, save = False, out_dir = None, fig = None, ax = None, annotate = True):
        if fig is None and ax is None:
            fig = plt.figure(figsize =(10, 7))
            ax = fig.add_subplot(111)
        elif ax is None and fig is not None:
            ax = fig.add_subplot(122)
        else:
            raise('Unexpected input of Matplotlib containers.')
        if self.invar_keys == ['MD','FA', 'uFA', 'C_c', 'C_MD']:
            ticklabels = [r'$\mathrm{MD}$',r'$\mathrm{FA}$', r'$\mathrm{\mu FA}$', r'$\mathrm{C_c}$', r'$\mathrm{C_{MD}}$']        
        #---------------------------------------------------------
        # different modes of averaging:
        if self.brain_wise:
            title_str = 'Brain-wise SSIM'; identifier = 'SSIM_bw'
        elif self.slice_wise:
            title_str = 'Slice-wise SSIM'; identifier = 'SSIM_sw'
        if not self.voxel_wise:
            # bp = ax.boxplot(self.SSIM, patch_artist = True)
            # ax.set_xticklabels(self.invar_keys)
        #---------------------------------------------------------
            #---------------------------------------------------------
            # plot customization:    
            boxprops = dict(linestyle='-', linewidth=1, facecolor='steelblue', edgecolor='steelblue')
            flierprops = dict(marker='o', markerfacecolor='none',
                            markeredgecolor='steelblue', markersize=6) # markersize=12
            capprops = dict(color='steelblue')
            whiskerpros =  dict(color='steelblue')
            medianprops = dict(linestyle='-', linewidth=.75, color='r')
            meanpointprops = dict(marker='^', markeredgecolor='none',
                                markerfacecolor='gold', markersize=6)
            meanlineprops = dict(linestyle='-', linewidth=.75, color='gold')
            #---------------------------------------------------------
            bp = ax.boxplot(self.SSIM, patch_artist = True, showmeans=True, meanline=False,\
                            boxprops=boxprops, flierprops=flierprops, medianprops=medianprops,\
                            meanprops=meanpointprops, capprops=capprops, whiskerprops=whiskerpros)
            ax.set_xticklabels(ticklabels)
            ax.set_xlim(0.3,5.8)         
        # annotate median to boxplot, if desired:
            if annotate:
                for line in bp['medians']:
                    # get position data for median line
                    x, y = line.get_xydata()[1] # top of median line
                    # overlay median value
                    text(x+0.23, y, '%.4f' % y, fontsize = 7.0,
                        horizontalalignment='right', verticalalignment='top') # draw above, right
        #---------------------------------------------------------
        # plot options:
        plt.title(title_str)
        #---------------------------------------------------------
        # save figure:
        if save:
            if out_dir is not None:
                out_pn = os.path.join(self.out_path, out_dir)
                if not os.path.exists(out_pn):
                    makedirs(out_pn)
            else:
                out_pn = self.out_path          
            plt.savefig(os.path.join(out_pn, identifier  + '.pdf'), format='pdf', dpi=1200, bbox_inches='tight')
        return fig, ax

def plot_map_from_nii(nii_path, slice_ind=None, save = True, vmax = 1.0, vmin = 0.0, out_dir = None):       
    # load data from nii:
    img = nib.load(nii_path)
    data = np.asarray(img.get_fdata())
    # x & y are interchanged?
    data = np.flip(np.transpose(data, (1, 0, 2)), axis=0)

    if slice_ind is None:
        slice_ind = data.shape[0]//2

    fig = plt.figure(figsize =(1.5, 1.5))
    ax = fig.add_subplot(111)
    lines = ax.imshow(data[..., slice_ind], cmap = 'gray', vmax = vmax , vmin = vmin)
    ax.axis('off')
    
    if save:
        if out_dir is not None:
            invar_name = os.path.split(nii_path)[1]
            out_pn = os.path.join(out_dir, invar_name[:-7] + '_s' + str(slice_ind))
            if not os.path.exists(out_dir):
                makedirs(out_dir)
        else:
            out_pn = nii_path[:-7] + '_s' + str(slice_ind)
        plt.savefig(out_pn + '.pdf', format='pdf', dpi=1200, bbox_inches='tight',transparent=True, pad_inches=0)

#-------------------------------------------------------------------------------------
# file reading functions:
#-------------------------------------------------------------------------------------
def read_mat(file_path, invar_keys):
    # always pass list as file_path, even when only one file
    # eventually list content of dir or so
    # affines for .nii saving are taken from y_target, y_pred, when fitted by MLP
    # has no field 'nii_h', so affines set to None
    if isinstance(file_path, list):
        data = []
        affines = []
        for path in file_path:
            values = []
            container = sio.loadmat(path)
            for key in invar_keys:
                values.append(container['dps'][key][0,0])
            data.append(values)
            try:
                nii_h = container['dps']['nii_h'][0,0][0,0]
                affines.append(np.transpose(np.concatenate((nii_h['srow_x'], nii_h['srow_y'], nii_h['srow_z'], np.array([0, 0, 0, 1])[:, None]), axis=-1)))
            except:
                affines = None
        data = np.asarray(data, dtype=np.float32) # changed dtype from 64 -> 32
    else:
        raise Exception("file_path argument has to be of type list.")
        # dim order: (file/head, slice z, x, y, n_invar)
    return torch.permute(torch.from_numpy(data), (0, 4, 2, 3, 1)), affines

def read_mask_nii(file_path):
    # always pass list as file_path, even when only one file
    if isinstance(file_path, list):
        data = []
        for path in file_path:
            img = nib.load(path)
            data.append(img.get_fdata())
        data = np.asarray(data, dtype=bool)
    else:
        raise Exception("file_path argument has to be of type list.")
    return torch.permute(torch.from_numpy(data), (0, 3, 1, 2))

# Define the Function
def plot_image_matrix(image_lists, row_labels = ['GT', 'NLLS', 'MLP', 'NLLS-GT', 'MLP-GT'] ,
                       col_labels=[r'$\mathrm{MD}$',r'$\mathrm{FA}$', r'$\mathrm{\mu FA}$', r'$\mathrm{C_c}$', r'$\mathrm{C_{MD}}$'] , identifier = 'Native',
                       slice_ind=None, cmap='gray', vmax=None, vmin=None, save_path=None, cmap_diff=cmr.iceburn, vmax_diff=0.3, vmin_diff=-0.3,
                       plot_diff = True, wspace = -0.225, hspace = 0.025):
    """
    Plots images from a list of lists in a matrix, each sublist gets its own row and every entry is a column.
    
    Parameters:
    - image_lists (list of list of str): List of lists of paths to the images.
    - row_labels (list of str): Labels for the rows.
    - col_labels (list of str): Labels for the columns.
    - slice_ind (int, optional): Index of the slice to display.
    - cmap (str, optional): Colormap to use for displaying the images.
    - vmax (list of float, optional): List of maximum values for colormap scaling.
    - vmin (list of float, optional): List of minimum values for colormap scaling.
    - save_path (str, optional): Path to save the resulting plot.
    - cmap_diff (str, optional): Colormap to use for displaying the difference images.
    - vmax_diff (float, optional): Maximum value for colormap scaling of difference images.
    - vmin_diff (float, optional): Minimum value for colormap scaling of difference images.
    """
    # Determine the number of rows and columns
    num_rows = len(image_lists)
    num_cols = len(image_lists[0]) if num_rows > 0 else 0
    
    # # Check if the number of rows is 3
    # if num_rows != 3:
    #     raise ValueError("The number of rows must be 3.")
    
    # Check if vmax and vmin are lists of the correct length
    if vmax is not None and len(vmax) != num_cols:
        raise ValueError("vmax must be a list with the same length as the number of columns.")
    if vmin is not None and len(vmin) != num_cols:
        raise ValueError("vmin must be a list with the same length as the number of columns.")
    
    # Create the figure with tighter grid spacing
    plt.style.use("dark_background")
    if plot_diff:
        fig, axes = plt.subplots(num_rows + 2, num_cols, figsize=(1.85 * num_cols, 3.0 * num_rows),
                 gridspec_kw={"wspace": wspace, "hspace": hspace})  # Negative spacing for overlap
    else:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(1.85 * num_cols, 3.0 * num_rows),
                                 gridspec_kw={"wspace": wspace, "hspace": hspace})  # Negative spacing for overlap

    font_size = 14

    for row_idx, image_list in enumerate(image_lists):
        for col_idx, image_path in enumerate(image_list):
            # Load the image
            img = np.flip(np.transpose(nib.load(image_path).get_fdata(), (1, 0, 2)), axis=0)
            # Plot the image
            ax = axes[row_idx, col_idx]
            if slice_ind is None:
                slice_ind = img.shape[2] // 2  # Default to the middle slice if not provided
            lines = ax.imshow(img[:, :, slice_ind], cmap=cmap, vmax=vmax[col_idx] if vmax else None, vmin=vmin[col_idx] if vmin else None)  # Display the specified slice with colormap and windowing
            ax.axis('off')
            
            # if row_idx == num_rows - 1:
            #     divider = make_axes_locatable(ax)
            #     cax = divider.append_axes("bottom", size="5%", pad=0.1)
            #     cbar = fig.colorbar(lines, cax=cax, orientation='horizontal')
            #     cbar.ax.tick_params(labelsize=font_size)
            # Add column labels
            if row_idx == 0:
                ax.set_title(col_labels[col_idx], fontsize=font_size, color="white")

            # if col_idx == 0:
            #     ax.set_ylabel(row_labels[row_idx], fontsize=font_size)

            # Add colorbars
        # for col_idx in range(num_cols):
        #     # Create a new axis for the colorbar
        #     cbar_ax = fig.add_axes([0.05 + col_idx * 0.185, 0.05, 0.015, 0.02])
        #     # Add the colorbar
        #     fig.colorbar(lines, cax=cbar_ax, orientation='horizontal')

    if plot_diff:
        # Compute and plot differences
        for col_idx in range(num_cols):
            # Load images for row 1, row 2, and row 3
            img_row1 = nib.load(image_lists[0][col_idx]).get_fdata()
            img_row2 = nib.load(image_lists[1][col_idx]).get_fdata()
            img_row3 = nib.load(image_lists[2][col_idx]).get_fdata()
            img_row1 = np.flip(np.transpose(img_row1, (1, 0, 2)), axis=0)
            img_row2 = np.flip(np.transpose(img_row2, (1, 0, 2)), axis=0)
            img_row3 = np.flip(np.transpose(img_row3, (1, 0, 2)), axis=0)
            # Compute differences
            diff_row2_row1 = img_row2 - img_row1
            diff_row3_row1 = img_row3 - img_row1
            
            # Plot differences
            ax_diff_row2_row1 = axes[num_rows, col_idx]
            ax_diff_row2_row1.imshow(diff_row2_row1[:, :, slice_ind], cmap=cmap_diff, vmax=vmax_diff, vmin=vmin_diff)
            ax_diff_row2_row1.axis('off')
            # if col_idx == 0:
            #     ax_diff_row2_row1.set_ylabel('Diff (Row 2 - Row 1)', fontsize=font_size)
            
            ax_diff_row3_row1 = axes[num_rows + 1, col_idx]
            ax_diff_row3_row1.imshow(diff_row3_row1[:, :, slice_ind], cmap=cmap_diff, vmax=vmax_diff, vmin=vmin_diff)
            ax_diff_row3_row1.axis('off')
            # if col_idx == 0:
            #     ax_diff_row3_row1.set_ylabel('Diff (Row 3 - Row 1)', fontsize=font_size)

    # Add row labels
    for i, label in enumerate(row_labels):
        axes[i, 0].text(
            -0.3, 0.5, label, va="center", ha="center", rotation=90,
            fontsize=font_size, color="white", transform=axes[i, 0].transAxes
        )

    # Save the plot if save_path is provided
    if save_path:
        if not os.path.exists(save_path):
            makedirs(save_path)
        out_pn = os.path.join(save_path, "matrix_plot_" + identifier)
        plt.savefig(out_pn + '.pdf', format='pdf', dpi=1200, bbox_inches='tight',pad_inches=0)
    
    plt.show()


#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# trash:
'''
obsolete:
    def save_diff(self, sep_files = False, out_dir = None):
        if out_dir is not None:
            out_pn = os.path.join(self.out_path, out_dir)
            if not os.path.exists(out_pn):
                makedirs(out_pn)
        else:
            out_pn = self.out_path
        if sep_files:
            for file_ind in range(self.diff.size()[0]):
                sio.savemat(out_pn + 'diff_f' + str(file_ind+1) + '.mat',\
                    {'diff': self.diff[file_ind, ...].detach().cpu().numpy()})
        else:
            sio.savemat(os.path.join(out_pn + 'diff' + '.mat'),\
                {'diff': self.diff.detach().cpu().numpy()})

    def save_abs_diff(self, sep_files = False, out_dir = None):
        if out_dir is not None:
            out_pn = os.path.join(self.out_path, out_dir)
            if not os.path.exists(out_pn):
                makedirs(out_pn)
        else:
            out_pn = self.out_path
        if sep_files:
            for file_ind in range(self.abs_diff.size(0)):
                sio.savemat(os.path.join(out_pn, 'abs_diff_f' + str(file_ind+1) + '.mat'),\
                    {'abs_diff': self.abs_diff[file_ind, ...].detach().cpu().numpy()})
        else:
            sio.savemat(os.path.join(out_pn, 'abs_diff' + '.mat'),\
                {'abs_diff': self.abs_diff.detach().cpu().numpy()})
            
    def save_rel_diff(self, sep_files = False, out_dir = None):
        if out_dir is not None:
            out_pn = os.path.join(self.out_path, out_dir)
            if not os.path.exists(out_pn):
                makedirs(out_pn)
        else:
            out_pn = self.out_path
        if sep_files:
            for file_ind in range(self.rel_diff.size()[0]):
                sio.savemat(out_pn + 'rel_diff_f' + str(file_ind+1) + '.mat',\
                    {'rel_diff': self.rel_diff[file_ind, ...].detach().cpu().numpy()})
        else:
            sio.savemat(os.path.join(out_pn + 'rel_diff' + '.mat'),\
                {'rel_diff': self.rel_diff.detach().cpu().numpy()})

'''
'''
    # def compute_nRMSE(self):
    #     # changed calc. to not scale with n, analogous to pSNR
    #     # still seems to scale with n..
    #     RSE = torch.square(self.rel_diff)
    #     RSE[~self.mask] = 0.0
    #     if self.slice_wise:
    #         # Summe über (x,y), nRMSE output hat dim (file, z, invar)
    #         dim=(2,3)
    #     elif self.brain_wise:
    #         # Summe über (z,x,y), nRMSE output hat dim (file, invar)
    #         dim=(1,2,3)
    #     elif self.voxel_wise:
    #         # Summe über (file,z,x,y), nRMSE output hat dim (invar)
    #         dim=(0,1,2,3)
    #     n = torch.count_nonzero(self.mask,  dim=dim)
    #     # carefull, 1/n introduces nan here, when entire slice is excluded in mask!
    #     self.nRMSE = torch.sqrt(torch.multiply(1/n,torch.sum(RSE, dim=dim)))
    #     del RSE, n
'''