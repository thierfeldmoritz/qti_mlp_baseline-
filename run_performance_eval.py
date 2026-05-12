#-------------------------------------------------------------------------------------
# script to run performance evaluation for QTI fits/prediction
# might be better in a jupyter notebook..
#-------------------------------------------------------------------------------------
# imports

from Eval_metrics import QTI_Fit_Dataset
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------
# paths

pred_path = [\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P1/qtip_sub_min_pp_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P2/qtip_sub_min_pp_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P17/qtip_sub_min_pp_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P3/qtip_sub_min_pp_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P5/qtip_sub_min_pp_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P6/qtip_sub_min_pp_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P8/qtip_sub_min_pp_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P9/qtip_sub_min_pp_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P10/qtip_sub_min_pp_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P11/qtip_sub_min_pp_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P12/qtip_sub_min_pp_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P13/qtip_sub_min_pp_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P14/qtip_sub_min_pp_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P15/qtip_sub_min_pp_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P16/qtip_sub_min_pp_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P7/qtip_sub_min_pp_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P18/qtip_sub_min_pp_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P4/qtip_sub_min_pp_pipe_2/dps3.mat',\
            ]
target_path = [\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P1/qtip_full_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P2/qtip_full_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P17/qtip_full_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P3/qtip_full_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P5/qtip_full_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P6/qtip_full_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P8/qtip_full_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P9/qtip_full_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P10/qtip_full_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P11/qtip_full_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P12/qtip_full_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P13/qtip_full_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P14/qtip_full_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P15/qtip_full_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P16/qtip_full_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P7/qtip_full_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P18/qtip_full_pipe_2/dps3.mat',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P4/qtip_full_pipe_2/dps3.mat',\
            ]
mask_path = [\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P1/manual_mask.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P2/manual_mask.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P17/manual_mask.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P3/manual_mask.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P5/manual_mask.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P6/manual_mask.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P8/manual_mask.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P9/manual_mask.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P10/manual_mask.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P11/manual_mask.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P12/manual_mask.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P13/manual_mask.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P14/manual_mask.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P15/manual_mask.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P16/manual_mask.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P7/manual_mask.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P18/manual_mask.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P4/manual_mask.nii.gz',\
            ]

#-------------------------------------------------------------------------------------
# top & bottom slice exclusion:
# slice_ind_t = [[0, 23], [0, 23], [1, 23], [1, 23],\
#               [0, 22], [0, 23], [0, 23], [0, 22], [0, 22],\
#                 [0, 23], [1, 22], [0, 22], [1, 23], [0, 22]] # 3:

# slice_ind_v = [[2, 23], [0, 23], [0, 22]] # :3

slice_ind = [[2, 23], [0, 23], [0, 22], [0, 23], [0, 23], [1, 23], [1, 23],\
              [0, 22], [0, 23], [0, 23], [0, 22], [0, 22],\
                [0, 23], [1, 22], [0, 22], [1, 23], [0, 22], [0, 23]] # 0:

#-------------------------------------------------------------------------------------
# create QTI_Fit_Dataset instance:
ds_eval = QTI_Fit_Dataset(pred_path, target_path, mask_path, invar_keys = ['FA', 'uFA'], slice_ind=slice_ind) # 's0', 'MD', 'FA', 'uFA', 'C_c', 'C_MD'

#-------------------------------------------------------------------------------------
# perform preparations:
ds_eval.expand_mask()
ds_eval.mask_nan_vals()
ds_eval.mask_zero_vals()

#-------------------------------------------------------------------------------------
# perform calculations:
ds_eval.compute_abs_diff()
ds_eval.compute_rel_diff()
ds_eval.compute_nRMSE()
ds_eval.compute_pSNR()

#-------------------------------------------------------------------------------------
# plotting:
fig1, ax1 = ds_eval.nRMSE_boxplot(save=True, out_dir='nRMSE_plots')
fig2, ax2 = ds_eval.pSNR_boxplot(save=True, out_dir='nRMSE_plots')

# toggle options: diff = 'rel', 'abs' (default = '' for diff)
# ds_eval.save_diff_2_mat(out_dir='Diff', diff='abs', sep_files=True, sep_invars=False)
# ds_eval.save_diff_2_nii(diff='abs', out_dir='Diff_nii')
figs, axs, sldrs_file, sldrs_slice = ds_eval.plot_diff(diff='abs', save=True, out_dir='Diff_plots')
print('piss')
# ds_eval.save_nRMSE(out_dir='nRMSE')

ds_eval.set_brain_wise(False)
ds_eval.set_slice_wise(True)
ds_eval.compute_nRMSE()
ds_eval.compute_pSNR()

ds_eval.set_slice_wise(False)
ds_eval.compute_nRMSE()
ds_eval.compute_pSNR()
