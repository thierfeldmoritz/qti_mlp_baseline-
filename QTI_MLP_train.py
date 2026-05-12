# main script for training QTI_MLP
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# imports
import torch
from torch import nn
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
from torch.optim import lr_scheduler

from os.path import split, join, exists
from os import makedirs

import numpy as np
import pandas as pd

from QTI_MLP import QTI_Dataset, QTI_MLP, QTIb_MLP, train_loop, test_loop, plot_hist

#-------------------------------------------------------------------------------------
# paths
'''
nii_path = [\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P1/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P2/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P17/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P3/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P5/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P6/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P8/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P9/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P10/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P11/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P12/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P13/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P14/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P15/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P16/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P7/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P18/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz'\
            ]#'/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P4/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz'
scalar_invars_path = [\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P1/dps3_nan_clmpd_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P2/dps3_nan_clmpd_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P17/dps3_nan_clmpd_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P3/dps3_nan_clmpd_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P5/dps3_nan_clmpd_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P6/dps3_nan_clmpd_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P8/dps3_nan_clmpd_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P9/dps3_nan_clmpd_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P10/dps3_nan_clmpd_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P11/dps3_nan_clmpd_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P12/dps3_nan_clmpd_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P13/dps3_nan_clmpd_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P14/dps3_nan_clmpd_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P15/dps3_nan_clmpd_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P16/dps3_nan_clmpd_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P7/dps3_nan_clmpd_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P18/dps3_nan_clmpd_clmpd.mat'\
            ]#'/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P4/manual_mask.nii.gz'
mask_path = [\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P1/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P2/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P17/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P3/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P5/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P6/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P8/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P9/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P10/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P11/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P12/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P13/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P14/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P15/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P16/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P7/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P18/manual_mask.nii.gz'\
            ]#'/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P4/manual_mask.nii.gz'
xps_path = [\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P1/xps_sub_min_pp.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P2/xps_sub_min_pp.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P17/xps_sub_min_pp.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P3/xps_sub_min_pp.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P5/xps_sub_min_pp.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P6/xps_sub_min_pp.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P8/xps_sub_min_pp.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P9/xps_sub_min_pp.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P10/xps_sub_min_pp.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P11/xps_sub_min_pp.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P12/xps_sub_min_pp.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P13/xps_sub_min_pp.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P14/xps_sub_min_pp.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P15/xps_sub_min_pp.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P16/xps_sub_min_pp.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P7/xps_sub_min_pp.mat',\
             '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P18/xps_sub_min_pp.mat'\
            ]#'/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Data/QTI_ML_TS/P4/xps_sub_min_pp.mat'
'''
#-------------------------------------------------------------------------------------
# paths test (original order of 11P5 exp, with ts, vs, trs):
'''
nii_path = [\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P15/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P16/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P7/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P18/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P4/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P1/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P17/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P2/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P3/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P5/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P6/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P8/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P9/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P10/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P11/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P12/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P13/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P14/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz'
            ]#
scalar_invars_path = [\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P15/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P16/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P7/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P18/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P4/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P1/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P17/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P2/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P3/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P5/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P6/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P8/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P9/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P10/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P11/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P12/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P13/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P14/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat'
             ]#
mask_path = [\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P15/manual_mask.nii.gz',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P16/manual_mask.nii.gz',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P7/manual_mask.nii.gz',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P18/manual_mask.nii.gz',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P4/manual_mask.nii.gz',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P1/manual_mask.nii.gz',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P17/manual_mask.nii.gz',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P2/manual_mask.nii.gz',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P3/manual_mask.nii.gz',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P5/manual_mask.nii.gz',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P6/manual_mask.nii.gz',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P8/manual_mask.nii.gz',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P9/manual_mask.nii.gz',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P10/manual_mask.nii.gz',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P11/manual_mask.nii.gz',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P12/manual_mask.nii.gz',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P13/manual_mask.nii.gz',\
             '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P14/manual_mask.nii.gz'
             ]#

'''
#-------------------------------------------------------------------------------------
# paths hard copy on HD
'''
nii_path = [\
            '/Volumes/Maxtor/ML_TS/P1/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P10/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P2/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P11/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P3/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P12/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P4/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P13/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P5/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P14/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P6/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P15/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P7/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P16/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P8/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P17/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P9/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P18/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz'
            ]#

scalar_invars_path = [\
             '/Volumes/Maxtor/ML_TS/P1/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/ML_TS/P10/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/ML_TS/P2/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/ML_TS/P11/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/ML_TS/P3/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/ML_TS/P12/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/ML_TS/P4/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/ML_TS/P13/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/ML_TS/P5/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/ML_TS/P14/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/ML_TS/P6/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/ML_TS/P15/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/ML_TS/P7/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/ML_TS/P16/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/ML_TS/P8/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/ML_TS/P17/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/ML_TS/P9/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
             '/Volumes/Maxtor/ML_TS/P18/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat'
             ]#
mask_path = [\
             '/Volumes/Maxtor/ML_TS/P1/manual_mask.nii.gz',\
             '/Volumes/Maxtor/ML_TS/P10/manual_mask.nii.gz',\
             '/Volumes/Maxtor/ML_TS/P2/manual_mask.nii.gz',\
             '/Volumes/Maxtor/ML_TS/P11/manual_mask.nii.gz',\
             '/Volumes/Maxtor/ML_TS/P3/manual_mask.nii.gz',\
             '/Volumes/Maxtor/ML_TS/P12/manual_mask.nii.gz',\
             '/Volumes/Maxtor/ML_TS/P4/manual_mask.nii.gz',\
             '/Volumes/Maxtor/ML_TS/P13/manual_mask.nii.gz',\
             '/Volumes/Maxtor/ML_TS/P5/manual_mask.nii.gz',\
             '/Volumes/Maxtor/ML_TS/P14/manual_mask.nii.gz',\
             '/Volumes/Maxtor/ML_TS/P6/manual_mask.nii.gz',\
             '/Volumes/Maxtor/ML_TS/P15/manual_mask.nii.gz',\
             '/Volumes/Maxtor/ML_TS/P7/manual_mask.nii.gz',\
             '/Volumes/Maxtor/ML_TS/P16/manual_mask.nii.gz',\
             '/Volumes/Maxtor/ML_TS/P8/manual_mask.nii.gz',\
             '/Volumes/Maxtor/ML_TS/P17/manual_mask.nii.gz',\
             '/Volumes/Maxtor/ML_TS/P9/manual_mask.nii.gz',\
             '/Volumes/Maxtor/ML_TS/P18/manual_mask.nii.gz'
             ]#
'''
#-------------------------------------------------------------------------------------
# XYZ fit GT on local HD

nii_path = [\
            '/Volumes/Maxtor/ML_TS/P1/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P10/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P2/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P11/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P3/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P12/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P4/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P13/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P5/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P14/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P6/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P15/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P7/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P16/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P8/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P17/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P9/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P18/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz'
            ]#

scalar_invars_path = [\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P1/qtip_pipe_2_XYZ/dps3_GT_nan_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P10/qtip_pipe_2_XYZ/dps3_GT_nan_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P2/qtip_pipe_2_XYZ/dps3_GT_nan_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P11/qtip_pipe_2_XYZ/dps3_GT_nan_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P3/qtip_pipe_2_XYZ/dps3_GT_nan_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P12/qtip_pipe_2_XYZ/dps3_GT_nan_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P4/qtip_pipe_2_XYZ/dps3_GT_nan_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P13/qtip_pipe_2_XYZ/dps3_GT_nan_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P5/qtip_pipe_2_XYZ/dps3_GT_nan_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P14/qtip_pipe_2_XYZ/dps3_GT_nan_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P6/qtip_pipe_2_XYZ/dps3_GT_nan_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P15/qtip_pipe_2_XYZ/dps3_GT_nan_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P7/qtip_pipe_2_XYZ/dps3_GT_nan_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P16/qtip_pipe_2_XYZ/dps3_GT_nan_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P8/qtip_pipe_2_XYZ/dps3_GT_nan_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P17/qtip_pipe_2_XYZ/dps3_GT_nan_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P9/qtip_pipe_2_XYZ/dps3_GT_nan_clmpd.mat',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P18/qtip_pipe_2_XYZ/dps3_GT_nan_clmpd.mat'
             ]#
mask_path = [\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P1/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P10/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P2/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P11/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P3/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P12/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P4/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P13/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P5/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P14/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P6/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P15/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P7/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P16/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P8/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P17/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P9/manual_mask.nii.gz',\
             '/Users/olivergoedicke/Documents/Uni/PhD/MA_ML_TS/qti-preprocess/P18/manual_mask.nii.gz'
             ]#

# hyperparams:
invar_keys = ['MD', 'FA', 'uFA', 'C_c', 'C_MD'] #'s0', 'MD', 'FA', 'uFA', 'C_c', 'C_MD', 'C_M', 'C_mu', 'MKa', 'MKi']
identifier = 'QTI_MLP_16P_sub_min'
model_dir = '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP'
out_dir = 'Paper_Base_Run'
#--------------------------------------------------------
learning_rate = 1e-3
batch_size = 512
epochs = 100
#--------------------------------------------------------
final_sigmoid = False
zscore_output = True
if zscore_output:
    norm_keys = invar_keys
else:
    norm_keys = ['None']
#--------------------------------------------------------
schedule = True
decay_const = 0.925
sched_start = 14; sched_end = 74
#--------------------------------------------------------
layer_norm = False
#--------------------------------------------------------
bias = True
#--------------------------------------------------------
# adjust MLP type and preparations below!
xps_path = None # must be None, when QTI_MLP is used
#--------------------------------------------------------
# thus far, cpu was faster than mps:
device = ("cpu")
# Initialize the loss function (squared L2-loss)
loss_fn = nn.MSELoss()# nn.L1Loss()
#--------------------------------------------------------
# top & bottom slice exclusion:
'''
# slice_ind_t = [[0, 23], [0, 23], [1, 23], [1, 23],\
#               [0, 22], [0, 23], [0, 23], [0, 22], [0, 22],\
#                 [0, 23], [1, 22], [0, 22], [1, 23], [0, 22]] # 3:

# slice_ind_v = [[2, 23], [0, 23], [0, 22]] # :2

# slice_ind_v = [[2, 23], [0, 22]] # :2

# slice_ind_t = [[0, 22], [0, 23], [1, 22], [0, 22], [1, 23], [0, 22]] # -6:
# slice_ind_v = [[2, 23], [0, 23]] # :2

slice_ind_11P5 = [[0, 23], [1, 22], [0, 22], [1, 23], [0, 22],\
                  [2, 23], [0, 22],\
                  [0, 23], [0, 23], [0, 23], [1, 23], [1, 23],\
                   [0, 22], [0, 23], [0, 23], [0, 22], [0, 22], [0, 23]] #  -5:, :2, 3:-4

'''

slice_ind = [[2, 21] for _ in range(len(nii_path))]

#-------------------------------------------------------------------------------------
# Ensemble Loop
#-------------------------------------------------------------------------------------
# Settings:
# # no ensembling (one run): ensemble factor = 0 
# ens_factor = 6
# # test set ensembling for protocol statistics
# # code should be adapted for training set ensembling

# # try: 
# #  vs_size = len(nii_path)//ens_factor
# # except:
# #   ts_size = 1
# #   ens_factor = 1

#-------------------------------------------------------------------------------------
# manual:
ts_size = 1
vs_size = 1
ens_factor = 18 # len(nii_path)//(ts_size+vs_size)
#-------------------------------------------------------------------------------------

if ts_size + vs_size >= len(nii_path):
  raise Exception('Invalid ensembling configuration, training set size is zero.')

for current_step in range(ens_factor):
  if ens_factor > 1:
    print('Training Ensemble {}:'.format(current_step + 1))
  #-------------------------------------------------------------------------------------
  # index selection
  current_ind = [i for i in range(len(nii_path))]
  current_ind = np.roll(current_ind, shift=-current_step*ts_size) # ts_size
  ts_ind = current_ind[0:ts_size]; vs_ind = current_ind[ts_size:ts_size+vs_size]
  tr_ind = current_ind[ts_size+vs_size:]  
  
  ts_path = [nii_path[ind] for ind in ts_ind]; vs_path = [nii_path[ind] for ind in vs_ind]
  #-------------------------------------------------------------------------------------
  # create QTI_Dataset instance:
  ds_train = QTI_Dataset([nii_path[ind] for ind in tr_ind],
                         [scalar_invars_path[ind] for ind in tr_ind],
                         [mask_path[ind] for ind in tr_ind],
                         [slice_ind[ind] for ind in tr_ind],
                        invar_keys, zscore_output = zscore_output) # xps_path, btens_der=True)
  ds_val = QTI_Dataset(vs_path,
                         [scalar_invars_path[ind] for ind in vs_ind],
                         [mask_path[ind] for ind in vs_ind],
                         [slice_ind[ind] for ind in vs_ind],
                        invar_keys, zscore_output = zscore_output)

  #-------------------------------------------------------------------------------------
  # perform preparations/transformations:
  # signal input:
  #--------------------------------------------------------
  ds_train.thresh_neg_vals(); ds_val.thresh_neg_vals()
  ds_train.mask_zero_vals(); ds_val.mask_zero_vals()
  ds_train.mask_nan_vals(); ds_val.mask_nan_vals()
  # input normalization:
  #--------------------------------------------------------
  ds_train.apply_masked_tensor(); ds_val.apply_masked_tensor()
  ds_train.zscore_norm_input(); ds_val.zscore_norm_input()

  # btens:
  #--------------------------------------------------------
  '''
  # ds_train.zscore_norm_btens; ds_test.zscore_norm_btens()
  # ds_train.min_max_norm_btens(); ds_test.min_max_norm_btens()
  # ds_train.scale_btens(1e-10); ds_test.scale_btens(1e-10)

  # btens derived quantities:
  #--------------------------------------------------------
  # ds_train.scale_bval(5e-10); ds_test.scale_bval(5e-10)
  # ds_train.bvec_cart2sph(); ds_test.bvec_cart2sph()
  # ds_train.btens_drop_b_eta(); ds_test.btens_drop_b_eta()

  # ds_train.cat_inputs(); ds_test.cat_inputs()
  '''
  # labels:
  #--------------------------------------------------------
  # pass list of invar keys that should be normalized here
  ds_train.zscore_norm_labels(norm_keys); ds_val.zscore_norm_labels(norm_keys)
  # ds_train.min_max_norm_labels(invar_keys); ds_test.min_max_norm_labels(invar_keys)

  # mask and flatten:
  #--------------------------------------------------------
  ds_train.flatten_slice_dim(); ds_val.flatten_slice_dim()
  ds_train.apply_mask(); ds_val.apply_mask()

  #-------------------------------------------------------------------------------------
  # create DataLoader
  train_dataloader = DataLoader(ds_train, batch_size=batch_size , shuffle=True, pin_memory=True)
  test_dataloader = DataLoader(ds_val, batch_size=batch_size, shuffle=True, pin_memory=True)

  #-------------------------------------------------------------------------------------
  # create QTI_MLP instance
  n_scans = ds_train.X.size()[-1]; n_invars = ds_train.y.size()[-1]

  mlp = QTI_MLP(n_scans, n_invars, layer_norm, final_sigmoid, bias).to(device)

  # optimizer
  #--------------------------------------------------------
  optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
  # lr scheduler
  #--------------------------------------------------------
  if schedule:
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=decay_const)

  #-------------------------------------------------------------------------------------
  # PyTorch TensorBoard support
  timestamp = datetime.now().strftime('%Y%m%d_%H%M')#_%H%M%S

  ts_names = [split(split(path)[0])[1] for path in ts_path]
  vs_names = [split(split(path)[0])[1] for path in vs_path]

  writer = SummaryWriter(join('runs', out_dir, identifier + '_ens_f{}_ts_'.format(ens_factor) +
                    ''.join(ts_names) + '_vs_'+ ''.join(vs_names)
                    + '_{}h_LR{:.0e}_bs{}'.format(timestamp[:11], learning_rate, batch_size)))
  epoch_number = 0

  #-------------------------------------------------------------------------------------
  # perform training:
  #-------------------------------------------------------------------------------------

  # best_vloss = 1_000_000.
  tic = time.time()
  for epoch in range(epochs):
      print('EPOCH {}:'.format(epoch_number + 1))
      mlp.to(device)
      # train
      avg_loss = train_loop(train_dataloader, mlp, loss_fn, optimizer, device, batch_size) # , epoch_number, writer
      # test
      avg_vloss = test_loop(test_dataloader, mlp, loss_fn, device, batch_size)

      if schedule and ((epoch >= sched_start) and (epoch <= sched_end)):
        scheduler.step()
        print('Current learning rate: {:.0e}'.format(optimizer.param_groups[0]["lr"]))
      print('Epoch loss/voxel train {} valid {}'.format(avg_loss, avg_vloss))

      # Log the running loss averaged per epoch per voxel
      # for both training and validation
      writer.add_scalars('Training vs. Validation Loss',
                      { 'Training' : avg_loss, 'Validation' : avg_vloss},
                      epoch_number + 1)
      # Log lr, if scheduler is used
      if schedule:
        writer.add_scalars('Learning Rate',
                        {'LR' : optimizer.param_groups[0]["lr"]},
                        epoch_number + 1)    
      writer.flush()

      '''
            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                    best_vloss = avg_vloss
                    model_path = '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/'\
                        + identifier + '_{}_{}_LR{}_bs{}'.format(timestamp[:11], epoch_number, learning_rate, batch_size)
                    torch.save(mlp.state_dict(), model_path)
      '''
      epoch_number += 1

  toc = time.time()
  print('Elapsed time: {}'.format(tic-toc))
  #-------------------------------------------------------------------------------------
  # save the model:
  #-------------------------------------------------------------------------------------
  # if ts_size == 0:
  #   ts_names = ['']

  model_path = join(model_dir, out_dir)
  if not exists(model_path):
    makedirs(model_path)
  model_path = join(model_path, identifier + '_ens_f{}_ts_'.format(ens_factor) +
                    ''.join(ts_names) + '_vs_'+ ''.join(vs_names)
                    + '_{}h_ep{}_LR{:.0e}_bs{}'.format(timestamp[:11], epoch_number, learning_rate, batch_size))
  if final_sigmoid:
    model_path += '_sigm'
  elif zscore_output:
    model_path += '_outnorm'
  if layer_norm:
    model_path += '_lnorm'
  if not bias:
    model_path += '_nobias'
  if schedule:
    model_path += '_sched_{}'.format(int(100*decay_const))

  torch.save(mlp.state_dict(), model_path)

  #-------------------------------------------------------------------------------------
  # write output zscore to .csv, if used:
  if ds_train.zscore_output:
    out = ds_train.zscore.numpy()
    df = pd.DataFrame(out, index=['mean', 'std'])
    out_pn = model_path + '_zscore.csv'
    df.to_csv(out_pn,index=False) #save to file
    print('Output z-score:\nmean =  {}\nstd =  {}\n'.format(ds_train.zscore[0,:], ds_train.zscore[1,:]))


#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
