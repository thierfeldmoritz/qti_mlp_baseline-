#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# main script for predicting with QTI_MLP
#-------------------------------------------------------------------------------------
# imports
import torch
from torch import nn

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as sio
from os.path import split, join, exists
from os import makedirs

import time

from QTI_MLP import QTI_Dataset, QTI_MLP, QTIb_MLP, rev_zscore, plot_hist

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# paths hard copy on HD
'''
model_path = [\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_17P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts__vs_P1_20231031_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_17P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts__vs_P10_20231031_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_17P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts__vs_P2_20231031_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_17P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts__vs_P11_20231031_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_17P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts__vs_P3_20231031_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_17P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts__vs_P12_20231031_15h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_17P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts__vs_P4_20231031_15h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_17P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts__vs_P13_20231031_15h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_17P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts__vs_P5_20231031_15h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_17P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts__vs_P14_20231031_15h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_17P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts__vs_P6_20231031_15h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_17P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts__vs_P15_20231031_16h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_17P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts__vs_P7_20231031_16h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_17P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts__vs_P16_20231031_16h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_17P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts__vs_P8_20231031_16h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_17P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts__vs_P17_20231031_16h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_17P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts__vs_P9_20231031_16h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_17P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts__vs_P18_20231031_17h_ep100_LR1e-03_bs512_outnorm_sched_92'
            ]#
'''
# paper base run xyz
'''
model_path = [\
            '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP/Paper_Base_Run/QTI_MLP_16P_sub_min_ens_f18_ts_P1_vs_P10_20241203_13h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP/Paper_Base_Run/QTI_MLP_16P_sub_min_ens_f18_ts_P10_vs_P2_20241203_13h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP/Paper_Base_Run/QTI_MLP_16P_sub_min_ens_f18_ts_P2_vs_P11_20241203_13h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP/Paper_Base_Run/QTI_MLP_16P_sub_min_ens_f18_ts_P11_vs_P3_20241203_13h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP/Paper_Base_Run/QTI_MLP_16P_sub_min_ens_f18_ts_P3_vs_P12_20241203_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP/Paper_Base_Run/QTI_MLP_16P_sub_min_ens_f18_ts_P12_vs_P4_20241203_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP/Paper_Base_Run/QTI_MLP_16P_sub_min_ens_f18_ts_P13_vs_P5_20241203_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP/Paper_Base_Run/QTI_MLP_16P_sub_min_ens_f18_ts_P5_vs_P14_20241203_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP/Paper_Base_Run/QTI_MLP_16P_sub_min_ens_f18_ts_P4_vs_P13_20241203_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP/Paper_Base_Run/QTI_MLP_16P_sub_min_ens_f18_ts_P14_vs_P6_20241203_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP/Paper_Base_Run/QTI_MLP_16P_sub_min_ens_f18_ts_P6_vs_P15_20241203_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP/Paper_Base_Run/QTI_MLP_16P_sub_min_ens_f18_ts_P15_vs_P7_20241203_15h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP/Paper_Base_Run/QTI_MLP_16P_sub_min_ens_f18_ts_P7_vs_P16_20241203_15h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP/Paper_Base_Run/QTI_MLP_16P_sub_min_ens_f18_ts_P16_vs_P8_20241203_15h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP/Paper_Base_Run/QTI_MLP_16P_sub_min_ens_f18_ts_P8_vs_P17_20241203_15h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP/Paper_Base_Run/QTI_MLP_16P_sub_min_ens_f18_ts_P17_vs_P9_20241203_15h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP/Paper_Base_Run/QTI_MLP_16P_sub_min_ens_f18_ts_P9_vs_P18_20241203_15h_ep100_LR1e-03_bs512_outnorm_sched_92',\
            '/Users/olivergoedicke/Documents/Uni/PhD/Results/QTI_ML/models/QTI_MLP/Paper_Base_Run/QTI_MLP_16P_sub_min_ens_f18_ts_P18_vs_P1_20241203_15h_ep100_LR1e-03_bs512_outnorm_sched_92'
            ]#

model_path = [\
              '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P1_vs_P10_20231029_12h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P10_vs_P2_20231029_12h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P2_vs_P11_20231029_12h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P11_vs_P3_20231029_12h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P3_vs_P12_20231029_12h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P12_vs_P4_20231029_12h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P4_vs_P13_20231029_13h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P13_vs_P5_20231029_13h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P5_vs_P14_20231029_13h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P14_vs_P6_20231029_13h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P6_vs_P15_20231029_13h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P15_vs_P7_20231029_13h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P7_vs_P16_20231029_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P16_vs_P8_20231029_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P8_vs_P17_20231029_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P17_vs_P9_20231029_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P9_vs_P18_20231029_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P18_vs_P1_20231029_14h_ep100_LR1e-03_bs512_outnorm_sched_92'
              ] 
'''
'''
model_path = [\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/ENSEMBLE_Prot_Vgl_sub_naive_min_3_pp/QTI_MLP_13P_ens_f_sub_naive_min_3_ens_f6_ts_P1P10P2_vs_P11P3_20240112_12h_ep50_LR1e-04_bs512_outnorm',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/ENSEMBLE_Prot_Vgl_sub_naive_min_3_pp/QTI_MLP_13P_ens_f_sub_naive_min_3_ens_f6_ts_P11P3P12_vs_P4P13_20240112_12h_ep50_LR1e-04_bs512_outnorm',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/ENSEMBLE_Prot_Vgl_sub_naive_min_3_pp/QTI_MLP_13P_ens_f_sub_naive_min_3_ens_f6_ts_P4P13P5_vs_P14P6_20240112_12h_ep50_LR1e-04_bs512_outnorm',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/ENSEMBLE_Prot_Vgl_sub_naive_min_3_pp/QTI_MLP_13P_ens_f_sub_naive_min_3_ens_f6_ts_P14P6P15_vs_P7P16_20240112_12h_ep50_LR1e-04_bs512_outnorm',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/ENSEMBLE_Prot_Vgl_sub_naive_min_3_pp/QTI_MLP_13P_ens_f_sub_naive_min_3_ens_f6_ts_P7P16P8_vs_P17P9_20240112_12h_ep50_LR1e-04_bs512_outnorm',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/ENSEMBLE_Prot_Vgl_sub_naive_min_3_pp/QTI_MLP_13P_ens_f_sub_naive_min_3_ens_f6_ts_P17P9P18_vs_P1P10_20240112_12h_ep50_LR1e-04_bs512_outnorm'
            ]#
'''
'''
model_path = [\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/ENSEMBLE_Prot_Vgl_sub_min_pp/QTI_MLP_14P_Prot_Vgl_sub_min_pp_ens_f3_ts_P1P10P2P11P3P12_vs_P4P13_20231017_20h_ep50_LR1e-04_bs512_outnorm',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/ENSEMBLE_Prot_Vgl_sub_min_pp/QTI_MLP_14P_Prot_Vgl_sub_min_pp_ens_f3_ts_P4P13P5P14P6P15_vs_P7P16_20231017_20h_ep50_LR1e-04_bs512_outnorm',\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/ENSEMBLE_Prot_Vgl_sub_min_pp/QTI_MLP_14P_Prot_Vgl_sub_min_pp_ens_f3_ts_P7P16P8P17P9P18_vs_P1P10_20231017_20h_ep50_LR1e-04_bs512_outnorm'
            ]#
model_path = [\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/Dropout_exp/QTI_MLP_14P_Dropout_exp_sub_min_pp_case_id_ens_f1_ts_P1P10P2_vs_P11P3_20231023_10h_ep150_LR1e-04_bs512_outnorm-'
            ]#
model_path = [\
            '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/NoBiasTest_sub_min/QTI_MLP_13P_sub_min_ens_f1_ts_P1P10P2_vs_P11P3_20240808_11h_ep50_LR1e-04_bs512_outnorm_nobias'
            ]#
'''

'''
#-------------------------------------------------------------------------------------
# paths hard copy on HD
nii_path = [\
            '/Volumes/Maxtor/ML_TS/P1/og_NII_dn_db_dg_tp_mc_b0_avg_sub_naive_min_3_pp.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P10/og_NII_dn_db_dg_tp_mc_b0_avg_sub_naive_min_3_pp.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P2/og_NII_dn_db_dg_tp_mc_b0_avg_sub_naive_min_3_pp.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P11/og_NII_dn_db_dg_tp_mc_b0_avg_sub_naive_min_3_pp.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P3/og_NII_dn_db_dg_tp_mc_b0_avg_sub_naive_min_3_pp.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P12/og_NII_dn_db_dg_tp_mc_b0_avg_sub_naive_min_3_pp.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P4/og_NII_dn_db_dg_tp_mc_b0_avg_sub_naive_min_3_pp.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P13/og_NII_dn_db_dg_tp_mc_b0_avg_sub_naive_min_3_pp.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P5/og_NII_dn_db_dg_tp_mc_b0_avg_sub_naive_min_3_pp.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P14/og_NII_dn_db_dg_tp_mc_b0_avg_sub_naive_min_3_pp.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P6/og_NII_dn_db_dg_tp_mc_b0_avg_sub_naive_min_3_pp.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P15/og_NII_dn_db_dg_tp_mc_b0_avg_sub_naive_min_3_pp.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P7/og_NII_dn_db_dg_tp_mc_b0_avg_sub_naive_min_3_pp.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P16/og_NII_dn_db_dg_tp_mc_b0_avg_sub_naive_min_3_pp.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P8/og_NII_dn_db_dg_tp_mc_b0_avg_sub_naive_min_3_pp.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P17/og_NII_dn_db_dg_tp_mc_b0_avg_sub_naive_min_3_pp.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P9/og_NII_dn_db_dg_tp_mc_b0_avg_sub_naive_min_3_pp.nii.gz',\
            '/Volumes/Maxtor/ML_TS/P18/og_NII_dn_db_dg_tp_mc_b0_avg_sub_naive_min_3_pp.nii.gz'
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
#-------------------------------------------------------------------------------------
'''

#-------------------------------------------------------------------------------------
# XYZ fit GT on local HD
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

# xps_path = ['']
'''
#-------------------------------------------------------------------------------------

# high res test
#-------------------------------------------------------------------------------------
# model_path = [\
#               '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_pp_case_m_ens_f18_ts_P7_vs_P16P8P17P9_20231029_09h_ep100_LR1e-03_bs512_outnorm_sched_92',\
#               '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_pp_case_m_ens_f18_ts_P16_vs_P8P17P9P18_20231029_09h_ep100_LR1e-03_bs512_outnorm_sched_92',\
#               '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_pp_case_m_ens_f18_ts_P8_vs_P17P9P18P1_20231029_10h_ep100_LR1e-03_bs512_outnorm_sched_92',\
#               '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_pp_case_m_ens_f18_ts_P17_vs_P9P18P1P10_20231029_10h_ep100_LR1e-03_bs512_outnorm_sched_92',\
#               '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_pp_case_m_ens_f18_ts_P9_vs_P18P1P10P2_20231029_10h_ep100_LR1e-03_bs512_outnorm_sched_92'           
#               ] 
# nii_path = ['/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P9/og_NII_dn_db_dg_tp_mc_b0_avg_sub_min_pp.nii.gz']
# mask_path = ['/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P9//manual_mask.nii.gz']
# scalar_invars_path = ['/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P9/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat']
# model_path = [\
#               '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/QTI_MLP/BENCHMARK_ABSTRACT/QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P9_vs_P18_20231029_14h_ep100_LR1e-03_bs512_outnorm_sched_92'
#               ]

# high res performance eval
#-------------------------------------------------------------------------------------
# nii_path = ['/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P20/full/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz']
# mask_path = ['/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P20/min/manual_mask.nii.gz']
# scalar_invars_path = ['/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P20/full/qtip_high_res_pipe_2/dps3_fit_nan_clmpd.mat']

# nii_path = ['/Volumes/Maxtor/ML_TS/P20/full/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz']
# mask_path = ['/Volumes/Maxtor/ML_TS/P20/min/manual_mask.nii.gz']
# scalar_invars_path = ['/Volumes/Maxtor/ML_TS/P20/full/qtip_full_XYZ_pipe_2/dps3fit_nan_clmpd.mat'
# ]

# high res measurements:
#--------------------------------------------------------
# 2mm measured P11

# 2.5 mm measured P19
# mask_path = ['/Volumes/Maxtor/ML_TS/P19/2_5mm/manual_mask.nii.gz']
# nii_path = ['/Volumes/Maxtor/ML_TS/P19/2_5mm/og_NII_dn_db_dg_tp_mc_b0_avg_reord.nii.gz']
# scalar_invars_path = ['/Volumes/Maxtor/ML_TS/P11/dps3_min.mat']
# 2.0 mm measured
# mask_path = ['/Volumes/Maxtor/ML_TS/P19/2mm/manual_mask.nii.gz']
# nii_path = ['/Volumes/Maxtor/ML_TS/P19/2mm/og_NII_dn_db_dg_tp_mc_b0_avg_reord.nii.gz']
# scalar_invars_path = ['/Volumes/Maxtor/ML_TS/P11/dps3_min.mat']
# 1.9 mm measured
# mask_path = ['/Volumes/Maxtor/ML_TS/P19/1_9mm/manual_mask.nii.gz']
# nii_path = ['/Volumes/Maxtor/ML_TS/P19/1_9mm/og_NII_dn_db_dg_tp_mc_b0_avg_reord.nii.gz']
# scalar_invars_path = ['/Volumes/Maxtor/ML_TS/P11/dps3_min.mat']
# 1.8 mm measured
# mask_path = ['/Volumes/Maxtor/ML_TS/P19/1_8mm/manual_mask.nii.gz']
# nii_path = ['/Volumes/Maxtor/ML_TS/P19/1_8mm/og_NII_dn_db_dg_tp_mc_b0_avg_reord.nii.gz']
# scalar_invars_path = ['/Volumes/Maxtor/ML_TS/P11/dps3_min.mat']
# # 1.8 b mm measured
# mask_path = ['/Volumes/Maxtor/ML_TS/P19/1_8mm_b/manual_mask.nii.gz']
# nii_path = ['/Volumes/Maxtor/ML_TS/P19/1_8mm_b/og_NII_dn_db_dg_tp_mc_b0_avg_reord.nii.gz']
# scalar_invars_path = ['/Volumes/Maxtor/ML_TS/P11/dps3_min.mat']
# 1.7 mm measured
# mask_path = ['/Volumes/Maxtor/ML_TS/P19/1_7mm/manual_mask.nii.gz']
# nii_path = ['/Volumes/Maxtor/ML_TS/P19/1_7mm/og_NII_dn_db_dg_tp_mc_b0_avg_reord.nii.gz']
# scalar_invars_path = ['/Volumes/Maxtor/ML_TS/P11/dps3_min.mat']
# 1.5 mm measured
# nii_path = ['/Volumes/Maxtor/ML_TS/P19/1_5mm/og_NII_dn_db_dg_tp_mc_b0_avg_reord.nii.gz']
# mask_path = ['/Volumes/Maxtor/ML_TS/P19/1_5mm/manual_mask.nii.gz']
# scalar_invars_path = ['/Volumes/Maxtor/ML_TS/P11/dps3_min.mat']
'''
1.8 mm opt measured
mask_path = ['/Volumes/Maxtor/ML_TS/P19/1_7mm/manual_mask.nii.gz']
nii_path = ['/Volumes/Maxtor/ML_TS/P19/1_7mm/og_NII_dn_db_dg_tp_mc_b0_avg_reord.nii.gz']
scalar_invars_path = ['/Volumes/Maxtor/ML_TS/P11/dps3_min.mat']
'''
# P11 noise simulation:
#--------------------------------------------------------

model_path = [\
              r'C:\QTI_ML\BENCHMARK_ABSTRACT\QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P1_vs_P10_20231029_12h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              r'C:\QTI_ML\BENCHMARK_ABSTRACT\QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P10_vs_P2_20231029_12h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              r'C:\QTI_ML\BENCHMARK_ABSTRACT\QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P2_vs_P11_20231029_12h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              r'C:\QTI_ML\BENCHMARK_ABSTRACT\QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P11_vs_P3_20231029_12h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              r'C:\QTI_ML\BENCHMARK_ABSTRACT\QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P3_vs_P12_20231029_12h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              r'C:\QTI_ML\BENCHMARK_ABSTRACT\QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P12_vs_P4_20231029_12h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              r'C:\QTI_ML\BENCHMARK_ABSTRACT\QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P4_vs_P13_20231029_13h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              r'C:\QTI_ML\BENCHMARK_ABSTRACT\QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P13_vs_P5_20231029_13h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              r'C:\QTI_ML\BENCHMARK_ABSTRACT\QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P5_vs_P14_20231029_13h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              r'C:\QTI_ML\BENCHMARK_ABSTRACT\QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P14_vs_P6_20231029_13h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              r'C:\QTI_ML\BENCHMARK_ABSTRACT\QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P6_vs_P15_20231029_13h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              r'C:\QTI_ML\BENCHMARK_ABSTRACT\QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P15_vs_P7_20231029_13h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              r'C:\QTI_ML\BENCHMARK_ABSTRACT\QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P7_vs_P16_20231029_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              r'C:\QTI_ML\BENCHMARK_ABSTRACT\QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P16_vs_P8_20231029_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              r'C:\QTI_ML\BENCHMARK_ABSTRACT\QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P8_vs_P17_20231029_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              r'C:\QTI_ML\BENCHMARK_ABSTRACT\QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P17_vs_P9_20231029_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              r'C:\QTI_ML\BENCHMARK_ABSTRACT\QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P9_vs_P18_20231029_14h_ep100_LR1e-03_bs512_outnorm_sched_92',\
              r'C:\QTI_ML\BENCHMARK_ABSTRACT\QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P18_vs_P1_20231029_14h_ep100_LR1e-03_bs512_outnorm_sched_92'
              ]

# # 2.5 bm:
# nii_path = [\
#     '/Volumes/Maxtor/ML_TS/P11/og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
#             ]
# scalar_invars_path = [\
#     '/Volumes/Maxtor/ML_TS/P11/qtip_full_pipe_2/dps3_nan_clmpd_clmpd.mat',\
#     ]
# mask_path = [\
#     '/Volumes/Maxtor/ML_TS/P11/manual_mask.nii.gz',\
#     ]

# set your real input files for prediction (same length for nii_path and mask_path)
nii_path = [\
            r'C:\patients\P1\og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz',\
            r'C:\patients\P14\og_NII_sub_min_dn_db_dg_tp_mc_b0_avg.nii.gz'
            ]

mask_path = [\
            r'C:\patients\P1\manual_mask.nii.gz',\
            r'C:\patients\P14\manual_mask.nii.gz'
            ]
#-------------------------------------------------------------------------------------
# patient measurements:
#--------------------------------------------------------
# model_path = [\
#               '/Users/olivergoedicke/Documents/Uni/Master_Ph/MA/Results/models/BENCHMARKs/QTI_MLP_14P_q_FA_uFA_C_c_C_MD_BENCHMARK_20230929_09_50_LR1e-04_bs512'
#               ]
# nii_path = ['/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P20/min_wb/og_NII_dn_db_dg_tp_mc_b0_avg_reord.nii.gz']
# mask_path = ['/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P20/min_wb/manual_mask.nii.gz']
# scalar_invars_path = '/Volumes/Maxtor/#Uni/Master_Ph/MA/ML_TS/P20/min_wb/qtip_high_res_pipe_2/dps3.mat'
# patient 3:
#--------------------------------------------------------
# nii_path = ['/Volumes/Maxtor/ML_TS/Pt_3/native/og_NII_dn_db_dg_tp_mc_b0_avg_reord.nii.gz']
# mask_path = ['/Volumes/Maxtor/ML_TS/Pt_3/native/manual_mask_t.nii.gz']
# scalar_invars_path = ['/Volumes/Maxtor/ML_TS/Pt_3/native/qtip_pipe_2/dps3.mat']
# patient 2:
#--------------------------------------------------------
# nii_path = ['/Volumes/Maxtor/ML_TS/Pt_2/native/og_NII_dn_db_dg_tp_mc_b0_avg_reord.nii.gz']
# mask_path = ['/Volumes/Maxtor/ML_TS/Pt_2/native/manual_mask_t.nii.gz']
# nii_path = ['/Volumes/Maxtor/ML_TS/Pt_2/high_res/og_NII_dn_db_dg_tp_mc_b0_avg_reord.nii.gz']
# mask_path = ['/Volumes/Maxtor/ML_TS/Pt_2/high_res/manual_mask_t.nii.gz']
# scalar_invars_path = ['/Volumes/Maxtor/ML_TS/Pt_2/native/qtip_pipe_2/dps3.mat']
#--------------------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# output paths:
# hyperparams:
results_path = 'runs/QTI_MLP_predict'
out_dir = 'BENCHMARK_ABSTRACT_prediction'
#-------------------------------------------------------------------------------------
# hyperparams:
invar_keys = ['MD', 'FA', 'uFA', 'C_c', 'C_MD'] #'s0', 'MD', 'FA', 'uFA', 'C_c', 'C_MD', 'C_M', 'C_mu', 'MKa', 'MKi']
# identifier = 'QTI_MLP_11P_output_norm_MD_sub_min_pp'
#--------------------------------------------------------
# configure MLP output:
final_sigmoid = False
#--------------------------------------------------------
zscore_output = True
if zscore_output:
    norm_keys = invar_keys
else:
    norm_keys = ['None']
#--------------------------------------------------------
device = ("cpu")
loss_fn = nn.MSELoss()
#--------------------------------------------------------
layer_norm = False
#--------------------------------------------------------
bias = True
#--------------------------------------------------------
# adjust MLP type and preparations below!
xps_path = None # must be None, when QTI_MLP is used
#--------------------------------------------------------
# pass zscore as tensor, filepath, or file <-> model path (default)
'''
# zscore = torch.tensor([[1.1519716,0.23112178,0.56032646,0.18088841,0.11575394],
#         [0.51073414,0.18647267,0.21798058,0.18778813,0.053884428]]) # MD invars, 14P
# zscore = torch.tensor([[1.1557629,0.22933668,0.557813,0.1785723,0.11590578],
#         [0.5101809,0.18595651,0.21704209,0.185997,0.05374427]]) # 11P
'''
zscore = None
zscore_path = None
#--------------------------------------------------------
# edge slice exclusion:
'''
slice_ind = [[0, 23], [1, 22], [0, 22], [1, 23], [0, 22]] # 3:

slice_ind_11P5 = [[0, 23], [1, 22], [0, 22], [1, 23], [0, 22],\
                  [2, 23], [0, 22],\
                  [0, 23], [0, 23], [0, 23], [1, 23], [1, 23],\
                   [0, 22], [0, 23], [0, 23], [0, 22], [0, 22], [0, 23]] #  -5:, :2, 3:-4

'''
slice_ind = [[2, 21] for _ in range(len(nii_path))]
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
# Ensemble Loop
# test set ensembling for protocol statistics
# code can be adapted for training set ensembling
#-------------------------------------------------------------------------------------
# use ensemble of models to return mean prediction for unseen data:
avg_ens_pred = True
if avg_ens_pred:
    ts_size = len(nii_path)
    vs_size = 0
    ens_factor = len(model_path)
    preds = []
#-------------------------------------------------------------------------------------
# Settings:
'''
# no ensembling (one run): ensemble factor = 0 
ens_factor = 0

try: 
  ts_size = len(nii_path)//ens_factor
except:
  ts_size = 0
  ens_factor = 1
vs_size = 1
'''
#-------------------------------------------------------------------------------------
# manual:
# ts_size = 6
# vs_size = 1
# ens_factor = 1 # len(model_path)
#-------------------------------------------------------------------------------------

if ts_size > len(nii_path):
    raise Exception('Invalid ensembling configuration, training set size is zero.')
if ens_factor != len(model_path) and not avg_ens_pred:
    raise Exception('Number of models and ensembles does not match.')

for current_step in range(ens_factor):
    if ens_factor > 1:
        print('\nFitting Ensemble {}:'.format(current_step + 1))
    #-------------------------------------------------------------------------------------
    # index selection
    current_ind = [i for i in range(len(nii_path))]
    current_ind = np.roll(current_ind, shift=-current_step*ts_size)
    ts_ind = current_ind[0:ts_size]

    ts_path = [nii_path[ind] for ind in ts_ind]
    #-------------------------------------------------------------------------------------
    # create QTI_Dataset instance: [mask_path[ind] for ind in ts_ind] [scalar_invars_path[ind] for ind in ts_ind] slice_ind=[slice_ind[ind] for ind in ts_ind]
    # mask_path=None: fit all voxels
    ds_pred = QTI_Dataset(ts_path, 
                         scalar_invars_path=None,
                         mask_path=[mask_path[ind] for ind in ts_ind],
                         slice_ind=None,
                        invar_keys=invar_keys, zscore_output = zscore_output)

    #-------------------------------------------------------------------------------------
    # perform preparations/transformations
    # signal input:
    #--------------------------------------------------------
    tic = time.time()
    ds_pred.thresh_neg_vals()
    # ds_pred.mask_zero_vals()
    # ds_pred.mask_nan_vals()
    ds_pred.apply_masked_tensor()
    ds_pred.zscore_norm_input()
    # btens:
    #--------------------------------------------------------
    '''    
    # ds_pred.zscore_norm_btens
    # ds_pred.min_max_norm_btens()
    # ds_pred.scale_btens(1e-10)

    # btens derived quantities:
    #--------------------------------------------------------
    # ds_pred.scale_bval(5e-10)
    # ds_pred.bvec_cart2sph()
    # ds_pred.btens_drop_b_eta()

    # ds_pred.cat_inputs()
    '''
    # labels:
    #--------------------------------------------------------
    # pass list of invar keys that should be normalized here
    if ds_pred.target_present:
        ds_pred.zscore_norm_labels(norm_keys) # ['s0', 'MD', 'FA', 'uFA', 'C_MD']
    #--------------------------------------------------------
    output_shape = [ds_pred.X.size(i) for i in range(len(ds_pred.X.size())-1)]
    #--------------------------------------------------------
    # do not apply the mask in order to retain num of elems to allow for reshaping into orig. 4D shape
    # output will still be masked, since masked tensor is used for normalization
    # after converting back, all voxels not in masked are set to NaN, s.t. when removing NaNs for zeros
    # in the network prediction, everything is masked again
    #--------------------------------------------------------
    ds_pred.flatten_slice_dim()

    #-------------------------------------------------------------------------------------
    # create QTI_MLP instance, load model:
    n_scans = ds_pred.X.size()[-1]; n_invars = len(invar_keys)
    model = QTI_MLP(n_scans, n_invars, layer_norm, final_sigmoid, bias)
    model.load_state_dict(torch.load(model_path[current_step]))
    model.to(device)
    model.eval()

    #-------------------------------------------------------------------------------------
    # perform prediction:
    X = ds_pred.X.to(device)
    if ds_pred.target_present:
        y = ds_pred.y.to(device)

    pred = model(X)

    output_shape.append(pred.size(-1))
    toc = time.time()
    print('Elapsed time: {}'.format(tic-toc))
    #--------------------------------------------------------
    # calculate loss per sample/voxel:
    if ds_pred.target_present:
        pred_loss = loss_fn(pred[ds_pred.mask], y[ds_pred.mask])/torch.count_nonzero(ds_pred.mask)
        print('\nTest loss per voxel = {}'.format(pred_loss))

    #--------------------------------------------------------
    # reverse z-score:
    if zscore_output:
        if zscore is None and zscore_path is None:
            df = pd.read_csv(model_path[current_step] + '_zscore.csv')
            zs = torch.tensor(df.values.astype(float)).to(torch.float32)
            print('\nz-score:\nmean =  {}\nstd =  {}'.format(zs[0,:], zs[1,:]))
        elif isinstance(zscore_path, list):
            df = pd.read_csv(zscore_path[current_step])
            zs = torch.tensor(df.values.astype(float)).to(torch.float32)
        elif isinstance(zscore, torch.Tensor):
            zs = zscore
        else:
            raise Exception('Incompatible input for zscore.')
        pred = rev_zscore(pred, zs)

    #--------------------------------------------------------
    # remove nan, if desired
    pred = torch.nan_to_num(pred, nan=0.0)
    if ds_pred.target_present:
        y = torch.nan_to_num(y, nan=0.0)

    #-------------------------------------------------------------------------------------
    # reshape to orig. dim.s: reorder dimensions to x,y,z
    pred = torch.permute(torch.reshape(pred, output_shape), (0,2,3,1,4)).detach().cpu().numpy()

    #-------------------------------------------------------------------------------------
    # save for average ensemble prediction:
    if avg_ens_pred and (current_step < ens_factor - 1):
        preds.append(pred)
        continue
    elif avg_ens_pred and (current_step == ens_factor - 1):
        preds.append(pred)
        # average prediction:
        pred = np.mean(preds, axis=0) # perhaps use nan mean
        if len(np.shape(pred)) < 5: # smh only 4 dims if one file
            pred = pred[None, ...]
    #-------------------------------------------------------------------------------------

    ts_names = [split(split(path)[0])[1] for path in ts_path]
    identifier = split(model_path[current_step])[1]
    if avg_ens_pred:
        identifier += '_avg'

    print('\nFitting model: \n' + identifier + '\n...for files:\n' + ''.join(ts_names))

    if output_shape[0] != len(ts_names):
        raise Exception('Dim and number of input files disagree.')

    #-------------------------------------------------------------------------------------
    # write output:

    for file_ind, name in enumerate(ts_names):
        out_path = join(results_path, out_dir, name)
        if not exists(out_path):
            makedirs(out_path)

        # single .nii with all quantities:
        #--------------------------------------------------------
        '''
        # drag affine from the y input along, write .nii with all outputs:
        affine = ds_pred.nii_affines[file_ind]
        output_img = nib.Nifti1Image(pred[file_ind, ...], affine)
        nib.save(output_img, (join(out_path,\
                identifier + '_prediction.nii')))
        '''

        # write .nii for each quantity:
        #--------------------------------------------------------
        dps_dict = {}
        for ind, invar in enumerate(ds_pred.invar_keys):
            affine = ds_pred.nii_affines[file_ind]
            output_img = nib.Nifti1Image(pred[file_ind, ..., ind], affine)
            nib.save(output_img, (join(out_path,\
                identifier + '_' + invar + '_pred.nii')))

            dps_dict[invar] = pred[file_ind, ..., ind]

        # write dps.mat:
        # --------------------------------------------------------
        sio.savemat((join(out_path,\
                identifier + '_' + 'dps_pred.mat')), {'dps': dps_dict})
        

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------