import sys
sys.path.append('..')  # Adjust the path as needed
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tqdm import tqdm

from src.read_data import *
# from src.frame_creation import *
from src.scenario2_modeling_functions import *
from src.scenario1_modeling_functions import *

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, balanced_accuracy_score
from xgboost import XGBRegressor
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

H = 730
t = 200
db_path = "../database.db"


def run_scenario1(loan_CF, loan_CF_filtered, batches, H, t):
    c_ = ['batch', 'pred_mu', 'pred_lo', 'pred_hi']

    print('\n------------------------------------------------------------------------------')

    print('SCENARIO 1:')
    print('\n MODEL 1: MODEL PREDICTS ROI(H)\n')
    print('')

    scenario1_snap_roi = scenario1_snap_roi_func(
        loan_CF, H = 730, t = 200
        )
    scenario1_snap_roi, reg_ = scenario1_data_split_model_train_roi(
        scenario1_snap_roi, RNG = 42
        )
    scenario1_mask_pred_roi, scenario1_cohort_roi, scenario1_eval_df_cohort_roi, scenario1_metrics_roi = scenario1_compute_metrics_roi(
        scenario1_snap_roi
        )

    print('\n MODEL 1 + p90 FILTER: MODEL PREDICTS ROI(H)\n')
    print('')

    scenario1_snap_roi_filtered = scenario1_snap_roi_func(
        loan_CF_filtered, H = 730, t = 200
        )
    scenario1_snap_roi_filtered, reg_ = scenario1_data_split_model_train_roi(
        scenario1_snap_roi_filtered, RNG = 42
        )
    scenario1_mask_pred_roi_filtered, scenario1_cohort_roi_filtered, scenario1_eval_df_cohort_roi_filtered, scenario1_metrics_roi_filtered = scenario1_compute_metrics_roi(
        scenario1_snap_roi_filtered
        )

    print('\n MODEL 2: MODEL PREDICTS NET CASH(H)\n')
    print('')

    scenario1_snap_cash = scenario1_snap_cash_func(
        loan_CF, H = 730, t = 200
        )
    scenario1_snap_cash, reg_ = scenario1_data_split_model_train_cash(
        scenario1_snap_cash, RNG = 42
        )
    scenario1_eval_b_net, scenario1_eval_b_roi, scenario1_metrics_cash = scenario1_compute_metrics_cash(
        scenario1_snap_cash
        )

    print('\n MODEL 2 + p90 FILTER: MODEL PREDICTS NET CASH(H)\n')
    print('')

    scenario1_snap_cash_filtered = scenario1_snap_cash_func(
        loan_CF_filtered, H = 730, t = 200
        )
    scenario1_snap_cash_filtered, reg_ = scenario1_data_split_model_train_cash(
        scenario1_snap_cash_filtered, RNG = 42
        )
    scenario1_eval_b_net_filtered, scenario1_eval_b_roi_filtered, scenario1_metrics_cash_filtered = scenario1_compute_metrics_cash(
        scenario1_snap_cash_filtered
        )
    
    out = [
        sort_final(scenario1_pre_err_roi(scenario1_cohort_roi, scenario1_eval_df_cohort_roi), batches)[c_],
        sort_final(scenario1_pre_err_roi(scenario1_cohort_roi_filtered, scenario1_eval_df_cohort_roi_filtered), batches)[c_],
        sort_final(scenario1_pre_err_cash(scenario1_snap_cash, scenario1_eval_b_net), batches)[c_],
        sort_final(scenario1_pre_err_cash(scenario1_snap_cash_filtered, scenario1_eval_b_net_filtered), batches)[c_]
    ]
    all_metrics = [
        scenario1_metrics_roi,
        scenario1_metrics_cash,
        scenario1_metrics_roi_filtered,
        scenario1_metrics_cash_filtered
        ]

    Model_n = [
        'Scenario 1: model 1',
        'Scenario 1: model 2',
        'Scenario 1: model 1 + p90 filter',
        'Scenario 1: model 2 + p90 filter'
        ]
    return out, all_metrics, Model_n


def run_scenario2(loan_CF, loan_CF_filtered, batches, H, t):
    c_ = ['batch', 'pred_mu', 'pred_lo', 'pred_hi']


    print('\n------------------------------------------------------------------------------')

    print('SCENARIO 2:')
    print('\n MODEL 1: MODEL PREDICTS ROI(H)\n')
    print('')

    scenario2_snap_roi = scenario2_snap_roi_func(
        loan_CF, H, t
        )
    scenario2_snap_roi, reg_ = scenario2_data_split_model_train_roi(
        scenario2_snap_roi, RNG = 42
        )
    scenario2_mask_pred_roi, scenario2_truth_can, scenario2_metrics_roi = scenario2_compute_metrics_roi(
        scenario2_snap_roi, loan_CF, H
        )

    print('\n MODEL 1 + p90 FILTER: MODEL PREDICTS ROI(H)\n')
    print('')

    scenario2_snap_roi_filtered = scenario2_snap_roi_func(
        loan_CF_filtered, H, t
        )
    scenario2_snap_roi_filtered, reg_ = scenario2_data_split_model_train_roi(
        scenario2_snap_roi_filtered, RNG = 42
        )
    scenario2_mask_pred_roi_filtered, scenario2_truth_can_filtered, scenario2_metrics_roi_filtered = scenario2_compute_metrics_roi(
        scenario2_snap_roi_filtered, loan_CF_filtered, H
        )

    print('\n MODEL 2: MODEL PREDICTS NET CASH(H)\n')
    print('')

    scenario2_snap_cash = scenario2_snap_cash_func(
        loan_CF, H, t
        )
    scenario2_snap_cash, reg_, test_idx_cash = scenario2_data_split_model_train_cash(
        scenario2_snap_cash, RNG = 42
        )
    scenario2_eval_b_net, scenario2_eval_b_roi, scenario2_metrics_cash = scenario2_compute_metrics_cash(
        loan_CF, scenario2_snap_cash, test_idx_cash, H
        )

    print('\n MODEL 2 + p90 FILTER: MODEL PREDICTS NET CASH(H)\n')
    print('')

    scenario2_snap_cash_filtered = scenario2_snap_cash_func(
        loan_CF_filtered, H, t)
    scenario2_snap_cash_filtered, reg_, test_idx_cash_filtered = scenario2_data_split_model_train_cash(
        scenario2_snap_cash_filtered, RNG = 42
        )
    scenario2_eval_b_net_filtered, scenario2_eval_b_roi_filtered, scenario2_metrics_cash_filtered = scenario2_compute_metrics_cash(
        loan_CF_filtered, scenario2_snap_cash_filtered, test_idx_cash_filtered, H
        )
        
    out = [
        sort_final(scenario2_pre_err_roi(scenario2_snap_roi, scenario2_truth_can), batches)[c_],
        sort_final(scenario2_pre_err_roi(scenario2_snap_roi_filtered, scenario2_truth_can_filtered), batches)[c_],
        sort_final(scenario2_pre_err_cash(scenario2_snap_cash, scenario2_eval_b_roi, test_idx_cash), batches)[c_],
        sort_final(scenario2_pre_err_cash(scenario2_snap_cash_filtered, scenario2_eval_b_roi_filtered, test_idx_cash_filtered), batches)[c_]
    ]

    all_metrics = [
    scenario2_metrics_roi,
    scenario2_metrics_cash,
    scenario2_metrics_roi_filtered,
    scenario2_metrics_cash_filtered
        ]

    Model_n = [
        'Scenario 2: model 1',
        'Scenario 2: model 2',
        'Scenario 2: model 1 + p90 filter',
        'Scenario 2: model 2 + p90 filter'
        ]
    return out, all_metrics, Model_n