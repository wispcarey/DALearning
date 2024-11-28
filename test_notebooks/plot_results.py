from utils import plot_results_3d, plot_results_2d
from utils import partial_obs_operator, DATASET_INFO, get_mean_std
from torch.utils.data import Dataset, DataLoader

from networks import ComplexAttentionModel, AttentionModel
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from localization import plot_GC, pairwise_distances
import os

BASE_INFO = {
    'lorenz96': {
        'Climatology': 3.62,
        'Climatology_std': 0.01,
        'OptInterp': 2.94,
        'OptInterp_std': 0.02,
        'Var3D': 3.18,
        'Var3D_std': 0.08
    }
}

BASE_METHODS = ['Climatology', 'OptInterp', 'Var3D']

ENSEMBLE_SIZE = [5,10,15,20,40,60,100]

def replace_nan_with_value(lst, value=6):
    return [value if np.isnan(x) else x for x in lst]

def get_benchmarks(args):
    file_path = f'save/benchmark/benchmarks_{args.dataset}.csv'
    df = pd.read_csv(file_path, usecols=['method', 'sigma_y', 'rmse_mean', 'rmse_std'])

    result_dict = {}

    methods = df['method'].unique()

    for method in methods:
        method_data = df[df['method'] == method]
        result_dict[method] = {
            "1_rmse": replace_nan_with_value(method_data[method_data['sigma_y'] == 1]['rmse_mean'].tolist()),
            "1_std": method_data[method_data['sigma_y'] == 1]['rmse_std'].tolist(),
            "0.7_rmse": replace_nan_with_value(method_data[method_data['sigma_y'] == 0.7]['rmse_mean'].tolist()),
            "0.7_std": method_data[method_data['sigma_y'] == 0.7]['rmse_std'].tolist(),
        }
    
    for method in BASE_METHODS:
        result_dict[method] = {
            "1_rmse": [BASE_INFO[args.dataset][method]] * len(ENSEMBLE_SIZE),
            "1_std": [BASE_INFO[args.dataset][f"{method}_std"]] * len(ENSEMBLE_SIZE),
            "0.7_rmse": [BASE_INFO[args.dataset][method]] * len(ENSEMBLE_SIZE),
            "0.7_std": [BASE_INFO[args.dataset][f"{method}_std"]] * len(ENSEMBLE_SIZE),
        }