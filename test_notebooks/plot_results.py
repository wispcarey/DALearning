import argparse
import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.utils.data import Dataset, DataLoader

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import plot_results_3d, plot_results_2d
from utils import partial_obs_operator, get_mean_std
from config.dataset_info import DATASET_INFO

from networks import ComplexAttentionModel, AttentionModel
from localization import plot_GC, pairwise_distances

# BASE_INFO = {
#     'lorenz96': {
#         'Climatology': 3.62,
#         'Climatology_std': 0.01,
#         'OptInterp': 2.94,
#         'OptInterp_std': 0.02,
#         'Var3D': 3.18,
#         'Var3D_std': 0.08
#     }
# }

# BASE_METHODS = ['Climatology', 'OptInterp', 'Var3D']

NAN_VALS = {
    'rmse':{
    'lorenz63': [1.5, 5],
    'lorenz96': [6, 3],
    'ks': [2, 2]},
    'rrmse':{
    'lorenz63': [1.5, 1.5],
    'lorenz96': [1.5, 1.5],
    'ks': [1.5, 1.5]}
}

ENSEMBLE_SIZE = [5,10,15,20,40,60,100]

DATASET_FOLDERS = {
    'lorenz63': [
        "2024-12-05_17-49lorenz63_1.0_10_60_8192_EnST_joint",
        "2024-12-05_19-57lorenz63_0.7_10_60_8192_EnST_joint",
        "2024-12-06_12-23lorenz63_1.0_20_60_8192_EnST_tuned_joint",
        "2024-12-09_13-38lorenz63_0.7_20_60_8192_EnST_tuned_joint"],
    'lorenz96': [
        "2024-12-01_16-40lorenz96_1.0_10_60_8192_EnST_joint",
        "2024-11-28_00-50lorenz96_0.7_10_60_8192_EnST_joint",
        "2024-12-09_14-55lorenz96_1.0_20_60_8192_EnST_tuned_joint",
        "2024-12-09_15-26lorenz96_0.7_20_60_8192_EnST_tuned_joint"],
    'ks': [
        "2025-02-20_19-00ks_1.0_10_60_8192_EnST_joint",
        "2025-02-21_03-11ks_0.7_10_60_8192_EnST_joint",
        "2025-02-21_11-24ks_1.0_20_60_8192_EnST_tuned_joint",
        "2025-02-21_12-23ks_0.7_20_60_8192_EnST_tuned_joint"],
}

TEST_DATA = {
    'lorenz63': '../data/lorenz63/test_0_96000_v_0.150step.npy',
    'lorenz96': '../data/lorenz96/test_0_96000_v_0.150step.npy',
    'ks': '../data/ks/test_0_128000_v_1.000step.npy'
}

COLOR_DICT = {
    'trained N=10': "green",
    'tuned': "blue",
    'LETKF': "red",
    'EnKF_Sqrt': "cyan",
    'EnKF_PertObs': "orange",
    'iEnKS': "Brown",
}

plt.rc('font', size=20)

def replace_nan_with_value(nparray, value=6):
    nparray[np.isnan(nparray)] = value
    return nparray

def collect_results(args):
    folder_list = DATASET_FOLDERS[args.dataset]
    
    results = np.zeros((4, len(ENSEMBLE_SIZE), 5), dtype=float)
    
    for i in range(4):
        for j, N in enumerate(ENSEMBLE_SIZE):
            data_dic = torch.load(f'../save/{folder_list[i]}/output_records_{N}.pt', weights_only=True)
            results[i, j, 0] = data_dic['nn']['mean_rmse']
            results[i, j, 1] = data_dic['nn']['std_rmse']
            results[i, j, 2] = data_dic['nn']['mean_rmv']
            results[i, j, 3] = data_dic['nn']['std_rmv']
            results[i, j, 4] = data_dic['nn']['valid_percent'] < 1
    
    nn_results = {'trained N=10': results[:2], 'tuned': results[2:]}
    return nn_results
        

def get_benchmarks(args):
    """
    This function processes benchmark data from a CSV file, splitting it by `sigma_y` values 1 and 0.7,
    and extracting specific columns for each `method`, then combining them into a 2*N*5 numpy array.
    
    Args:
        args: An object or namespace with a `dataset` attribute specifying the dataset name.
        
    Returns:
        result_dict: A dictionary where each key is a method, and the value is a 2*N*5 numpy array.
    """
    file_path = f'../save/benchmark/benchmarks_{args.dataset}.csv'
    df = pd.read_csv(file_path, usecols=['method', 'sigma_y', 'rmse_mean', 'rmse_std', 'rmv_mean', 'rmv_std', 'nan_exist'])

    result_dict = {}

    methods = df['method'].unique()

    for method in methods:
        method_data = df[df['method'] == method]
        
        # Filter rows where sigma_y == 1 and 0.7
        sigma_y_1 = method_data[method_data['sigma_y'] == 1][['rmse_mean', 'rmse_std', 'rmv_mean', 'rmv_std', 'nan_exist']]
        sigma_y_0_7 = method_data[method_data['sigma_y'] == 0.7][['rmse_mean', 'rmse_std', 'rmv_mean', 'rmv_std', 'nan_exist']]
        
        # Convert to numpy arrays
        sigma_y_1_array = sigma_y_1.to_numpy()
        sigma_y_0_7_array = sigma_y_0_7.to_numpy()
        
        # Combine into a 2*N*5 array
        combined_array = np.stack([sigma_y_1_array, sigma_y_0_7_array], axis=0)
        
        # Store in result dictionary
        result_dict[method] = combined_array.astype(float)

    return result_dict

def plot_results_all(args):
    if args.rrmse:
        print("Use the relative metric = rmse/rms")
        test_traj = np.load(TEST_DATA[args.dataset])
        RMS = np.mean(np.sqrt(np.mean(test_traj ** 2, axis=2)))
        print("RMS:", RMS)
    
    x_inds = ENSEMBLE_SIZE
    
    nn_results = collect_results(args)
    benchmark_results = get_benchmarks(args)
    
    titles = [f"{args.dataset.upper()}:"+r"$\sigma_y=1$; RMSE Mean $\pm$ 1std", 
            f"{args.dataset.upper()}:"+r"$\sigma_y=1$; RMV Mean $\pm$ 1std",
            f"{args.dataset.upper()}:"+r"$\sigma_y=0.7$; RMSE Mean $\pm$ 1std",
            f"{args.dataset.upper()}:"+r"$\sigma_y=0.7$; RMV Mean $\pm$ 1std"]
    if args.rrmse:
        y_labels = ["RRMSE", "RRMV", "RRMSE", "RRMV"]
    else:
        y_labels = ["RMSE", "RMV", "RMSE", "RMV"]
    save_figure_suffix = ["RMSE_10", "RMV_10", "RMSE_07", "RMV_07"]
    if args.rrmse:
        nan_rmse_val, nan_rmv_val = NAN_VALS['rrmse'][args.dataset]
        # nan_rmse_val, nan_rmv_val = nan_rmse_val * RMS, nan_rmv_val * RMS
    else:
        nan_rmse_val, nan_rmv_val = NAN_VALS['rmse'][args.dataset]
    nan_replace_vals = [nan_rmse_val, 0, nan_rmv_val, 0]
    
    fig1 = plt.figure(1, figsize=(12, 6)) 
    fig2 = plt.figure(2, figsize=(12, 6)) 
    fig3 = plt.figure(3, figsize=(12, 6))
    fig4 = plt.figure(4, figsize=(12, 6))
    value_max = np.zeros((2,2))
    value_min = 100 * np.ones((2,2))
    nan_in_ours = np.zeros(2, dtype=bool)
    for key, value in nn_results.items():
        if key == 'tuned':
            ours_best = value
        # change nan to values
        nan_in_ours = np.logical_or(nan_in_ours, np.any(np.isnan(value), axis=(1,2)))
        
        for i in range(4):
            if args.rrmse:
                value[:, :, i] = replace_nan_with_value(value[:, :, i], nan_replace_vals[i] * RMS)
            else:
                value[:, :, i] = replace_nan_with_value(value[:, :, i], nan_replace_vals[i])

        max_val = np.max(value[:,:,[0,2]], axis=1)
        value_max = np.maximum(max_val, value_max)
                
        mean_rmse_1, std_rmse_1, mean_rmv_1, std_rmv_1, nan_exists_1 = value[0, :, 0], value[0, :, 1], value[0, :, 2], value[0, :, 3], value[0, :, 4]
        
        if args.rrmse:
            mean_rmse_1, std_rmse_1, mean_rmv_1, std_rmv_1 = mean_rmse_1 / RMS, std_rmse_1 / RMS, mean_rmv_1 / RMS, std_rmv_1 / RMS
        
        print(f"RMSE {key} for {args.dataset} with " + r"$\sigma_y=1$:", mean_rmse_1)

        plt.figure(1)
        plt.errorbar(x_inds, mean_rmse_1, yerr=std_rmse_1,
            linestyle='-', capsize=3, capthick=2, color=COLOR_DICT[key], marker='D', linewidth=2, alpha=0.75, label=f"Ours {key}")
        # for x, y, is_star in zip(x_inds, mean_rmv_1, nan_exists_1):
        #     marker = "*" if is_star else "D"
        #     plt.scatter(x, y, marker=marker, color=COLOR_DICT[key], s=50)
        
        plt.figure(2)
        plt.errorbar(x_inds, mean_rmv_1, yerr=std_rmv_1,
            linestyle='-', capsize=3, capthick=2, color=COLOR_DICT[key], marker='D', linewidth=2, alpha=0.75, label=f"Ours {key}")
        # for x, y, is_star in zip(x_inds, mean_rmv_1, nan_exists_1):
        #     marker = "*" if is_star else "D"
        #     plt.scatter(x, y, marker=marker, color=COLOR_DICT[key], s=50)
        
        mean_rmse_07, std_rmse_07, mean_rmv_07, std_rmv_07, nan_exists_07 = value[1, :, 0], value[1, :, 1], value[1, :, 2], value[1, :, 3], value[1, :, 4]
        
        if args.rrmse:
            mean_rmse_07, std_rmse_07, mean_rmv_07, std_rmv_07 = mean_rmse_07 / RMS, std_rmse_07 / RMS, mean_rmv_07 / RMS, std_rmv_07 / RMS
        
        print(f"RMSE {key} for {args.dataset} with " + r"$\sigma_y=0.7$:", mean_rmse_07)


        plt.figure(3)
        plt.errorbar(x_inds, mean_rmse_07, yerr=std_rmse_07,
            linestyle='-', capsize=3, capthick=2, color=COLOR_DICT[key], marker='D', linewidth=2, alpha=0.75, label=f"Ours {key}")
        # for x, y, is_star in zip(x_inds, mean_rmv_07, nan_exists_07):
        #     marker = "*" if is_star else "D"
        #     plt.scatter(x, y, marker=marker, color=COLOR_DICT[key], s=50)
        
        plt.figure(4)
        plt.errorbar(x_inds, mean_rmv_07, yerr=std_rmv_07,
            linestyle='-', capsize=3, capthick=2, color=COLOR_DICT[key], marker='D', linewidth=2, alpha=0.75, label=f"Ours {key}")
        # for x, y, is_star in zip(x_inds, mean_rmv_07, nan_exists_07):
        #     marker = "*" if is_star else "D"
        #     plt.scatter(x, y, marker=marker, color=COLOR_DICT[key], s=50)
            
    nan_in_others = np.zeros(2, dtype=bool)
    for key, value in benchmark_results.items():
        if key == "EnKF_PertObs":
            others_enkf = value
        elif key == "LETKF":
            others_letkf = value
        
        nan_in_others = np.logical_or(nan_in_others, np.any(np.isnan(value), axis=(1,2)))
        for i in range(4):
            if args.rrmse:
                value[:, :, i] = replace_nan_with_value(value[:, :, i], nan_replace_vals[i] * RMS)
            else:
                value[:, :, i] = replace_nan_with_value(value[:, :, i], nan_replace_vals[i])
        
        max_val = np.max(value[:,:,[0,2]], axis=1)
        value_max = np.maximum(max_val, value_max)
        min_val = np.min(value[:,:,[0,2]], axis=1)
        value_min = np.minimum(min_val, value_min)
        
        mean_rmse_1, std_rmse_1, mean_rmv_1, std_rmv_1, nan_exists_1 = value[0, :, 0], value[0, :, 1], value[0, :, 2], value[0, :, 3], value[0, :, 4]
        
        if args.rrmse:
            mean_rmse_1, std_rmse_1, mean_rmv_1, std_rmv_1 = mean_rmse_1 / RMS, std_rmse_1 / RMS, mean_rmv_1 / RMS, std_rmv_1 / RMS

        plt.figure(1)
        plt.errorbar(x_inds, mean_rmse_1, yerr=std_rmse_1,
            linestyle='-', capsize=3, capthick=2, color=COLOR_DICT[key], marker='D', linewidth=2, alpha=0.75, label=f"{key}")
        # for x, y, is_star in zip(x_inds, mean_rmse_1, nan_exists_1):
        #     marker = "*" if is_star else "D"
        #     plt.scatter(x, y, marker=marker, color=COLOR_DICT[key], s=50)
        
        plt.figure(2)
        plt.errorbar(x_inds, mean_rmv_1, yerr=std_rmv_1,
            linestyle='-', capsize=3, capthick=2, color=COLOR_DICT[key], marker='D', linewidth=2, alpha=0.75, label=f"{key}")
        # for x, y, is_star in zip(x_inds, mean_rmv_1, nan_exists_1):
        #     marker = "*" if is_star else "D"
        #     plt.scatter(x, y, marker=marker, color=COLOR_DICT[key], s=50)
        
        mean_rmse_07, std_rmse_07, mean_rmv_07, std_rmv_07, nan_exists_07 = value[1, :, 0], value[1, :, 1], value[1, :, 2], value[1, :, 3], value[1, :, 4]
        if args.rrmse:
            mean_rmse_07, std_rmse_07, mean_rmv_07, std_rmv_07 = mean_rmse_07 / RMS, std_rmse_07 / RMS, mean_rmv_07 / RMS, std_rmv_07 / RMS

        plt.figure(3)
        plt.errorbar(x_inds, mean_rmse_07, yerr=std_rmse_07,
            linestyle='-', capsize=3, capthick=2, color=COLOR_DICT[key], marker='D', linewidth=2, alpha=0.75, label=f"{key}")
        # for x, y, is_star in zip(x_inds, mean_rmse_07, nan_exists_07):
        #     marker = "*" if is_star else "D"
        #     plt.scatter(x, y, marker=marker, color=COLOR_DICT[key], s=50)
        
        plt.figure(4)
        plt.errorbar(x_inds, mean_rmv_07, yerr=std_rmv_07,
            linestyle='-', capsize=3, capthick=2, color=COLOR_DICT[key], marker='D', linewidth=2, alpha=0.75, label=f"{key}")
        # for x, y, is_star in zip(x_inds, mean_rmv_07, nan_exists_07):
        #     marker = "*" if is_star else "D"
        #     plt.scatter(x, y, marker=marker, color=COLOR_DICT[key], s=50)
    
    for i in range(4):
        plt.figure(i+1)

        if i in [0,2]:
            nan_val = nan_rmse_val
        else:
            nan_val = nan_rmv_val        
        
        if args.rrmse:
            if args.dataset == 'lorenz63':
                ticksize = 0.02
            else:
                ticksize = 0.25
        else:
            ticksize = 0.5
            
        if args.rrmse:
            vmax = value_max[i//2, i - 2 * i//2] / RMS
            vmin = value_min[i//2, i - 2 * i//2] / RMS
        else:
            vmax = value_max[i//2, i - 2 * i//2]
            vmin = value_min[i//2, i - 2 * i//2] 
        if nan_in_ours[i//2] or nan_in_others[i//2]:
            y_ticks = list(np.arange(int(vmin / ticksize) * ticksize, nan_val + ticksize, ticksize))
            if int(nan_val) != nan_val:
                y_ticks.append(nan_val)
            new_y_ticks = ["NaN" if y == nan_val else y for y in y_ticks]  
        else:
            y_ticks = list(np.arange(int(vmin / ticksize) * ticksize, vmax + ticksize, ticksize))
            new_y_ticks = y_ticks
            
        plt.yticks(ticks=y_ticks, labels=new_y_ticks)
        plt.legend(loc='upper right')
        plt.title(titles[i])
        plt.xlabel("Ensemble Size")
        plt.ylabel(y_labels[i])
        plt.xticks(ENSEMBLE_SIZE)
        plt.grid(True)
        
        if args.rrmse:
            suffix = '_R'
        else:
            suffix = ''
        plt.savefig(os.path.join('../save/figures', f"{args.dataset}_{save_figure_suffix[i]}{suffix}.png"), bbox_inches="tight", dpi=200)
        print("Image saved to", os.path.join('../save/figures', f"{args.dataset}_{save_figure_suffix[i]}.png"))
        plt.close()  
    
    # Relative improvement plot
    plt.figure(figsize=(12, 6))

    # Calculate relative improvement
    relative_imp_1_enkf = (others_enkf[0,:,0] - ours_best[0,:,0]) / others_enkf[0,:,0]
    relative_imp_07_enkf = (others_enkf[1,:,0] - ours_best[1,:,0]) / others_enkf[1,:,0]
    # Plot relative improvement
    plt.plot(x_inds, relative_imp_1_enkf, linestyle='-', marker='o', markersize=8, linewidth=3, label=r"EnKF $\sigma_y=1$")
    plt.plot(x_inds, relative_imp_07_enkf, linestyle='-', marker='o', markersize=8, linewidth=3, label=r"EnKF $\sigma_y=0.7$")
    
    if args.dataset != 'lorenz63':
        relative_imp_1_letkf = (others_letkf[0,:,0] - ours_best[0,:,0]) / others_letkf[0,:,0]
        relative_imp_07_letkf = (others_letkf[1,:,0] - ours_best[1,:,0]) / others_letkf[1,:,0]
        plt.plot(x_inds, relative_imp_1_letkf, linestyle='-', marker='o', markersize=8, linewidth=3, label=r"LETKF $\sigma_y=1$")
        plt.plot(x_inds, relative_imp_07_letkf, linestyle='-', marker='o', markersize=8, linewidth=3, label=r"LETKF $\sigma_y=0.7$")
    plt.legend(loc='upper right')

    # Set x and y labels
    plt.xlabel("Ensemble Size")
    plt.ylabel("Relative Improvement (%)")

    # Set x ticks
    plt.xticks([5, 10, 15, 20, 40, 60, 100])

    # Set y axis to start at 0 and format as percentage
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))  # Convert values to percentage format
    if args.dataset != 'lorenz63':
        if np.min(np.stack((relative_imp_1_enkf, relative_imp_07_enkf, relative_imp_1_letkf, relative_imp_07_letkf))) > 0:
            plt.gca().set_ylim(0, None)  # Ensure y-axis starts from 0
    else:
        if np.min(np.stack((relative_imp_1_enkf, relative_imp_07_enkf))) > 0:
            plt.gca().set_ylim(0, None)  # Ensure y-axis starts from 0

    # Add grid
    plt.grid(True)

    # Set the title based on dataset
    # plt.title(f"{args.dataset.upper()}: Relative improvement"
    #         r"$\dfrac{\mathrm{RMSE}_{\mathrm{Benchmark}} - \mathrm{RMSE}_{\mathrm{Ours}}}{\mathrm{RMSE}_{\mathrm{Benchmark}}}$ v.s. Ensemble size")
    plt.title(f"{args.dataset.upper()}: Relative improvement v.s. Ensemble size")

    # Save the figure
    plt.savefig(os.path.join('../save/figures', f"{args.dataset}_Relative_Imp.png"), bbox_inches="tight", dpi=200)
    print("Image saved to", os.path.join('../save/figures', f"{args.dataset}_Relative_Imp.png"))
    plt.close()

        
    
    

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Process benchmark dataset.")
    
    # Add dataset argument with choices and default value
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["ks", "lorenz96", "lorenz63"], 
        default="lorenz96", 
        help="Specify the dataset to process. Options: 'ks', 'lorenz96', 'lorenz63'. Default is 'lorenz96'."
    )
    parser.add_argument(
        "--rrmse", 
        action="store_true",
        help="Use relative rmse = rmse/rms"
    )

    
    args = parser.parse_args()
    
    plot_results_all(args)