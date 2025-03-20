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

ENSEMBLE_SIZE = [5,10,15,20,40,60,100]

DATASET_FOLDERS = {
    'lorenz63': [
        "2024-12-06_12-23lorenz63_1.0_20_60_8192_EnST_tuned_joint",
        "2024-12-09_13-38lorenz63_0.7_20_60_8192_EnST_tuned_joint"],
    'lorenz96': [
        "2024-12-09_14-55lorenz96_1.0_20_60_8192_EnST_tuned_joint",
        "2024-12-09_15-26lorenz96_0.7_20_60_8192_EnST_tuned_joint"],
    'ks': [
        "2025-02-21_11-24ks_1.0_20_60_8192_EnST_tuned_joint",
        "2025-02-21_12-23ks_0.7_20_60_8192_EnST_tuned_joint"],
}

TEST_DATA = {
    'lorenz63': '../data/lorenz63/test_0_96000_v_0.150step.npy',
    'lorenz96': '../data/lorenz96/test_0_96000_v_0.150step.npy',
    'ks': '../data/ks/test_0_128000_v_1.000step.npy'
}

COLOR_DICT = {
    'Normal Infl': "blue",
    'No Infl': "red",
}

plt.rc('font', size=20)

def replace_nan_with_value(nparray, value=6):
    nparray[np.isnan(nparray)] = value
    return nparray

def collect_results(args):
    folder_list = DATASET_FOLDERS[args.dataset]
    
    results = np.zeros((2, len(ENSEMBLE_SIZE), 5), dtype=float)
    nn_results = {}
    
    for i in range(2):
        for j, N in enumerate(ENSEMBLE_SIZE):
            data_dic = torch.load(f'../save/{folder_list[i]}/output_records_{N}.pt', weights_only=True)
            results[i, j, 0] = data_dic['nn']['mean_rmse']
            results[i, j, 1] = data_dic['nn']['std_rmse']
            results[i, j, 2] = data_dic['nn']['mean_rmv']
            results[i, j, 3] = data_dic['nn']['std_rmv']
            results[i, j, 4] = data_dic['nn']['valid_percent'] < 1
    
    nn_results['Normal Infl'] = results.copy()
    
    for i in range(2):
        for j, N in enumerate(ENSEMBLE_SIZE):
            data_dic = torch.load(f'../save/{folder_list[i]}/output_records_zero_infl_{N}.pt', weights_only=True)
            results[i, j, 0] = data_dic['nn']['mean_rmse']
            results[i, j, 1] = data_dic['nn']['std_rmse']
            results[i, j, 2] = data_dic['nn']['mean_rmv']
            results[i, j, 3] = data_dic['nn']['std_rmv']
            results[i, j, 4] = data_dic['nn']['valid_percent'] < 1
    
    nn_results['No Infl'] = results.copy()
    
    return nn_results
        

def plot_results_all(args):
    if args.rrmse:
        print("Use the relative metric = rmse/rms")
        test_traj = np.load(TEST_DATA[args.dataset])
        RMS = np.mean(np.sqrt(np.mean(test_traj ** 2, axis=2)))
        print("RMS:", RMS)
    
    x_inds = ENSEMBLE_SIZE
    
    nn_results_all = collect_results(args)
    
    titles = [f"{args.dataset.upper()}:"+r"$\sigma_y=1$; RMSE Mean $\pm$ 1std", 
            f"{args.dataset.upper()}:"+r"$\sigma_y=1$; RMV Mean $\pm$ 1std",
            f"{args.dataset.upper()}:"+r"$\sigma_y=0.7$; RMSE Mean $\pm$ 1std",
            f"{args.dataset.upper()}:"+r"$\sigma_y=0.7$; RMV Mean $\pm$ 1std"]
    if args.rrmse:
        y_labels = ["RRMSE", "RRMSE"]
    else:
        y_labels = ["RMSE", "RMSE"]
    save_figure_suffix = ["RMSE_10", "RMSE_07"]
    
    fig1 = plt.figure(1, figsize=(12, 6)) 
    fig2 = plt.figure(2, figsize=(12, 6)) 
    # fig3 = plt.figure(3, figsize=(12, 6))
    # fig4 = plt.figure(4, figsize=(12, 6))
    value_max = 0 * np.ones(2)
    value_min = 100 * np.ones(2)
    
    mean_rmse_dict = {}
    for key, value in nn_results_all.items():

        mean_rmse_record = {}

        max_val = np.max(value[:,:,0], axis=1)
        value_max = np.maximum(max_val, value_max)
        
        min_val = np.min(value[:,:,0], axis=1)
        print("min val:", min_val)
        value_min = np.minimum(min_val, value_min)
                
        mean_rmse_1, std_rmse_1, mean_rmv_1, std_rmv_1, nan_exists_1 = value[0, :, 0], value[0, :, 1], value[0, :, 2], value[0, :, 3], value[0, :, 4]
        mean_rmse_record["10"] = mean_rmse_1
        
        if args.rrmse:
            mean_rmse_1, std_rmse_1, mean_rmv_1, std_rmv_1 = mean_rmse_1 / RMS, std_rmse_1 / RMS, mean_rmv_1 / RMS, std_rmv_1 / RMS
        
        print(f"RMSE {key} for {args.dataset} with " + r"$\sigma_y=1$:", mean_rmse_1)

        plt.figure(1)
        plt.errorbar(x_inds, mean_rmse_1, yerr=std_rmse_1,
            linestyle='-', capsize=3, capthick=2, color=COLOR_DICT[key], marker='D', linewidth=2, alpha=0.75, label=f"{key}")
        # for x, y, is_star in zip(x_inds, mean_rmv_1, nan_exists_1):
        #     marker = "*" if is_star else "D"
        #     plt.scatter(x, y, marker=marker, color=COLOR_DICT[key], s=50)
        
        mean_rmse_07, std_rmse_07, mean_rmv_07, std_rmv_07, nan_exists_07 = value[1, :, 0], value[1, :, 1], value[1, :, 2], value[1, :, 3], value[1, :, 4]
        mean_rmse_record["07"] = mean_rmse_07
        
        if args.rrmse:
            mean_rmse_07, std_rmse_07, mean_rmv_07, std_rmv_07 = mean_rmse_07 / RMS, std_rmse_07 / RMS, mean_rmv_07 / RMS, std_rmv_07 / RMS
        
        print(f"RMSE {key} for {args.dataset} with " + r"$\sigma_y=0.7$:", mean_rmse_07)

        plt.figure(2)
        plt.errorbar(x_inds, mean_rmse_07, yerr=std_rmse_07,
            linestyle='-', capsize=3, capthick=2, color=COLOR_DICT[key], marker='D', linewidth=2, alpha=0.75, label=f"{key}")
        # for x, y, is_star in zip(x_inds, mean_rmv_07, nan_exists_07):
        #     marker = "*" if is_star else "D"
        #     plt.scatter(x, y, marker=marker, color=COLOR_DICT[key], s=50)
        
        mean_rmse_dict[key] = mean_rmse_record
            
    
    
    for i in range(2):
        plt.figure(i+1)

        if args.rrmse:
            if args.dataset == 'lorenz63':
                ticksize = 0.02
            else:
                ticksize = 0.25
        else:
            ticksize = 0.5
            
        if args.rrmse:
            vmax = value_max[i] / RMS
            vmin = value_min[i] / RMS
        else:
            vmax = value_max[i]
            vmin = value_min[i]

        y_ticks = list(np.arange(int(vmin / ticksize) * ticksize, vmax + ticksize, ticksize))
        print(vmax,vmin)
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
        plt.savefig(os.path.join('../save/figures', f"{args.dataset}_{save_figure_suffix[i]}_infl_{suffix}.png"), bbox_inches="tight", dpi=200)
        print("Image saved to", os.path.join('../save/figures', f"{args.dataset}_{save_figure_suffix[i]}.png"))
        plt.close()  
    
    # Relative improvement plot
    plt.figure(figsize=(12, 6))

    # Calculate relative improvement
    relative_imp_1 = np.abs(mean_rmse_dict['Normal Infl']['10'] - mean_rmse_dict['No Infl']['10']) / mean_rmse_dict['Normal Infl']['10']
    relative_imp_07 = np.abs(mean_rmse_dict['Normal Infl']['07'] - mean_rmse_dict['No Infl']['07']) / mean_rmse_dict['Normal Infl']['07']
    # Plot relative improvement
    plt.plot(x_inds, relative_imp_1, linestyle='-', marker='o', markersize=8, linewidth=3, label=r"$\sigma_y=1$")
    plt.plot(x_inds, relative_imp_07, linestyle='-', marker='o', markersize=8, linewidth=3, label=r"$\sigma_y=0.7$")
    
    plt.legend(loc='upper right')

    # Set x and y labels
    plt.xlabel("Ensemble Size")
    plt.ylabel("Relative Difference")

    # Set x ticks
    plt.xticks([5, 10, 15, 20, 40, 60, 100])

    # Set y axis to start at 0 and format as percentage
    # plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))  # Convert values to percentage format

    if np.min(np.stack((relative_imp_1, relative_imp_07))) > 0:
        plt.gca().set_ylim(0, None)  # Ensure y-axis starts from 0

    # Add grid
    plt.grid(True)

    # Set the title based on dataset
    # plt.title(f"{args.dataset.upper()}: Relative improvement"
    #         r"$\dfrac{\mathrm{RMSE}_{\mathrm{Benchmark}} - \mathrm{RMSE}_{\mathrm{Ours}}}{\mathrm{RMSE}_{\mathrm{Benchmark}}}$ v.s. Ensemble size")
    plt.title(f"{args.dataset.upper()}: Relative RMSE v.s. Ensemble size")

    # Save the figure
    plt.savefig(os.path.join('../save/figures', f"{args.dataset}_Relative_Imp_Infl.png"), bbox_inches="tight", dpi=200)
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