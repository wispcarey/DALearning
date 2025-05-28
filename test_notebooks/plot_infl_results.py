import argparse
import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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
        "2025-04-11_12-18lorenz63_1.0_20_60_8192_norm_EnST_tuned_joint",
        "2025-04-11_12-39lorenz63_0.7_20_60_8192_norm_EnST_tuned_joint"],
    'lorenz96': [
        "2025-04-11_12-59lorenz96_1.0_20_60_8192_norm_EnST_tuned_joint",
        "2025-04-11_13-31lorenz96_0.7_20_60_8192_norm_EnST_tuned_joint"],
    'ks': [
        "2025-04-11_14-02ks_1.0_20_60_8192_norm_EnST_tuned_joint",
        "2025-04-11_15-09ks_0.7_20_60_8192_norm_EnST_tuned_joint"],
}

TEST_DATA = {
    'lorenz63': '../data/lorenz63/test_0_96000_v_0.150step.npy',
    'lorenz96': '../data/lorenz96/test_0_96000_v_0.150step.npy',
    'ks': '../data/ks/test_0_128000_v_1.000step.npy'
}

COLOR_DICT = {
    'Trained Infl': "blue",
    'No Infl': "red",
}

plt.rc('font', size=25)

def replace_nan_with_value(nparray, value=6):
    nparray[np.isnan(nparray)] = value
    return nparray

def collect_results(args):
    folder_list = DATASET_FOLDERS[args.dataset]
    
    results = np.zeros((2, len(ENSEMBLE_SIZE), 3), dtype=float)
    nn_results = {}
    
    # Load results for trained inflation
    for i in range(2):
        for j, N in enumerate(ENSEMBLE_SIZE):
            data_dic = torch.load(f'../save/{folder_list[i]}/output_records_{N}.pt', weights_only=True)
            results[i, j, 0] = data_dic['nn']['mean_rrmse']
            results[i, j, 1] = data_dic['nn']['std_rrmse']
            results[i, j, 2] = data_dic['nn']['valid_percent'] < 1
    
    nn_results['Trained Infl'] = results.copy()
    
    # Load results for no inflation
    for i in range(2):
        for j, N in enumerate(ENSEMBLE_SIZE):
            data_dic = torch.load(f'../save/{folder_list[i]}/output_records_zero_infl_{N}.pt', weights_only=True)
            results[i, j, 0] = data_dic['nn']['mean_rrmse']
            results[i, j, 1] = data_dic['nn']['std_rrmse']
            results[i, j, 2] = data_dic['nn']['valid_percent'] < 1
    
    nn_results['No Infl'] = results.copy()
    
    return nn_results

def save_legend_separately(fig_num, filename_prefix):
    """Save legend as a separate horizontal image"""
    # Get the legend from the current figure
    fig = plt.figure(fig_num)
    ax = fig.gca()
    
    # Create a new figure for the legend only
    legend_fig = plt.figure(figsize=(12, 2))
    
    # Get handles and labels from the original plot
    handles, labels = ax.get_legend_handles_labels()
    
    # Create horizontal legend
    legend_fig.legend(handles, labels, loc='center', ncol=len(labels), frameon=False)
    legend_fig.gca().set_axis_off()
    
    # Save legend
    legend_fig.savefig(os.path.join('../save/figures', f"{filename_prefix}_legend.png"), 
                      bbox_inches="tight", dpi=200, pad_inches=0)
    legend_fig.savefig(os.path.join('../save/figures', f"{filename_prefix}_legend.pdf"), 
                      bbox_inches="tight", dpi=200, pad_inches=0)
    
    plt.close(legend_fig)
        

def plot_results_all(args):
    
    x_inds = ENSEMBLE_SIZE
    
    nn_results_all = collect_results(args)
    
    save_figure_suffix = ["R_RMSE_10", "R_RMSE_07"]
    
    fig1 = plt.figure(1, figsize=(14, 6)) 
    fig2 = plt.figure(2, figsize=(14, 6)) 
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
                
        mean_rmse_1, std_rmse_1 = value[0, :, 0], value[0, :, 1]
        mean_rmse_record["10"] = mean_rmse_1
        
        print(f"RMSE {key} for {args.dataset} with " + r"$\sigma_y=1$:", mean_rmse_1)

        plt.figure(1)
        plt.errorbar(x_inds, mean_rmse_1, yerr=std_rmse_1,
            linestyle='-', capsize=3, capthick=2, color=COLOR_DICT[key], marker='D', linewidth=2, alpha=0.75, 
            label=f"{key} (mean ± std)")
        
        mean_rmse_07, std_rmse_07 = value[1, :, 0], value[1, :, 1]
        mean_rmse_record["07"] = mean_rmse_07
        
        
        print(f"RMSE {key} for {args.dataset} with " + r"$\sigma_y=0.7$:", mean_rmse_07)

        plt.figure(2)
        plt.errorbar(x_inds, mean_rmse_07, yerr=std_rmse_07,
            linestyle='-', capsize=3, capthick=2, color=COLOR_DICT[key], marker='D', linewidth=2, alpha=0.75, 
            label=f"{key} (mean ± std)")
        
        mean_rmse_dict[key] = mean_rmse_record
            
    
    
    for i in range(2):
        plt.figure(i+1)

        ticksize = 0.2
            
        vmax = value_max[i]
        vmin = value_min[i]

        y_ticks = list(np.arange(int(vmin / ticksize) * ticksize, vmax + ticksize, ticksize))
        new_y_ticks = [f"{y:.2f}".rstrip('0').rstrip('.') if '.' in f"{y:.2f}" else f"{y}" for y in y_ticks]
            
        plt.yticks(ticks=y_ticks, labels=new_y_ticks)
        # Remove legend from main plot
        # plt.legend(loc='upper right')
        plt.xticks(ENSEMBLE_SIZE)
        plt.grid(True)
        
        # Save main plot without labels and title - no white borders
        plt.savefig(os.path.join('../save/figures', f"{args.dataset}_{save_figure_suffix[i]}_infl.png"), bbox_inches="tight", dpi=200, pad_inches=0)
        plt.savefig(os.path.join('../save/figures', f"{args.dataset}_{save_figure_suffix[i]}_infl.pdf"), bbox_inches="tight", dpi=200, pad_inches=0)
        print("Image saved to", os.path.join('../save/figures', f"{args.dataset}_{save_figure_suffix[i]}_infl.png"))
        
        # Save legend separately only for the first plot
        if i == 0:
            save_legend_separately(i+1, f"{args.dataset}_legend")
        
        plt.close()  
    
    
    # No relative improvement plot needed

        
    
    

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

    
    args = parser.parse_args()
    
    plot_results_all(args)