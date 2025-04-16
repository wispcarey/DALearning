import argparse
import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import plot_results_3d, plot_results_2d
from utils import partial_obs_operator, get_mean_std
from config.dataset_info import DATASET_INFO

from networks import ComplexAttentionModel, AttentionModel
from localization import plot_GC, pairwise_distances

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

ENSEMBLE_SIZE = [5, 10, 15, 20, 40, 60, 100]

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
    
    sample_data = torch.load(f'../save/{folder_list[0]}/output_records_{ENSEMBLE_SIZE[0]}.pt', weights_only=True)
    loc_dist_num = len(sample_data['nn']['loc_diff_dist'])
    
    results = np.zeros((2, len(ENSEMBLE_SIZE), loc_dist_num, 3), dtype=float)
    
    for i in range(2):
        for j, N in enumerate(ENSEMBLE_SIZE):
            data_dic = torch.load(f'../save/{folder_list[i]}/output_records_{N}.pt', weights_only=True)
            results[i, j, :, 0] = data_dic['nn']['loc_diff_dist'].cpu().numpy()
            results[i, j, :, 1] = data_dic['nn']['loc_mean'].cpu().numpy()
            results[i, j, :, 2] = data_dic['nn']['loc_std'].cpu().numpy()
    
    return results
        
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
    df = pd.read_csv(file_path, usecols=['method', 'sigma_y', 'best_loc_rad'])

    method = "LETKF"

    method_data = df[df['method'] == method]
    
    # Filter rows where sigma_y == 1 and 0.7
    loc_y_1 = method_data[method_data['sigma_y'] == 1][['best_loc_rad']]
    loc_y_0_7 = method_data[method_data['sigma_y'] == 0.7][['best_loc_rad']]
    
    # Convert to numpy arrays
    sigma_y_1_array = loc_y_1.to_numpy()
    sigma_y_0_7_array = loc_y_0_7.to_numpy()
    
    # Combine into a 2*N*1 array
    combined_array = np.stack([sigma_y_1_array, sigma_y_0_7_array], axis=0)

    return combined_array

def calculate_gc(radius, x):
    """
    Calculate the GC function based on the given radius and x values.
    """
    dists = np.abs(x)
    coeffs = np.zeros_like(dists)
    
    R = radius * 1.82  # =np.sqrt(10/3). Sakov: 1.7386
    
    # 1st segment
    ind1 = dists <= R
    r2 = (dists[ind1] / R) ** 2
    r3 = (dists[ind1] / R) ** 3
    coeffs[ind1] = 1 + r2 * (-r3 / 4 + r2 / 2) + r3 * (5 / 8) - r2 * (5 / 3)
    
    # 2nd segment
    ind2 = np.logical_and(R < dists, dists <= 2 * R)
    r1 = dists[ind2] / R
    r2 = (dists[ind2] / R) ** 2
    r3 = (dists[ind2] / R) ** 3
    coeffs[ind2] = (
        r2 * (r3 / 12 - r2 / 2)
        + r3 * (5 / 8)
        + r2 * (5 / 3)
        - r1 * 5
        + 4
        - (2 / 3) / r1
    )
    
    return coeffs

def plot_gc_and_curves(A, B, save_dir='../save/figures'):
    """
    Plot GC functions and curves based on the given arrays A and B.
    A: array of shape (2, 7, M, 3)
    B: array of shape (2, 7, 1)
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Output filenames for reference
    output_files = {}
    
    for i in range(2):  # First dimension
        output_files[i] = {}
        sigma_y = 1.0 if i == 0 else 0.7
        
        for j in range(7):  # Second dimension (ENSEMBLE_SIZE)
            # Get radius from B
            radius = B[i, j, 0]
            
            # Get x values from A
            x = A[i, j, :, 0]
            
            # Calculate GC function
            gc_values = calculate_gc(radius, x)
            
            # Get center and range values from A
            centers = A[i, j, :, 1]
            ranges = A[i, j, :, 2]
            
            # Create figure with minimal borders
            fig = plt.figure(figsize=(4, 3))
            ax = plt.axes([0, 0, 1, 1])  # Remove all padding
            
            # Plot GC function
            ax.plot(x, gc_values, 'b-', linewidth=2)
            
            # Plot curves with center and range
            ax.fill_between(x, centers - ranges, centers + ranges, alpha=0.3, color='red')
            ax.plot(x, centers, 'r-', linewidth=1.5)
            
            # Remove all axes elements
            ax.set_axis_off()
            
            # Create filename
            if sigma_y == 1.0:
                suffix = "10"
            else:
                suffix = "07"
            filename = f"loc_plot_{ENSEMBLE_SIZE[j]}_{suffix}.png"
            filepath = os.path.join(save_dir, filename)
            
            # Save figure with no border and no padding
            plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Store filename for reference
            output_files[i][ENSEMBLE_SIZE[j]] = filepath
    
    return output_files

def create_combined_grid(dataset, ens_list, save_dir='../save/figures', add_ticks=True):
    """
    Create a combined grid of plots for the specified ensemble sizes.
    Add ticks to the outer edges of the combined grid.
    
    Args:
        ens_list: List of ensemble sizes to include (must be a subset of ENSEMBLE_SIZE)
        save_dir: Directory where individual plots are saved
        add_ticks: Whether to add x and y ticks to the combined grid
    """
    # Validate input
    for size in ens_list:
        if size not in ENSEMBLE_SIZE:
            raise ValueError(f"Ensemble size {size} not in ENSEMBLE_SIZE list")
    
    # Get list of images to combine
    image_grid = []
    
    # Two rows for sigma_y values
    for i in range(2):
        row_images = []
        sigma_y = 1.0 if i == 0 else 0.7
        suffix = "10" if sigma_y == 1.0 else "07"
        
        # Columns for ensemble sizes
        for ens in ens_list:
            # Get corresponding image file
            img_path = os.path.join(save_dir, f"loc_plot_{ens}_{suffix}.png")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image {img_path} not found. Run plot_gc_and_curves first.")
            
            # Open image
            img = Image.open(img_path)
            row_images.append(img)
        
        image_grid.append(row_images)
    
    # Determine dimensions of individual images for consistency
    # Assuming all images have the same size
    img_width, img_height = image_grid[0][0].size
    
    # Create combined image using PIL first
    combined_width = len(ens_list) * img_width
    combined_height = 2 * img_height
    combined_img = Image.new('RGB', (combined_width, combined_height), color='white')
    
    # Paste images into grid
    for i in range(2):
        for j, img in enumerate(image_grid[i]):
            combined_img.paste(img, (j * img_width, i * img_height))
    
    if not add_ticks:
        # Without ticks, just save the PIL combined image
        combined_path = os.path.join(save_dir, f"{dataset}_loc_grid_{'_'.join(map(str, ens_list))}.png")
        combined_img.save(combined_path)
    else:
        # Add ticks using matplotlib
        fig, ax = plt.subplots(figsize=(combined_width/100, combined_height/100))  # Convert pixels to inches (approximate)
        
        # Display the combined image
        ax.imshow(np.array(combined_img))
        
        # Add vertical grid lines between columns
        for j in range(1, len(ens_list)):
            ax.axvline(x=j * img_width, color='black', linestyle='-', linewidth=1, alpha=0.7)
        
        # Add horizontal grid line between rows
        ax.axhline(y=img_height, color='black', linestyle='-', linewidth=1, alpha=0.7)
        
        # Add ticks on the bottom for each ensemble size
        x_ticks = []
        for j, ens in enumerate(ens_list):
            # Place tick at the center of each column
            x_ticks.append(j * img_width + img_width / 2)
        
        ax.set_xticks(x_ticks)
        # Use ensemble sizes as x tick labels with larger font
        ax.set_xticklabels(ens_list, fontsize=14)
        
        # Remove y ticks completely
        ax.set_yticks([])
        
        # Add title instead of ensemble size labels (since we now use them as x-ticks)
        ax.set_xlabel('Ensemble Size', fontsize=20)
        ax.xaxis.set_label_position('bottom')
        
        # Add sigma_y labels on the left
        for i in range(2):
            sigma_y = 1.0 if i == 0 else 0.7
            mid_point = i * img_height + img_height / 2
            ax.text(-20, mid_point, r"$\sigma_y = " + f"{sigma_y}" + "$", 
                    verticalalignment='center', fontsize=20, fontweight='bold', rotation=90)
        
        # Remove the axis frame
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Adjust layout to make room for titles and labels
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
        
        # Save combined image with ticks
        plt.savefig(os.path.join(save_dir, f"{dataset}_loc_grid_{'_'.join(map(str, ens_list))}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f"{dataset}_loc_grid_{'_'.join(map(str, ens_list))}.pdf"), dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_results_all(args):
    """
    Main function to collect results, get benchmarks, generate individual plots,
    and create a combined grid.
    """
    # Create figures directory if it doesn't exist
    os.makedirs('../save/figures', exist_ok=True)
    
    # Collect neural network results
    nn_results = collect_results(args)
    
    # Get benchmark results
    benchmark_results = get_benchmarks(args)
    
    # Generate individual plots
    plot_gc_and_curves(nn_results, benchmark_results)
    
    # Create combined grid for selected ensemble sizes
    create_combined_grid(args.dataset, args.ensemble_sizes)


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
    
    # Add optional argument for ensemble sizes to include in the combined grid
    parser.add_argument(
        "--ensemble_sizes",
        nargs="+",
        type=int,
        default=ENSEMBLE_SIZE,
        help="Specify which ensemble sizes to include in the combined grid. Default is all sizes."
    )
    
    args = parser.parse_args()
    
    # Run the main function
    plot_results_all(args)
    
    # # If specific ensemble sizes were provided, create a custom combined grid
    # if args.ensemble_sizes:
    #     valid_sizes = [size for size in args.ensemble_sizes if size in ENSEMBLE_SIZE]
    #     if valid_sizes:
    #         print(f"Creating combined grid for ensemble sizes: {valid_sizes}")
    #         create_combined_grid(args.dataset, valid_sizes)
    #     else:
    #         print(f"Warning: None of the provided ensemble sizes {args.ensemble_sizes} are valid. Must be from {ENSEMBLE_SIZE}")