import torch
import matplotlib.pyplot as plt
import os
import argparse
import sys

def plot_comparison(pt_file1, pt_file2, save_dir='../save/figures/', filename='ft_comparison'):
    """
    Read two PT files, extract 'test_rrmse' and 'test_epochs', and plot comparison curves.
    
    Args:
        pt_file1 (str): Path to first PT file (Our FT)
        pt_file2 (str): Path to second PT file (Full FT)
        save_dir (str): Directory to save output figures
        filename (str): Base filename for saved figures
    """
    # Load PT files
    data1 = torch.load(pt_file1, weights_only=True)
    data2 = torch.load(pt_file2, weights_only=True)
    
    # Create figure with larger size
    plt.figure(figsize=(11, 8))
    
    # Plot first file data - 'Our FT'
    plt.plot(data1['test_epochs'], data1['test_rrmse'], 
             label='Our FT', 
             linewidth=4.0, 
             marker='o', 
             markersize=8)
    
    # Plot second file data - 'Full FT'
    plt.plot(data2['test_epochs'], data2['test_rrmse'], 
             label='Full FT', 
             linewidth=4.0, 
             marker='s',  # Square marker to differentiate
             markersize=8)
    
    # Add vertical line at x=20
    plt.axvline(x=20, color='red', linestyle='--', linewidth=2.0)
    
    # Set labels and legend with larger font sizes
    plt.xlabel('FT Epoch', fontsize=24)
    plt.ylabel('R-RMSE', fontsize=24)
    plt.legend(fontsize=26)
    
    # Remove grid
    # plt.grid(True, linestyle='--', alpha=0.7)  # Commented out to remove grid
    
    # Increase tick label size
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    # Remove spines
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.5)
    
    # Tight layout
    plt.tight_layout()
    
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as PNG
    png_path = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    # Save as PDF
    pdf_path = os.path.join(save_dir, f"{filename}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    
    print(f"Figures saved to {png_path} and {pdf_path}")
    
    # Close the plot to free memory
    plt.close()

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Process benchmark dataset.")
    
    # Add dataset argument with choices and default value
    parser.add_argument(
        "--N", 
        type=int, 
        default=20, 
        help="Ensemble size."
    )
    args = parser.parse_args()
    
    # Use file path
    pt_path_1 = f"../save/2025-04-02_19-50lorenz96_1.0_20_60_8192_EnST_tuned_joint/ft_records_{args.N}.pt"
    pt_path_2 = f"../save/2025-04-02_20-12lorenz96_1.0_20_60_8192_EnST_tuned_joint/ft_records_{args.N}.pt"
    
    plot_comparison(pt_path_1, pt_path_2, save_dir='../save/figures/', filename=f'ft_comparison_{args.N}')