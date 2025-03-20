import re
import matplotlib.pyplot as plt
import numpy as np

def parse_training_log(file_path):
    # Read file content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Extract batch information and average batch times
    batch_time_pattern = r'Training epoch : \[(\d+)\]\[(\d+)/(\d+)\].*?Batch time \d+\.\d+ \(Avg: (\d+\.\d+)\)'
    batch_matches = re.findall(batch_time_pattern, content)
    
    # Process batch times - multiply by number of batches to get total epoch time
    batch_times = []
    for epoch, current_batch, total_batches, avg_time in batch_matches:
        # Convert to appropriate types
        epoch = int(epoch)
        total_batches = int(total_batches)
        avg_time = float(avg_time)
        
        # Total epoch time = average batch time × number of batches
        total_epoch_time = avg_time * total_batches
        batch_times.append(total_epoch_time)
    
    # Extract RMSE values and corresponding epochs
    rmse_pattern = r'Average Test RMSE: (\d+\.\d+)'
    epoch_pattern = r'Checkpoint saved to .*?ft_cp_(\d+)_(\d+)\.pth'
    
    rmse_values = re.findall(rmse_pattern, content)
    epochs = re.findall(epoch_pattern, content)
    
    # Round RMSE to 4 decimal places
    rmse_values = [round(float(rmse), 4) for rmse in rmse_values]
    epochs = [int(epoch) for _, epoch in epochs]
    
    
    epochs = [0] + epochs
    
    # # Make sure epochs and RMSE values lists have the same length
    # min_len = min(len(rmse_values), len(epochs))
    # rmse_values = rmse_values[:min_len]
    # epochs = epochs[:min_len]
    
    return batch_times, rmse_values, epochs

if __name__ == "__main__":
    # Use file path
    file_path_1 = "../save/2025-03-18_14-35lorenz96_1.0_20_60_8192_EnST_tuned_joint/ft_output.txt"
    file_path_2 = "../save/2025-03-18_14-45lorenz96_1.0_20_60_8192_EnST_tuned_joint/ft_output.txt"
    
    # file_path_1 = "../save/2025-03-18_15-34lorenz96_1.0_20_60_8192_EnST_tuned_joint/ft_output.txt"
    # file_path_2 = "../save/2025-03-18_15-47lorenz96_1.0_20_60_8192_EnST_tuned_joint/ft_output.txt"

    # Parse log files
    batch_times_1, rmse_values_1, epochs_1 = parse_training_log(file_path_1)
    batch_times_2, rmse_values_2, epochs_2 = parse_training_log(file_path_2)

    print('Single Epoch Time (Model 1):', np.mean(np.array(batch_times_1)))
    print('Single Epoch Time (Model 2):', np.mean(np.array(batch_times_2)))

    # Create high-resolution figure
    plt.figure(figsize=(10, 6), dpi=300)

    # Plot RMSE curves with thicker lines
    plt.plot(epochs_1, rmse_values_1, 'o-', color='#1f77b4', linewidth=4.0, markersize=8, label='Our FT')
    plt.plot(epochs_2, rmse_values_2, 'o-', color='#ff7f0e', linewidth=4.0, markersize=8, label='Full FT')

    # Add vertical dashed line at x=20
    plt.axvline(x=20, color='red', linestyle='--', linewidth=2.0)

    # Set labels and legend with larger font sizes
    plt.xlabel('FT Epoch', fontsize=24)
    plt.ylabel('RMSE', fontsize=24)
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

    # Save figure with high resolution
    plt.savefig('../save/figures/ft_rmse_comparison_20.png', dpi=300, bbox_inches='tight')