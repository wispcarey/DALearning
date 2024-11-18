#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=00:10:00     # walltime (10 minutes)
#SBATCH --nodes=1           # number of nodes (1 node)
#SBATCH --gres=gpu:v100:2   # 2 V100 GPUs with 16GB memory each
#SBATCH --partition=gpu     # use GPU partition
#SBATCH --constraint="v100" # ensure 16GB V100 GPUs
#SBATCH --ntasks=2          # 2 tasks (1 task per GPU)
#SBATCH -J "bohan-gpu-test-LearnKalmanGain"   # job name
#SBATCH --mail-user=bhchen@caltech.edu # email address
#SBATCH --mail-type=BEGIN   # email notification at start
#SBATCH --mail-type=END     # email notification at end
#SBATCH --mail-type=FAIL    # email notification on failure

# Optional: specify output and error files
#SBATCH -o slurm.%N.%j.out  # STDOUT
#SBATCH -e slurm.%N.%j.err  # STDERR

# Load modules if necessary (e.g., CUDA or other dependencies)
# module load cuda/12.2  # Adjusted to CUDA version 12.2

# Change to the directory containing v2_run_fine_tuning.sh
cd ../script  # Assuming `script` folder is one level up from `slurm_script`

# Run your program
bash v2_run_enscorrection_train.sh