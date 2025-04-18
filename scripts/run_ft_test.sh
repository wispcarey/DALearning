#!/bin/bash

cd ..

# lorenz 63
# dataset="lorenz63"

# sigma_y=1
# seed=42
# save_dir="2025-04-11_12-18lorenz63_1.0_20_60_8192_norm_EnST_tuned_joint"

# # sigma_y = $sigma_y, EnST
# for N in 5 10 15 20 40 60 100; do
#     python evaluate.py \
#         --dataset $dataset \
#         --N $N \
#         --sigma_y $sigma_y \
#         --seed $seed \
#         --cp_load_path save/${save_dir}/ft_cp_${N}_20.pth \
#         --no_localization 
# done

# sigma_y=0.7
# seed=42
# save_dir="2025-04-11_12-39lorenz63_0.7_20_60_8192_norm_EnST_tuned_joint"

# # sigma_y = $sigma_y, EnST
# for N in 5 10 15 20 40 60 100; do
#     python evaluate.py \
#         --dataset $dataset \
#         --N $N \
#         --sigma_y $sigma_y \
#         --seed $seed \
#         --cp_load_path save/${save_dir}/ft_cp_${N}_20.pth \
#         --no_localization 
# done

# lorenz 96
dataset="lorenz96"

sigma_y=1
seed=42
# save_dir="2024-12-09_14-55lorenz96_1.0_20_60_8192_EnST_tuned_joint"
save_dir="2025-04-11_12-59lorenz96_1.0_20_60_8192_norm_EnST_tuned_joint"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python evaluate.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $sigma_y \
        --seed $seed \
        --cp_load_path save/${save_dir}/ft_cp_${N}_20.pth 
done

sigma_y=0.7
seed=42
save_dir="2025-04-11_13-31lorenz96_0.7_20_60_8192_norm_EnST_tuned_joint"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python evaluate.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $sigma_y \
        --seed $seed \
        --cp_load_path save/${save_dir}/ft_cp_${N}_20.pth 
done

# ks
dataset="ks"

sigma_y=1
seed=42
save_dir="2025-04-11_14-02ks_1.0_20_60_8192_norm_EnST_tuned_joint"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python evaluate.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $sigma_y \
        --seed $seed \
        --cp_load_path save/${save_dir}/ft_cp_${N}_20.pth
done

sigma_y=0.7
seed=42
save_dir="2025-04-11_15-09ks_0.7_20_60_8192_norm_EnST_tuned_joint"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
# for N in 15; do
    python evaluate.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $sigma_y \
        --seed $seed \
        --cp_load_path save/${save_dir}/ft_cp_${N}_20.pth
done

