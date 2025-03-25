#!/bin/bash

cd ..

# # lorenz 63
# dataset="lorenz63"

# sigma_y=1
# seed=42
# save_dir="2024-12-06_12-23lorenz63_1.0_20_60_8192_EnST_tuned_joint"

# # sigma_y = $sigma_y, EnST
# for N in 5 10 15 20 40 60 100; do
#     python test_enscorrection_set_transformer_local.py \
#         --dataset $dataset \
#         --N $N \
#         --sigma_y $sigma_y \
#         --seed $seed \
#         --sigma_ens 1 \
#         --cp_load_path save/${save_dir}/ft_cp_${N}_20.pth \
#         --no_localization 
# done

# sigma_y=0.7
# seed=42
# save_dir="2024-12-09_13-38lorenz63_0.7_20_60_8192_EnST_tuned_joint"

# # sigma_y = $sigma_y, EnST
# for N in 5 10 15 20 40 60 100; do
#     python test_enscorrection_set_transformer_local.py \
#         --dataset $dataset \
#         --N $N \
#         --sigma_y $sigma_y \
#         --seed $seed \
#         --cp_load_path save/${save_dir}/ft_cp_${N}_20.pth \
#         --no_localization 
# done

# # lorenz 96
dataset="lorenz96"

sigma_y=1
seed=42
# save_dir="2024-12-09_14-55lorenz96_1.0_20_60_8192_EnST_tuned_joint"
save_dir="2025-03-24_10-19lorenz96_1.0_20_60_8192_EnST_tuned_joint"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python test_enscorrection_set_transformer_local.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $sigma_y \
        --seed $seed \
        --cp_load_path save/${save_dir}/ft_cp_${N}_20.pth
done

sigma_y=0.7
seed=42
save_dir="2024-12-09_15-26lorenz96_0.7_20_60_8192_EnST_tuned_joint"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python test_enscorrection_set_transformer_local.py \
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
save_dir="2025-02-21_11-24ks_1.0_20_60_8192_EnST_tuned_joint"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python test_enscorrection_set_transformer_local.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $sigma_y \
        --seed $seed \
        --cp_load_path save/${save_dir}/ft_cp_${N}_20.pth
done

sigma_y=0.7
seed=42
save_dir="2025-02-21_12-23ks_0.7_20_60_8192_EnST_tuned_joint"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
# for N in 15; do
    python test_enscorrection_set_transformer_local.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $sigma_y \
        --seed $seed \
        --cp_load_path save/${save_dir}/ft_cp_${N}_20.pth
done

