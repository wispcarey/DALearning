#!/bin/bash

cd ..

# # lorenz 63
dataset="lorenz63"

sigma_y=1
seed=42
save_dir="2025-03-25_15-33lorenz63_1.0_20_60_8192_EnST_tuned_joint"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python test_enscorrection_set_transformer_local.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $sigma_y \
        --seed $seed \
        --sigma_ens 1 \
        --cp_load_path save/${save_dir}/ft_cp_${N}_20.pth \
        --no_localization 
done

sigma_y=0.7
seed=42
save_dir="2025-03-25_15-50lorenz63_0.7_20_60_8192_EnST_tuned_joint"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python test_enscorrection_set_transformer_local.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $sigma_y \
        --seed $seed \
        --cp_load_path save/${save_dir}/ft_cp_${N}_20.pth \
        --no_localization 
done

# # lorenz 96
dataset="lorenz96"

sigma_y=1
seed=42
# save_dir="2024-12-09_14-55lorenz96_1.0_20_60_8192_EnST_tuned_joint"
save_dir="2025-03-25_16-07lorenz96_1.0_20_60_8192_EnST_tuned_joint"

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
save_dir="2025-03-25_16-36lorenz96_0.7_20_60_8192_EnST_tuned_joint"

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
save_dir="2025-03-25_17-05ks_1.0_20_60_8192_EnST_tuned_joint"

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
save_dir="2025-03-25_18-02ks_0.7_20_60_8192_EnST_tuned_joint"

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

