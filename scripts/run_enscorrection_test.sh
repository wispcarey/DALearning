#!/bin/bash

cd ..

# lorenz 63
dataset="lorenz63"

sigma_y=1
seed=42
save_dir="2025-03-25_03-08lorenz63_1.0_10_60_8192_EnST_joint"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python test_enscorrection_set_transformer_local.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $sigma_y \
        --seed $seed \
        --no_localization \
        --cp_load_path save/${save_dir}/cp_1000.pth
done

sigma_y=0.7
seed=42
save_dir="2025-03-25_05-08lorenz63_0.7_10_60_8192_EnST_joint"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python test_enscorrection_set_transformer_local.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $sigma_y \
        --seed $seed \
        --no_localization \
        --cp_load_path save/${save_dir}/cp_1000.pth
done

# # lorenz 96
dataset="lorenz96"

sigma_y=1
seed=42
save_dir="2025-03-25_07-10lorenz96_1.0_10_60_8192_EnST_joint"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python test_enscorrection_set_transformer_local.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $sigma_y \
        --seed $seed \
        --cp_load_path save/${save_dir}/cp_1000.pth
done

sigma_y=0.7
seed=42
save_dir="2025-03-25_11-19lorenz96_0.7_10_60_8192_EnST_joint"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python test_enscorrection_set_transformer_local.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $sigma_y \
        --seed $seed \
        --cp_load_path save/${save_dir}/cp_1000.pth
done

# ks
dataset="ks"

sigma_y=1
seed=42
save_dir="2025-03-24_10-26ks_1.0_10_60_8192_EnST_joint"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python test_enscorrection_set_transformer_local.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $sigma_y \
        --seed $seed \
        --cp_load_path save/${save_dir}/cp_1000.pth
done

sigma_y=0.7
seed=42
save_dir="2025-03-24_18-50ks_0.7_10_60_8192_EnST_joint"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python test_enscorrection_set_transformer_local.py \
        --dataset $dataset \
        --N $N \
        --sigma_y $sigma_y \
        --seed $seed \
        --cp_load_path save/${save_dir}/cp_1000.pth
done

