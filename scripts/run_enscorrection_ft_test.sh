#!/bin/bash

cd ..

sigma_y=0.7
seed=42
save_dir="2024-11-14_13-26ks_0.7_20_50_4096_EnST_tuned_normalized_l2"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
# for N in 100; do
    python test_enscorrection_set_transformer_local.py \
        --dataset ks \
        --N $N \
        --sigma_y $sigma_y \
        --seed $seed \
        --cp_load_path save/${save_dir}/ft_cp_${N}_20.pth
done

