#!/bin/bash

cd ..

sigma_y=0.7
test_steps=1500
test_traj_num=100
seed=42
save_dir="2024-11-11_15-28lorenz96_0.7_20_50_4096_EnST_tuned_normalized_l2"

# sigma_y = $sigma_y, EnST
for N in 5 10 15 20 40 60 100; do
    python test_enscorrection_set_transformer_local.py \
        --N $N \
        --test_steps $test_steps \
        --test_traj_num $test_traj_num \
        --sigma_y $sigma_y \
        --seed $seed \
        --hidden_dim 64 \
        --cp_load_path save/${save_dir}/ft_cp_${N}_20.pth
done

