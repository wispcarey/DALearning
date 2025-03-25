#!/bin/bash

cd ..

# # Lorenz 63
# python finetune_enscorrection_set_transformer_local.py \
#     --epochs 20 \
#     --save_epoch 20 \
#     --dataset lorenz63 \
#     --train_steps 60 \
#     --train_traj_num 8192 \
#     --sigma_y 1 \
#     --seed 42 \
#     --learning_rate 1e-4 \
#     --sigma_ens 1 \
#     --cp_load_path save/2024-12-05_17-49lorenz63_1.0_10_60_8192_EnST_joint/cp_1000.pth \
#     --no_localization 

# python finetune_enscorrection_set_transformer_local.py \
#     --epochs 20 \
#     --save_epoch 20 \
#     --dataset lorenz63 \
#     --train_steps 60 \
#     --train_traj_num 8192 \
#     --sigma_y 0.7 \
#     --seed 42 \
#     --learning_rate 1e-4 \
#     --sigma_ens 1 \
#     --cp_load_path save/2024-12-05_19-57lorenz63_0.7_10_60_8192_EnST_joint/cp_1000.pth \
#     --no_localization 

# # Lorenz 96
python finetune_enscorrection_set_transformer_local.py \
    --epochs 20 \
    --save_epoch 20 \
    --dataset lorenz96 \
    --train_steps 60 \
    --train_traj_num 8192 \
    --sigma_y 1 \
    --seed 42 \
    --learning_rate 1e-4 \
    --cp_load_path save/2024-12-01_16-40lorenz96_1.0_10_60_8192_EnST_joint/cp_1000.pth

# python finetune_enscorrection_set_transformer_local.py \
#     --epochs 20 \
#     --save_epoch 20 \
#     --dataset lorenz96 \
#     --train_steps 60 \
#     --train_traj_num 8192 \
#     --sigma_y 0.7 \
#     --seed 42 \
#     --learning_rate 1e-4 \
#     --cp_load_path save/2024-11-28_00-50lorenz96_0.7_10_60_8192_EnST_joint/cp_1000.pth

# KS
# python finetune_enscorrection_set_transformer_local.py \
#     --epochs 20 \
#     --save_epoch 20 \
#     --dataset ks \
#     --train_steps 60 \
#     --train_traj_num 8192 \
#     --sigma_y 1 \
#     --seed 42 \
#     --learning_rate 5e-5 \
#     --cp_load_path save/2025-02-20_19-00ks_1.0_10_60_8192_EnST_joint/cp_1000.pth

# python finetune_enscorrection_set_transformer_local.py \
#     --epochs 20 \
#     --save_epoch 20 \
#     --dataset ks \
#     --train_steps 60 \
#     --train_traj_num 8192 \
#     --sigma_y 0.7 \
#     --seed 42 \
#     --learning_rate 5e-5 \
#     --cp_load_path save/2025-02-21_03-11ks_0.7_10_60_8192_EnST_joint/cp_1000.pth



