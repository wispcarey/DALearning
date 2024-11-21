#!/bin/bash

cd ..

python finetune_enscorrection_set_transformer_local.py \
    --epochs 20 \
    --save_epoch 20 \
    --dataset lorenz63 \
    --train_steps 50 \
    --train_traj_num 4096 \
    --sigma_y 1 \
    --seed 42 \
    --learning_rate 1e-4 \
    --loss_type normalized_l2 \
    --no_localization \
    --cp_load_path save/2024-11-18_15-50lorenz63_1.0_10_60_8192_EnST_normalized_l2/cp_1000.pth

python finetune_enscorrection_set_transformer_local.py \
    --epochs 20 \
    --save_epoch 20 \
    --dataset ks \
    --train_steps 50 \
    --train_traj_num 4096 \
    --sigma_y 0.7 \
    --batch_size 256 \
    --seed 42 \
    --learning_rate 1e-4 \
    --loss_type normalized_l2 \
    --no_localization \
    --cp_load_path save/2024-11-18_15-50lorenz63_1.0_10_60_8192_EnST_normalized_l2/cp_1000.pth



