#!/bin/bash

cd ..

python finetune_enscorrection_set_transformer_local_v2.py \
    --epochs 20 \
    --save_epoch 10 \
    --dataset lorenz96 \
    --train_steps 50 \
    --train_traj_num 4096 \
    --sigma_y 1 \
    --batch_size 256 \
    --seed 42 \
    --learning_rate 5e-4 \
    --loss_type normalized_l2 \
    --st_output_dim 128 \
    --cp_load_path save/2024-11-14_18-32lorenz96_1.0_10_60_8192_EnST_normalized_l2_v2/cp_1000.pth

python finetune_enscorrection_set_transformer_local_v2.py \
    --epochs 20 \
    --save_epoch 10 \
    --dataset lorenz96 \
    --train_steps 50 \
    --train_traj_num 4096 \
    --sigma_y 0.7 \
    --batch_size 256 \
    --seed 42 \
    --learning_rate 5e-4 \
    --loss_type normalized_l2 \
    --st_output_dim 128 \
    --cp_load_path save/2024-11-14_22-35lorenz96_0.7_10_60_8192_EnST_normalized_l2_v2/cp_1000.pth



