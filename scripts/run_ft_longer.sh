#!/bin/bash

cd ..

# # Lorenz 96
python finetune.py \
    --epochs 50 \
    --save_epoch 5 \
    --dataset lorenz96 \
    --train_steps 60 \
    --train_traj_num 8192 \
    --sigma_y 1 \
    --seed 42 \
    --learning_rate 1e-4 \
    --cp_load_path save/2025-04-10_09-31lorenz96_1.0_10_60_8192_norm_EnST_joint/cp_1000.pth
