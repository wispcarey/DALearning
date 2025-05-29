#!/bin/bash

cd ..

# # Lorenz 96
python finetune.py \
    --epochs 20 \
    --save_epoch 20 \
    --dataset lorenz96 \
    --sigma_y 1 \
    --seed 42 \
    --learning_rate 1e-4 \
    --cp_load_path save/2025-04-10_09-31lorenz96_1.0_10_60_8192_norm_EnST_joint/cp_1000.pth

python finetune.py \
    --epochs 20 \
    --save_epoch 20 \
    --dataset lorenz96 \
    --sigma_y 0.7 \
    --seed 42 \
    --learning_rate 1e-4 \
    --cp_load_path save/2025-04-10_13-18lorenz96_0.7_10_60_8192_norm_EnST_joint/cp_1000.pth

# # KS
python finetune.py \
    --epochs 20 \
    --save_epoch 20 \
    --dataset ks \
    --sigma_y 1 \
    --seed 42 \
    --learning_rate 5e-5 \
    --cp_load_path save/2025-04-09_18-18ks_1.0_10_60_8192_norm_EnST_joint/cp_1000.pth

python finetune.py \
    --epochs 20 \
    --save_epoch 20 \
    --dataset ks \
    --sigma_y 0.7 \
    --seed 42 \
    --learning_rate 5e-5 \
    --cp_load_path save/2025-04-10_01-52ks_0.7_10_60_8192_norm_EnST_joint/cp_1000.pth

# # Lorenz 63
python finetune.py \
    --epochs 20 \
    --save_epoch 20 \
    --dataset lorenz63 \
    --sigma_y 1 \
    --seed 42 \
    --learning_rate 1e-4 \
    --cp_load_path save/2025-04-10_17-18lorenz63_1.0_10_60_8192_norm_EnST_joint/cp_1000.pth \
    --no_localization 

python finetune.py \
    --epochs 20 \
    --save_epoch 20 \
    --dataset lorenz63 \
    --sigma_y 0.7 \
    --seed 42 \
    --learning_rate 1e-4 \
    --cp_load_path save/2025-04-10_19-40lorenz63_0.7_10_60_8192_norm_EnST_joint/cp_1000.pth \
    --no_localization 


