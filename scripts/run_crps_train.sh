#!/bin/bash

cd ..

# python train.py \
#     --dataset ks \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 \
#     --loss_type crps

# python train.py \
#     --dataset ks \
#     --N 10 \
#     --sigma_y 0.7 \
#     --seed 42 \
#     --loss_type crps

python train.py \
    --dataset lorenz96 \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --loss_type crps \
    --crps_p 1.5

# python train.py \
#     --dataset lorenz96 \
#     --N 10 \
#     --sigma_y 0.7 \
#     --seed 42 \
#     --loss_type crps

# python train.py \
#     --dataset lorenz63 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 1 \
#     --no_localization 

# python train.py \
#     --dataset lorenz63 \
#     --N 10 \
#     --sigma_y 0.7 \
#     --seed 1 \
#     --no_localization 

# python train.py \
#     --dataset lorenz96 \
#     --epochs 500 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 \
#     --st_num_seeds 16 \
#     --adjust_lr

# python train.py \
#     --dataset lorenz96 \
#     --epochs 500 \
#     --N 20 \
#     --sigma_y 1 \
#     --seed 42 \
#     --st_num_seeds 1 \
#     --adjust_lr

# python train.py \
#     --dataset lorenz96 \
#     --epochs 500 \
#     --N 20 \
#     --sigma_y 1 \
#     --seed 42 \
#     --st_num_seeds 8 \
#     --adjust_lr

# python train.py \
#     --dataset lorenz96 \
#     --epochs 500 \
#     --N 20 \
#     --sigma_y 1 \
#     --seed 42 \
#     --st_num_seeds 16 \
#     --adjust_lr


# python train.py \
#     --dataset lorenz96 \
#     --N 10 \
#     --sigma_y 0.7 \
#     --seed 42 \
#     --adjust_lr


