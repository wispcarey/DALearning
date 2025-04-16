#!/bin/bash

cd ..

python train.py \
    --dataset lorenz96 \
    --learning_rate 1e-4 \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --loss_type kes \
    --lr_decay_epochs 400,800,1200,1600,2000,2400 \
    --lr_decay_rate 0.7 \
    --epochs 3000 \
    --v EtE \
    --warm_up

python train.py \
    --dataset lorenz96 \
    --learning_rate 1e-4 \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --loss_type nkes \
    --lr_decay_epochs 400,800,1200,1600,2000,2400 \
    --lr_decay_rate 0.7 \
    --epochs 3000 \
    --v EtE \
    --warm_up

python train.py \
    --dataset lorenz96 \
    --learning_rate 1e-4 \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --loss_type tnkes \
    --lr_decay_epochs 400,800,1200,1600,2000,2400 \
    --lr_decay_rate 0.7 \
    --epochs 3000 \
    --v EtE \
    --warm_up