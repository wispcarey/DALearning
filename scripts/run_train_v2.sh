#!/bin/bash

cd ..

python train.py \
    --dataset lorenz96 \
    --learning_rate 1e-4 \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --loss_type crps \
    --lr_decay_epochs 500,1000,1500,2000 \
    --epochs 3000 \
    --v EtE \
    --warm_up