#!/bin/bash

cd ..

python train.py \
    --dataset lorenz96 \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --loss_type nes \
    --es_p 2



