#!/bin/bash

cd ..

python learn_enscorrection_set_transformer_local.py \
    --dataset lorenz63 \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --no_localization \
    --adjust_lr

python learn_enscorrection_set_transformer_local.py \
    --dataset lorenz63 \
    --N 10 \
    --sigma_y 0.7 \
    --seed 42 \
    --no_localization \
    --adjust_lr

python learn_enscorrection_set_transformer_local.py \
    --dataset lorenz96 \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --adjust_lr

python learn_enscorrection_set_transformer_local.py \
    --dataset lorenz96 \
    --N 10 \
    --sigma_y 0.7 \
    --seed 42 \
    --adjust_lr

python learn_enscorrection_set_transformer_local.py \
    --dataset ks \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --adjust_lr

python learn_enscorrection_set_transformer_local.py \
    --dataset ks \
    --N 10 \
    --sigma_y 0.7 \
    --seed 42 \
    --adjust_lr


