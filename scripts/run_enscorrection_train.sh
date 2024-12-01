#!/bin/bash

cd ..

# python learn_enscorrection_set_transformer_local.py \
#     --epochs 50 \
#     --dataset lorenz96 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 \
#     --adjust_lr

# python learn_enscorrection_set_transformer_local.py \
#     --epochs 50 \
#     --dataset lorenz96 \
#     --N 10 \
#     --sigma_y 1 \
#     --seed 42 \
#     --adjust_lr \
#     --unfreeze_WQ


python learn_enscorrection_set_transformer_local.py \
    --dataset lorenz63 \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --no_localization \
    --num_loader_workers 1 \
    --st_type separate \
    --adjust_lr

python learn_enscorrection_set_transformer_local.py \
    --dataset lorenz63 \
    --N 10 \
    --sigma_y 0.7 \
    --seed 42 \
    --no_localization \
    --num_loader_workers 1 \
    --st_type separate \
    --adjust_lr

python learn_enscorrection_set_transformer_local.py \
    --dataset lorenz96 \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --num_loader_workers 1 \
    --st_type separate \
    --adjust_lr

python learn_enscorrection_set_transformer_local.py \
    --dataset lorenz96 \
    --N 10 \
    --sigma_y 0.7 \
    --seed 42 \
    --num_loader_workers 1 \
    --st_type separate \
    --adjust_lr

python learn_enscorrection_set_transformer_local.py \
    --dataset ks \
    --N 10 \
    --sigma_y 1 \
    --seed 42 \
    --num_loader_workers 1 \
    --st_type separate \
    --adjust_lr

python learn_enscorrection_set_transformer_local.py \
    --dataset ks \
    --N 10 \
    --sigma_y 0.7 \
    --seed 42 \
    --num_loader_workers 1 \
    --st_type separate \
    --adjust_lr


