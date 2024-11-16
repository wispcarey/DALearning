#!/bin/bash

cd ..

python learn_enscorrection_set_transformer_local.py \
    --dataset ks \
    --N 10 \
    --train_steps 60 \
    --train_traj_num 8192 \
    --sigma_y 1 \
    --batch_size 256 \
    --seed 42 \
    --learning_rate 1e-4 \
    --loss_type normalized_l2 \
    --adjust_lr

python learn_enscorrection_set_transformer_local.py \
    --dataset ks \
    --N 10 \
    --train_steps 60 \
    --train_traj_num 8192 \
    --sigma_y 0.7 \
    --batch_size 256 \
    --seed 42 \
    --learning_rate 1e-4 \
    --loss_type normalized_l2 \
    --adjust_lr


