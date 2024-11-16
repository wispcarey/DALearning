#!/bin/bash

cd ..

python learn_enscorrection_set_transformer_local.py \
    --N 10 \
    --train_steps 60 \
    --train_traj_num 8192 \
    --sigma_y 2.5 \
    --batch_size 512 \
    --seed 42 \
    --adjust_lr

python learn_enscorrection_set_transformer_local.py \
    --N 10 \
    --train_steps 60 \
    --train_traj_num 8192 \
    --sigma_y 1 \
    --batch_size 512 \
    --seed 42 \
    --adjust_lr

python learn_full_dependences_local.py \
    --N 10 \
    --train_steps 60 \
    --train_traj_num 8192 \
    --sigma_y 2.5 \
    --batch_size 512 \
    --seed 42 \
    --adjust_lr

python learn_full_dependences_local.py \
    --N 10 \
    --train_steps 60 \
    --train_traj_num 8192 \
    --sigma_y 1 \
    --batch_size 512 \
    --seed 42 \
    --adjust_lr
