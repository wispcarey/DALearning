#!/bin/bash

cd ..

python learn_enscorrection_set_transformer_local_v2.py \
    --dataset lorenz63 \
    --N 10 \
    --train_steps 60 \
    --train_traj_num 8192 \
    --sigma_y 1 \
    --batch_size 512 \
    --seed 42 \
    --learning_rate 1e-4 \
    --loss_type normalized_l2 \
    --adjust_lr \
    --st_output_dim 128 \
    --no_localization

python learn_enscorrection_set_transformer_local_v2.py \
    --dataset lorenz63 \
    --N 10 \
    --train_steps 60 \
    --train_traj_num 8192 \
    --sigma_y 0.7 \
    --batch_size 512 \
    --seed 42 \
    --learning_rate 1e-4 \
    --loss_type normalized_l2 \
    --adjust_lr \
    --st_output_dim 128 \
    --no_localization

# python learn_enscorrection_set_transformer_local_v2.py \
#     --dataset ks \
#     --N 10 \
#     --train_steps 60 \
#     --train_traj_num 8192 \
#     --sigma_y 1 \
#     --batch_size 256 \
#     --seed 42 \
#     --learning_rate 1e-3 \
#     --loss_type normalized_l2 \
#     --adjust_lr \
#     --st_output_dim 128

# python learn_enscorrection_set_transformer_local_v2.py \
#     --dataset ks \
#     --N 10 \
#     --train_steps 60 \
#     --train_traj_num 8192 \
#     --sigma_y 0.7 \
#     --batch_size 256 \
#     --seed 42 \
#     --learning_rate 1e-3 \
#     --loss_type normalized_l2 \
#     --adjust_lr \
#     --st_output_dim 128


