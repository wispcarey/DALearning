# DALearning

There are two models provided.

1. **EnGA**: The key idea here is to approximate the ensemble dataset and observation set using Gaussian Approximation, replacing these two sets with their mean and covariance as input.

2. **EnST**: This approach employs a set transformer to process both the ensemble dataset and the observation set.

## EnGA
To train our EnGA model based on an Ensemble Size of 10 and the Lorenz 96 model:

```bash
python learn_full_dependences_local.py \
    --N 10 \
    --train_steps 60 \
    --train_traj_num 8192 \
    --sigma_y 1 \
    --batch_size 1024 \
    --seed 42 \
    --adjust_lr
``` 

To test the trained EnGA model (You can use your own checkpoint):

```bash
python test_full_dependences_local.py \
    --N 10 \
    --test_steps 1500 \
    --test_traj_num 100 \
    --sigma_y 1 \
    --seed 42 \
    --cp_load_path save/2024-10-09_22-04lorenz96_1.0_10_60_8192_EnLoc/cp_1000.pth
```

## EnST
To train our EnST model based on an Ensemble Size of 10 and the Lorenz 96 model:

```bash
python learn_enscorrection_set_transformer_local.py \
    --N 10 \
    --train_steps 60 \
    --train_traj_num 8192 \
    --sigma_y 2.5 \
    --batch_size 1024 \
    --seed 42 \
    --adjust_lr
``` 

To test the trained EnST model (You can use your own checkpoint):

```bash
python test_enscorrection_set_transformer_local.py \
    --N 10 \
    --test_steps 1500 \
    --test_traj_num 100 \
    --sigma_y 1 \
    --seed 42 \
    --cp_load_path save/2024-10-09_18-16lorenz96_1.0_10_60_8192_EnST/cp_1000.pth
```