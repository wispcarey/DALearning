# DALearning

This repository provides a collection of methods that enhance data assimilation using various machine learning techniques. Most of the approaches are based on the Set Transformer architecture.

## Quick Start Examples

To quickly start training a model, refer to the script: `scripts/run_train.sh`.

For example, if you want to train a model on the Lorenz 96 system with an ensemble size of N=10 and observation noise σ<sub>y</sub>=1, you can use the following command:

```bash
python train.py --dataset lorenz96 --N 10 --sigma_y 1
```

To fine-tune a pretrained model on a different ensemble size, refer to the script: `scripts/run_fine_tuning.sh`.

For example, if you want to fine-tune a model on the Lorenz 96 system with an observation noise σ<sub>y</sub>=1, based on a pretrained checkpoint, you can use the following command:

```bash
python finetune.py --epochs 20 --save_epoch 20 --dataset lorenz96 --sigma_y 1 --learning_rate 1e-4 --cp_load_path PATH_TO_YOUR_CHECKPOINT
```

## Project Structure

The main scripts for training, fine-tuning, and evaluation are:

- `train.py`: used for training models from scratch.
- `finetune.py`: used for fine-tuning pre-trained models.
- `evaluate.py`: used for evaluating trained models.

Utility functions are mainly located in `utils.py`. This file includes:

- Definitions for different dynamical systems,
- Data generation and preprocessing routines,
- Optimizer setup and other helper functions.

Core training-related functions are implemented in `training_utils.py`, including:

- Functions for training and testing models,
- Model setup routines.

The file `loss.py` defines various loss functions used in the training process.

Different neural network architectures are defined in `networks.py`.

Training configurations are organized in `config/cli`, where all training parameters are specified.

Model-specific configurations for the three dynamical systems — Lorenz63, Lorenz96, and Kuramoto–Sivashinsky (KS) — are provided in `config/dataset_info`.

### Remark: Data Generation

The function `gen_data` in `utils.py` is used to generate trajectories for different dynamical systems. 

Note that generating data for the first time can be time-consuming, especially for complex systems. However, once the data is generated, it will be automatically saved in the `data` directory. 

If the same parameters are used in future runs, the data will be loaded directly from the saved files, avoiding redundant computation.
