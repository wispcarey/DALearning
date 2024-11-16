import torch

DATASET_INFO = {
    'lorenz63': {
        'dim': 3,
        'obs_dim': 2,
        'obs_inds': torch.tensor([0, 1]),
        'dt': 0.15,
        'dt_iter': 5,
        'test_steps': 1500,
        'test_traj_num': 100,
        'hidden_dim': 32,
    },
    'lorenz96': {
        'dim': 40,
        'obs_dim': 10,
        'obs_inds': torch.arange(0, 40, 4),
        'dt': 0.15,
        'dt_iter': 5,
        'test_steps': 1500,
        'test_traj_num': 100,
        'hidden_dim': 64,
    },
    'ks': {
        'dim': 128,
        'obs_dim': 32,
        'obs_inds': torch.arange(0, 128, 4),
        'dt': 1,
        'dt_iter': 4,
        'test_steps': 2000,
        'test_traj_num': 100,
        'hidden_dim': 64,
    }
}