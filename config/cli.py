import argparse
import math
import torch

from .dataset_info import DATASET_INFO

def float_or_random(value):
    try:
        return float(value)
    except ValueError:
        if value == "random":
            return value
        raise argparse.ArgumentTypeError(f"Invalid value for dt: {value}")

def float_or_str(value):
    try:
        return float(value)
    except ValueError:
        return value

def float_or_default(value):
    try:
        return float(value)
    except ValueError:
        return "default"
    
def int_or_default(value):
    try:
        return int(value)
    except ValueError:
        return "default"

def get_parameters():
    parser = argparse.ArgumentParser()
    # dataset setting
    parser.add_argument('--dataset', type=str, default='lorenz96', choices=['lorenz63', 'lorenz96', 'ks'],
                        help='Dataset name')
    parser.add_argument('--num_loader_workers', type=int, default=16,
                        help='number of workers for the data loader')
    parser.add_argument('--N', type=int, default=20,
                        help='Number of ensemble sample')
    parser.add_argument('--train_steps', type=int_or_default, default="default",
                        help='Training time steps')
    parser.add_argument('--train_traj_num', type=int_or_default, default="default",
                        help='Number of training trajectories')
    parser.add_argument('--test_steps', type=int_or_default, default="default",
                        help='Testing time steps')
    parser.add_argument('--test_traj_num', type=int_or_default, default="default",
                        help='Number of testing trajectories')
    parser.add_argument('--valid_steps', type=int, default=0,
                        help='Validation time steps')
    parser.add_argument('--burn_steps', type=int, default=1000, help='Test time steps')
    parser.add_argument('--dt', type=float_or_default, default="default", help='Time stepsize')
    parser.add_argument('--dt_iter', type=int_or_default, default="default",
                        help='Separate dt into multiple steps of rk4 forward')
    parser.add_argument('--overlap_rate', type=float, default=0.75,
                        help='Overlap rate when segmenting learning trajectories')
    parser.add_argument('--new_data', action='store_false',
                        help='do not use the check_dist option when generating training data')
    parser.add_argument('--sigma_ens', type=float_or_default, default="default",
                        help='std of initial ensemble samples')
    parser.add_argument('--sigma_v', type=float, default=0.0,
                        help='noise std in the latent process')
    parser.add_argument('--sigma_y', type=float, default=1,
                        help='noise std in the observation')
    
    # loss function
    parser.add_argument('--ignore_first', type=int, default=0,
                        help='number of first steps ignored in calculating the loss in each training traj')
    parser.add_argument('--loss_warm_up', action='store_true',
                        help='warm-up the loss according to epochs')

    # training setting
    parser.add_argument('--cp_load_path', type=str, default="no",
                        help='checkpoint load path. "no" for training from scratch')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='training epochs')
    parser.add_argument('--hidden_dim', type=int_or_default, default="default",
                        help='Hidden dimension of the NN')
    parser.add_argument('--batch_size', type=int_or_default, default="default",
                        help='training batch size')
    parser.add_argument('--test_batch_size', type=int_or_default, default="default",
                        help='test batch size')
    parser.add_argument('--detach_training_epoch', type=int, default=10000,
                        help='detach training epochs')
    parser.add_argument('--loss_type', type=str, default="normalized_l2", choices=["l2", "normalized_l2", "rmse", "crps"],
                        help='the type of loss function')
    parser.add_argument('--no_localization', action='store_true',
                        help='do not apply localization')
    parser.add_argument('--st_output_dim', type=int, default=64,
                        help='dimension of set transformer output')
    parser.add_argument('--st_num_seeds', type=int, default=16,
                        help='number of seeds in PMA layer')
    parser.add_argument('--st_type', type=str, default='joint', choices=['state_only', 'separate', 'joint'],
        help=(
            'Specifies how the Set Transformer is applied to distributions. '
            '"state_only" is used to process only the state distribution, '
            '"separate" processes the state distribution and observation distribution independently, '
            'and "joint" is used to process the joint distribution of state and observation.'
        )
    )
    parser.add_argument('--unfreeze_WQ', action='store_true',
                        help='unfreeze Q weights in the PMA layer')
    parser.add_argument('--loc_max_val', type=float, default=2,
                        help='max value of the localization weights')


    

    # output setting
    parser.add_argument('--print_batch', type=float_or_str, default="auto",
                        help='Number of batchs to print out the results')
    parser.add_argument('--save_epoch', type=int, default=25,
                        help='Number of epoch between save and test')
    parser.add_argument('--trail', type=str, default="",
                        help='trail name')
    parser.add_argument('--zero_infl', action='store_true',
                        help='set inflation to zero')
    

    # optimization setting
    parser.add_argument('--learning_rate', type=float_or_default, default='default',
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='200,300,400',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--adjust_lr', action='store_true',
                        help='adjusting learning rate')
    parser.add_argument('--warm_up', action='store_true',
                        help='adjusting learning rate')
    parser.add_argument('--warm_up_rate', type=float, default=1.02,
                        help='warm-up rate')
    parser.add_argument('--warm_up_epochs', type=int, default=50,
                        help='number of epochs in the beginning to warm up the learning rate')
    parser.add_argument('--SGD', action='store_true',
                        help='use SGD optimizer, otherwise use Adam')

    # others
    parser.add_argument('--seed', type=str, default=None, help='Random Seed')
    parser.add_argument('--test_only', action='store_true', help='Only do the test part')
    parser.add_argument('--test_rounds', type=int, default=1, help='Number of test rounds when selecting --test_only')
    parser.add_argument('--GPU_memory', type=int, default=16, help='GPU memory in GB')

    args = parser.parse_args()

    # iterations = args.lr_decay_epochs.split(',')
    # args.lr_decay_epochs = list([])
    # for it in iterations:
    #     args.lr_decay_epochs.append(int(it))

    if not args.warm_up:
        args.warm_up_rate = 1
        args.warm_up_epochs = 1

    if not args.adjust_lr:
        args.lr_decay_epochs = "0"
        args.lr_decay_rate = 1
        
    if args.dt == 'default':
        args.dt = DATASET_INFO[args.dataset]['dt']
    
    if args.dt_iter == 'default':
        args.dt_iter = DATASET_INFO[args.dataset]['dt_iter']
    
    if args.test_steps == 'default':
        args.test_steps = DATASET_INFO[args.dataset]['test_steps']
    
    if args.test_traj_num == 'default':
        args.test_traj_num = DATASET_INFO[args.dataset]['test_traj_num']
    
    if args.train_steps == 'default':
        args.train_steps = DATASET_INFO[args.dataset]['train_steps']
    
    if args.train_traj_num == 'default':
        args.train_traj_num = DATASET_INFO[args.dataset]['train_traj_num']
    
    if args.hidden_dim == 'default':
        args.hidden_dim = DATASET_INFO[args.dataset]['hidden_dim']
    
    if args.learning_rate == 'default':
        args.learning_rate = DATASET_INFO[args.dataset]['learning_rate']
    
    if args.batch_size == 'default':
        args.batch_size = DATASET_INFO[args.dataset]['batch_size']
        
    if args.test_batch_size == 'default':
        args.test_batch_size = args.test_traj_num
        
    if args.sigma_ens == 'default':
        args.sigma_ens = DATASET_INFO[args.dataset]['sigma_ens']
    
    args.ori_dim = DATASET_INFO[args.dataset]['dim']
    args.obs_dim = DATASET_INFO[args.dataset]['obs_dim']
    args.obs_inds = DATASET_INFO[args.dataset]['obs_inds']
    args.clamp = DATASET_INFO[args.dataset]['clamp']
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Detected {torch.cuda.device_count()} GPUs")
    if num_gpus > 1:
        args.use_data_parallel = True
        args.batch_size = args.batch_size * num_gpus
        print(f"Use DataParallel. Adjusted Batch Size: {args.batch_size}")
    else:
        args.use_data_parallel = False
        print(f"Do not Use DataParallel")
        
    if args.GPU_memory != 16:
        args.batch_size = args.batch_size * int(args.GPU_memory / 16)
    
    if args.print_batch == "auto":
        args.print_batch = math.ceil(args.train_traj_num / args.batch_size)
    

    return args