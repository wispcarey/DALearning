import numpy as np
import time
import datetime
import os

import torch
import torch.nn as nn

from config.cli import get_parameters

from utils import plot_results_2d, setup_optimizer_and_scheduler, save_checkpoint, load_checkpoint, DATASET_INFO
from utils import L63, L96, rk4, mystery_operator, partial_obs_operator, get_dataloader, batch_covariance, get_mean_std
from EnKF_utils import loc_EnKF_analysis, EnKF_analysis, post_process, mrdiv, mean0
from networks import ComplexAttentionModel, AttentionModel, NaiveNetwork
from localization import pairwise_distances, dist2coeff, create_loc_mat


def test_model(loader, model_list, args, infl=1, H_info=None):
    model, local_model = model_list

    m = args.N
    
    if args.dataset == "lorenz63":
        forward_fun = L63.forward
    elif args.dataset == "lorenz96":
        forward_fun = L96.forward
    else:
        raise NotImplementedError

    if H_info is None:
        H_fun, H = mystery_operator((args.ori_dim, args.obs_dim), args.device)
    else:
        H_fun, H = H_info
        
    # loc_mat_vy = dist2coeff(args.Lvy, radius=4).unsqueeze(0)
    # loc_mat_yy = dist2coeff(args.Lyy, radius=4).unsqueeze(0)

    with torch.no_grad():
        for batch_v in loader:
            batch_v = batch_v.to(device=args.device)

            # Sample from prior
            ens_v_a = batch_v[0].unsqueeze(1).repeat(1, m, 1)
            ens_v_a = ens_v_a + torch.randn_like(ens_v_a, device=args.device) * args.sigma_ens

            # Store everything
            ens_list = [ens_v_a]
            K_list = []
            loc_records = []

            # Iterate over timesteps in batch
            for i in range(len(batch_v) - 1):
                t_start = time.time()
                # get next observation
                obs_y = H_fun(batch_v[i + 1].unsqueeze(1))
                obs_y += args.sigma_y * torch.randn_like(obs_y, device=args.device)

                # forecast step
                ens_v_a = ens_v_a.view(-1, args.ori_dim)
                for j in range(args.dt_iter):
                    ens_v_a = rk4(forward_fun, ens_v_a, i * args.dt + j * args.dt / args.dt_iter,
                                  args.dt / args.dt_iter)
                ens_v_f = ens_v_a.view(-1, m, args.ori_dim)
                
                # add forward noise
                ens_v_f = ens_v_f + torch.randn_like(ens_v_f, device=args.device) * args.sigma_v

                # preparation for individual ensemble data
                hv = H_fun(ens_v_f)
                B, N, D = ens_v_f.shape
                d = hv.shape[2]
                
                # generate a random variable for the observation noise
                r = mean0(args.sigma_y * torch.randn_like(hv, device=args.device))
                
                ens_i = obs_y - hv - r
                
                # for the ensemble dataset
                mean_hv = torch.mean(hv, dim=1, keepdim=True).expand(-1, N, -1)
                mean_ens_v_f = torch.mean(ens_v_f, dim=1, keepdim=True).expand(-1, N, -1)
                Cvv = batch_covariance(ens_v_f).view(-1, 1, D ** 2).expand(-1, N, -1)
                Cvy = batch_covariance(ens_v_f, hv).view(-1, 1, D * d).expand(-1, N, -1)
                Cyy = batch_covariance(hv) + args.sigma_y * torch.eye(d).unsqueeze(0).expand(B, -1, -1).to(args.device)
                Cyy = Cyy.view(-1, 1, d ** 2).expand(-1, N, -1)
                ensemble_info = torch.cat([mean_ens_v_f, mean_hv, Cvv, Cvy, Cyy], dim=-1)
                
                nn_input = torch.cat([ens_v_f, hv, ens_i, ensemble_info], dim=-1).view(-1, args.input_dim)
                local_nn_input = torch.cat([obs_y.squeeze(1), ensemble_info[:, 0, :]], dim=-1)

                # execute model
                nn_output = model(nn_input).view(hv.shape[0], hv.shape[1], -1)
                Vnn1 = ens_v_f + nn_output[:, :, :D]
                Vnn2 = ens_v_f - mean_ens_v_f + nn_output[:, :, D:2 * D]
                Ynn = hv - mean_hv + nn_output[:, :, 2 * D:]
                R = args.sigma_y ** 2 * torch.eye(d).unsqueeze(0).expand(B, -1, -1).to(args.device)
                
                # get localization matrices
                loc_nn_output = torch.sigmoid(local_model(local_nn_input)) * 2
                loc_mat_vy = create_loc_mat(loc_nn_output, args.diff_dist, args.Lvy)
                loc_mat_yy = create_loc_mat(loc_nn_output, args.diff_dist, args.Lyy)
                loc_records.append(loc_nn_output)
                
                # Kalman Gain
                K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn) * loc_mat_vy
                K2 = torch.bmm(Ynn.transpose(1, 2), Ynn) * loc_mat_yy + R * (N - 1)
                K = torch.bmm(K1, torch.inverse(K2))
                ens_v_a = Vnn1 + torch.bmm(ens_i, K.transpose(1, 2))
                
                ens_v_a = post_process(ens_v_a, infl=infl)
                
                ens_v_a = torch.clamp(ens_v_a, min=-20, max=20)

                ens_list.append(ens_v_a)
                K_list.append(K)

            # Concat outputs
            ens_tensor = torch.stack(ens_list)
            K_tensor = torch.stack(K_list)
            loc_tensor = torch.stack(loc_records)

            # Loss functions
            loss_tensor = torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2))

        return loss_tensor, ens_tensor, K_tensor, loc_tensor


def test_SequentialEnKF(loader, args, infl=1, H_info=None, localization=False):
    m = args.N
    
    if args.dataset == "lorenz63":
        forward_fun = L63.forward
    elif args.dataset == "lorenz96":
        forward_fun = L96.forward
    else:
        raise NotImplementedError

    if H_info is None:
        H_fun, H = mystery_operator((args.ori_dim, args.obs_dim), args.device)
    else:
        H_fun, H = H_info
    
    Lvy = dist2coeff(args.Lvy, radius=4)
    Lyy = dist2coeff(args.Lyy, radius=4)

    with torch.no_grad():
        for batch_v in loader:
            batch_v = batch_v.to(device=args.device)

            # Sample from prior
            ens_v_a = batch_v[0].unsqueeze(1).repeat(1, m, 1)
            ens_v_a = ens_v_a + torch.randn_like(ens_v_a, device=args.device) * args.sigma_ens

            # Store everything
            ens_list = [ens_v_a]
            K_list = []

            # Iterate over timesteps in batch
            for i in range(len(batch_v) - 1):
                t_start = time.time()
                # get next observation
                obs_y = H_fun(batch_v[i + 1].unsqueeze(1))
                obs_y += args.sigma_y * torch.randn_like(obs_y, device=args.device)

                # forecast step
                ens_v_a = ens_v_a.view(-1, args.ori_dim)
                for j in range(args.dt_iter):
                    ens_v_a = rk4(forward_fun, ens_v_a, i * args.dt + j * args.dt / args.dt_iter,
                                  args.dt / args.dt_iter)
                ens_v_f = ens_v_a.view(-1, m, args.ori_dim)

                # add forward noise
                ens_v_f = ens_v_f + torch.randn_like(ens_v_f, device=args.device) * args.sigma_v 
                
                ens_yo = H_fun(ens_v_f)
                if localization:
                    ens_v_a, K = loc_EnKF_analysis(ens_v_f, ens_yo, obs_y, args.sigma_y, Lvy, Lyy, a_method="PertObs")
                else:
                    ens_v_a, K = EnKF_analysis(ens_v_f, ens_yo, obs_y, args.sigma_y, a_method="PertObs")
                ens_v_a = post_process(ens_v_a, infl=infl)

                ens_list.append(ens_v_a)
                K_list.append(K)
            # Concat outputs
            ens_tensor = torch.stack(ens_list)
            K_tensor = torch.stack(K_list)
            loc_tensor = torch.unique(torch.cat((Lvy.flatten(), Lyy.flatten())))

            # Loss functions
            loss_tensor = torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2))

        return loss_tensor, ens_tensor, K_tensor, loc_tensor


if __name__ == "__main__":
    args = get_parameters()
    
    # dimension of problem
    args.ori_dim = DATASET_INFO[args.dataset]['dim']
    args.obs_dim = DATASET_INFO[args.dataset]['obs_dim']
    obs_inds = DATASET_INFO[args.dataset]['obs_inds']
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args.input_dim = 2 * args.ori_dim + 3 * args.obs_dim + args.ori_dim * args.obs_dim + args.ori_dim ** 2 + args.obs_dim ** 2 # D + d + d + D + d + D^2 + Dd + d^2
    args.local_input_dim = args.ori_dim + 2 * args.obs_dim + args.ori_dim * args.obs_dim + args.ori_dim ** 2 + args.obs_dim ** 2 # d + D + d + D^2 + Dd + d^2

    if args.cp_load_path == "no":
        raise ValueError("The parameter cp_load_path is invalid.")
    folder_name = os.path.dirname(args.cp_load_path)
    
    if args.seed is not None and args.seed != "None":
        torch.manual_seed(int(args.seed))

    # H_info
    H_info = partial_obs_operator(args.ori_dim, obs_inds, args.device)

    test_loader = get_dataloader(args, test_only=True)
    
    # localization
    full_inds = torch.arange(0, args.ori_dim)
    Lvy = pairwise_distances(full_inds[:, None], obs_inds[:, None], domain=(args.ori_dim,)).to(args.device)
    Lyy = pairwise_distances(obs_inds[:, None], obs_inds[:, None], domain=(args.ori_dim,)).to(args.device)
    args.diff_dist = torch.unique(torch.cat((Lvy.flatten(), Lyy.flatten())))
    args.num_dist = len(args.diff_dist)
    args.Lvy = Lvy
    args.Lyy = Lyy
    
    # print test information
    print(f"Test on {args.test_traj_num} trajectories with the length {args.test_steps}. Observation noise sigma_y={args.sigma_y}.")
    
    # EnKF
    print("Original EnKF with localization")

    loss_tensor_enkf, ens_tensor_enkf, K_tensor_enkf, loc_enkf = test_SequentialEnKF(test_loader, args, infl=1.06, H_info=H_info, localization=True)
    print("Shape of EnKF loss tensor:", loss_tensor_enkf.shape)
    mean_enkf, std_enkf = get_mean_std(loss_tensor_enkf)
    print(f"RMSE: {mean_enkf:.3f} ± {std_enkf:.3f}")

    # set model
    model = AttentionModel(input_dim=args.input_dim, output_dim=args.obs_dim + 2 * args.ori_dim, num_attention_layers=1, hidden_dim=32).to(args.device)
    local_model = AttentionModel(input_dim=args.local_input_dim, output_dim=args.num_dist, num_attention_layers=1, hidden_dim=32).to(args.device)
    model_list = [model, local_model]
    total_params = sum(sum(p.numel() for p in model.parameters()) for model in model_list)
    print(f'Total number of parameters: {total_params}')
    
    # optimizer
    optimizer, scheduler = setup_optimizer_and_scheduler(model_list, args)

    # load checkpoint
    if args.cp_load_path != "no":
        load_checkpoint(model_list, optimizer, scheduler, filename=args.cp_load_path)

    # test
    print("Test NN Results")
    loss_list_nn = []
    loss_tensor_nn, ens_tensor_nn, K_tensor_nn, loc_nn = test_model(test_loader, model_list, args, H_info=H_info)
    print("Shape of NN loss tensor:", loss_tensor_nn.shape)
    mean_nn, std_nn = get_mean_std(loss_tensor_nn)
    print(f"RMSE: {mean_nn:.3f} ± {std_nn:.3f}")


    # save results
    tensor_dict = {
        'enkf':{
            'loss':loss_tensor_enkf,
            'ens':ens_tensor_enkf,
            'k':K_tensor_enkf,
            'loc':loc_enkf,
        },
        'nn':{
            'loss':loss_tensor_nn,
            'ens':ens_tensor_nn,
            'k':K_tensor_nn,
            'loc':loc_nn,
        },
        'cp_load_path': args.cp_load_path,
        'sigma_y': args.sigma_y,
    }
    
    torch.save(tensor_dict, os.path.join(folder_name, f"output_records_{args.N}.pt"))


