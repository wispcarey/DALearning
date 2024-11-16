import numpy as np
import time
import datetime
import os

import torch
import torch.nn as nn

from config.cli import get_parameters
from config.dataset_info import DATASET_INFO

from utils import plot_results_2d, setup_optimizer_and_scheduler, save_checkpoint, load_checkpoint
from utils import L63, L96, rk4, etd_rk4_wrapper
from utils import mystery_operator, partial_obs_operator, get_dataloader, batch_covariance, get_mean_std
from EnKF_utils import loc_EnKF_analysis, EnKF_analysis, post_process, mrdiv, mean0
from networks import ComplexAttentionModel, AttentionModel, NaiveNetwork, Simple_MLP, SetTransformer
from localization import pairwise_distances, dist2coeff, create_loc_mat


def test_model(loader, model_list, args, infl=1, H_info=None):
    model, local_model, st_model1, st_model2 = model_list

    m = args.N
    
    if args.dataset == "lorenz63":
        forward_fun = L63.forward
    elif args.dataset == "lorenz96":
        forward_fun = L96.forward
    elif args.dataset == "ks":
        forward_fun = etd_rk4_wrapper(device = args.device, dt = args.dt / args.dt_iter)
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
                    if args.dataset == 'ks':
                        ens_v_a = forward_fun(ens_v_a, None, args.dt / args.dt_iter)
                    else:
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
                
                # for the ensemble dataset and observations
                mean_hv = torch.mean(hv, dim=1, keepdim=True).expand(-1, N, -1)
                mean_ens_v_f = torch.mean(ens_v_f, dim=1, keepdim=True).expand(-1, N, -1)
                ens_nn_output = st_model1(ens_v_f)
                ens_o_nn_output = st_model2(hv)
                
                # nn_inputs
                nn_input = torch.cat([ens_v_f, hv, ens_i, 
                                    ens_nn_output.unsqueeze(1).expand(-1, N, -1), ens_o_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
                local_nn_input = torch.cat([obs_y.squeeze(1), ens_nn_output, ens_o_nn_output], dim=-1)

                # execute model
                nn_output = model(nn_input).view(hv.shape[0], hv.shape[1], -1)
                Vnn1 = ens_v_f + nn_output[:, :, :D]
                Vnn2 = ens_v_f - mean_ens_v_f + nn_output[:, :, D:2 * D]
                Ynn = hv - mean_hv + nn_output[:, :, 2 * D:]
                R = args.sigma_y ** 2 * torch.eye(d).unsqueeze(0).expand(B, -1, -1).to(args.device)
                
                # get localization matrices
                if args.no_localization:
                    loc_mat_vy = torch.ones(B, D, d, device=args.device)
                    loc_mat_yy = torch.ones(B, d, d, device=args.device)
                else:
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
            if args.no_localization:
                loc_tensor = torch.empty(1)
            else:
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
    elif args.dataset == "ks":
        forward_fun = etd_rk4_wrapper(device = args.device, dt = args.dt / args.dt_iter)
    else:
        raise NotImplementedError

    if H_info is None:
        H_fun, H = mystery_operator((args.ori_dim, args.obs_dim), args.device)
    else:
        H_fun, H = H_info
    
    if args.no_localization:
        Lvy = torch.ones_like(args.Lvy)
        Lyy = torch.ones_like(args.Lyy)
    else:
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
                    if args.dataset == 'ks':
                        ens_v_a = forward_fun(ens_v_a, None, args.dt / args.dt_iter)
                    else:
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
    args.input_dim = args.ori_dim + 2 * args.obs_dim + args.st_output_dim * 2 
    args.local_input_dim = args.obs_dim + args.st_output_dim * 2 
    
    # if args.cp_load_path == "no":
    #     raise ValueError("The parameter cp_load_path is invalid.")
    folder_name = os.path.dirname(args.cp_load_path)
    
    if args.seed is not None and args.seed != "None":
        torch.manual_seed(int(args.seed))

    # H_info
    H_info = partial_obs_operator(args.ori_dim, args.obs_inds, args.device)

    test_loader = get_dataloader(args, test_only=True)
    
    # localization
    full_inds = torch.arange(0, args.ori_dim)
    Lvy = pairwise_distances(full_inds[:, None], args.obs_inds[:, None], domain=(args.ori_dim,)).to(args.device)
    Lyy = pairwise_distances(args.obs_inds[:, None], args.obs_inds[:, None], domain=(args.ori_dim,)).to(args.device)
    args.diff_dist = torch.unique(torch.cat((Lvy.flatten(), Lyy.flatten())))
    args.num_dist = len(args.diff_dist)
    args.Lvy = Lvy
    args.Lyy = Lyy
    
    # print test information
    print(f"Test on {args.test_traj_num} trajectories with the length {args.test_steps}. Observation noise sigma_y={args.sigma_y}.")
    
    # EnKF
    print("Original EnKF with localization")

    loss_tensor_enkf, ens_tensor_enkf, K_tensor_enkf, loc_enkf = test_SequentialEnKF(test_loader, args, infl=1.5, H_info=H_info, localization=True)
    print("Shape of EnKF loss tensor:", loss_tensor_enkf.shape)
    mean_enkf, std_enkf = get_mean_std(torch.mean(loss_tensor_enkf,dim=0))
    print(f"RMSE: {mean_enkf:.3f} ± {std_enkf:.3f}")

    # set model
    model = Simple_MLP(d_input=args.input_dim, d_output=args.obs_dim + 2 * args.ori_dim, num_hidden_layers=2).to(args.device)
    if args.no_localization:
        local_model = NaiveNetwork(1)
    else:
        local_model = Simple_MLP(d_input=args.local_input_dim, d_output=args.num_dist, num_hidden_layers=2).to(args.device)
    st_model1 = SetTransformer(input_dim=args.ori_dim, num_heads=8, num_inds=16, output_dim=args.st_output_dim, hidden_dim=args.hidden_dim, num_layers=1).to(args.device)
    st_model2 = SetTransformer(input_dim=args.obs_dim, num_heads=8, num_inds=16, output_dim=args.st_output_dim, hidden_dim=args.hidden_dim, num_layers=1).to(args.device)
    model_list = [model, local_model, st_model1, st_model2]
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
    mean_nn, std_nn = get_mean_std(torch.mean(loss_tensor_nn,dim=0))
    print(f"RMSE: {mean_nn:.3f} ± {std_nn:.3f}")


    # save results
    tensor_dict = {
        # 'enkf':{
        #     'loss':loss_tensor_enkf,
        #     'ens':ens_tensor_enkf,
        #     'k':K_tensor_enkf,
        #     'loc':loc_enkf,
        # },
        'nn':{
            'loss':loss_tensor_nn,
            'ens':ens_tensor_nn,
            'k':K_tensor_nn,
            'loc':loc_nn,
        },
        'cp_load_path': args.cp_load_path,
        'sigma_y': args.sigma_y,
    }
    
    if args.cp_load_path != "no":
        torch.save(tensor_dict, os.path.join(folder_name, f"output_records_{args.N}.pt"))


