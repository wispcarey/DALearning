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
from utils import redirect_output
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
        for batch_ind, batch_v in enumerate(loader):
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
                
                # ens_v_a, K = EnKF_analysis(ens_v_f, hv, obs_y, args.sigma_y, a_method="PertObs")
                B, N, D = ens_v_f.shape
                d = hv.shape[2]
                
                # generate a random variable for the observation noise
                r = mean0(args.sigma_y * torch.randn_like(hv, device=args.device))
                
                ens_i = obs_y - hv - r
                
                # for the ensemble dataset and observations
                mean_hv = torch.mean(hv, dim=1, keepdim=True).expand(-1, N, -1)
                mean_ens_v_f = torch.mean(ens_v_f, dim=1, keepdim=True).expand(-1, N, -1)
                
                if args.st_type == 'state_only':
                    ens_nn_output = st_model1(ens_v_f)
                    nn_input = torch.cat([ens_v_f, hv, ens_i, 
                                        ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
                    local_nn_input = torch.cat([obs_y.squeeze(1), ens_nn_output], dim=-1)
                elif args.st_type == 'separate':
                    ens_nn_output = st_model1(ens_v_f)
                    ens_o_nn_output = st_model2(hv)                
                    nn_input = torch.cat([ens_v_f, hv, ens_i, 
                                        ens_nn_output.unsqueeze(1).expand(-1, N, -1), ens_o_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
                    local_nn_input = torch.cat([obs_y.squeeze(1), ens_nn_output, ens_o_nn_output], dim=-1)
                elif args.st_type == 'joint':
                    ens_nn_output = st_model1(torch.cat([ens_v_f, hv], dim=-1))
                    nn_input = torch.cat([ens_v_f, hv, ens_i, 
                                        ens_nn_output.unsqueeze(1).expand(-1, N, -1)], dim=-1).view(-1, args.input_dim)
                    local_nn_input = torch.cat([obs_y.squeeze(1), ens_nn_output], dim=-1)
                    
                # execute model
                nn_output = model(nn_input).view(hv.shape[0], hv.shape[1], -1)
                if args.zero_infl:
                    Vnn1 = ens_v_f
                else:
                    Vnn1 = ens_v_f + nn_output[:, :, :D]
                
                # Vnn1 = ens_v_f
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
                
                ens_v_a = torch.clamp(ens_v_a, min=-args.clamp, max=args.clamp)

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
            # absolute rmse
            rmse_tensor = torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2)) 
            # relative rmse
            # rmse_tensor = torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2)) / torch.sqrt(torch.mean((batch_v) ** 2, dim=2))
            rmv_tensor = torch.sqrt(N / (N-1) * torch.mean((ens_tensor - batch_v.unsqueeze(2)) ** 2, dim=(2,3)))
            
            if batch_ind == 0:
                rmse_tensor_all, rmv_tensor_all = rmse_tensor, rmv_tensor
            else:
                # Concatenate tensors along the first dimension
                rmse_tensor_all = torch.cat((rmse_tensor_all, rmse_tensor), dim=0)
                rmv_tensor_all = torch.cat((rmv_tensor_all, rmv_tensor), dim=0)
        
        # non-nan trajs
        nan_mask = torch.isnan(rmse_tensor_all).any(dim=0)  
        valid_B_mask = ~nan_mask
            
        mean_rmse, std_rmse = get_mean_std(torch.mean(rmse_tensor_all[:, valid_B_mask], dim=0))
        mean_rmv, std_rmv = get_mean_std(torch.mean(rmv_tensor_all[:, valid_B_mask], dim=0))
        
        no_nan_percent = torch.sum(valid_B_mask) / args.test_traj_num

    return mean_rmse, std_rmse, mean_rmv, std_rmv, no_nan_percent, loc_tensor


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
        for batch_ind, batch_v in enumerate(loader):
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
                
                B, N, D = ens_v_f.shape

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
            rmse_tensor = torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2))
            rmv_tensor = torch.sqrt(N / (N-1) * torch.mean((ens_tensor - batch_v.unsqueeze(2)) ** 2, dim=(2,3)))
        
            if batch_ind == 0:
                rmse_tensor_all, rmv_tensor_all = rmse_tensor, rmv_tensor
            else:
                # Concatenate tensors along the first dimension
                rmse_tensor_all = torch.cat((rmse_tensor_all, rmse_tensor), dim=0)
                rmv_tensor_all = torch.cat((rmv_tensor_all, rmv_tensor), dim=0)
            
        # non-nan trajs
        nan_mask = torch.isnan(rmse_tensor_all).any(dim=0)  
        valid_B_mask = ~nan_mask
            
        mean_rmse, std_rmse = get_mean_std(torch.mean(rmse_tensor_all[:, valid_B_mask], dim=0))
        mean_rmv, std_rmv = get_mean_std(torch.mean(rmv_tensor_all[:, valid_B_mask], dim=0))
        
        no_nan_percent = torch.sum(valid_B_mask) / args.test_traj_num

    return mean_rmse, std_rmse, mean_rmv, std_rmv, no_nan_percent


if __name__ == "__main__":
    args = get_parameters()
    if args.st_type == 'state_only':
        print("Only apply an ST on the ensemble state data.")
        args.input_dim = args.ori_dim + 2 * args.obs_dim + args.st_output_dim
        args.local_input_dim = args.obs_dim + args.st_output_dim
    elif args.st_type == "separate": 
        print("Apply STs separately on the ensemble state data and observation data.")
        args.input_dim = args.ori_dim + 2 * args.obs_dim + args.st_output_dim * 2 
        args.local_input_dim = args.obs_dim + args.st_output_dim * 2
    elif args.st_type == 'joint':
        print("Apply an ST on the joint distribution of ensemble state data and observation data.")
        args.input_dim = args.ori_dim + 2 * args.obs_dim + args.st_output_dim * 2 
        args.local_input_dim = args.obs_dim + args.st_output_dim * 2
    else:
        raise ValueError("Please use a valid st_type.")
    
    # if args.cp_load_path == "no":
    #     raise ValueError("The parameter cp_load_path is invalid.")
    folder_name = os.path.dirname(args.cp_load_path)
    
    # redirect output
    with redirect_output(folder_name, filename="test_output.txt"):
        if args.seed is not None and args.seed != "None":
            torch.manual_seed(int(args.seed))

        # H_info
        H_info = partial_obs_operator(args.ori_dim, args.obs_inds, args.device)

        # modify test_batch_size
        if args.N == 100:
            args.test_batch_size = args.test_batch_size // 2
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
        print(f"Test on {args.test_traj_num} trajectories with the length {args.test_steps} and ensemble size {args.N}. Observation noise sigma_y={args.sigma_y}.")
        
        # EnKF
        # print("Original EnKF with localization")

        # mean_rmse_enkf, std_rmse_enkf, mean_rmv_enkf, std_rmv_enkf, no_nan_percent_enkf = test_SequentialEnKF(test_loader, args, infl=1.09, H_info=H_info, localization=False)
        # print(f"RMSE: {mean_rmse_enkf:.3f} ± {std_rmse_enkf:.3f}")
        # print(f"RMV: {mean_rmv_enkf:.3f} ± {std_rmv_enkf:.3f}")
        # print(f'No NAN Percentage: {no_nan_percent_enkf * 100: .2f}%')

        # set model
        model = Simple_MLP(d_input=args.input_dim, d_output=args.obs_dim + 2 * args.ori_dim, num_hidden_layers=2).to(args.device)
        if args.no_localization:
            local_model = NaiveNetwork(1)
        else:
            local_model = Simple_MLP(d_input=args.local_input_dim, d_output=args.num_dist, num_hidden_layers=2).to(args.device)
        if args.st_type == 'separate':
            st_model1 = SetTransformer(input_dim=args.ori_dim, num_heads=8, num_inds=args.st_num_seeds, output_dim=args.st_output_dim, 
                                        hidden_dim=args.hidden_dim, num_layers=1, freeze_WQ=not args.unfreeze_WQ).to(args.device)
            st_model2 = SetTransformer(input_dim=args.obs_dim, num_heads=8, num_inds=args.st_num_seeds, output_dim=args.st_output_dim, 
                                        hidden_dim=args.hidden_dim, num_layers=1, freeze_WQ=not args.unfreeze_WQ).to(args.device)
        elif args.st_type == 'state_only':
            st_model1 = SetTransformer(input_dim=args.ori_dim, num_heads=8, num_inds=args.st_num_seeds, output_dim=args.st_output_dim, 
                                        hidden_dim=args.hidden_dim, num_layers=2, freeze_WQ=not args.unfreeze_WQ).to(args.device)
            st_model2 = NaiveNetwork(1)
        elif args.st_type == 'joint':
            st_model1 = SetTransformer(input_dim=args.ori_dim + args.obs_dim, num_heads=8, num_inds=args.st_num_seeds, output_dim=args.st_output_dim * 2, 
                                        hidden_dim=args.hidden_dim, num_layers=2, freeze_WQ=not args.unfreeze_WQ).to(args.device)
            st_model2 = NaiveNetwork(1)
        if args.use_data_parallel:
            model, local_model, st_model1, st_model2 = nn.DataParallel(model), nn.DataParallel(local_model), nn.DataParallel(st_model1), nn.DataParallel(st_model2)
        model_list = [model, local_model, st_model1, st_model2]
        total_params = sum(sum(p.numel() for p in model.parameters()) for model in model_list)
        print(f'Total number of parameters: {total_params}')

        
        # optimizer
        optimizer, scheduler = setup_optimizer_and_scheduler(model_list, args)

        # load checkpoint
        if args.cp_load_path != "no":
            load_checkpoint(model_list, None, None, filename=args.cp_load_path, use_data_parallel=args.use_data_parallel)

        # test
        print("Test NN Results")
        loss_list_nn = []
        mean_rmse_nn, std_rmse_nn, mean_rmv_nn, std_rmv_nn, no_nan_percent_nn, loc_tensor = test_model(test_loader, model_list, args, H_info=H_info)
        print(f"RMSE: {mean_rmse_nn:.3f} ± {std_rmse_nn:.3f}")
        print(f"RMV: {mean_rmv_nn:.3f} ± {std_rmv_nn:.3f}")
        print(f'No NAN Percentage: {no_nan_percent_nn * 100: .2f}%')

        loc_mean = torch.mean(loc_tensor, dim=(0,1)) 
        loc_std = torch.std(loc_tensor, dim=(0,1)) 
        # save results
        tensor_dict = {
            # 'enkf':{
            #     'mean_rmse':mean_rmse_enkf,
            #     'std_rmse':std_rmse_enkf,
            #     'mean_rmv':mean_rmv_enkf,
            #     'std_rmv':std_rmv_enkf,
            #     'valid_percent':no_nan_percent_enkf,
            # },
            'nn':{
                'mean_rmse':mean_rmse_nn,
                'std_rmse':std_rmse_nn,
                'mean_rmv':mean_rmv_nn,
                'std_rmv':std_rmv_nn,
                'valid_percent':no_nan_percent_nn,
                'loc_mean':loc_mean,
                'loc_std':loc_std,
                'loc_diff_dist':args.diff_dist,
            },
            'cp_load_path': args.cp_load_path,
            'sigma_y': args.sigma_y,
        }
        
        # print(torch.mean((ens_tensor_enkf.mean(dim=2) - ens_tensor_nn.mean(dim=2))**2, dim=(1,2))[:100])
        
        if args.cp_load_path != "no":
            if args.zero_infl:
                torch.save(tensor_dict, os.path.join(folder_name, f"output_records_zero_infl_{args.N}.pt"))
            else:
                torch.save(tensor_dict, os.path.join(folder_name, f"output_records_{args.N}.pt"))


