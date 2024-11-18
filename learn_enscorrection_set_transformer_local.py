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
from utils import AverageMeter, mystery_operator, partial_obs_operator, get_dataloader, batch_covariance
from EnKF_utils import StochasticENKF_analysis, loc_EnKF_analysis, EnKF_analysis, post_process, mrdiv, mean0
from networks import ComplexAttentionModel, AttentionModel, NaiveNetwork, SetTransformer, Simple_MLP
from localization import pairwise_distances, dist2coeff, create_loc_mat


def train_model(epoch, loader, model_list, optimizer, scheduler, args, H_info=None):
    model, local_model, st_model1, st_model2 = model_list
    
    m = args.N

    losses = AverageMeter()
    batch_time = AverageMeter()
    
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

    model.train()
    for batch_ind, batch_v in enumerate(loader):
        t_start = time.time()
        batch_v = batch_v.to(device=args.device)

        # Sample from prior
        ens_v_a = batch_v[0].unsqueeze(1).repeat(1, m, 1)
        ens_v_a = ens_v_a + torch.randn_like(ens_v_a, device=args.device) * args.sigma_ens

        # Store everything
        ens_list = [ens_v_a]

        # Iterate over timesteps in batch
        if args.loss_warm_up:
            end_ind = np.minimum(epoch + 1, len(batch_v) - 1)
        else:
            end_ind = len(batch_v) - 1
        
        for i in range(end_ind):
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
                K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn) 
                K2 = torch.bmm(Ynn.transpose(1, 2), Ynn) + R * (N - 1)
            else:
                loc_nn_output = torch.sigmoid(local_model(local_nn_input)) * 2
                loc_mat_vy = create_loc_mat(loc_nn_output, args.diff_dist, args.Lvy)
                loc_mat_yy = create_loc_mat(loc_nn_output, args.diff_dist, args.Lyy)
                
                K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn) * loc_mat_vy
                K2 = torch.bmm(Ynn.transpose(1, 2), Ynn) * loc_mat_yy + R * (N - 1)
            
            # Kalman Gain
            K = torch.bmm(K1, torch.inverse(K2))
            ens_v_a = Vnn1 + torch.bmm(ens_i, K.transpose(1, 2))
            
            # ens_v_a = post_process(ens_v_a, infl=1.06)
            
            ens_v_a = torch.clamp(ens_v_a, min=-20, max=20)

            ens_list.append(ens_v_a)
            
            if epoch <= args.detach_training_epoch:
                ens_v_a = ens_v_a.detach()

        # Concat outputs
        ens_tensor = torch.stack(ens_list)

        # Loss functions
        if args.loss_warm_up:
            if epoch >= len(batch_v) - 1:
                ignore_first = args.ignore_first
            else:
                ignore_first = 0
        else:
            ignore_first = args.ignore_first
            
        if args.loss_type == "l2":
            loss = torch.mean((ens_tensor.mean(dim=2)[ignore_first:, :, :] - batch_v[ignore_first:end_ind + 1, :, :]) ** 2)
        elif args.loss_type == 'normalized_l2':
            error_norm_2 = torch.sum((ens_tensor.mean(dim=2)[ignore_first:, :, :] - batch_v[ignore_first:end_ind + 1, :, :]) ** 2, dim=2)
            true_norm_2 = torch.sum(batch_v[ignore_first:end_ind + 1, :, :] ** 2, dim=2)
            loss = torch.mean(error_norm_2 / true_norm_2)
        elif args.loss_type == 'rmse':
            loss = torch.mean(torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2)))
        else:
            raise NotImplementedError("loss_type is not implemented")


        losses.update(loss.item(), batch_v.shape[0])

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # batch time
        batch_time.update(time.time() - t_start)

        if (batch_ind + 1) % args.print_batch == 0:
            print(f'Training epoch : [{epoch}][{batch_ind + 1}/{len(loader)}]\t'
                  f'Batch time {batch_time.val:.3f} (Avg: {batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.3f} (Avg: {losses.avg:.3f})\t'
                  f'Current learning rate: {optimizer.param_groups[0]["lr"]:.2e}'
                  )

    scheduler.step()
    return losses.avg


def test_model(loader, model_list, args, infl=1, verbose_test=True, H_info=None):
    # kalman_layer = KalmanFilterLayer.apply
    
    model, local_model, st_model1, st_model2 = model_list

    m = args.N

    losses = AverageMeter()
    step_time = AverageMeter()
    
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
                    K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn)
                    K2 = torch.bmm(Ynn.transpose(1, 2), Ynn) + R * (N - 1)
                else:
                    loc_nn_output = torch.sigmoid(local_model(local_nn_input)) * 2
                    loc_mat_vy = create_loc_mat(loc_nn_output, args.diff_dist, args.Lvy)
                    loc_mat_yy = create_loc_mat(loc_nn_output, args.diff_dist, args.Lyy)
                    
                    K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn) * loc_mat_vy
                    K2 = torch.bmm(Ynn.transpose(1, 2), Ynn) * loc_mat_yy + R * (N - 1)
                
                K = torch.bmm(K1, torch.inverse(K2))
                ens_v_a = Vnn1 + torch.bmm(ens_i, K.transpose(1, 2))
                
                ens_v_a = post_process(ens_v_a, infl=infl)
                
                ens_v_a = torch.clamp(ens_v_a, min=-20, max=20)

                ens_list.append(ens_v_a)
                K_list.append(K)
                step_time.update(time.time() - t_start)
                
            # Concat outputs
            ens_tensor = torch.stack(ens_list)
            K_tensor = torch.stack(K_list)

            # Loss functions
            loss = torch.mean(torch.sqrt(torch.mean((ens_tensor.mean(dim=2) - batch_v) ** 2, dim=2)))
            losses.update(loss.item(), batch_v.shape[0])

        if verbose_test:
            print(f'Average Test RMSE: {losses.avg}\t'
                  f'Average One-step Time: {step_time.avg}\t'
                  )
            plot_results_2d(ens_tensor.mean(dim=2)[:, 0, :].cpu().numpy().T, batch_v[:, 0, :].cpu().numpy().T, 0, 499, 
                            plot_inds=[0,1,2], save_path=os.path.join(args.save_folder, "EnKF_results.png"))

        return losses.avg, step_time.avg, ens_tensor, K_tensor


if __name__ == "__main__":
    args = get_parameters()
    args.input_dim = args.ori_dim + 2 * args.obs_dim + args.st_output_dim * 2 
    args.local_input_dim = args.obs_dim + args.st_output_dim * 2 
    
    if args.seed is not None and args.seed != "None":
        torch.manual_seed(int(args.seed))

    # H_info
    H_info = partial_obs_operator(args.ori_dim, args.obs_inds, args.device)

    train_loader, test_loader = get_dataloader(args)
    
    # localization
    full_inds = torch.arange(0, args.ori_dim)
    Lvy = pairwise_distances(full_inds[:, None], args.obs_inds[:, None], domain=(args.ori_dim,)).to(args.device)
    Lyy = pairwise_distances(args.obs_inds[:, None], args.obs_inds[:, None], domain=(args.ori_dim,)).to(args.device)
    args.diff_dist = torch.unique(torch.cat((Lvy.flatten(), Lyy.flatten())))
    args.num_dist = len(args.diff_dist)
    args.Lvy = Lvy
    args.Lyy = Lyy

    # set models
    model = Simple_MLP(d_input=args.input_dim, d_output=args.obs_dim + 2 * args.ori_dim, num_hidden_layers=2).to(args.device)
    if args.no_localization:
        local_model = NaiveNetwork(1)
    else:
        local_model = Simple_MLP(d_input=args.local_input_dim, d_output=args.num_dist, num_hidden_layers=2).to(args.device)
    st_model1 = SetTransformer(input_dim=args.ori_dim, num_heads=8, num_inds=16, output_dim=args.st_output_dim, hidden_dim=args.hidden_dim, num_layers=1).to(args.device)
    st_model2 = SetTransformer(input_dim=args.obs_dim, num_heads=8, num_inds=16, output_dim=args.st_output_dim, hidden_dim=args.hidden_dim, num_layers=1).to(args.device)
    model, local_model, st_model1, st_model2 = nn.DataParallel(model), nn.DataParallel(local_model), nn.DataParallel(st_model1), nn.DataParallel(st_model2)
    model_list = [model, local_model, st_model1, st_model2]
    total_params = sum(sum(p.numel() for p in model.parameters()) for model in model_list)
    print(f'Total number of parameters: {total_params}')


    # Save folder
    if args.cp_load_path != "no":
        suffix = "_tuned"
    else:
        suffix = ""
    folder_name = os.path.join("save",
                               datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    folder_name += f"{args.dataset}_{args.sigma_y}_{args.N}_{args.train_steps}_{args.train_traj_num}_EnST{suffix}_{args.loss_type}"
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    args.save_folder = folder_name

    # optimizer
    optimizer, scheduler = setup_optimizer_and_scheduler(model_list, args)

    # load checkpoint
    if args.cp_load_path != "no":
        load_checkpoint(model_list, None, None, filename=args.cp_load_path)
        # for param in st_model1.parameters():
        #     param.requires_grad = False
        # for param in st_model2.parameters():
        #     param.requires_grad = False

    # training
    train_loss_list = []
    test_loss_list = []
    test_epochs = []
    if args.test_only:
        print("Test Only")
        loss_list_nn = []
        for i in range(args.test_rounds):
            loss_val, _, ens_tensor_nn, K_tensor_nn = test_model(test_loader, model_list, args, verbose_test=False, H_info=H_info)
            loss_list_nn.append(loss_val)
        print("Average RMSE:", torch.mean(torch.tensor(loss_list_nn)))
    else:
        print("Training Start")
        test_loss, _, _, _ = test_model(test_loader, model_list, args, verbose_test=True, H_info=H_info)
        test_loss_list.append(test_loss)
        for epoch in range(1, 1 + args.epochs):
            train_loss = train_model(epoch, train_loader, model_list, optimizer, scheduler, args, H_info=H_info)
            train_loss_list.append(train_loss)
            if epoch % args.save_epoch == 0:
                test_loss, _, _, _ = test_model(test_loader, model_list, args, verbose_test=True, H_info=H_info)
                test_epochs.append(epoch)
                test_loss_list.append(test_loss)
                train_records = {"train_loss": train_loss_list, "test_loss": test_loss_list, "test_epochs": test_epochs}
                torch.save(train_records, os.path.join(folder_name, f"training_records.pt"))
                save_checkpoint(model_list, optimizer, scheduler, filename=os.path.join(folder_name, f"cp_{epoch}.pth"))




