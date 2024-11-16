import numpy as np
import time
import datetime
import os

import torch
import torch.nn as nn

from config.cli import get_parameters

from utils import plot_results_2d, setup_optimizer_and_scheduler, save_checkpoint, load_checkpoint, DATASET_INFO
from utils import L63, L96, rk4, AverageMeter, mystery_operator, partial_obs_operator, get_dataloader, batch_covariance
from EnKF_utils import StochasticENKF_analysis, loc_EnKF_analysis, EnKF_analysis, post_process, mrdiv, mean0
from networks import ComplexAttentionModel, AttentionModel, NaiveNetwork
from localization import pairwise_distances, dist2coeff, create_loc_mat


def train_model(epoch, loader, model, optimizer, scheduler, args, H_info=None):
    m = args.N

    losses = AverageMeter()
    batch_time = AverageMeter()
    
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
                ens_v_a_new = rk4(forward_fun, ens_v_a, i * args.dt + j * args.dt / args.dt_iter,
                              args.dt / args.dt_iter)
                # if torch.isnan(ens_v_a_new).any():
                #     print(torch.max(abs(ens_v_a)), torch.quantile(abs(ens_v_a), 0.98), torch.quantile(abs(ens_v_a), 0.95))
                #     raise ValueError("NAN value after RK4")
                ens_v_a = ens_v_a_new
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
            mean_hv = torch.mean(hv, dim=1)
            mean_ens_v_f = torch.mean(ens_v_f, dim=1)
            Cvv = batch_covariance(ens_v_f).view(-1, D ** 2)
            Cvy = batch_covariance(ens_v_f, hv).view(-1, D * d)
            Cyy = batch_covariance(hv) + args.sigma_y * torch.eye(d).unsqueeze(0).expand(B, -1, -1).to(args.device)
            Cyy = Cyy.view(-1, d ** 2)
            ensemble_info = torch.cat([mean_ens_v_f, mean_hv, Cvv, Cvy, Cyy], dim=-1)
            
            nn_input = torch.cat([obs_y.squeeze(1), ensemble_info], dim=-1)

            # execute model
            nn_output = model(nn_input)
            # infl = 2 * torch.sigmoid(nn_output[:, :1]).view(B, 1, 1).expand(-1, N, D)
            infl = 1.1
            dK = nn_output[:, 1:].view(B, D, d)
            Vnn1 = ens_v_f
            Vnn2 = ens_v_f - mean_ens_v_f.unsqueeze(1)
            Ynn = hv - mean_hv.unsqueeze(1)
            R = args.sigma_y ** 2 * torch.eye(d).unsqueeze(0).expand(B, -1, -1).to(args.device)
            
            # Kalman Gain
            K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn)
            K2 = torch.bmm(Ynn.transpose(1, 2), Ynn) + R * (N - 1)
            K = torch.bmm(K1, torch.inverse(K2)) + dK
            ens_v_a = Vnn1 + torch.bmm(ens_i, K.transpose(1, 2))
            
            ens_v_a = post_process(ens_v_a, infl=infl)
            
            ens_v_a = torch.clamp(ens_v_a, min=-40, max=40)

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
        loss = torch.mean(torch.mean((ens_tensor.mean(dim=2)[ignore_first:, :, :] - batch_v[ignore_first:end_ind + 1, :, :]) ** 2, dim=2))
        # loss = torch.mean(torch.sqrt(torch.mean((ens_tensor.mean(dim=2)[args.ignore_first:, :, :] - batch_v[args.ignore_first:, :, :]) ** 2, dim=2)))
        losses.update(loss.item(), batch_v.shape[0])

        # optimization
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


def test_model(loader, model, args, verbose_test=True, H_info=None):
    # kalman_layer = KalmanFilterLayer.apply

    m = args.N

    losses = AverageMeter()
    step_time = AverageMeter()
    
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
                mean_hv = torch.mean(hv, dim=1)
                mean_ens_v_f = torch.mean(ens_v_f, dim=1)
                Cvv = batch_covariance(ens_v_f).view(-1, D ** 2)
                Cvy = batch_covariance(ens_v_f, hv).view(-1, D * d)
                Cyy = batch_covariance(hv) + args.sigma_y * torch.eye(d).unsqueeze(0).expand(B, -1, -1).to(args.device)
                Cyy = Cyy.view(-1, d ** 2)
                ensemble_info = torch.cat([mean_ens_v_f, mean_hv, Cvv, Cvy, Cyy], dim=-1)
                
                nn_input = torch.cat([obs_y.squeeze(1), ensemble_info], dim=-1)

                # execute model
                nn_output = model(nn_input)
                # infl = 2 * torch.sigmoid(nn_output[:, :1]).view(B, 1, 1).expand(-1, N, D)
                infl = 1.1
                dK = nn_output[:, 1:].view(B, D, d)
                Vnn1 = ens_v_f
                Vnn2 = ens_v_f - mean_ens_v_f.unsqueeze(1)
                Ynn = hv - mean_hv.unsqueeze(1)
                R = args.sigma_y ** 2 * torch.eye(d).unsqueeze(0).expand(B, -1, -1).to(args.device)
                
                # Kalman Gain
                K1 = torch.bmm(Vnn2.transpose(1, 2), Ynn)
                K2 = torch.bmm(Ynn.transpose(1, 2), Ynn) + R * (N - 1)
                K = torch.bmm(K1, torch.inverse(K2)) + dK
                ens_v_a = Vnn1 + torch.bmm(ens_i, K.transpose(1, 2))
                
                ens_v_a = post_process(ens_v_a, infl=infl)
                
                ens_v_a = torch.clamp(ens_v_a, min=-40, max=40)

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


def test_SequentialEnKF(loader, args, infl=1, verbose_test=True, H_info=None, localization=False):
    m = args.N

    losses = AverageMeter()
    step_time = AverageMeter()
    
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
                if torch.isnan(ens_v_a).any():
                    raise ValueError(f"{i}, nan 1")
                ens_v_a = ens_v_a.view(-1, args.ori_dim)
                for j in range(args.dt_iter):
                    ens_v_a = rk4(forward_fun, ens_v_a, i * args.dt + j * args.dt / args.dt_iter,
                                  args.dt / args.dt_iter)
                if torch.isnan(ens_v_a).any():
                    raise ValueError(f"{i}, nan 2")
                ens_v_f = ens_v_a.view(-1, m, args.ori_dim)

                # add forward noise
                ens_v_f = ens_v_f + torch.randn_like(ens_v_f, device=args.device) * args.sigma_v

                # analysis
                if torch.isnan(ens_v_f).any():
                    raise ValueError(f"{i}, nan 3") 
                
                ens_yo = H_fun(ens_v_f)
                if localization:
                    ens_v_a, K = loc_EnKF_analysis(ens_v_f, ens_yo, obs_y, args.sigma_y, Lvy, Lyy, a_method="PertObs")
                else:
                    ens_v_a, K = EnKF_analysis(ens_v_f, ens_yo, obs_y, args.sigma_y, a_method="PertObs")
                ens_v_a = post_process(ens_v_a, infl=infl)

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
            # plot_results_3d(ens_tensor.mean(dim=2)[:, 0, :].cpu().numpy().T, batch_v[:, 0, :].cpu().numpy().T, 0, 499)

        return losses.avg, step_time.avg, ens_tensor, K_tensor


if __name__ == "__main__":
    args = get_parameters()
    
    # dimension of problem
    args.ori_dim = DATASET_INFO[args.dataset]['dim']
    args.obs_dim = DATASET_INFO[args.dataset]['obs_dim']
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args.input_dim = args.ori_dim + 2 * args.obs_dim + args.ori_dim * args.obs_dim + args.ori_dim ** 2 + args.obs_dim ** 2 # d + D + d + D^2 + Dd + d^2


    folder_name = os.path.join("save",
                               datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    folder_name += f"{args.dataset}_{args.sigma_y}_{args.N}"
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    args.save_folder = folder_name


    if args.seed is not None and args.seed != "None":
        torch.manual_seed(int(args.seed))

    # H_info
    # H_info = mystery_operator((args.test_traj_num, args.ori_dim, args.obs_dim), args.device)
    obs_inds = torch.arange(0, args.ori_dim, 2)
    H_info = partial_obs_operator(args.ori_dim, obs_inds, args.device)
    print(H_info[1].shape)

    train_loader, test_loader = get_dataloader(args)
    
    # localization
    full_inds = torch.arange(0, args.ori_dim)
    Lvy = pairwise_distances(full_inds[:, None], obs_inds[:, None], domain=(args.ori_dim,)).to(args.device)
    Lyy = pairwise_distances(obs_inds[:, None], obs_inds[:, None], domain=(args.ori_dim,)).to(args.device)
    args.diff_dist = torch.unique(torch.cat((Lvy.flatten(), Lyy.flatten())))
    args.num_dist = len(args.diff_dist)
    args.Lvy = Lvy
    args.Lyy = Lyy

    # # EnKF
    # print("Original EnKF without localization")
    # if args.test_only:
    #     loss_list = []
    #     for i in range(args.test_rounds):
    #         loss_val, _, ens_tensor_enkf, K_tensor_enkf = test_SequentialEnKF(test_loader, args, infl=1.1, verbose_test=False, H_info=H_info)
    #         loss_list.append(loss_val)
    #     loss_val = torch.mean(torch.tensor(loss_list))
    # else:
    #     loss_val, _, ens_tensor_enkf, K_tensor_enkf = test_SequentialEnKF(test_loader, args, infl=1.1, verbose_test=False, H_info=H_info)
    # print("Average RMSE:", loss_val)
    
    # print("Original EnKF with localization")
    # if args.test_only:
    #     loss_list = []
    #     for i in range(args.test_rounds):
    #         loss_val, _, ens_tensor_enkf, K_tensor_enkf = test_SequentialEnKF(test_loader, args, infl=1.1, verbose_test=False, H_info=H_info, localization=True)
    #         loss_list.append(loss_val)
    #     loss_val = torch.mean(torch.tensor(loss_list))
    # else:
    #     loss_val, _, ens_tensor_enkf, K_tensor_enkf = test_SequentialEnKF(test_loader, args, infl=1.1, verbose_test=False, H_info=H_info, localization=True)
    # print("Average RMSE:", loss_val)
    
    # # Naive model test
    # print("Naive NN test")
    # model = NaiveNetwork(1 + args.obs_dim * args.ori_dim).to(args.device)
    # local_model = NaiveNetwork(args.num_dist).to(args.device)
    # loss_val, _, ens_tensor_naive, K_tensor_naive = test_model(test_loader, model, args, verbose_test=True, H_info=H_info)

    # train model
    model = AttentionModel(args.input_dim, 1 + args.obs_dim * args.ori_dim, num_attention_layers=1, hidden_dim=64).to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

    # optimizer
    optimizer, scheduler = setup_optimizer_and_scheduler(model, args)

    # load checkpoint
    if args.cp_load_path != "no":
        load_checkpoint(model, optimizer, scheduler, filename=args.cp_load_path)
        

    # training
    train_loss_list = []
    test_loss_list = []
    test_epochs = []
    if args.test_only:
        print("Test Only")
        loss_list_nn = []
        for i in range(args.test_rounds):
            loss_val, _, ens_tensor_nn, K_tensor_nn = test_model(test_loader, model, args, verbose_test=False, H_info=H_info)
            loss_list_nn.append(loss_val)
        print("Average RMSE:", torch.mean(torch.tensor(loss_list_nn)))
    else:
        print("Training Start")
        test_model(test_loader, model, args, verbose_test=True, H_info=H_info)
        for epoch in range(1, 1 + args.epochs):
            train_loss = train_model(epoch, train_loader, model, optimizer, scheduler, args, H_info=H_info)
            if epoch % args.save_epoch == 0:
                test_model(test_loader, model, args, verbose_test=True, H_info=H_info)
                save_checkpoint(model, optimizer, scheduler, filename=os.path.join(folder_name, f"cp_{epoch}.pth"))



