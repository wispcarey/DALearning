import torch
import torch.nn as nn

def compute_es(ens_states, true_states, norm_p=1):
    """
    Memory-efficient Energy Score (ES) computation for inputs with shape [T, B, N, D].
    
    The Energy Score (for β = 1) is defined as:
    
        ES(P, y) = (1 / (N*(N-1))) * sum_{i != j} ||x_i - x_j|| 
                   - (2 / N) * sum_{i=1}^N ||x_i - y||
    
    Args:
        ens_states: Tensor of shape [T, B, N, D] - Ensemble predictions.
        true_states: Tensor of shape [T, B, D]   - Ground truth.
        norm_p: Norm type (e.g., 2 for L2) used in distance computation.
        
    Returns:
        es: Tensor of shape [T, B] - Energy Score per time step and batch element.
    """
    T, B, N, D = ens_states.shape

    # Compute the average distance from each ensemble member to the ground truth.
    true_expanded = true_states.unsqueeze(2)  # [T, B, 1, D]
    # Compute distances: shape [T, B, N]
    dist_to_true = torch.norm(ens_states - true_expanded, p=norm_p, dim=-1)
    # Compute (2 / N) * sum_{i=1}^N ||x_i - y||
    term_obs = (2.0 / N) * torch.sum(dist_to_true, dim=2)  # [T, B]
    
    # Compute the average pairwise distance among ensemble members.
    total_pair = torch.zeros(T, B, device=ens_states.device)
    # Sum over distinct pairs (i, j) with i < j; note that sum_{i != j} = 2 * sum_{i<j}
    for i in range(N):
        xi = ens_states[:, :, i, :]  # [T, B, D]
        for j in range(i + 1, N):
            xj = ens_states[:, :, j, :]  # [T, B, D]
            dist_pair = torch.norm(xi - xj, p=norm_p, dim=-1)  # [T, B]
            total_pair += dist_pair
    # There are N*(N-1)/2 distinct pairs; compute average of all distinct pairs,
    # then multiply by 2 to obtain (1/(N*(N-1))) * sum_{i != j} ||x_i - x_j||
    term_pair = (2.0 * total_pair) / (N * (N - 1))  # [T, B]
    
    # Energy Score: ES = (pairwise term) - (observation term)
    es = term_pair - term_obs  # [T, B]
    return es

def compute_kernel_es(ens_states, true_states, sigma=None):
    """
    Compute the kernel version of the Energy Score (kernel ES) using Gaussian kernel.
    
    The Gaussian kernel is defined as:
        k(x, y) = exp(-||x - y||^2/(2 * sigma^2))
    
    The kernel Energy Score is computed as:
    
        kernel_ES = 1/2 * (average kernel value over all distinct ensemble pairs)
                    - 1/N * (average kernel value between ensemble members and ground truth)
    
    If sigma is not provided, for each (T, B) it is computed as the median of 
    ||x_i - y|| over the ensemble members.
    
    Args:
        ens_states: Tensor of shape [T, B, N, D] - Ensemble predictions.
        true_states: Tensor of shape [T, B, D]   - Ground truth.
        sigma: Optional scalar specifying the kernel bandwidth.
               If None, sigma is computed as the median of ||x_i - y|| for each (T, B).
    
    Returns:
        kernel_es: Tensor of shape [T, B] - Kernel Energy Score per time step and batch element.
    """
    T, B, N, D = ens_states.shape
    device = ens_states.device

    # Compute sigma if not provided, per (T, B)
    if sigma is None:
        # Expand true_states to [T, B, 1, D] for broadcasting.
        true_expanded = true_states.unsqueeze(2)  # [T, B, 1, D]
        # Compute Euclidean distances for each ensemble member: shape [T, B, N]
        distances = torch.norm(ens_states - true_expanded, dim=-1)
        # Sigma for each (T, B) is set as the median over the ensemble dimension.
        sigma_val = torch.median(distances, dim=2)[0]  # [T, B]
    else:
        sigma_val = torch.tensor(sigma, device=device).expand(T, B)  # [T, B]
    
    # Expand sigma_val to shape [T, B, 1, 1] for kernel computation.
    sigma_val = sigma_val.unsqueeze(-1).unsqueeze(-1)
    
    # Compute kernel values between ensemble members and the true state.
    true_expanded = true_states.unsqueeze(2)  # [T, B, 1, D]
    diff_obs = ens_states - true_expanded      # [T, B, N, D]
    dist_sq_obs = torch.sum(diff_obs ** 2, dim=-1)  # [T, B, N]
    k_obs = torch.exp(-dist_sq_obs / (2 * sigma_val ** 2))  # [T, B, N]
    # Average over ensemble members.
    observed_term = torch.mean(k_obs, dim=2)  # [T, B]
    
    # Compute average kernel value for distinct ensemble pairs.
    total = torch.zeros(T, B, device=device)
    for i in range(N):
        xi = ens_states[:, :, i:i+1, :]  # [T, B, 1, D]
        for j in range(i + 1, N):
            xj = ens_states[:, :, j:j+1, :]  # [T, B, 1, D]
            diff_pair = xi - xj              # [T, B, 1, D]
            dist_sq_pair = torch.sum(diff_pair ** 2, dim=-1)  # [T, B, 1]
            k_pair = torch.exp(-dist_sq_pair / (2 * sigma_val ** 2))  # [T, B, 1]
            total += k_pair.squeeze(-1)  # [T, B]
    
    # Number of distinct pairs is N*(N-1)/2, so the average kernel value is:
    first_term_avg = (2 * total) / (N * (N - 1))
    
    # Final kernel energy score as per the formula:
    # kernel_ES = 0.5 * (first_term_avg) - (1/N) * (observed_term)
    kernel_es = 0.5 * first_term_avg - (1.0 / N) * observed_term
    return kernel_es

def compute_loss(ens_tensor, batch_v, loss_type, ignore_first=0, end_ind=None, valid_B_mask=None, norm_p=1):
    """
    Comprehensive loss function supporting multiple loss types.
    
    Parameters:
        ens_tensor: Ensemble predictions with shape [time_steps, batch_size, ensemble_size, feature_dim].
        batch_v: Ground truth values with shape [time_steps, batch_size, feature_dim].
        loss_type: Type of loss ('l2', 'normalized_l2', 'rmse', 'es', or combinations).
                   For kernel version, use 'kes' for kernel energy score and 'nkes' for normalized version.
        ignore_first: Number of initial time steps to ignore.
        end_ind: Last time step to consider (inclusive), defaults to all.
        valid_B_mask: Boolean mask for valid batch elements, defaults to all valid.
        norm_p: Norm type (e.g., 2 for L2).
    
    Returns:
        Computed loss value.
    """
    # Ensure end_ind is valid
    if end_ind is None:
        end_ind = batch_v.size(0) - 1
    
    # If no valid batch mask provided, consider all batches valid.
    if valid_B_mask is None:
        valid_B_mask = torch.ones(batch_v.size(1), dtype=torch.bool, device=batch_v.device)
    
    # Extract valid data for loss computation.
    ens_states = ens_tensor[ignore_first:, valid_B_mask, :, :]  # [time, valid_batch, ensemble_size, feature_dim]
    true_states = batch_v[ignore_first:end_ind + 1, valid_B_mask, :]  # [time, valid_batch, feature_dim]
    
    # Calculate ensemble mean for non-ES losses.
    ens_mean = torch.mean(ens_states, dim=2)  # [time, valid_batch, feature_dim]
    
    # Compute loss based on specified type.
    if loss_type == "l2":
        # Mean squared error.
        loss = torch.mean((ens_mean - true_states) ** 2)
    elif loss_type == 'nl2':
        # Normalized mean squared error.
        error_norm_2 = torch.sum((ens_mean - true_states) ** 2, dim=2)
        true_norm_2 = torch.sum(true_states ** 2, dim=2)
        eps = 1e-8
        loss = torch.mean(error_norm_2 / (true_norm_2 + eps))
    elif loss_type == 'rmse':
        # Root mean squared error.
        mse = torch.mean((ens_mean - true_states) ** 2, dim=2)
        loss = torch.mean(torch.sqrt(mse + 1e-8))
    elif loss_type == 'es':
        # Energy Score.
        es_values = compute_es(ens_states, true_states, norm_p)  # [time, valid_batch]
        loss = torch.mean(es_values)
    elif loss_type == 'nes':
        # Normalized Energy Score.
        es_values = compute_es(ens_states, true_states, norm_p)  # [time, valid_batch]
        true_norm = torch.norm(true_states, p=norm_p, dim=2)
        loss = torch.mean(es_values / (true_norm + 1e-8))
    elif loss_type == 'tnes':
        # Trajectory Normalized Energy Score.
        es_values = compute_es(ens_states, true_states, norm_p)  # [time, valid_batch]
        true_norm = torch.norm(true_states, p=norm_p, dim=2)
        loss = torch.mean(torch.sum(es_values, dim=0) / (torch.sum(true_norm, dim=0) + 1e-8))
    elif loss_type == 'kes':
        # Kernel Energy Score (using Gaussian kernel).
        es_values = compute_kernel_es(ens_states, true_states, sigma=None)
        loss = torch.mean(es_values)
    elif loss_type == 'nkes':
        # Normalized Kernel Energy Score.
        es_values = compute_kernel_es(ens_states, true_states, sigma=None)
        true_norm = torch.norm(true_states, p=norm_p, dim=2)
        loss = torch.mean(es_values / (true_norm + 1e-8))
    elif loss_type == 'tnkes':
        # Trajectory Normalized Kernel Energy Score.
        es_values = compute_kernel_es(ens_states, true_states, sigma=None)
        true_norm = torch.norm(true_states, p=norm_p, dim=2)
        loss = torch.mean(torch.sum(es_values, dim=0) / (torch.sum(true_norm, dim=0) + 1e-8))
    else:
        raise NotImplementedError(f"Loss type '{loss_type}' is not implemented")
    
    return loss
class MultiLossUncertaintyWeight(nn.Module):
    def __init__(self, num_losses):
        super(MultiLossUncertaintyWeight, self).__init__()
        # learnable log(sigma) for each loss
        self.log_sigma = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        # losses: list or tensor of individual losses
        total_loss = 0
        for i, loss in enumerate(losses):
            weight = torch.exp(-self.log_sigma[i])
            total_loss += weight * loss + self.log_sigma[i]
        return total_loss

if __name__ == "__main__":
    # Test dimensions of compute_crps output
    
    # Setup test parameters
    time_steps = 8
    batch_size = 4
    ensemble_size = 10
    feature_dim = 3
    
    # Create test data
    ens_states = torch.randn(time_steps, batch_size, ensemble_size, feature_dim)
    true_states = torch.randn(time_steps, batch_size, feature_dim)
    
    # Compute CRPS
    crps_result = compute_crps(ens_states, true_states)
    
    # Print input and output dimensions
    print(f"Input dimensions:")
    print(f"  ens_states: {ens_states.shape}")
    print(f"  true_states: {true_states.shape}")
    print(f"Output dimension:")
    print(f"  crps_result: {crps_result.shape}")
    
    # Verify output dimension is as expected
    expected_shape = torch.Size([time_steps, batch_size])
    assert crps_result.shape == expected_shape, f"Expected shape {expected_shape}, got {crps_result.shape}"
    
    print("Test passed! Output dimensions are correct.")