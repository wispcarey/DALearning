import torch

def compute_crps(ens_states, true_states, norm_p=1):
    """
    Compute the discrete Continuous Ranked Probability Score (CRPS)
    
    Parameters:
        ens_states: Ensemble forecasts with shape [..., ensemble_size, feature_dim]
        true_states: Ground truth observations with shape [..., feature_dim]
    
    Returns:
        CRPS values averaged over feature dimensions
    """
    # Get the number of ensemble members
    n_samples = ens_states.size(-2)
    
    # Expand dimensions for broadcasting
    true_expanded = true_states.unsqueeze(-2)  # [..., 1, feature_dim]
    
    # Compute first term: mean Euclidean norm between each ensemble member and ground truth
    # Using L2 norm along the feature dimension
    term1 = torch.mean(torch.norm(ens_states - true_expanded, p=norm_p, dim=-1), dim=-1)
    
    # Compute second term: mean Euclidean norm between all pairs of ensemble members
    ens_i = ens_states.unsqueeze(-3)  # [..., 1, ensemble_size, feature_dim]
    ens_j = ens_states.unsqueeze(-2)  # [..., ensemble_size, 1, feature_dim]
    
    # Compute L2 norm along feature dimension for all pairs
    pairwise_dists = torch.norm(ens_i - ens_j, p=norm_p, dim=-1)  # [..., ensemble_size, ensemble_size]
    
    # Average over all pairs
    term2 = torch.sum(pairwise_dists, dim=(-2, -1)) / (n_samples * n_samples)
    
    # CRPS
    crps = term1 - 0.5 * term2
    
    return crps

def compute_loss(ens_tensor, batch_v, loss_type, ignore_first=0, end_ind=None, valid_B_mask=None):
    """
    Comprehensive loss function supporting multiple loss types
    
    Parameters:
        ens_tensor: Ensemble predictions with shape [time_steps, batch_size, ensemble_size, feature_dim]
        batch_v: Ground truth values with shape [time_steps, batch_size, feature_dim]
        loss_type: Type of loss ('l2', 'normalized_l2', 'rmse', 'crps', or combinations)
        ignore_first: Number of initial time steps to ignore
        end_ind: Last time step to consider (inclusive), defaults to all
        valid_B_mask: Boolean mask for valid batch elements, defaults to all valid
    
    Returns:
        Computed loss value
    """
    # Ensure end_ind is valid
    if end_ind is None:
        end_ind = batch_v.size(0) - 1
    
    # If no valid batch mask provided, consider all batches valid
    if valid_B_mask is None:
        valid_B_mask = torch.ones(batch_v.size(1), dtype=torch.bool, device=batch_v.device)
    
    # Extract valid data for loss computation
    ens_states = ens_tensor[ignore_first:, valid_B_mask, :, :]  # [time, valid_batch, ensemble_size, feature_dim]
    true_states = batch_v[ignore_first:end_ind + 1, valid_B_mask, :]  # [time, valid_batch, feature_dim]
    
    # Calculate ensemble mean for non-CRPS losses
    ens_mean = torch.mean(ens_states, dim=2)  # [time, valid_batch, feature_dim]
    
    # Compute loss based on specified type
    if loss_type == "l2":
        # Mean squared error
        loss = torch.mean((ens_mean - true_states) ** 2)
    elif loss_type == 'normalized_l2':
        # Normalized mean squared error
        error_norm_2 = torch.sum((ens_mean - true_states) ** 2, dim=2)
        true_norm_2 = torch.sum(true_states ** 2, dim=2)
        # Avoid division by zero
        eps = 1e-8
        loss = torch.mean(error_norm_2 / (true_norm_2 + eps))
    elif loss_type == 'rmse':
        # Root mean squared error
        mse = torch.mean((ens_mean - true_states) ** 2, dim=2)
        loss = torch.mean(torch.sqrt(mse + 1e-8))  # Add small constant for numerical stability
    elif loss_type == 'crps':
        # Continuous Ranked Probability Score
        crps_values = compute_crps(ens_states, true_states)  # [time, valid_batch, feature_dim]
        loss = torch.mean(crps_values)
    
    else:
        raise NotImplementedError(f"Loss type '{loss_type}' is not implemented")
    
    return loss

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