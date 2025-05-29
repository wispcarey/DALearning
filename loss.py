import torch
import torch.nn as nn

def compute_loss(ens_tensor, batch_v, loss_type, ignore_first=0, end_ind=None, valid_B_mask=None, norm_p=1, kes_sigma=1):
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