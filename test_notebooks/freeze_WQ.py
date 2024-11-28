import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from networks import MAB  # Assuming MAB is defined in networks.py


if __name__ == "__main__":
    # Define input dimensions and parameters
    dim_Q = 64
    dim_KV = 64
    num_heads = 8
    batch_size = 32
    seq_len_Q = 10
    seq_len_K = 15
    num_epochs = 5

    # Instantiate the MAB with freeze_WQ=True
    mab = MAB(dim_Q=dim_Q, dim_KV=dim_KV, num_heads=num_heads, freeze_WQ=True)

    # Function to extract W_Q, W_K, W_V from in_proj_weight
    def get_WQ_WK_WV(mab):
        # Extract in_proj_weight
        in_proj_weight = mab.multihead_attn.in_proj_weight
        embed_dim = mab.multihead_attn.embed_dim
        
        # Split in_proj_weight into W_Q, W_K, and W_V
        W_Q = in_proj_weight[:embed_dim, :].detach().clone()
        W_K = in_proj_weight[embed_dim:2*embed_dim, :].detach().clone()
        W_V = in_proj_weight[2*embed_dim:, :].detach().clone()
        
        return W_Q, W_K, W_V

    # Function to get other parameters
    def get_other_params(mab):
        params = {}
        for name, param in mab.named_parameters():
            if param.requires_grad and 'multihead_attn.in_proj_weight' not in name:
                params[name] = param.detach().clone()
        return params

    # Get initial W_Q, W_K, W_V and other parameters
    initial_WQ, initial_WK, initial_WV = get_WQ_WK_WV(mab)
    initial_other_params = get_other_params(mab)

    # Print the initial W_Q
    print("Initial W_Q (should be identity matrix):")
    print(initial_WQ)

    # Define a simple loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mab.parameters(), lr=0.001)

    # Generate dummy data and train the model
    for epoch in range(num_epochs):
        # Random input tensors Q and K
        Q = torch.randn(batch_size, seq_len_Q, dim_Q)
        K = torch.randn(batch_size, seq_len_K, dim_KV)
        # Target tensor
        target = torch.randn(batch_size, seq_len_Q, dim_Q)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = mab(Q, K)
        
        # Compute loss
        loss = criterion(output, target)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # Get final W_Q, W_K, W_V and other parameters
    final_WQ, final_WK, final_WV = get_WQ_WK_WV(mab)
    final_other_params = get_other_params(mab)

    # Print final W_Q
    print("\nFinal W_Q (should be unchanged):")
    print(final_WQ)

    # Verify if W_Q has changed
    if torch.allclose(initial_WQ, final_WQ):
        print("\nW_Q has not changed during training. It is successfully frozen.")
    else:
        print("\nW_Q has changed during training. There might be an issue.")

    # Verify if W_K and W_V have changed
    if not torch.allclose(initial_WK, final_WK):
        print("\nW_K has changed during training.")
    else:
        print("\nW_K has NOT changed during training.")

    if not torch.allclose(initial_WV, final_WV):
        print("\nW_V has changed during training.")
    else:
        print("\nW_V has NOT changed during training.")

    # Verify that other parameters have changed
    params_changed = False
    for name in initial_other_params:
        initial_param = initial_other_params[name]
        final_param = final_other_params[name]
        if not torch.allclose(initial_param, final_param):
            params_changed = True
            print(f"Parameter '{name}' has changed during training.")
        else:
            print(f"Parameter '{name}' has NOT changed during training.")

    if params_changed:
        print("\nOther parameters have changed during training.")
    else:
        print("\nOther parameters have NOT changed during training.")