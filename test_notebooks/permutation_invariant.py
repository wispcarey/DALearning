import sys
import os
import torch

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from networks import SetTransformer

if __name__ == "__main__":
    torch.manual_seed(0)

    # Define model parameters
    input_dim = 16
    num_heads = 4
    num_inds = 8
    output_dim = 10
    hidden_dim = 32
    num_layers = 2

    # Create the SetTransformer model
    model = SetTransformer(input_dim, num_heads, num_inds, output_dim, hidden_dim, num_layers)

    # Generate random input data
    batch_size = 5
    set_size = 20  # Number of elements in each set
    X = torch.randn(batch_size, set_size, input_dim)

    # Create a permuted version of the input data
    X_permuted = X.clone()
    for i in range(batch_size):
        perm = torch.randperm(set_size)
        X_permuted[i] = X_permuted[i, perm, :]

    # Pass both inputs through the model
    output_original = model(X)
    output_permuted = model(X_permuted)

    # Compare the outputs
    difference = torch.abs(output_original - output_permuted)
    max_difference = difference.max().item()
    print(f"Maximum difference between outputs: {max_difference}")

    # Check if the outputs are the same within a tolerance
    if torch.allclose(output_original, output_permuted, atol=1e-6):
        print("Test Passed: The model is permutation invariant.")
    else:
        print("Test Failed: The model is not permutation invariant.")