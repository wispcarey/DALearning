import torch
import torch.nn as nn
import torch.nn.functional as F

class MyMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MyMultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Define linear layers for Q, K, V with bias
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=True)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # Define the output linear layer with bias
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=True)
    
    def forward(self, query, key, value):
        # query, key, value shape: [seq_len, batch_size, embed_dim]
        batch_size = query.size(1)
        seq_len = query.size(0)
        
        # Compute Q, K, V
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)
        
        # Reshape to support multi-head attention
        Q = Q.view(seq_len, batch_size, self.num_heads, self.head_dim)
        K = K.view(seq_len, batch_size, self.num_heads, self.head_dim)
        V = V.view(seq_len, batch_size, self.num_heads, self.head_dim)
        
        # Transpose to match matrix multiplication dimensions
        Q = Q.permute(1, 2, 0, 3)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.permute(1, 2, 0, 3)
        V = V.permute(1, 2, 0, 3)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        # Compute context vectors
        context = torch.matmul(attn, V)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Merge heads
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.embed_dim)
        
        # Pass through the output linear layer
        output = self.W_O(context)
        
        # Transpose back to [seq_len, batch_size, embed_dim]
        output = output.permute(1, 0, 2)
        
        return output

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(2)

    seq_len = 10
    batch_size = 2
    embed_dim = 64
    num_heads = 8

    # Create random input tensors
    query = torch.randn(seq_len, batch_size, embed_dim)
    key = torch.randn(seq_len, batch_size, embed_dim)
    value = torch.randn(seq_len, batch_size, embed_dim)

    # Create the built-in MultiheadAttention module with bias
    attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    # Create the custom MultiheadAttention module
    my_attn = MyMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    # Copy weights from the built-in module to the custom module
    with torch.no_grad():
        # Get the built-in module's W_Q, W_K, W_V weights and biases
        W_QKV = attn.in_proj_weight  # [3 * embed_dim, embed_dim]
        b_QKV = attn.in_proj_bias    # [3 * embed_dim]
        
        W_Q = W_QKV[:embed_dim, :]
        W_K = W_QKV[embed_dim:2*embed_dim, :]
        W_V = W_QKV[2*embed_dim:, :]
        
        b_Q = b_QKV[:embed_dim]
        b_K = b_QKV[embed_dim:2*embed_dim]
        b_V = b_QKV[2*embed_dim:]
        
        W_O = attn.out_proj.weight  # [embed_dim, embed_dim]
        b_O = attn.out_proj.bias    # [embed_dim]
        
        # Assign weights and biases to the custom module
        my_attn.W_Q.weight.copy_(W_Q)
        my_attn.W_Q.bias.copy_(b_Q)
        my_attn.W_K.weight.copy_(W_K)
        my_attn.W_K.bias.copy_(b_K)
        my_attn.W_V.weight.copy_(W_V)
        my_attn.W_V.bias.copy_(b_V)
        my_attn.W_O.weight.copy_(W_O)
        my_attn.W_O.bias.copy_(b_O)

    # Get outputs from both modules
    output_attn, _ = attn(query, key, value)
    output_my_attn = my_attn(query, key, value)

    # Compare the outputs
    diff = (output_my_attn - output_attn).abs().max()
    print('Maximum difference:', diff.item())
