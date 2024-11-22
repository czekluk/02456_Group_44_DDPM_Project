import torch
import math
import torch.nn as nn
import torch.nn.functional as F

def get_timestep_embedding(timesteps: torch.Tensor, dimension: int = 256) -> torch.Tensor:
    """Generate sinusoidal embeddings for a given tensor of timesteps.

    Args:
        timesteps (torch.Tensor): Tensor containing timesteps to generate embeddings for. Shape: [batch_size]
        dimension (int): Dimensionality of the embeddings to generate.

    Returns:
        torch.Tensor: Sinusoidal embeddings for the given timesteps. Shape: [batch_size, dimension]
    """
    device = timesteps.device

    # Loop over half of the requested dim
    half_dim = dimension // 2
    i = torch.arange(half_dim, dtype=torch.float32) # shape: [half_dim]
    exponent = math.log(10000) / (half_dim - 1) # subtract 1 to account for 0-indexing
    frequencies = torch.exp(i * -exponent)
    frequencies = frequencies.to(timesteps.device)
    # Reshape to allow for element-wise multiplication
    timesteps = timesteps.view(-1, 1).cpu() # shape: [batch_size, 1]
    frequencies = frequencies.view(1, -1) # shape: [1, half_dim]
    angles = timesteps * frequencies # shape: [batch_size, half_dim]
    
    # Apply sin to even indices and cos to odd indices
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) # shape: [batch_size, dimension]
    
    # If dimension is odd, pad last dimension with zeros
    if dimension % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    
    return emb.to(device)

def test_timestep_embedding():
    """Test the get_timestep_embedding function."""
    # generate 64 random timesteps to calculate embeddings for
    timesteps = torch.randint(0, 1000, (10,)) 
    print("Generating timesteps:")
    print(timesteps.shape) # shape: [batch_size]
    print(timesteps)
    print("-"*80)
    
    # Define requested embedding dimension and calculate embeddings
    embedding_dim = 10
    print(f"Calculating embeddings with dimensionality: [{embedding_dim}]")
    timestep_embeddings = get_timestep_embedding(timesteps, embedding_dim)

    print("Output embeddings shape:")
    print(timestep_embeddings.shape)  # shape: [batch_size, embedding_dim]
    print("-"*80)
    
    print(f"Example embedding for first timestep ({timesteps[0]}):")
    print(timestep_embeddings[0])  # shape: [embedding_dim]


def normalize(x, temb, name):
    # Replacing group_norm from tf.contrib
    num_groups = 32  # You may change this as needed
    return nn.GroupNorm(num_groups=num_groups, num_channels=x.shape[1])(x)

def upsample(x, with_conv):
    B, C, H, W = x.shape
    x = nn.Upsample(size=(H * 2, W * 2), mode='nearest')(x)
    if with_conv:
        x = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=3, stride=1, padding=1)(x)
    return x

def downsample(x, with_conv):
    B, C, H, W = x.shape
    if with_conv:
        x = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=3, stride=2, padding=1)(x)
    else:
        x = F.avg_pool2d(x, 2, stride=2)
    return x


if __name__ == "__main__":
    test_timestep_embedding()