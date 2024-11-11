import torch
import math

def get_embedding_for_timesteps(timesteps: torch.Tensor, dimension: int = 256) -> torch.Tensor:
    """Generate sinusoidal embeddings for a given set of timesteps.

    Args:
        timesteps (torch.Tensor): Tensor containing timesteps to generate embeddings for. Shape: [batch_size]
        dimension (int): Dimensionality of the embeddings to generate.

    Returns:
        torch.Tensor: Sinusoidal embeddings for the given timesteps. Shape: [batch_size, dimension]
    """
    # Loop over half of the requested dim
    half_dim = dimension // 2
    i = torch.arange(half_dim, dtype=torch.float32) # shape: [half_dim]
    exponent = math.log(10000) / (half_dim - 1) # subtract 1 to account for 0-indexing
    frequencies = torch.exp(i * -exponent)
    
    # Reshape to allow for element-wise multiplication
    timesteps = timesteps.view(-1, 1) # shape: [batch_size, 1]
    frequencies = frequencies.view(1, -1) # shape: [1, half_dim]
    angles = timesteps * frequencies # shape: [batch_size, half_dim]
    
    # Apply sin to even indices and cos to odd indices
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) # shape: [batch_size, dimension]
    
    # If dimension is odd, pad last dimension with zeros
    if dimension % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    
    return emb
    

if __name__ == "__main__":
    # ---------------------------
    # TEST TIMESTEP EMBEDDING
    # ---------------------------
    # generate 64 random timesteps to calculate embeddings for
    timesteps = torch.randint(0, 1000, (10,)) 
    print("Generating timesteps:")
    print(timesteps.shape) # shape: [batch_size]
    print(timesteps)
    print("-"*80)
    
    # Define requested embedding dimension and calculate embeddings
    embedding_dim = 10
    print(f"Calculating embeddings with dimensionality: [{embedding_dim}]")
    timestep_embeddings = get_embedding_for_timesteps(timesteps, embedding_dim)

    print("Output embeddings shape:")
    print(timestep_embeddings.shape)  # shape: [batch_size, embedding_dim]
    print("-"*80)
    
    print(f"Example embedding for first timestep ({timesteps[0]}):")
    print(timestep_embeddings[0])  # shape: [embedding_dim]
