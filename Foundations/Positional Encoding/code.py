## Code: Sinusoidal PE

import numpy as np
import torch

def sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    """
    Returns positional encoding matrix of shape (1, seq_len, d_model).
    Ready to broadcast over a batch.
    """
    PE = torch.zeros(seq_len, d_model)
    positions = torch.arange(seq_len).unsqueeze(1).float()         # (seq_len, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
    )                                                               # (d_model/2,)

    PE[:, 0::2] = torch.sin(positions * div_term)   # even dims
    PE[:, 1::2] = torch.cos(positions * div_term)   # odd dims

    return PE.unsqueeze(0)   # (1, seq_len, d_model)

# Usage
pe = sinusoidal_pe(seq_len=50, d_model=512)
print(pe.shape)   # torch.Size([1, 50, 512])
