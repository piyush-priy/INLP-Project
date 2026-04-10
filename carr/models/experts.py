import torch
import torch.nn as nn
import torch.nn.functional as F


class VExpertMLP(nn.Module):
    """
    Small 2-layer MLP that transforms value states.
    expert(V) = W2 @ GELU(W1 @ V)
    W2 is zero-initialized so expert output starts at zero.
    """

    def __init__(self, v_dim: int, inner_dim: int):
        super().__init__()
        self.w1 = nn.Linear(v_dim, inner_dim, bias=False)
        self.w2 = nn.Linear(inner_dim, v_dim, bias=False)
        nn.init.zeros_(self.w2.weight)

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(V)))
