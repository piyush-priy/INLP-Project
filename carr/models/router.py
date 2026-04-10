import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CARRRouter(nn.Module):
    """
    Capability-Aware Residual Router.

    Routing pipeline (per token x):
        1. r_e = x @ W_g.T                       (standard router logits)
        2. c_e = ||W_probe_e @ x||_2 / sqrt(p)   (capability score)
        3. s_e = LN(r_e) + sigmoid(alpha) * LN(c_e)   (fused score)
        4. top-k on s_e
        5. softmax + renormalize over selected experts

    Shapes:
        x:   (tokens, hidden_dim)
        r_e: (tokens, num_experts)
        c_e: (tokens, num_experts)
        s_e: (tokens, num_experts)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        probe_dim: int,
        top_k: int = 2,
        alpha_init: float = 0.0,
        scale_capability: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.probe_dim = probe_dim
        self.top_k = top_k
        self.scale_capability = scale_capability

        # --- Trainable components ---
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # LN across the expert dimension (last dim of shape (T, E))
        self.ln_router = nn.LayerNorm(num_experts)
        self.ln_capability = nn.LayerNorm(num_experts)

        # --- Frozen probe buffer (set after extraction) ---
        self.register_buffer(
            "W_probe", torch.zeros(num_experts, probe_dim, hidden_dim)
        )

    # ------------------------------------------------------------------
    def set_probes(self, probes: torch.Tensor):
        """Store extracted probes as a frozen buffer.
        probes: (num_experts, probe_dim, hidden_dim) fp16
        """
        expected = (self.num_experts, self.probe_dim, self.hidden_dim)
        assert probes.shape == expected, (
            f"Expected probe shape {expected}, got {probes.shape}"
        )
        self.W_probe.copy_(probes)

    # ------------------------------------------------------------------
    def compute_capability_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        c_e(x) = ||W_probe_e @ x||_2 / sqrt(p)

        Args:   x  – (T, D)
        Returns: c_e – (T, E)
        """
        # projected[e, t, :] = W_probe[e] @ x[t]  → (E, T, P)
        projected = torch.einsum(
            "epd,td->etp", self.W_probe.float(), x.float()
        )
        scores = torch.norm(projected, dim=2)       # (E, T)
        scores = scores.transpose(0, 1)             # (T, E)

        if self.scale_capability:
            scores = scores / math.sqrt(self.probe_dim)

        return scores

    # ------------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: (tokens, hidden_dim)

        Returns:
            routing_weights:  (tokens, top_k)
            selected_experts: (tokens, top_k)
            fused_scores:     (tokens, num_experts)
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()             # fp16 → fp32 for gate + LN

        r_e = self.gate(hidden_states)                          # (T, E)
        c_e = self.compute_capability_scores(hidden_states)     # (T, E)

        gate_value = torch.sigmoid(self.alpha)
        s_e = self.ln_router(r_e) + gate_value * self.ln_capability(c_e)

        routing_weights = F.softmax(s_e, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        )
        routing_weights = routing_weights.to(input_dtype)

        return routing_weights, selected_experts, s_e
