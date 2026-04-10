import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from carr.models.router import CARRRouter
from carr.models.experts import VExpertMLP
from carr.core.probe import extract_probes_from_v_experts


class CARRVProj(nn.Module):
    """
    Replaces v_proj in attention with CARR-routed V transformation.

    Pipeline per token:
        1. V = original_v_proj(x)
        2. Route V:  r_e = gate(V),  c_e = ||W_probe_e @ V||_2 / sqrt(p)
        3. Fuse:     s_e = LN(r_e) + sigmoid(alpha) * LN(c_e)
        4. Top-k on s_e
        5. V_dynamic = V + sum(g_i * expert_i(V))   (residual)

    Shapes:
        x:         (batch, seq, hidden_dim)
        V:         (batch, seq, v_dim)
        V_dynamic: (batch, seq, v_dim)

    The rest of attention (Q, K, RoPE, softmax) is untouched.
    """

    def __init__(
        self,
        original_v_proj: nn.Module,
        v_dim: int,
        num_experts: int = 8,
        expert_inner_dim: int = 32,
        probe_dim: int = 8,
        top_k: int = 2,
        alpha_init: float = 0.0,
        scale_capability: bool = True,
        use_shared_expert: bool = False,
        shared_expert_idx: int = 0,
    ):
        super().__init__()

        self.v_proj = original_v_proj  # frozen
        self.v_dim = v_dim
        self.num_experts = num_experts
        self.use_shared_expert = use_shared_expert
        self.shared_expert_idx = shared_expert_idx

        # Routing decision tracking (for metrics — detached, no grad)
        self._last_selected = None       # (tokens, top_k)
        self._last_routing_weights = None # (tokens, top_k)

        # V-transformation experts
        self.experts = nn.ModuleList(
            [VExpertMLP(v_dim, expert_inner_dim) for _ in range(num_experts)]
        )

        # Routed expert indices
        if use_shared_expert:
            self.routed_indices = [
                i for i in range(num_experts) if i != shared_expert_idx
            ]
        else:
            self.routed_indices = list(range(num_experts))

        num_routed = len(self.routed_indices)

        # CARR router (operates on V, routes to V-experts)
        self.router = CARRRouter(
            hidden_dim=v_dim,
            num_experts=num_routed,
            probe_dim=probe_dim,
            top_k=top_k,
            alpha_init=alpha_init,
            scale_capability=scale_capability,
        )

        # Extract initial probes from V-expert w1 weights
        routed_experts = nn.ModuleList(
            [self.experts[i] for i in self.routed_indices]
        )
        probes = extract_probes_from_v_experts(routed_experts, probe_dim)
        self.router.set_probes(probes)

    def refresh_probes(self):
        """Re-extract probes from current V-expert weights (call periodically)."""
        routed_experts = nn.ModuleList(
            [self.experts[i] for i in self.routed_indices]
        )
        probes = extract_probes_from_v_experts(routed_experts, self.router.probe_dim)
        self.router.set_probes(probes.to(self.router.W_probe.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  x — (batch, seq, hidden_dim)  [input to original v_proj]
        Returns:   V_dynamic — (batch, seq, v_dim)
        """
        V = self.v_proj(x)                              # (B, S, v_dim)
        orig_shape = V.shape
        orig_dtype = V.dtype
        V_flat = V.view(-1, self.v_dim).float()         # (T, v_dim) — fp32 for CARR ops
        num_tokens = V_flat.shape[0]

        # CARR routing on V
        routing_weights, selected, _ = self.router(V_flat)
        # routing_weights: (T, top_k)  selected: (T, top_k)

        # Store for metric collection (detached — no grad impact)
        self._last_selected = selected.detach()
        self._last_routing_weights = routing_weights.detach()

        # Map router indices to actual expert indices
        routed_map = torch.tensor(
            self.routed_indices, device=selected.device, dtype=selected.dtype
        )
        actual = routed_map[selected]                   # (T, top_k)

        # Compute weighted expert outputs (residual delta)
        V_delta = torch.zeros_like(V_flat)

        expert_mask = F.one_hot(
            actual, num_classes=self.num_experts
        ).permute(2, 1, 0)                              # (E, top_k, T)

        for eidx in range(self.num_experts):
            if self.use_shared_expert and eidx == self.shared_expert_idx:
                continue
            idx, top_x = torch.where(expert_mask[eidx])
            if top_x.numel() == 0:
                continue
            out = self.experts[eidx](V_flat[top_x])
            weighted = out * routing_weights[top_x, idx, None].float()
            V_delta.index_add_(0, top_x, weighted)

        # Shared expert (always active, outside routing)
        if self.use_shared_expert:
            V_delta = V_delta + self.experts[self.shared_expert_idx](V_flat)

        # Residual: V_dynamic = V + expert contributions
        V_dynamic = V_flat + V_delta
        return V_dynamic.to(orig_dtype).view(orig_shape)
