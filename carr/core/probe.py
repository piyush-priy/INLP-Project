import torch
import torch.nn as nn
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def dequantize_weight(linear_layer) -> torch.Tensor:
    """Dequantize a potentially 4-bit quantized linear layer weight."""
    try:
        import bitsandbytes as bnb
        if isinstance(linear_layer, bnb.nn.Linear4bit):
            return bnb.functional.dequantize_4bit(
                linear_layer.weight.data, linear_layer.weight.quant_state
            ).float().cpu()
    except ImportError:
        pass
    return linear_layer.weight.data.float().cpu()


def extract_probes_from_v_experts(
    experts: nn.ModuleList, probe_dim: int
) -> torch.Tensor:
    """
    Extract frozen capability probes from V-expert w1 weights.
    w1 is the first linear layer capturing input feature sensitivity.
    Probes shape: (num_experts, probe_dim, v_dim) in fp16.
    """
    probes = []
    for idx, expert in enumerate(experts):
        w1 = expert.w1.weight.data.float()  # (inner_dim, v_dim)
        probe = w1[:probe_dim, :].clone().half()
        probes.append(probe)
        logger.debug(f"Probe from V-expert {idx}: norm={probe.float().norm():.4f}")
    stacked = torch.stack(probes)
    logger.info(f"Extracted {len(probes)} V-expert probes: shape={stacked.shape}")
    return stacked
