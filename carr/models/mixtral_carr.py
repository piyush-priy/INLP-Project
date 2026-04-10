import torch
import logging
from typing import Dict

from carr.core.modules import CARRVProj

logger = logging.getLogger(__name__)


def patch_mixtral_with_carr(
    model,
    num_v_experts: int = 8,
    expert_inner_dim: int = 32,
    probe_dim: int = 8,
    top_k: int = 2,
    alpha_init: float = 0.0,
    scale_capability: bool = True,
    use_shared_expert: bool = False,
    shared_expert_idx: int = 0,
) -> Dict[str, int]:
    """
    Patch a pretrained Mixtral model: replace v_proj in every attention
    layer with CARRVProj (capability-aware V transformation).

    Implements attention-level CARR-Calibrate:
        - ALL original model parameters are frozen.
        - Only new V-experts, router gate, alpha, and LN are trainable.

    The attention equation changes from:
        output = softmax(QK^T) V
    to:
        output = softmax(QK^T) phi(V)
        where phi(V) = V + sum(g_i * expert_i(V))
    """
    cfg = model.config
    v_dim = cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads)

    # Step 1: Freeze entire model
    for param in model.parameters():
        param.requires_grad = False
    logger.info("Froze all model parameters")

    # Step 2: Replace v_proj in each attention layer
    num_replaced = 0
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if not hasattr(attn, "v_proj"):
            continue

        device = next(attn.v_proj.parameters()).device

        carr_v = CARRVProj(
            original_v_proj=attn.v_proj,
            v_dim=v_dim,
            num_experts=num_v_experts,
            expert_inner_dim=expert_inner_dim,
            probe_dim=probe_dim,
            top_k=top_k,
            alpha_init=alpha_init,
            scale_capability=scale_capability,
            use_shared_expert=use_shared_expert,
            shared_expert_idx=shared_expert_idx,
        )
        carr_v.to(device)
        attn.v_proj = carr_v

        num_replaced += 1
        logger.info(f"Replaced v_proj at layer {layer_idx}")

    # Step 3: Ensure correct grad status
    # Original v_proj inside CARRVProj stays frozen (was frozen in step 1).
    # New experts + router are trainable (nn.Module defaults).
    # Explicit pass to be safe:
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        is_carr = any(
            tag in name for tag in [".experts.", ".router."]
        )
        is_original_v = ".v_proj.v_proj." in name
        if is_carr and not is_original_v:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    stats = {
        "num_replaced_layers": num_replaced,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": 100.0 * trainable_params / total_params if total_params else 0,
        "v_dim": v_dim,
    }

    logger.info(
        f"CARR patching complete: {num_replaced} attention layers | "
        f"Trainable: {trainable_params:,} / {total_params:,} "
        f"({stats['trainable_pct']:.4f}%)"
    )
    return stats
