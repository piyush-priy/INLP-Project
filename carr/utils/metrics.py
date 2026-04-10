import math
import torch
import torch.nn.functional as F
from typing import Dict, List
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ── Primitive metric functions ────────────────────────────────────────

def compute_load_entropy(expert_counts: torch.Tensor) -> float:
    """H = -sum(p_e * log2(p_e)).  Higher = better balanced."""
    p = expert_counts.float() / expert_counts.sum()
    p = p[p > 0]
    return -(p * torch.log2(p)).sum().item()


def compute_cov(expert_counts: torch.Tensor) -> float:
    """CoV = std / mean of expert load counts.  Lower = better balanced."""
    c = expert_counts.float()
    mean = c.mean()
    return (c.std() / mean).item() if mean > 0 else 0.0


def compute_jaccard_overlap(expert_token_sets: List[set]) -> float:
    """Mean pairwise Jaccard similarity.  Lower = better specialised."""
    n = len(expert_token_sets)
    if n < 2:
        return 0.0
    total, pairs = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            union = len(expert_token_sets[i] | expert_token_sets[j])
            if union > 0:
                total += len(expert_token_sets[i] & expert_token_sets[j]) / union
            pairs += 1
    return total / pairs


# ── Routing decision collector ────────────────────────────────────────

@torch.no_grad()
def collect_routing_decisions(model, dataloader, num_batches: int = 30) -> Dict:
    """
    Run forward passes and collect routing decisions from all CARRVProj layers.

    Returns:
        {layer_idx: {
            "counts": Tensor(num_experts),      # total tokens per expert
            "token_sets": [set(), ...],          # which tokens each expert saw
        }}
    """
    from carr.core.modules import CARRVProj

    model.eval()

    # Identify CARR layers
    carr_layers = {}
    for lidx, layer in enumerate(model.model.layers):
        vp = layer.self_attn.v_proj
        if isinstance(vp, CARRVProj):
            num_e = vp.num_experts
            carr_layers[lidx] = {
                "counts": torch.zeros(num_e),
                "token_sets": [set() for _ in range(num_e)],
                "module": vp,
            }

    global_token_offset = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        batch = {k: v.to(model.device) for k, v in batch.items()}
        _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        batch_tokens = batch["input_ids"].numel()

        for lidx, info in carr_layers.items():
            vp = info["module"]
            if vp._last_selected is None:
                continue

            selected = vp._last_selected.cpu()  # (T, top_k)

            for eidx in range(info["counts"].shape[0]):
                mask = (selected == eidx).any(dim=1)
                count = mask.sum().item()
                info["counts"][eidx] += count

                token_ids = torch.where(mask)[0] + global_token_offset
                info["token_sets"][eidx].update(token_ids.tolist())

        global_token_offset += batch_tokens

    # Strip module refs before returning
    return {
        lidx: {"counts": info["counts"], "token_sets": info["token_sets"]}
        for lidx, info in carr_layers.items()
    }


# ── Aggregate routing metrics ─────────────────────────────────────────

@torch.no_grad()
def compute_routing_metrics(model, dataloader, num_batches: int = 30) -> Dict:
    """
    Compute full routing metrics across all CARR layers:
        - load_entropy (per-layer + mean)
        - cov          (per-layer + mean)
        - jaccard      (per-layer + mean)
        - expert_usage (per-layer counts)
    """
    decisions = collect_routing_decisions(model, dataloader, num_batches)

    per_layer = {}
    entropies, covs, jaccards = [], [], []

    for lidx in sorted(decisions.keys()):
        info = decisions[lidx]
        counts = info["counts"]
        token_sets = info["token_sets"]

        ent = compute_load_entropy(counts)
        cov = compute_cov(counts)
        jac = compute_jaccard_overlap(token_sets)

        per_layer[lidx] = {
            "entropy": ent,
            "cov": cov,
            "jaccard": jac,
            "expert_usage": counts.tolist(),
        }

        entropies.append(ent)
        covs.append(cov)
        jaccards.append(jac)

        logger.info(
            f"  Layer {lidx:2d}  "
            f"H={ent:.3f}  CoV={cov:.3f}  Jaccard={jac:.3f}  "
            f"usage={[int(c) for c in counts.tolist()]}"
        )

    return {
        "load_entropy": sum(entropies) / len(entropies) if entropies else 0,
        "cov": sum(covs) / len(covs) if covs else 0,
        "jaccard": sum(jaccards) / len(jaccards) if jaccards else 0,
        "per_layer": per_layer,
    }


# ── Perplexity ────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, dataloader, num_batches: int = 100) -> float:
    """Compute validation perplexity."""
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
        )
        total_loss += outputs.loss.item() * batch["input_ids"].numel()
        total_tokens += batch["input_ids"].numel()

    avg = total_loss / total_tokens if total_tokens > 0 else 0
    return math.exp(avg)
