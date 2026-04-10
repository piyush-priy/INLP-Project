"""Evaluate a CARR-calibrated model.  Usage: python scripts/run_eval.py --config configs/default.yaml --checkpoint carr_output/final"""

import argparse, yaml, sys, os, torch, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carr.utils.model_utils import load_mixtral_4bit, print_trainable_summary
from carr.models.mixtral_carr import patch_mixtral_with_carr
from carr.utils.data_utils import load_calibration_data
from carr.utils.metrics import compute_perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("carr.eval")


def main():
    parser = argparse.ArgumentParser(description="CARR Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_batches", type=int, default=100)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    m, c, ev = cfg["model"], cfg["carr"], cfg["evaluation"]

    model, tokenizer = load_mixtral_4bit(model_name=m["name"])
    patch_mixtral_with_carr(
        model, num_v_experts=c["num_v_experts"], expert_inner_dim=c["expert_inner_dim"],
        probe_dim=c["probe_dim"], top_k=c["top_k"],
        use_shared_expert=c.get("use_shared_expert", False),
    )

    if args.checkpoint:
        state = torch.load(os.path.join(args.checkpoint, "carr_state.pt"), map_location="cpu")
        for name, param in model.named_parameters():
            if name in state:
                param.data.copy_(state[name].to(param.device))
        logger.info(f"Loaded CARR state from {args.checkpoint}")

    _, val_loader = load_calibration_data(
        tokenizer, batch_size=ev.get("batch_size", 4), max_seq_length=ev.get("max_seq_length", 512),
    )

    ppl = compute_perplexity(model, val_loader, num_batches=args.num_batches)
    logger.info(f"Perplexity: {ppl:.2f}")

    alphas = [torch.sigmoid(p).item() for n, p in model.named_parameters() if "router.alpha" in n]
    if alphas:
        logger.info(f"Alpha (sigmoid): mean={sum(alphas)/len(alphas):.4f}  [{min(alphas):.4f}, {max(alphas):.4f}]")

    print_trainable_summary(model)


if __name__ == "__main__":
    main()
