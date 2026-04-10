"""
CARR-Calibrate on Mixtral 8x7B (4-bit) — Colab Quickstart

Colab setup:
    !pip install -q torch transformers bitsandbytes accelerate datasets tqdm pyyaml
    # Upload / clone the repo, then:
    !python colab_quickstart.py
"""

import os, sys, time, logging, argparse, torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("carr.colab")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from carr.utils.model_utils import load_mixtral_4bit, print_trainable_summary
from carr.models.mixtral_carr import patch_mixtral_with_carr
from carr.utils.data_utils import load_calibration_data
from carr.trainer.calibrator import CARRCalibrator
from carr.utils.metrics import compute_perplexity, compute_routing_metrics

# ── CONFIG ────────────────────────────────────────────────────────────
MODEL_NAME    = "mistralai/Mistral-7B-v0.1"
NUM_V_EXPERTS = 8
INNER_DIM     = 32
PROBE_DIM     = 8
TOP_K         = 2
LR            = 1e-4
NUM_EPOCHS    = 3
BATCH_SIZE    = 2
GRAD_ACCUM    = 4
MAX_SEQ_LEN   = 512
MAX_TOKENS    = 50_000
OUTPUT_DIR    = "./carr_output"


def parse_args():
    parser = argparse.ArgumentParser(description="CARR-Calibrate on Mixtral 8x7B")
    parser.add_argument(
        "--debug", action="store_true",
        help="Fast end-to-end test: 1 epoch, 1k tokens, seq_len=128.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    wall_start = time.time()

    # Apply debug overrides
    global NUM_EPOCHS, MAX_TOKENS, MAX_SEQ_LEN, GRAD_ACCUM
    if args.debug:
        logger.warning("\n" + "\u26a1" * 30)
        logger.warning("  DEBUG MODE \u2014 1 epoch, 1k tokens, seq_len=128")
        logger.warning("\u26a1" * 30 + "\n")
        NUM_EPOCHS  = 1
        MAX_TOKENS  = 1000
        MAX_SEQ_LEN = 128
        GRAD_ACCUM  = 1

    num_baseline_batches = 5 if args.debug else 50
    num_eval_batches     = 5 if args.debug else 100
    num_routing_batches  = 3 if args.debug else 20
    num_final_routing    = 3 if args.debug else 30

    # ── 1. Load ───────────────────────────────────────────────────────
    logger.info("=" * 60 + "\n  Step 1: Loading Mixtral 8x7B in 4-bit\n" + "=" * 60)
    model, tokenizer = load_mixtral_4bit(model_name=MODEL_NAME)

    # ── 2. Patch ──────────────────────────────────────────────────────
    logger.info("=" * 60 + "\n  Step 2: Patching attention with CARR V-experts\n" + "=" * 60)
    stats = patch_mixtral_with_carr(
        model, num_v_experts=NUM_V_EXPERTS, expert_inner_dim=INNER_DIM,
        probe_dim=PROBE_DIM, top_k=TOP_K,
    )
    print_trainable_summary(model)

    # ── 3. Sanity: probe norms ────────────────────────────────────────
    logger.info("=" * 60 + "\n  Step 3: Sanity — V-expert probe norms\n" + "=" * 60)
    for lidx, layer in enumerate(model.model.layers):
        vp = layer.self_attn.v_proj
        if hasattr(vp, "router") and hasattr(vp.router, "W_probe"):
            norms = vp.router.W_probe.float().norm(dim=(1, 2))
            logger.info(f"  Layer {lidx:2d}  mean={norms.mean():.4f}  std={norms.std():.4f}")
            if norms.std() < 1e-6:
                logger.warning(f"  ⚠ Layer {lidx} probes identical!")

    # ── 4. Data ───────────────────────────────────────────────────────
    logger.info("=" * 60 + "\n  Step 4: Loading calibration data\n" + "=" * 60)
    train_loader, val_loader = load_calibration_data(
        tokenizer, max_seq_length=MAX_SEQ_LEN, max_tokens=MAX_TOKENS, batch_size=BATCH_SIZE,
    )

    # ── 5. Baseline perplexity + routing ──────────────────────────────
    logger.info("=" * 60 + "\n  Step 5: Baseline metrics (before calibration)\n" + "=" * 60)
    baseline_ppl = compute_perplexity(model, val_loader, num_batches=num_baseline_batches)
    logger.info(f"  Baseline perplexity: {baseline_ppl:.2f}")

    logger.info("  Baseline routing metrics:")
    baseline_routing = compute_routing_metrics(model, val_loader, num_batches=num_routing_batches)
    logger.info(
        f"  H={baseline_routing['load_entropy']:.3f}  "
        f"CoV={baseline_routing['cov']:.3f}  "
        f"Jaccard={baseline_routing['jaccard']:.3f}"
    )

    # ── 6. Calibrate ──────────────────────────────────────────────────
    logger.info("=" * 60 + "\n  Step 6: Running CARR-Calibrate\n" + "=" * 60)
    calibrator = CARRCalibrator(
        model=model, tokenizer=tokenizer, learning_rate=LR,
        num_epochs=NUM_EPOCHS, gradient_accumulation_steps=GRAD_ACCUM,
        output_dir=OUTPUT_DIR, probe_refresh_epochs=1,
    )
    history = calibrator.calibrate(train_loader, val_loader)

    # ── 7. Final comparison ───────────────────────────────────────────
    logger.info("=" * 60 + "\n  Step 7: Final evaluation\n" + "=" * 60)
    final_ppl = compute_perplexity(model, val_loader, num_batches=num_eval_batches)
    final_routing = compute_routing_metrics(model, val_loader, num_batches=num_final_routing)

    improvement = (1 - final_ppl / baseline_ppl) * 100

    logger.info("\n" + "=" * 60)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Perplexity:  baseline={baseline_ppl:.2f}  final={final_ppl:.2f}  ({improvement:+.2f}%)")
    logger.info(f"  Entropy:     baseline={baseline_routing['load_entropy']:.3f}  final={final_routing['load_entropy']:.3f}")
    logger.info(f"  CoV:         baseline={baseline_routing['cov']:.3f}  final={final_routing['cov']:.3f}")
    logger.info(f"  Jaccard:     baseline={baseline_routing['jaccard']:.3f}  final={final_routing['jaccard']:.3f}")
    logger.info(f"  Trainable:   {stats['trainable_params']:,} / {stats['total_params']:,} ({stats['trainable_pct']:.4f}%)")
    logger.info(f"  Wall time:   {history['wall_time_seconds']:.1f}s ({history['wall_time_seconds']/60:.1f} min)")

    alphas = [torch.sigmoid(p).item() for n, p in model.named_parameters() if "router.alpha" in n]
    if alphas:
        logger.info(f"  Alpha:       mean={sum(alphas)/len(alphas):.4f}  [{min(alphas):.4f}, {max(alphas):.4f}]")

    logger.info("=" * 60)
    logger.info(f"  Checkpoint: {OUTPUT_DIR}")
    logger.info(f"  Total time: {time.time() - wall_start:.1f}s")


if __name__ == "__main__":
    main()
