"""
Run CARR-Calibrate across all 4 modes (or a single mode).

Usage:
    python scripts/run_all_modes.py                          # run all 4 modes
    python scripts/run_all_modes.py --mode full_carr         # run one mode
    python scripts/run_all_modes.py --results_dir ./results  # custom output dir
    python scripts/run_all_modes.py --debug                  # fast e2e test
"""

import argparse
import json
import math
import os
import sys
import time
import logging
import yaml
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carr.utils.model_utils import load_mixtral_4bit, print_trainable_summary
from carr.models.mixtral_carr import patch_mixtral_with_carr
from carr.utils.data_utils import load_calibration_data
from carr.trainer.calibrator import CARRCalibrator
from carr.utils.metrics import compute_perplexity, compute_routing_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("carr.runner")

# ── Mode definitions ─────────────────────────────────────────────────
MODE_CONFIGS = {
    "gate_only":     "configs/mode2_gate_only.yaml",
    "full_carr":     "configs/mode3_full_carr.yaml",
    "shared_expert": "configs/mode4_shared_expert.yaml",
}

MODE_DISPLAY = {
    "gate_only":     "Mode 1: Gate-Only CARR",
    "full_carr":     "Mode 2: Full CARR (Gate + Capability)",
    "shared_expert": "Mode 3: Full CARR + Shared Expert",
}

# ── Debug overrides ──────────────────────────────────────────────────
DEBUG_OVERRIDES = {
    "calibration": {
        "num_epochs": 1,
        "max_calibration_tokens": 1000,
        "max_seq_length": 128,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "logging_steps": 1,
        "save_steps": 0,           # disable intermediate checkpoints
        "warmup_steps": 2,
    },
    "evaluation": {
        "num_eval_batches": 5,
        "max_seq_length": 128,
    },
}


def apply_debug_overrides(cfg):
    """Override config values for fast end-to-end testing."""
    for section, overrides in DEBUG_OVERRIDES.items():
        if section not in cfg:
            cfg[section] = {}
        for key, val in overrides.items():
            cfg[section][key] = val
    logger.warning("⚡ DEBUG MODE: using tiny dataset (1k tokens, 1 epoch, seq_len=128)")
    return cfg



def run_carr_mode(cfg, results_dir):
    """Run a CARR-patched mode: load → patch → calibrate → evaluate."""
    mode_name = cfg["mode"]
    m, c, cal = cfg["model"], cfg["carr"], cfg["calibration"]

    logger.info("=" * 60)
    logger.info(f"  {MODE_DISPLAY.get(mode_name, mode_name)}")
    logger.info("=" * 60)

    # 1. Load model
    model, tokenizer = load_mixtral_4bit(model_name=m["name"])

    # 2. Patch with CARR
    stats = patch_mixtral_with_carr(
        model,
        num_v_experts=c["num_v_experts"],
        expert_inner_dim=c["expert_inner_dim"],
        probe_dim=c["probe_dim"],
        top_k=c["top_k"],
        alpha_init=c.get("alpha_init", 0.0),
        scale_capability=c.get("scale_capability", True),
        use_shared_expert=c.get("use_shared_expert", False),
        shared_expert_idx=c.get("shared_expert_idx", 0),
    )
    print_trainable_summary(model)

    # 3. Load data
    train_loader, val_loader = load_calibration_data(
        tokenizer,
        dataset_name=cal.get("dataset_name", "wikitext"),
        dataset_config=cal.get("dataset_config", "wikitext-103-raw-v1"),
        max_seq_length=cal.get("max_seq_length", 512),
        max_tokens=cal.get("max_calibration_tokens", 50000),
        batch_size=cal.get("batch_size", 2),
    )

    # 4. Pre-calibration metrics
    logger.info("  Computing baseline metrics (before calibration)...")
    baseline_ppl = compute_perplexity(model, val_loader, num_batches=50)
    baseline_routing = compute_routing_metrics(model, val_loader, num_batches=20)
    logger.info(f"  Baseline perplexity: {baseline_ppl:.2f}")

    # 5. Calibrate
    calibrator = CARRCalibrator(
        model=model,
        tokenizer=tokenizer,
        learning_rate=cal["learning_rate"],
        weight_decay=cal.get("weight_decay", 0.01),
        num_epochs=cal["num_epochs"],
        gradient_accumulation_steps=cal.get("gradient_accumulation_steps", 4),
        output_dir=results_dir,
        seed=cal.get("seed", 42),
        probe_refresh_epochs=c.get("probe_refresh_epochs", 1),
        mode=mode_name,
    )
    history = calibrator.calibrate(train_loader, val_loader)

    # 6. Final evaluation
    num_eval = cfg.get("evaluation", {}).get("num_eval_batches", 100)
    final_ppl = compute_perplexity(model, val_loader, num_batches=num_eval)
    final_routing = compute_routing_metrics(model, val_loader, num_batches=30)

    # 7. Append final comparison to history
    history["baseline_metrics"] = {
        "perplexity": baseline_ppl,
        "load_entropy": baseline_routing["load_entropy"],
        "cov": baseline_routing["cov"],
        "jaccard": baseline_routing["jaccard"],
    }
    history["final_metrics"] = {
        "perplexity": final_ppl,
        "load_entropy": final_routing["load_entropy"],
        "cov": final_routing["cov"],
        "jaccard": final_routing["jaccard"],
    }
    history["trainable_params"] = stats["trainable_params"]
    history["total_params"] = stats["total_params"]
    history["trainable_pct"] = stats["trainable_pct"]

    improvement = (1 - final_ppl / baseline_ppl) * 100

    logger.info("\n" + "=" * 60)
    logger.info(f"  {mode_name.upper()} RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Perplexity:  baseline={baseline_ppl:.2f}  final={final_ppl:.2f}  ({improvement:+.2f}%)")
    logger.info(f"  Entropy:     baseline={baseline_routing['load_entropy']:.3f}  final={final_routing['load_entropy']:.3f}")
    logger.info(f"  CoV:         baseline={baseline_routing['cov']:.3f}  final={final_routing['cov']:.3f}")
    logger.info(f"  Jaccard:     baseline={baseline_routing['jaccard']:.3f}  final={final_routing['jaccard']:.3f}")

    # Re-save history with final metrics
    hist_path = os.path.join(results_dir, "history.json")

    # Serialize routing metrics safely
    serializable_routing = []
    for epoch_metrics in history.get("routing_metrics", []):
        entry = {
            "load_entropy": epoch_metrics.get("load_entropy", 0),
            "cov": epoch_metrics.get("cov", 0),
            "jaccard": epoch_metrics.get("jaccard", 0),
        }
        if "per_layer" in epoch_metrics:
            per_layer_safe = {}
            for lidx, ldata in epoch_metrics["per_layer"].items():
                per_layer_safe[str(lidx)] = {
                    "entropy": ldata["entropy"],
                    "cov": ldata["cov"],
                    "jaccard": ldata["jaccard"],
                    "expert_usage": [
                        int(x) if isinstance(x, (int, float)) else x
                        for x in ldata.get("expert_usage", [])
                    ],
                }
            entry["per_layer"] = per_layer_safe
        serializable_routing.append(entry)

    save_history = dict(history)
    save_history["routing_metrics"] = serializable_routing

    with open(hist_path, "w") as f:
        json.dump(save_history, f, indent=2)
    logger.info(f"  History saved: {hist_path}")

    # Cleanup
    del model, calibrator
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return history


def main():
    parser = argparse.ArgumentParser(description="Run CARR in all 4 modes")
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=list(MODE_CONFIGS.keys()),
        help="Run a single mode. If not specified, runs all 4.",
    )
    parser.add_argument(
        "--results_dir", type=str, default="./carr_output",
        help="Base directory for results (each mode gets a subdirectory).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Fast end-to-end test: 1 epoch, 1k tokens, seq_len=128.",
    )
    args = parser.parse_args()

    if args.debug:
        logger.warning("\n" + "⚡" * 30)
        logger.warning("  DEBUG MODE ENABLED — tiny dataset, 1 epoch")
        logger.warning("⚡" * 30 + "\n")

    modes = [args.mode] if args.mode else list(MODE_CONFIGS.keys())
    all_results = {}

    total_start = time.time()

    for mode_name in modes:
        config_path = MODE_CONFIGS[mode_name]
        logger.info(f"\n{'#' * 60}")
        logger.info(f"  Starting: {MODE_DISPLAY[mode_name]}")
        logger.info(f"  Config:   {config_path}")
        logger.info(f"{'#' * 60}\n")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        if args.debug:
            cfg = apply_debug_overrides(cfg)

        mode_dir = os.path.join(args.results_dir, mode_name)

        history = run_carr_mode(cfg, mode_dir)

        all_results[mode_name] = history
        logger.info(f"  ✓ {mode_name} complete\n")

    total_time = time.time() - total_start

    # Print final summary table
    logger.info("\n" + "=" * 80)
    logger.info("  FINAL COMPARISON ACROSS ALL MODES")
    logger.info("=" * 80)
    logger.info(f"  {'Mode':<20} {'Perplexity':>12} {'Entropy':>10} {'CoV':>10} {'Jaccard':>10}")
    logger.info("  " + "-" * 62)

    for mode_name, hist in all_results.items():
        fm = hist.get("final_metrics", {})
        ppl = fm.get("perplexity", None)
        ent = fm.get("load_entropy", None)
        cov = fm.get("cov", None)
        jac = fm.get("jaccard", None)

        ppl_s = f"{ppl:.2f}" if ppl is not None else "N/A"
        ent_s = f"{ent:.3f}" if ent is not None else "N/A"
        cov_s = f"{cov:.3f}" if cov is not None else "N/A"
        jac_s = f"{jac:.3f}" if jac is not None else "N/A"

        logger.info(f"  {mode_name:<20} {ppl_s:>12} {ent_s:>10} {cov_s:>10} {jac_s:>10}")

    logger.info("=" * 80)
    logger.info(f"  Total wall time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  Results directory: {args.results_dir}")
    logger.info(f"  Run 'python scripts/plot_comparison.py --results_dir {args.results_dir}' to generate plots")


if __name__ == "__main__":
    main()
