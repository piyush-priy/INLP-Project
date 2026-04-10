"""CLI entry point for CARR-Calibrate.  Usage: python scripts/run_calibrate.py --config configs/default.yaml"""

import argparse, yaml, sys, os, logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carr.utils.model_utils import load_mixtral_4bit, print_trainable_summary
from carr.models.mixtral_carr import patch_mixtral_with_carr
from carr.utils.data_utils import load_calibration_data
from carr.trainer.calibrator import CARRCalibrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")


def main():
    parser = argparse.ArgumentParser(description="CARR-Calibrate")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    m, c, cal = cfg["model"], cfg["carr"], cfg["calibration"]

    model, tokenizer = load_mixtral_4bit(model_name=m["name"], torch_dtype=m.get("torch_dtype", "float16"))

    patch_mixtral_with_carr(
        model, num_v_experts=c["num_v_experts"], expert_inner_dim=c["expert_inner_dim"],
        probe_dim=c["probe_dim"], top_k=c["top_k"], alpha_init=c["alpha_init"],
        scale_capability=c.get("scale_capability", True),
        use_shared_expert=c.get("use_shared_expert", False),
        shared_expert_idx=c.get("shared_expert_idx", 0),
    )
    print_trainable_summary(model)

    train_loader, val_loader = load_calibration_data(
        tokenizer, dataset_name=cal.get("dataset_name", "wikitext"),
        dataset_config=cal.get("dataset_config", "wikitext-103-raw-v1"),
        max_seq_length=cal.get("max_seq_length", 512),
        max_tokens=cal.get("max_calibration_tokens", 50000), batch_size=cal.get("batch_size", 2),
    )

    calibrator = CARRCalibrator(
        model=model, tokenizer=tokenizer, learning_rate=cal["learning_rate"],
        weight_decay=cal.get("weight_decay", 0.01), num_epochs=cal["num_epochs"],
        gradient_accumulation_steps=cal.get("gradient_accumulation_steps", 4),
        output_dir=cal.get("output_dir", "./carr_output"), seed=cal.get("seed", 42),
    )
    history = calibrator.calibrate(train_loader, val_loader)
    print("Calibration complete.")


if __name__ == "__main__":
    main()
