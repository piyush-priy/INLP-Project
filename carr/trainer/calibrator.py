import os
import json
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from tqdm import tqdm
from typing import Optional, Dict, List

from carr.utils.metrics import compute_perplexity, compute_routing_metrics

logger = logging.getLogger(__name__)


class CARRCalibrator:
    """
    CARR-Calibrate trainer.

    Only updates V-expert weights, router gate, alpha, and LN params
    on a fully-frozen pretrained backbone.

    No auxiliary load-balancing loss — only standard LM cross-entropy.
    Logs: alpha convergence, routing metrics, timing, param counts.
    """

    def __init__(
        self,
        model,
        tokenizer,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        warmup_steps: int = 50,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        logging_steps: int = 10,
        save_steps: int = 500,
        output_dir: str = "./carr_output",
        seed: int = 42,
        probe_refresh_epochs: int = 0,
        mode: str = "full_carr",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.probe_refresh_epochs = probe_refresh_epochs
        self.mode = mode

        os.makedirs(output_dir, exist_ok=True)
        torch.manual_seed(seed)

        self.trainable_params = [p for p in model.parameters() if p.requires_grad]
        self._trainable_count = sum(p.numel() for p in self.trainable_params)
        self._total_count = sum(p.numel() for p in model.parameters())

        logger.info(
            f"Trainable: {self._trainable_count:,} / {self._total_count:,} "
            f"({100 * self._trainable_count / self._total_count:.4f}%)"
        )

        self.optimizer = AdamW(self.trainable_params, lr=learning_rate, weight_decay=weight_decay)
        self.global_step = 0
        self.best_val_loss = float("inf")

    # ── Alpha tracking ────────────────────────────────────────────────
    def _get_alpha_values(self) -> List[float]:
        return [
            torch.sigmoid(p).item()
            for n, p in self.model.named_parameters()
            if "router.alpha" in n
        ]

    def _log_alpha_analysis(self, tag: str = ""):
        alphas = self._get_alpha_values()
        if not alphas:
            return
        mean_a = sum(alphas) / len(alphas)
        logger.info(
            f"  Alpha analysis {tag}: "
            f"mean={mean_a:.4f}  min={min(alphas):.4f}  max={max(alphas):.4f}  "
            f"(paper expects ~0.49-0.50)"
        )

    # ── Probe refresh ─────────────────────────────────────────────────
    def _refresh_probes(self):
        from carr.core.modules import CARRVProj
        for layer in self.model.model.layers:
            vp = layer.self_attn.v_proj
            if isinstance(vp, CARRVProj):
                vp.refresh_probes()
        logger.info("  Probes refreshed from current V-expert weights")

    # ── Checkpointing ─────────────────────────────────────────────────
    def _save_checkpoint(self, tag: str):
        path = os.path.join(self.output_dir, tag)
        os.makedirs(path, exist_ok=True)

        state = {
            name: param.data.cpu()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        torch.save(state, os.path.join(path, "carr_state.pt"))

        meta = {
            "global_step": self.global_step,
            "alpha_values": self._get_alpha_values(),
            "trainable_params": self._trainable_count,
            "total_params": self._total_count,
            "trainable_pct": 100 * self._trainable_count / self._total_count,
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Checkpoint saved: {path}")

    # ── Main calibration loop ─────────────────────────────────────────
    def calibrate(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Dict:
        total_steps = (
            len(train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
        )
        scheduler = CosineAnnealingLR(self.optimizer, T_max=max(total_steps, 1), eta_min=1e-6)

        history = {
            "mode": self.mode,
            "train_loss": [], "val_perplexity": [],
            "routing_metrics": [], "alpha_history": [],
            "wall_time_seconds": 0,
        }

        cal_start = time.time()
        self.model.train()

        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            epoch_loss, num_batches = 0.0, 0

            progress = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

            for step, batch in enumerate(progress):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["input_ids"],
                )

                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % self.logging_steps == 0:
                        cur = loss.item() * self.gradient_accumulation_steps
                        alphas = self._get_alpha_values()
                        progress.set_postfix({
                            "loss": f"{cur:.4f}",
                            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                            "α": f"{sum(alphas)/len(alphas):.3f}" if alphas else "?",
                        })
                        history["train_loss"].append({"step": self.global_step, "loss": cur})

                    if self.save_steps > 0 and self.global_step % self.save_steps == 0:
                        self._save_checkpoint(f"checkpoint-{self.global_step}")

                epoch_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1

            epoch_time = time.time() - epoch_start
            avg = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch + 1} — avg loss: {avg:.4f} — time: {epoch_time:.1f}s")

            # ── Alpha analysis ──
            self._log_alpha_analysis(f"[epoch {epoch + 1}]")
            history["alpha_history"].append(self._get_alpha_values())

            # ── Probe refresh ──
            if self.probe_refresh_epochs > 0 and (epoch + 1) % self.probe_refresh_epochs == 0:
                self._refresh_probes()

            # ── Validation + routing metrics ──
            if val_dataloader is not None:
                val_ppl = compute_perplexity(self.model, val_dataloader)
                history["val_perplexity"].append(val_ppl)
                logger.info(f"  Validation perplexity: {val_ppl:.2f}")

                logger.info("  Computing routing metrics...")
                rmets = compute_routing_metrics(self.model, val_dataloader, num_batches=20)
                history["routing_metrics"].append(rmets)
                logger.info(
                    f"  Routing — H={rmets['load_entropy']:.3f}  "
                    f"CoV={rmets['cov']:.3f}  Jaccard={rmets['jaccard']:.3f}"
                )

                self.model.train()

        total_time = time.time() - cal_start
        history["wall_time_seconds"] = total_time
        logger.info(f"Calibration complete in {total_time:.1f}s ({total_time/60:.1f} min)")

        self._save_checkpoint("final")

        # Save full history (including routing metrics)
        hist_path = os.path.join(self.output_dir, "history.json")

        # Serialize routing metrics — convert per_layer dicts to JSON-safe format
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
        logger.info(f"History saved: {hist_path}")

        return history
