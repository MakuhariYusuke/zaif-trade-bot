#!/usr/bin/env python3
"""
Action Mask/Logit/Probability Diagnostics for PPO Training.

Provides tools to visualize and debug:
- Mask application before/after
- Logits and probabilities with/without temperature
- Deterministic action selection rationale
- Entropy, KL divergence, losses
- Action-wise advantages and policy gradients
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from ztb.analysis.common.plot_utils import save_plot
from ztb.utils.file_utils import safe_json_dump

if TYPE_CHECKING:
    pass

class ActionDiagnostics:
    """Diagnostic tool for action selection debugging."""

    def __init__(self, save_dir: str = "plots/diagnostics"):
        """Initialize diagnostics.

        Args:
            save_dir: Directory to save diagnostic plots and logs
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.step_counter = 0
        self.batch_logs: list[dict[str, Any]] = []

    def log_batch_diagnostics(
        self,
        step: int,
        logits_raw: "torch.Tensor",
        logits_masked: "torch.Tensor",
        action_masks: "torch.Tensor" | None,
        probs_before_temp: "torch.Tensor",
        probs_after_temp: "torch.Tensor",
        temperature: float,
        actions_selected: torch.Tensor,
        advantages: torch.Tensor | None = None,
        policy_grad_norm: float | None = None,
        entropy: float | None = None,
        approx_kl: float | None = None,
        value_loss: float | None = None,
        policy_loss: float | None = None,
        phase: str = "train",
    ) -> None:
        """Log diagnostic data for a single batch.

        Args:
            step: Current training step
            logits_raw: Raw logits before masking [batch_size, n_actions]
            logits_masked: Logits after mask application [batch_size, n_actions]
            action_masks: Action masks [batch_size, n_actions] (1=legal, 0=illegal)
            probs_before_temp: Probabilities before temperature [batch_size, n_actions]
            probs_after_temp: Probabilities after temperature [batch_size, n_actions]
            temperature: Temperature value used
            actions_selected: Selected actions [batch_size]
            advantages: Advantage values [batch_size]
            policy_grad_norm: Policy gradient norm
            entropy: Policy entropy
            approx_kl: Approximate KL divergence
            value_loss: Value function loss
            policy_loss: Policy loss
            phase: "train" or "eval"
        """
        # Import torch lazily (avoid requiring torch at module import time)
        import importlib

        tmod = importlib.import_module("torch")

        # Convert to numpy for analysis
        logits_raw_np = logits_raw.detach().cpu().numpy()
        logits_masked_np = logits_masked.detach().cpu().numpy()
        probs_before_np = probs_before_temp.detach().cpu().numpy()
        probs_after_np = probs_after_temp.detach().cpu().numpy()
        actions_np = actions_selected.detach().cpu().numpy()

        if action_masks is not None:
            masks_np = action_masks.detach().cpu().numpy()
        else:
            masks_np = np.ones_like(logits_raw_np)

        if advantages is not None:
            advantages_np = advantages.detach().cpu().numpy()
        else:
            advantages_np = None

        # Compute action distribution
        action_counts = np.bincount(actions_np, minlength=3)
        action_dist = (
            action_counts / len(actions_np) if len(actions_np) > 0 else action_counts
        )

        # Compute action-wise advantages
        action_advantages = {}
        if advantages_np is not None:
            for a in range(3):
                mask = actions_np == a
                if mask.sum() > 0:
                    action_advantages[f"action_{a}_adv_mean"] = float(
                        advantages_np[mask].mean()
                    )
                    action_advantages[f"action_{a}_adv_std"] = float(
                        advantages_np[mask].std()
                    )

        # Log entry
        log_entry = {
            "step": step,
            "phase": phase,
            "temperature": temperature,
            "action_distribution": {
                "HOLD": float(action_dist[0]),
                "BUY": float(action_dist[1]),
                "SELL": float(action_dist[2]),
            },
            "logits_raw_mean": float(logits_raw_np.mean(axis=0).tolist()),
            "logits_masked_mean": float(logits_masked_np.mean(axis=0).tolist()),
            "probs_before_temp_mean": float(probs_before_np.mean(axis=0).tolist()),
            "probs_after_temp_mean": float(probs_after_np.mean(axis=0).tolist()),
            "mask_legal_rate": float(masks_np.mean()),
            **action_advantages,
        }

        if entropy is not None:
            log_entry["entropy"] = float(entropy)
        if approx_kl is not None:
            log_entry["approx_kl"] = float(approx_kl)
        if value_loss is not None:
            log_entry["value_loss"] = float(value_loss)
        if policy_loss is not None:
            log_entry["policy_loss"] = float(policy_loss)
        if policy_grad_norm is not None:
            log_entry["policy_grad_norm"] = float(policy_grad_norm)

        self.batch_logs.append(log_entry)

        # Save sample details for first 10 steps
        if step < 10:
            sample_file = self.save_dir / f"{phase}_step_{step}_samples.json"
            sample_data = {
                "step": step,
                "temperature": temperature,
                "samples": [
                    {
                        "index": int(i),
                        "logits_raw": logits_raw_np[i].tolist(),
                        "logits_masked": logits_masked_np[i].tolist(),
                        "mask": masks_np[i].tolist(),
                        "probs_before_temp": probs_before_np[i].tolist(),
                        "probs_after_temp": probs_after_np[i].tolist(),
                        "action_selected": int(actions_np[i]),
                        "advantage": (
                            float(advantages_np[i])
                            if advantages_np is not None
                            else None
                        ),
                    }
                    for i in range(min(5, len(actions_np)))
                ],
            }
            safe_json_dump(sample_data, sample_file, indent=2)

    def save_logs(self, filename: str = "diagnostics_log.json") -> None:
        """Save all logged diagnostics to file."""
        log_file = self.save_dir / filename
        safe_json_dump(self.batch_logs, str(log_file), indent=2)
        print(f"Diagnostics saved to {log_file}")

    def plot_diagnostics(self) -> None:
        """Generate diagnostic plots from logged data."""
        if not self.batch_logs:
            print("No diagnostic data to plot")
            return

        # Extract data
        steps = [log["step"] for log in self.batch_logs]
        hold_dist = [log["action_distribution"]["HOLD"] for log in self.batch_logs]
        buy_dist = [log["action_distribution"]["BUY"] for log in self.batch_logs]
        sell_dist = [log["action_distribution"]["SELL"] for log in self.batch_logs]
        entropy = [log.get("entropy", 0) for log in self.batch_logs]
        approx_kl = [log.get("approx_kl", 0) for log in self.batch_logs]
        value_loss = [log.get("value_loss", 0) for log in self.batch_logs]
        policy_loss = [log.get("policy_loss", 0) for log in self.batch_logs]

        # Create plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # Action distribution
        axes[0, 0].plot(steps, hold_dist, label="HOLD", alpha=0.7)
        axes[0, 0].plot(steps, buy_dist, label="BUY", alpha=0.7)
        axes[0, 0].plot(steps, sell_dist, label="SELL", alpha=0.7)
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Action Probability")
        axes[0, 0].set_title("Action Distribution Over Training")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Entropy
        axes[0, 1].plot(steps, entropy, alpha=0.7, color="purple")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Entropy")
        axes[0, 1].set_title("Policy Entropy")
        axes[0, 1].grid(True)

        # KL divergence
        axes[1, 0].plot(steps, approx_kl, alpha=0.7, color="orange")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Approx KL")
        axes[1, 0].set_title("KL Divergence")
        axes[1, 0].grid(True)

        # Value loss
        axes[1, 1].plot(steps, value_loss, alpha=0.7, color="blue")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Value Loss")
        axes[1, 1].set_title("Value Function Loss")
        axes[1, 1].grid(True)

        # Policy loss
        axes[2, 0].plot(steps, policy_loss, alpha=0.7, color="red")
        axes[2, 0].set_xlabel("Step")
        axes[2, 0].set_ylabel("Policy Loss")
        axes[2, 0].set_title("Policy Loss")
        axes[2, 0].grid(True)

        # Action-wise advantages (if available)
        adv_hold = [log.get("action_0_adv_mean", 0) for log in self.batch_logs]
        adv_buy = [log.get("action_1_adv_mean", 0) for log in self.batch_logs]
        adv_sell = [log.get("action_2_adv_mean", 0) for log in self.batch_logs]
        axes[2, 1].plot(steps, adv_hold, label="HOLD Adv", alpha=0.7)
        axes[2, 1].plot(steps, adv_buy, label="BUY Adv", alpha=0.7)
        axes[2, 1].plot(steps, adv_sell, label="SELL Adv", alpha=0.7)
        axes[2, 1].set_xlabel("Step")
        axes[2, 1].set_ylabel("Mean Advantage")
        axes[2, 1].set_title("Action-wise Advantages")
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        plt.tight_layout()
        plot_file = self.save_dir / "diagnostics_plots.png"
        save_plot(plot_file, dpi=150)
        plt.close()
        print(f"Diagnostic plots saved to {plot_file}")

def analyze_deterministic_selection(
    logits: torch.Tensor,
    masks: torch.Tensor,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Analyze deterministic action selection process.

    Args:
        logits: Raw logits [batch_size, n_actions]
        masks: Action masks [batch_size, n_actions]
        temperature: Temperature for softmax

    Returns:
        Dictionary with analysis results
    """
    # Apply mask
    logits_masked = logits.clone()
    logits_masked[masks == 0] = float("-inf")

    # Apply temperature
    logits_temp = logits_masked / temperature

    # Softmax
    probs = torch.softmax(logits_temp, dim=-1)

    # Argmax
    actions = torch.argmax(probs, dim=-1)

    # Gather statistics
    action_counts = torch.bincount(actions, minlength=3)
    action_dist = action_counts.float() / len(actions)

    max_probs = probs.max(dim=-1)[0]

    return {
        "action_distribution": {
            "HOLD": float(action_dist[0]),
            "BUY": float(action_dist[1]),
            "SELL": float(action_dist[2]),
        },
        "mean_max_prob": float(max_probs.mean()),
        "std_max_prob": float(max_probs.std()),
        "mean_entropy": float(
            -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
        ),
    }
