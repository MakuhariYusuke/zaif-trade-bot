"""
Action Gradient Probe & Failsafe.

Monitors specific action gradients and advantages during training.
Triggers early stop if action gradient dies (grad_norm ≈ 0) or
advantage remains outside target range for extended period.

Supports HOLD/BUY/SELL action monitoring.
"""

from typing import Dict, Optional, Tuple, Literal, TextIO, Any
import numpy as np
import torch
from pathlib import Path
import csv
from numpy.typing import NDArray


ActionType = Literal["HOLD", "BUY", "SELL"]


class ActionGradientProbe:
    """
    Probe for monitoring specific action health during training.
    
    Replaces SELLGradientProbe with generalized version.
    """
    
    def __init__(
        self,
        target_action: ActionType = "SELL",
        grad_norm_threshold: float = 1e-6,
        advantage_threshold: float = 0.0,
        consecutive_failures: int = 200,
        moving_window: int = 50,
        save_path: Optional[Path] = None,
    ):
        """
        Initialize gradient probe.
        
        Args:
            target_action: Action to monitor ("HOLD", "BUY", or "SELL")
            grad_norm_threshold: Minimum grad norm (below = unhealthy)
            advantage_threshold: Minimum advantage (below = unhealthy)
            consecutive_failures: Max consecutive unhealthy updates before stop
            moving_window: Window for moving average
            save_path: Path to save probe CSV
        """
        self.target_action = target_action
        self.action_idx = {"HOLD": 0, "BUY": 1, "SELL": 2}[target_action]
        self.grad_norm_threshold = grad_norm_threshold
        self.advantage_threshold = advantage_threshold
        self.consecutive_failures = consecutive_failures
        self.moving_window = moving_window
        self.save_path = save_path
        
        # Tracking
        self.step_count = 0
        self.grad_norms: list[float] = []
        self.advantages: list[float] = []
        self.consecutive_unhealthy = 0
        self.triggered = False
        
        # CSV writer
        self.csv_file: Optional[TextIO] = None
        self.csv_writer: Optional['csv.DictWriter[str]'] = None
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = open(save_path, "w", newline="")
            self.csv_writer = csv.DictWriter(
                self.csv_file,
                fieldnames=[
                    "step",
                    f"grad_norm_{target_action.lower()}",
                    "grad_norm_ma",
                    f"advantage_{target_action.lower()}",
                    "advantage_ma",
                    "is_healthy",
                    "consecutive_unhealthy",
                ],
            )
            self.csv_writer.writeheader()
        else:
            self.csv_file = None
            self.csv_writer = None
    
    def probe(
        self,
        action_logits: torch.Tensor,
        advantages: NDArray[np.floating[Any]],
        actions: NDArray[np.int64],
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Probe target action gradient and advantage.
        
        Args:
            action_logits: Action logits tensor [batch_size, n_actions] (requires_grad=True)
            advantages: Advantage values [batch_size]
            actions: Actual actions taken [batch_size]
            
        Returns:
            Tuple of (is_healthy, info_dict)
            - is_healthy: False if failsafe should trigger
            - info_dict: Probe statistics
        """
        self.step_count += 1
        
        # Extract target action logits
        target_logits = action_logits[:, self.action_idx]
        
        # Compute gradient norm for target action
        if target_logits.requires_grad:
            # Create dummy loss (sum of target logits)
            target_loss = target_logits.sum()
            target_loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]
            
            # Get gradient
            if action_logits.grad is not None:
                target_grad = action_logits.grad[:, self.action_idx]
                grad_norm = torch.norm(target_grad).item()
            else:
                grad_norm = 0.0
            
            # Zero gradients for next computation
            action_logits.grad = None
        else:
            grad_norm = 0.0
        
        # Compute average advantage for target actions
        target_mask = actions == self.action_idx
        if np.any(target_mask):
            advantage_target = float(np.mean(advantages[target_mask]))
        else:
            advantage_target = 0.0
        
        # Record
        self.grad_norms.append(grad_norm)
        self.advantages.append(advantage_target)
        
        # Compute moving averages
        recent_grad_norms = self.grad_norms[-self.moving_window:]
        recent_advantages = self.advantages[-self.moving_window:]
        
        grad_norm_ma = float(np.mean(recent_grad_norms))
        advantage_ma = float(np.mean(recent_advantages))
        
        # Check health
        is_healthy = (
            grad_norm_ma > self.grad_norm_threshold and
            advantage_ma > self.advantage_threshold
        )
        
        if not is_healthy:
            self.consecutive_unhealthy += 1
        else:
            self.consecutive_unhealthy = 0
        
        # Check if failsafe should trigger
        should_stop = (
            self.consecutive_unhealthy >= self.consecutive_failures
            and not self.triggered
        )
        
        if should_stop:
            self.triggered = True
        
        # Build info dict
        info = {
            "step": self.step_count,
            f"grad_norm_{self.target_action.lower()}": grad_norm,
            "grad_norm_ma": grad_norm_ma,
            f"advantage_{self.target_action.lower()}": advantage_target,
            "advantage_ma": advantage_ma,
            "is_healthy": is_healthy,
            "consecutive_unhealthy": self.consecutive_unhealthy,
        }
        
        # Write to CSV
        if self.csv_writer:
            self.csv_writer.writerow(info)
            if self.csv_file:
                self.csv_file.flush()
        
        return not should_stop, info
    
    def get_statistics(self) -> Dict[str, float]:
        """Get probe statistics."""
        if len(self.grad_norms) == 0:
            return {
                "grad_norm_mean": 0.0,
                "advantage_mean": 0.0,
                "is_healthy": False,
            }
        
        recent_grad_norms = self.grad_norms[-self.moving_window:]
        recent_advantages = self.advantages[-self.moving_window:]
        
        return {
            "grad_norm_mean": float(np.mean(recent_grad_norms)),
            "grad_norm_std": float(np.std(recent_grad_norms)),
            "advantage_mean": float(np.mean(recent_advantages)),
            "advantage_std": float(np.std(recent_advantages)),
            "consecutive_unhealthy": self.consecutive_unhealthy,
            "triggered": self.triggered,
        }
    
    def close(self) -> None:
        """Close CSV file."""
        if self.csv_file:
            self.csv_file.close()


def create_failsafe_dump(
    model: Any,
    probe: ActionGradientProbe,
    output_dir: Path,
) -> None:
    """
    Create failsafe dump when probe triggers.
    
    Args:
        model: PPO model to save
        probe: Gradient probe
        output_dir: Directory to save dumps
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / "failsafe_model.zip"
    model.save(str(model_path))
    print(f"Failsafe: Model saved to {model_path}")
    
    # Save probe statistics
    stats = probe.get_statistics()
    stats_path = output_dir / "failsafe_stats.txt"
    
    with open(stats_path, "w") as f:
        f.write(f"{probe.target_action} Gradient Probe Failsafe Triggered\n")
        f.write("=" * 60 + "\n\n")
        
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nDiagnostics:\n")
        if stats["grad_norm_mean"] < probe.grad_norm_threshold:
            f.write(f"- {probe.target_action} gradient norm too low (dead gradient)\n")
        if stats["advantage_mean"] <= probe.advantage_threshold:
            f.write(f"- {probe.target_action} advantage non-positive (policy not learning value)\n")
        
        f.write(f"\nConsecutive unhealthy updates: {stats['consecutive_unhealthy']}\n")
    
    print(f"Failsafe: Statistics saved to {stats_path}")
    print(f"\n⚠️  {probe.target_action} gradient probe failsafe triggered!")
    print("Review probe CSV and failsafe dump for diagnosis.")


# Backward compatibility alias
SELLGradientProbe = ActionGradientProbe


def test_action_gradient_probe() -> None:
    """Test Action gradient probe with synthetic data."""
    print("Testing Action Gradient Probe...")
    
    # Create probe for SELL action
    probe = ActionGradientProbe(
        target_action="SELL",
        grad_norm_threshold=1e-6,
        advantage_threshold=0.0,
        consecutive_failures=10,
        moving_window=5,
    )
    
    # Scenario 1: Healthy gradients
    print("\nScenario 1: Healthy gradients")
    for i in range(20):
        # Mock logits with gradients
        logits = torch.randn(32, 3, requires_grad=True)
        advantages = np.random.randn(32) * 0.5 + 0.2  # Mostly positive
        actions = np.random.choice([0, 1, 2], size=32)
        
        is_healthy, info = probe.probe(logits, advantages, actions)
        
        if (i + 1) % 5 == 0:
            print(f"  Step {i+1}: healthy={is_healthy}, "
                  f"grad_norm_ma={info['grad_norm_ma']:.6f}, "
                  f"advantage_ma={info['advantage_ma']:.3f}")
    
    # Reset
    probe = ActionGradientProbe(
        target_action="SELL",
        grad_norm_threshold=1e-6,
        advantage_threshold=0.0,
        consecutive_failures=10,
        moving_window=5,
    )
    
    # Scenario 2: Dead gradients (should trigger failsafe)
    print("\nScenario 2: Dead gradients (should trigger failsafe)")
    for i in range(15):
        # Mock logits with zero gradients
        logits = torch.zeros(32, 3, requires_grad=True)
        advantages = np.random.randn(32) * 0.5 - 0.5  # Mostly negative
        actions = np.random.choice([0, 1, 2], size=32)
        
        is_healthy, info = probe.probe(logits, advantages, actions)
        
        if (i + 1) % 3 == 0:
            print(f"  Step {i+1}: healthy={is_healthy}, "
                  f"consecutive_unhealthy={info['consecutive_unhealthy']}, "
                  f"grad_norm_ma={info['grad_norm_ma']:.6f}")
        
        if not is_healthy and probe.triggered:
            print(f"\n  ⚠️  Failsafe triggered at step {i+1}")
            break
    
    # Get final statistics
    stats = probe.get_statistics()
    print("\nFinal statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Action gradient probe test complete")


# Keep old function name for backward compatibility
test_sell_gradient_probe = test_action_gradient_probe


if __name__ == "__main__":
    test_action_gradient_probe()
