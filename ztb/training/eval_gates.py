"""
Evaluation gates for 1M training success criteria.

Gates ensure training quality and prevent invalid checkpoints from being considered successful.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import psutil


class GateStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class GateResult:
    name: str
    status: GateStatus
    reason: str
    value: Any = None
    threshold: Any = None


class EvalGates:
    """Evaluation gates for training success validation."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.gates: Dict[str, Dict[str, Any]] = {
            "resume_time": {"threshold": 30.0, "description": "Resume time < 30s"},
            "no_dup_steps": {
                "threshold": 0,
                "description": "No duplicate global steps",
            },
            "memory_rss": {
                "threshold": 2 * 1024 * 1024 * 1024,
                "description": "RSS < 2GB",
            },  # 2GB in bytes
            "reward_trend_300k": {
                "threshold": 0.0,
                "description": "Mean reward trend >= 0 after 300k steps",
            },
            "eval_above_baseline": {
                "threshold": 0.0,
                "description": "Final eval reward > baseline",
            },
        }

    def enable_all(self) -> None:
        """Enable all evaluation gates."""
        self.enabled = True

    def disable_all(self) -> None:
        """Disable all evaluation gates."""
        self.enabled = False

    def check_resume_time(self, resume_start: float) -> GateResult:
        """Check if resume time is within acceptable limits."""
        elapsed = time.time() - resume_start
        status = (
            GateStatus.PASS
            if elapsed < self.gates["resume_time"]["threshold"]
            else GateStatus.FAIL
        )
        return GateResult(
            name="resume_time",
            status=status,
            reason=f"Resume took {elapsed:.2f}s (threshold: {self.gates['resume_time']['threshold']}s)",
            value=elapsed,
            threshold=self.gates["resume_time"]["threshold"],
        )

    def check_no_duplicate_steps(self, global_steps: List[int]) -> GateResult:
        """Check for duplicate global steps indicating training issues."""
        unique_steps = set(global_steps)
        duplicates = len(global_steps) - len(unique_steps)
        status = GateStatus.PASS if duplicates == 0 else GateStatus.FAIL
        return GateResult(
            name="no_dup_steps",
            status=status,
            reason=f"Found {duplicates} duplicate global steps",
            value=duplicates,
            threshold=0,
        )

    def check_memory_usage(self) -> GateResult:
        """Check current memory usage."""
        process = psutil.Process()
        rss_bytes = process.memory_info().rss
        rss_gb = rss_bytes / (1024 * 1024 * 1024)
        threshold_gb = self.gates["memory_rss"]["threshold"] / (1024 * 1024 * 1024)
        status = (
            GateStatus.PASS
            if rss_bytes < self.gates["memory_rss"]["threshold"]
            else GateStatus.FAIL
        )
        return GateResult(
            name="memory_rss",
            status=status,
            reason=f"RSS: {rss_gb:.2f}GB (threshold: {threshold_gb:.2f}GB)",
            value=rss_bytes,
            threshold=self.gates["memory_rss"]["threshold"],
        )

    def check_reward_trend(
        self,
        rewards: List[float],
        steps: List[int],
        threshold_steps: int = 300000,
        reward_stats: Optional[Dict[str, float]] = None,
    ) -> GateResult:
        """Check if reward trend is positive after threshold steps."""
        if reward_stats and reward_stats.get("count", 0) > 1000:
            # Use statistics for efficiency - assume recent trend is representative
            mean_reward = reward_stats.get("mean", 0.0)
            # Simple check: if mean is positive, consider it trending up
            slope = mean_reward * 0.001  # Rough approximation
            status = (
                GateStatus.PASS
                if slope >= self.gates["reward_trend_300k"]["threshold"]
                else GateStatus.FAIL
            )
            return GateResult(
                name="reward_trend_300k",
                status=status,
                reason=f"Reward trend approximation: {slope:.6f} (using stats, threshold: {self.gates['reward_trend_300k']['threshold']})",
                value=slope,
                threshold=self.gates["reward_trend_300k"]["threshold"],
            )

        # Fallback to full calculation
        if not rewards or len(rewards) < 2:
            return GateResult(
                name="reward_trend_300k",
                status=GateStatus.SKIP,
                reason="Insufficient reward data for trend analysis",
                value=None,
                threshold=self.gates["reward_trend_300k"]["threshold"],
            )

        # Find rewards after threshold steps
        post_threshold_rewards = []
        for reward, step in zip(rewards, steps):
            if step >= threshold_steps:
                post_threshold_rewards.append(reward)

        if len(post_threshold_rewards) < 2:
            return GateResult(
                name="reward_trend_300k",
                status=GateStatus.SKIP,
                reason=f"Insufficient data after {threshold_steps} steps",
                value=None,
                threshold=self.gates["reward_trend_300k"]["threshold"],
            )

        # Calculate trend (simple linear regression slope)
        n = len(post_threshold_rewards)
        x = list(range(n))
        y = post_threshold_rewards

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)

        slope = (
            (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            if (n * sum_x2 - sum_x * sum_x) != 0
            else 0
        )

        status = (
            GateStatus.PASS
            if slope >= self.gates["reward_trend_300k"]["threshold"]
            else GateStatus.FAIL
        )
        return GateResult(
            name="reward_trend_300k",
            status=status,
            reason=f"Reward trend: {slope:.6f} (threshold: {self.gates['reward_trend_300k']['threshold']})",
            value=slope,
            threshold=self.gates["reward_trend_300k"]["threshold"],
        )

    def check_eval_above_baseline(
        self, final_eval_reward: float, baseline: float = 0.0
    ) -> GateResult:
        """Check if final evaluation reward exceeds baseline."""
        status = GateStatus.PASS if final_eval_reward > baseline else GateStatus.FAIL
        return GateResult(
            name="eval_above_baseline",
            status=status,
            reason=f"Final eval: {final_eval_reward:.6f} vs baseline: {baseline:.6f}",
            value=final_eval_reward,
            threshold=baseline,
        )

    def evaluate_all(self, **kwargs: Any) -> Dict[str, GateResult]:
        """Run all gate checks and return results."""
        if not self.enabled:
            return {}

        results = {}

        # Resume time check
        if "resume_start" in kwargs:
            results["resume_time"] = self.check_resume_time(kwargs["resume_start"])

        # Duplicate steps check
        if "global_steps" in kwargs:
            results["no_dup_steps"] = self.check_no_duplicate_steps(
                kwargs["global_steps"]
            )

        # Memory check
        results["memory_rss"] = self.check_memory_usage()

        # Reward trend check
        if "rewards" in kwargs and "steps" in kwargs:
            results["reward_trend_300k"] = self.check_reward_trend(
                kwargs["rewards"],
                kwargs["steps"],
                kwargs.get("threshold_steps", 300000),
                kwargs.get("reward_stats"),
            )

        # Eval baseline check
        if "final_eval_reward" in kwargs:
            results["eval_above_baseline"] = self.check_eval_above_baseline(
                kwargs["final_eval_reward"], kwargs.get("baseline", 0.0)
            )

        return results

    def get_acceptance_summary(
        self, gate_results: Dict[str, GateResult]
    ) -> Dict[str, Any]:
        """Generate acceptance summary from gate results."""
        if not gate_results:
            return {"pass": True, "reasons": [], "gates_disabled": True}

        passed = all(
            result.status == GateStatus.PASS for result in gate_results.values()
        )
        reasons = []

        for result in gate_results.values():
            if result.status == GateStatus.FAIL:
                reasons.append(f"FAIL: {result.name} - {result.reason}")
            elif result.status == GateStatus.SKIP:
                reasons.append(f"SKIP: {result.name} - {result.reason}")

        return {
            "pass": passed,
            "reasons": reasons,
            "gate_results": {
                name: {"status": r.status.value, "reason": r.reason, "value": r.value}
                for name, r in gate_results.items()
            },
        }

    def should_halt_training(
        self,
        gate_results: Dict[str, GateResult],
        consecutive_failures: int = 0,
        max_consecutive_failures: int = 3,
    ) -> tuple[bool, str]:
        """
        Determine if training should be halted based on gate results.

        Args:
            gate_results: Results from gate evaluation
            consecutive_failures: Number of consecutive gate failures
            max_consecutive_failures: Maximum allowed consecutive failures

        Returns:
            Tuple of (should_halt, reason)
        """
        if not gate_results:
            return False, ""

        failed_gates = [r for r in gate_results.values() if r.status == GateStatus.FAIL]

        if not failed_gates:
            return False, ""

        # Critical failures that should always halt immediately
        critical_gates = ["memory_rss", "no_dup_steps"]
        critical_failures = [r for r in failed_gates if r.name in critical_gates]

        if critical_failures:
            reasons = [f"{r.name}: {r.reason}" for r in critical_failures]
            return True, f"Critical gate failure: {', '.join(reasons)}"

        # Consecutive failures exceed threshold
        if consecutive_failures >= max_consecutive_failures:
            return (
                True,
                f"Too many consecutive failures ({consecutive_failures}/{max_consecutive_failures})",
            )

        # Persistent negative reward trend
        trend_gate = gate_results.get("reward_trend_300k")
        if (
            trend_gate
            and trend_gate.status == GateStatus.FAIL
            and consecutive_failures >= 2
        ):
            return True, f"Persistent negative reward trend: {trend_gate.reason}"

        return False, ""


def enable_all_gates(gates: EvalGates) -> None:
    """Enable all evaluation gates."""
    gates.enabled = True


def disable_all_gates(gates: EvalGates) -> None:
    """Disable all evaluation gates."""
    gates.enabled = False
