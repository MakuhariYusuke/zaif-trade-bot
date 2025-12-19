"""Minimal reward function optimizer shim for tests."""
from typing import Any, Dict
import json
import random
from types import SimpleNamespace


class RewardFunctionOptimizer:
    """Very small stub used only to satisfy import-time expectations."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self._console_verbose = False
        self._show_progress = False

    def search(self):
        return {"best": None, "status": "noop"}

    def set_console_output(self, verbose: bool = False, show_progress: bool = False) -> None:
        """Configure console output verbosity and progress display."""
        self._console_verbose = bool(verbose)
        self._show_progress = bool(show_progress)

    def create_backtest_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a simple backtest config from parameter dict for testing."""
        return {"stage": "test", "params": params}

    def run_backtest_evaluation(self, config: Dict[str, Any]) -> float:
        """Run a mock backtest evaluation and return a score (float)."""
        # Deterministic-ish mock: sum of numeric params or random fallback
        params = config.get("params", {})
        score = 0.0
        for v in params.values():
            try:
                score += float(v)
            except Exception:
                score += random.random()
        return score

    def optimize_reward_function(
        self,
        stage: str,
        evaluation_function: Any,
        n_trials: int = 5,
        objectives: Any = None,
    ) -> SimpleNamespace:
        """Run a quick mock optimization loop for tests."""
        best_score = float("-inf")
        best_params = {}
        for i in range(max(1, int(n_trials))):
            # Generate simple random parameter set
            params = {"profit_weight": random.random(), "risk_weight": random.random()}
            score = evaluation_function(params)
            if score > best_score:
                best_score = score
                best_params = params

        result = SimpleNamespace()
        result.best_scores = {"profit": best_score}
        result.best_config = SimpleNamespace(parameters=best_params)
        return result

    def optimize_from_config_file(self, config_file_path: str, n_trials: int = 3) -> SimpleNamespace:
        try:
            with open(config_file_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            params = cfg.get("parameters", {})
            # Use the same simple evaluation as optimize_reward_function
            result = self.optimize_reward_function("config_stage", lambda p: sum(map(float, params.values())), n_trials=n_trials)
            return result
        except Exception:
            return self.optimize_reward_function("config_stage", lambda p: 0.0, n_trials=n_trials)


__all__ = ["RewardFunctionOptimizer"]
