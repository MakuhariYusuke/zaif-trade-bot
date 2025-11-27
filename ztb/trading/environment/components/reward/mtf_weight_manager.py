from typing import Dict, Any, Optional
import math

class MTFWeightManager:
    """
    Simple, safe MTF weight manager for Layer 5.

    This minimal implementation is a placeholder that returns static default
    weights and enforces min/max constraints. A future optimizer can be
    implemented behind this component.
    """

    def __init__(self, config: Any):
        self.config = config
        # defaults - safe initial values
        self._weights = {"1min": 0.30, "5min": 0.55, "15min": 0.15}
        self._min_weights = {"1min": 0.10, "5min": 0.10, "15min": 0.01}
        self._max_weights = {"1min": 0.50, "5min": 0.80, "15min": 0.50}

    def get_weights(self) -> Dict[str, float]:
        return dict(self._weights)

    def update(self, step: int, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Conservative update rule for MTF weights.

        Expected metrics format (optional):
        {
            "tf_metrics": {
                "1min": {"sharpe": 0.2},
                "5min": {"sharpe": 0.5},
                "15min": {"sharpe": 0.3}
            }
        }

        The method computes a normalized score per timeframe based on provided metrics
        and performs a small alpha-weighted update towards those scores. It enforces
        min/max constraints and renormalizes weights to sum to 1.0.
        """
        if metrics is None:
            return None

        tf_metrics = metrics.get("tf_metrics") if isinstance(metrics, dict) else None
        if not tf_metrics or not isinstance(tf_metrics, dict):
            # Nothing to update
            return None

        # Build an unnormalized score per timeframe using simple rules (sharpe preferred)
        scores: Dict[str, float] = {}
        for tf, m in tf_metrics.items():
            if not isinstance(m, dict):
                scores[tf] = 0.0
                continue
            # Accept either sharpe or avg_return otherwise 0
            score = m.get("sharpe") if (m.get("sharpe") is not None) else m.get("avg_return", 0.0)
            try:
                score = float(score)
            except Exception:
                score = 0.0
            scores[tf] = max(0.0, score)

        # If all zeros, do nothing
        if all(s == 0.0 for s in scores.values()):
            return None

        # Normalize scores to sum to 1
        total_score = sum(scores.values())
        norm_scores = {tf: s / (total_score + 1e-12) for tf, s in scores.items()}

        # Conservative update param (configurable via config, default 0.05)
        alpha = 0.05
        try:
            alpha_config = float(getattr(self.config, "mtf_weight_alpha", alpha))
            if 0 < alpha_config < 1:
                alpha = alpha_config
        except Exception:
            pass

        # Update weights towards norm_scores only for known timeframes
        for tf in list(self._weights.keys()):
            target = norm_scores.get(tf, self._weights.get(tf, 0.0))
            new_w = (1 - alpha) * self._weights.get(tf, 0.0) + alpha * target
            # enforce min/max
            min_w = self._min_weights.get(tf, 0.0)
            max_w = self._max_weights.get(tf, 1.0)
            new_w = max(min_w, min(max_w, new_w))
            self._weights[tf] = new_w

        # Renormalize to sum to 1
        total = sum(self._weights.values())
        if total > 0:
            for tf in list(self._weights.keys()):
                self._weights[tf] = self._weights[tf] / total

    def reset(self) -> None:
        # Reset to default values
        self._weights = {"1min": 0.30, "5min": 0.55, "15min": 0.15}
