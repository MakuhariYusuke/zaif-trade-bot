"""Minimal optimizer features shim used in tests."""


class OptimizerFeatureTracker:
    def __init__(self, *args, **kwargs):
        self.features = []

    def track(self, feature_name: str, metric: float) -> None:
        self.features.append((feature_name, metric))


def get_optimizer_tracker() -> OptimizerFeatureTracker:
    return OptimizerFeatureTracker()


def update_optimizer_features(tracker: OptimizerFeatureTracker, features: dict) -> None:
    """Apply update to optimizer features; minimal implementation used by tests."""
    for k, v in (features or {}).items():
        try:
            tracker.track(k, float(v))
        except Exception:
            tracker.track(k, 0.0)


__all__ = ["OptimizerFeatureTracker"]
