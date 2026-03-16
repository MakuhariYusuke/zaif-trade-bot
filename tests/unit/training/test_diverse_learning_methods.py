from __future__ import annotations

from ztb.training.diverse_learning_methods import DiverseLearningMethods


class _OptimizerStub:
    def optimize(self, **kwargs):
        return {"best_value": 1.23, "payload": kwargs["max_evals"]}


class TestDiverseLearningMethodsCache:
    def test_results_cache_is_bounded(self) -> None:
        methods = DiverseLearningMethods()
        methods.max_results_cache_entries = 2
        methods.frameworks = {
            "ray_tune": lambda: _OptimizerStub(),
            "hyperopt": lambda: _OptimizerStub(),
            "bohb": lambda: _OptimizerStub(),
        }

        methods.optimize_hyperparameters(lambda _: 0.0, {"a": 1}, framework="ray_tune", max_evals=1)
        methods.optimize_hyperparameters(lambda _: 0.0, {"b": 1}, framework="hyperopt", max_evals=2)
        methods.optimize_hyperparameters(lambda _: 0.0, {"c": 1}, framework="bohb", max_evals=3)

        stats = methods.get_results_cache_stats()
        assert stats["entries"] == 2
        assert stats["max_entries"] == 2

    def test_clear_results_cache(self) -> None:
        methods = DiverseLearningMethods()
        methods.frameworks = {"ray_tune": lambda: _OptimizerStub()}

        methods.optimize_hyperparameters(lambda _: 0.0, {"a": 1}, framework="ray_tune", max_evals=1)
        assert methods.get_results_cache_stats()["entries"] == 1

        methods.clear_results_cache()

        assert methods.get_results_cache_stats()["entries"] == 0
