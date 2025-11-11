"""
最適化手法の実装モジュール

各種ハイパーパラメータ最適化手法を提供します。
"""

from ztb.optimization.methods.binary_search import BinarySearchOptimizer
from ztb.optimization.methods.grid_search import GridSearchOptimizer
from ztb.optimization.methods.random_search import RandomSearchOptimizer

try:
    from ztb.optimization.methods.bayesian_optimization import BayesianOptimizer

    __all__ = [
        "GridSearchOptimizer",
        "RandomSearchOptimizer",
        "BinarySearchOptimizer",
        "BayesianOptimizer",
    ]
except ImportError:
    # scikit-optimize がインストールされていない場合
    __all__ = ["GridSearchOptimizer", "RandomSearchOptimizer", "BinarySearchOptimizer"]
