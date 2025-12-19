"""
Binary Search: 二分探索法（単一パラメータ向け）

単一の連続パラメータを効率的に探索します。
目的関数が単峰性（unimodal）の場合に最適です。

利点:
- 非常に効率的（O(log n)）
- 実装がシンプル
- 少ない試行回数で最適値に近づける

欠点:
- 単一パラメータのみ
- 単峰性の仮定が必要
- 多峰性の場合は局所最適解に陥る

推奨:
- Learning Rateなど、単一パラメータの微調整
- 目的関数が単峰性であることが分かっている場合
"""

import time
from typing import Any, Callable

from ztb.optimization.base import (
    OptimizationResult,
    OptimizerBase,
    ParameterSpace,
    ParameterType,
    TrialResult,
)


class BinarySearchOptimizer(OptimizerBase):
    """
    Binary Search最適化（単一パラメータ向け）

    Example:
        >>> param_space = ParameterSpace('learning_rate',
        ...                              ParameterType.LOG_UNIFORM,
        ...                              low=1e-5, high=1e-2)
        >>> optimizer = BinarySearchOptimizer(
        ...     parameter_space=param_space,
        ...     objective_function=my_objective,
        ...     tolerance=1e-5,
        ...     max_iterations=20
        ... )
        >>> result = optimizer.optimize()
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_function: Callable[[dict[str, Any]], TrialResult],
        tolerance: float = 1e-5,
        max_iterations: int = 20,
        random_state: int = 42,
    ):
        """
        Args:
            parameter_space: 探索するパラメータ空間（単一のみ）
            objective_function: 目的関数
            tolerance: 収束判定の閾値
            max_iterations: 最大反復回数
            random_state: 乱数シード
        """
        if not isinstance(parameter_space, ParameterSpace):
            raise ValueError(
                "parameter_spaceはParameterSpaceインスタンスである必要があります"
            )

        if parameter_space.param_type not in [
            ParameterType.CONTINUOUS,
            ParameterType.LOG_UNIFORM,
        ]:
            raise ValueError("Binary Searchは連続値パラメータのみサポートします")

        super().__init__(
            parameter_spaces=[parameter_space],
            objective_function=objective_function,
            n_trials=max_iterations,
            random_state=random_state,
        )

        self.param_name = parameter_space.name
        self.param_space = parameter_space
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        # 対数スケールの場合
        self.use_log_scale = parameter_space.param_type == ParameterType.LOG_UNIFORM

    def _to_search_space(self, value: float) -> float:
        """元の空間から探索空間への変換"""
        if self.use_log_scale:
            import math

            return math.log10(value)
        return value

    def _from_search_space(self, value: float) -> float:
        """探索空間から元の空間への変換"""
        if self.use_log_scale:
            import math

            return math.pow(10.0, value)
        return value

    def _golden_section_search(self) -> OptimizationResult:
        """
        黄金分割探索（Golden Section Search）

        二分探索の改良版。3点評価により、より効率的に探索します。
        """
        import numpy as np

        # 黄金比
        phi = (1 + np.sqrt(5)) / 2
        resphi = 2 - phi

        # 探索空間での境界
        assert self.param_space.low is not None
        assert self.param_space.high is not None
        a = self._to_search_space(self.param_space.low)
        b = self._to_search_space(self.param_space.high)

        # 初期の2点を配置
        x1 = a + resphi * (b - a)
        x2 = b - resphi * (b - a)

        # 初期評価
        f1_result = self._run_trial(1, {self.param_name: self._from_search_space(x1)})
        f1 = f1_result.objective_value if f1_result.success else float("inf")

        f2_result = self._run_trial(2, {self.param_name: self._from_search_space(x2)})
        f2 = f2_result.objective_value if f2_result.success else float("inf")

        trial_count = 2

        # 反復
        while abs(b - a) > self.tolerance and trial_count < self.max_iterations:
            trial_count += 1

            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = a + resphi * (b - a)
                f1_result = self._run_trial(
                    trial_count, {self.param_name: self._from_search_space(x1)}
                )
                f1 = f1_result.objective_value if f1_result.success else float("inf")
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = b - resphi * (b - a)
                f2_result = self._run_trial(
                    trial_count, {self.param_name: self._from_search_space(x2)}
                )
                f2 = f2_result.objective_value if f2_result.success else float("inf")

            # 進捗表示
            current_best = min(f1, f2)
            abs(b - a)
            print(
                f"  区間: [{self._from_search_space(a):.6e}, "
                f"{self._from_search_space(b):.6e}], "
                f"現在のベスト: {current_best:.6f}"
            )

        # 最終結果
        if f1 < f2:
            best_x = x1
        else:
            best_x = x2

        print()
        print("収束完了！")
        print(f"  最適値: {self._from_search_space(best_x):.6e}")
        print(f"  反復回数: {trial_count}")
        print(f"  最終区間幅: {abs(b - a):.6e}")

        return self._create_result()

    def optimize(self) -> OptimizationResult:
        """Binary Searchで最適化を実行"""
        print("=" * 80)
        print("  Binary Search (Golden Section) 最適化開始")
        print("=" * 80)
        print()
        print(f"パラメータ: {self.param_name}")
        print(f"範囲: [{self.param_space.low:.6e}, {self.param_space.high:.6e}]")
        print(f"対数スケール: {self.use_log_scale}")
        print(f"許容誤差: {self.tolerance:.6e}")
        print(f"最大反復: {self.max_iterations}")
        print()

        self.start_time = time.time()

        result = self._golden_section_search()
        result.print_summary()

        return result
