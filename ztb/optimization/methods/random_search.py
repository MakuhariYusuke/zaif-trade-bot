"""
Random Search: ランダムサンプリングによる探索

パラメータ空間からランダムにサンプリングして最良のものを見つけます。
Bergstra & Bengio (2012)の研究で、Grid Searchより効率的であることが示されています。

利点:
- Grid Searchより効率的（特に高次元空間）
- 重要なパラメータに対して密に探索
- 連続値パラメータを直接扱える
- 並列化が容易

欠点:
- 最適解を見逃す可能性
- 運が必要

推奨:
- パラメータ数が3つ以上
- 連続値パラメータを含む
- 計算リソースが限られている
"""

from typing import List
import time
import random
import numpy as np

from ztb.optimization.base import (
    OptimizerBase,
    ParameterSpace,
    OptimizationResult
)


class RandomSearchOptimizer(OptimizerBase):
    """
    Random Search最適化
    
    Example:
        >>> param_spaces = [
        ...     ParameterSpace('learning_rate', ParameterType.LOG_UNIFORM, 
        ...                    low=1e-4, high=1e-3),
        ...     ParameterSpace('batch_size', ParameterType.CATEGORICAL, 
        ...                    choices=[64, 128, 256]),
        ...     ParameterSpace('gamma', ParameterType.CONTINUOUS,
        ...                    low=0.95, high=0.999)
        ... ]
        >>> optimizer = RandomSearchOptimizer(
        ...     parameter_spaces=param_spaces,
        ...     objective_function=my_objective,
        ...     n_trials=20
        ... )
        >>> result = optimizer.optimize()
    """
    
    def __init__(
        self,
        parameter_spaces: List[ParameterSpace],
        objective_function,
        n_trials: int = 20,
        random_state: int = 42
    ):
        """
        Args:
            parameter_spaces: 探索するパラメータ空間
            objective_function: 目的関数
            n_trials: ランダムサンプリングの試行回数
            random_state: 乱数シード
        """
        super().__init__(
            parameter_spaces=parameter_spaces,
            objective_function=objective_function,
            n_trials=n_trials,
            random_state=random_state
        )
        
        # 乱数シードの設定
        random.seed(random_state)
        np.random.seed(random_state)
    
    def _sample_parameters(self) -> dict:
        """パラメータをランダムにサンプリング"""
        parameters = {}
        for param_name, param_space in self.parameter_spaces.items():
            parameters[param_name] = param_space.sample()
        return parameters
    
    def optimize(self) -> OptimizationResult:
        """Random Searchで最適化を実行"""
        print("=" * 80)
        print("  Random Search 最適化開始")
        print("=" * 80)
        print()
        print(f"試行回数: {self.n_trials}")
        print(f"探索パラメータ: {list(self.parameter_spaces.keys())}")
        print()
        
        self.start_time = time.time()
        
        # ランダムサンプリング
        for trial_id in range(1, self.n_trials + 1):
            parameters = self._sample_parameters()
            self._run_trial(trial_id, parameters)
        
        result = self._create_result()
        result.print_summary()
        
        return result
