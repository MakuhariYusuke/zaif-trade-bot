"""
Grid Search: 全組み合わせを網羅的に探索

パラメータの全組み合わせを試行し、最良のものを見つけます。
小規模な探索空間で確実に最適解を見つけたい場合に有効です。

利点:
- 最適解を見逃さない（探索空間内で）
- 実装がシンプル
- 各パラメータの効果を体系的に評価可能

欠点:
- パラメータ数が増えると組み合わせ爆発
- 計算コストが高い
- 連続値パラメータは離散化が必要
"""

from typing import List, Dict, Any, Callable, Optional
import itertools
import time

from ztb.optimization.base import (
    OptimizerBase,
    ParameterSpace,
    ParameterType,
    TrialResult,
    OptimizationResult
)


class GridSearchOptimizer(OptimizerBase):
    """
    Grid Search最適化
    
    Example:
        >>> param_spaces = [
        ...     ParameterSpace('learning_rate', ParameterType.LOG_UNIFORM, 
        ...                    low=1e-4, high=1e-3),
        ...     ParameterSpace('batch_size', ParameterType.CATEGORICAL, 
        ...                    choices=[64, 128, 256])
        ... ]
        >>> optimizer = GridSearchOptimizer(
        ...     parameter_spaces=param_spaces,
        ...     objective_function=my_objective,
        ...     grid_resolution={'learning_rate': [1e-4, 3e-4, 5e-4, 1e-3]}
        ... )
        >>> result = optimizer.optimize()
    """
    
    def __init__(
        self,
        parameter_spaces: List[ParameterSpace],
        objective_function: Callable[[dict[str, Any]], TrialResult],
        grid_resolution: Optional[Dict[str, List[Any]]] = None,
        random_state: int = 42
    ):
        """
        Args:
            parameter_spaces: 探索するパラメータ空間
            objective_function: 目的関数
            grid_resolution: 各パラメータのグリッド値
                            例: {'learning_rate': [1e-4, 3e-4, 1e-3],
                                 'batch_size': [64, 128, 256]}
            random_state: 乱数シード
        """
        super().__init__(
            parameter_spaces=parameter_spaces,
            objective_function=objective_function,
            random_state=random_state
        )
        
        # Ensure n_trials is int for type safety
        self.n_trials: int = 0
        
        # グリッド解像度の設定
        self.grid_resolution = grid_resolution or {}
        self._build_grid()
    
    def _build_grid(self) -> None:
        """パラメータグリッドを構築"""
        self.grid = {}
        
        for param_name, param_space in self.parameter_spaces.items():
            if param_name in self.grid_resolution:
                # ユーザー指定のグリッド値を使用
                self.grid[param_name] = self.grid_resolution[param_name]
            else:
                # デフォルトのグリッド値を生成
                self.grid[param_name] = self._generate_default_grid(param_space)
        
        # 総試行回数を計算
        self.n_trials = 1
        for values in self.grid.values():
            self.n_trials *= len(values)
        
        print(f"Grid Search: {self.n_trials} 通りの組み合わせを探索")
    
    def _generate_default_grid(self, param_space: ParameterSpace) -> List[Any]:
        """デフォルトのグリッド値を生成"""
        import numpy as np
        
        if param_space.param_type == ParameterType.CATEGORICAL:
            return param_space.choices or []
        
        elif param_space.param_type == ParameterType.INTEGER:
            # 整数: 最大5ステップ
            if param_space.low is None or param_space.high is None:
                return []
            n_steps = min(5, int(param_space.high - param_space.low + 1))
            return [int(x) for x in np.linspace(param_space.low, param_space.high, n_steps)]
        
        elif param_space.param_type == ParameterType.CONTINUOUS:
            # 連続値: 5ステップ
            if param_space.low is None or param_space.high is None:
                return []
            return list(np.linspace(param_space.low, param_space.high, 5))
        
        elif param_space.param_type == ParameterType.LOG_UNIFORM:
            # 対数スケール: 5ステップ
            if param_space.low is None or param_space.high is None:
                return []
            log_values = np.linspace(np.log10(param_space.low), 
                                     np.log10(param_space.high), 5)
            return [10 ** x for x in log_values]
        
        return []
    
    def optimize(self) -> OptimizationResult:
        """Grid Searchで最適化を実行"""
        print("=" * 80)
        print("  Grid Search 最適化開始")
        print("=" * 80)
        print()
        
        # グリッドの表示
        print("探索グリッド:")
        for param_name, values in self.grid.items():
            print(f"  {param_name}: {values}")
        print()
        
        self.start_time = time.time()
        
        # 全組み合わせを生成
        param_names = list(self.grid.keys())
        param_values = [self.grid[name] for name in param_names]
        combinations = list(itertools.product(*param_values))
        
        # 各組み合わせで試行
        for trial_id, combination in enumerate(combinations, 1):
            parameters = dict(zip(param_names, combination))
            self._run_trial(trial_id, parameters)
        
        result = self._create_result()
        result.print_summary()
        
        return result
