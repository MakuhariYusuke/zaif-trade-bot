"""
Bayesian Optimization: ベイズ最適化による効率的探索

ガウス過程（Gaussian Process）を使ってパラメータ空間をモデル化し、
獲得関数（Acquisition Function）を使って次に試すべきパラメータを賢く選択します。

利点:
- 少ない試行回数で良い結果を得られる
- 過去の試行結果を活用して効率的に探索
- 高コストな評価関数に最適
- 探索と活用のバランスを自動調整

欠点:
- 低次元空間（~20次元）で効果的
- 高次元では性能低下
- カテゴリカル変数の扱いが難しい
- 実装が複雑

実装: scikit-optimizeを使用
"""

import time
from typing import Any, Callable, List

from ztb.optimization.base import (
    OptimizationResult,
    OptimizerBase,
    ParameterSpace,
    ParameterType,
    TrialResult,
)


class BayesianOptimizer(OptimizerBase):
    """
    Bayesian Optimization

    scikit-optimize (skopt) を使用した実装

    Example:
        >>> param_spaces = [
        ...     ParameterSpace('learning_rate', ParameterType.LOG_UNIFORM,
        ...                    low=1e-4, high=1e-3),
        ...     ParameterSpace('gamma', ParameterType.CONTINUOUS,
        ...                    low=0.95, high=0.999)
        ... ]
        >>> optimizer = BayesianOptimizer(
        ...     parameter_spaces=param_spaces,
        ...     objective_function=my_objective,
        ...     n_trials=30,
        ...     n_initial_points=10
        ... )
        >>> result = optimizer.optimize()
    """

    def __init__(
        self,
        parameter_spaces: List[ParameterSpace],
        objective_function: Callable[[dict[str, Any]], TrialResult],
        n_trials: int = 30,
        n_initial_points: int = 10,
        acquisition_function: str = "EI",  # 'EI', 'PI', 'LCB'
        random_state: int = 42,
    ):
        """
        Args:
            parameter_spaces: 探索するパラメータ空間
            objective_function: 目的関数
            n_trials: 総試行回数
            n_initial_points: ランダム探索の初期試行回数（ガウス過程の初期化用）
            acquisition_function: 獲得関数 ('EI'=Expected Improvement,
                                           'PI'=Probability of Improvement,
                                           'LCB'=Lower Confidence Bound)
            random_state: 乱数シード
        """
        super().__init__(
            parameter_spaces=parameter_spaces,
            objective_function=objective_function,
            n_trials=n_trials,
            random_state=random_state,
        )

        self.n_initial_points = n_initial_points
        self.acquisition_function = acquisition_function

        # skoptが利用可能かチェック
        self._check_skopt()

    def _check_skopt(self) -> None:
        """scikit-optimizeのインストール確認"""
        try:
            import skopt

            self.skopt = skopt
        except ImportError:
            raise ImportError(
                "Bayesian Optimizationにはscikit-optimizeが必要です。\n"
                "インストール: pip install scikit-optimize"
            )

    def _convert_to_skopt_space(self) -> List[Any]:
        """ParameterSpaceをskoptのスペースに変換"""
        from skopt.space import Categorical, Integer, Real

        skopt_space = []
        self.param_names = []

        for param_name, param_space in self.parameter_spaces.items():
            self.param_names.append(param_name)

            if param_space.param_type == ParameterType.CONTINUOUS:
                assert param_space.low is not None and param_space.high is not None
                skopt_space.append(
                    Real(param_space.low, param_space.high, name=param_name)
                )

            elif param_space.param_type == ParameterType.LOG_UNIFORM:
                assert param_space.low is not None and param_space.high is not None
                skopt_space.append(
                    Real(
                        param_space.low,
                        param_space.high,
                        prior="log-uniform",
                        name=param_name,
                    )
                )

            elif param_space.param_type == ParameterType.INTEGER:
                assert param_space.low is not None and param_space.high is not None
                skopt_space.append(
                    Integer(
                        int(param_space.low), int(param_space.high), name=param_name
                    )
                )

            elif param_space.param_type == ParameterType.CATEGORICAL:
                assert param_space.choices is not None
                skopt_space.append(Categorical(param_space.choices, name=param_name))

        return skopt_space

    def _objective_wrapper(self, params: List[Any]) -> float:
        """skopt用の目的関数ラッパー"""
        # リストを辞書に変換
        parameters = dict(zip(self.param_names, params))

        # トライアルを実行
        trial_result = self._run_trial(len(self.trials) + 1, parameters)

        # 失敗した場合は大きなペナルティ
        if not trial_result.success:
            return 1e10

        return trial_result.objective_value

    def optimize(self) -> OptimizationResult:
        """Bayesian Optimizationで最適化を実行"""
        from skopt import gp_minimize

        print("=" * 80)
        print("  Bayesian Optimization 開始")
        print("=" * 80)
        print()
        print(f"試行回数: {self.n_trials}")
        print(f"初期ランダム探索: {self.n_initial_points}")
        print(f"獲得関数: {self.acquisition_function}")
        print(f"探索パラメータ: {list(self.parameter_spaces.keys())}")
        print()

        self.start_time = time.time()

        # skopt空間に変換
        skopt_space = self._convert_to_skopt_space()

        # Bayesian Optimization実行
        result_skopt = gp_minimize(
            func=self._objective_wrapper,
            dimensions=skopt_space,
            n_calls=self.n_trials,
            n_initial_points=self.n_initial_points,
            acq_func=self.acquisition_function,
            random_state=self.random_state,
            verbose=False,
        )

        print()
        print("最適化完了！")
        print(f"  最良の目的関数値: {result_skopt.fun:.6f}")
        print(f"  最良のパラメータ: {dict(zip(self.param_names, result_skopt.x))}")
        print()

        result = self._create_result()
        result.print_summary()

        return result
