"""
Phase 3-2: パラメータ最適化 - ウォークフォワード分析器

ウォークフォワード分析によるパラメータ最適化システムを実装します。
スライディングウィンドウ方式でパラメータを最適化し、アウトオブサンプル性能を検証します。
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ztb.metrics.metrics import max_drawdown, sharpe_ratio
from ztb.utils.performance_profiler import PerformanceProfiler


@dataclass
class WalkForwardWindow:
    """ウォークフォワードウィンドウ設定"""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    window_id: int

    @property
    def train_days(self) -> int:
        """トレーニング期間の日数"""
        return (self.train_end - self.train_start).days

    @property
    def test_days(self) -> int:
        """テスト期間の日数"""
        return (self.test_end - self.test_start).days


@dataclass
class ParameterSet:
    """パラメータセット"""

    stop_loss_atr_multiplier: float
    take_profit_risk_multiplier: float
    position_size_kelly_fraction: float
    confidence_threshold: float
    max_positions: int
    name: str

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "stop_loss_atr_multiplier": self.stop_loss_atr_multiplier,
            "take_profit_risk_multiplier": self.take_profit_risk_multiplier,
            "position_size_kelly_fraction": self.position_size_kelly_fraction,
            "confidence_threshold": self.confidence_threshold,
            "max_positions": self.max_positions,
            "name": self.name,
        }


@dataclass
class OptimizationResult:
    """最適化結果"""

    window: WalkForwardWindow
    best_parameters: ParameterSet
    in_sample_performance: Dict[str, float]
    out_of_sample_performance: Dict[str, float]
    parameter_scores: List[Tuple[ParameterSet, Dict[str, float]]]

    @property
    def sharpe_ratio_improvement(self) -> float:
        """Sharpe Ratioの改善度"""
        in_sample = self.in_sample_performance.get("sharpe_ratio", 0)
        out_sample = self.out_of_sample_performance.get("sharpe_ratio", 0)
        return out_sample - in_sample

    @property
    def is_overfitted(self) -> bool:
        """過学習の判定"""
        return self.sharpe_ratio_improvement < -0.2  # 0.2以上の悪化は過学習


class ParameterSpace:
    """パラメータ空間定義"""

    def __init__(self):
        # ATRベースのストップロス乗数
        self.stop_loss_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]

        # リスク倍率ベースのテイクプロフィット
        self.take_profit_multipliers = [1.5, 2.0, 2.5, 3.0, 4.0]

        # Kelly基準のポジションサイズ（分数）
        self.kelly_fractions = [0.1, 0.2, 0.3, 0.5, 0.7]

        # 信頼度閾値
        self.confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

        # 最大ポジション数
        self.max_positions_list = [1, 3, 5, 10]

    def generate_parameter_sets(self) -> List[ParameterSet]:
        """全パラメータセットを生成"""
        parameter_sets = []

        for sl_mult in self.stop_loss_multipliers:
            for tp_mult in self.take_profit_multipliers:
                for kelly_frac in self.kelly_fractions:
                    for conf_thresh in self.confidence_thresholds:
                        for max_pos in self.max_positions_list:
                            name = f"SL{sl_mult}_TP{tp_mult}_KF{kelly_frac}_CT{conf_thresh}_MP{max_pos}"
                            param_set = ParameterSet(
                                stop_loss_atr_multiplier=sl_mult,
                                take_profit_risk_multiplier=tp_mult,
                                position_size_kelly_fraction=kelly_frac,
                                confidence_threshold=conf_thresh,
                                max_positions=max_pos,
                                name=name,
                            )
                            parameter_sets.append(param_set)

        return parameter_sets

    def get_conservative_defaults(self) -> ParameterSet:
        """保守的なデフォルトパラメータ"""
        return ParameterSet(
            stop_loss_atr_multiplier=2.0,
            take_profit_risk_multiplier=2.0,
            position_size_kelly_fraction=0.2,
            confidence_threshold=0.7,
            max_positions=3,
            name="conservative_defaults",
        )

    def get_aggressive_defaults(self) -> ParameterSet:
        """積極的なデフォルトパラメータ"""
        return ParameterSet(
            stop_loss_atr_multiplier=1.5,
            take_profit_risk_multiplier=3.0,
            position_size_kelly_fraction=0.5,
            confidence_threshold=0.6,
            max_positions=5,
            name="aggressive_defaults",
        )


class WalkForwardAnalyzer:
    """ウォークフォワード分析器"""

    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.logger = logging.getLogger(__name__)
        self.parameter_space = ParameterSpace()

    def create_sliding_windows(
        self,
        data: pd.DataFrame,
        train_days: int = 90,
        test_days: int = 30,
        step_days: int = 15,
        min_samples: int = 1000,
    ) -> List[WalkForwardWindow]:
        """
        スライディングウィンドウを作成

        Args:
            data: 市場データ
            train_days: トレーニング期間の日数
            test_days: テスト期間の日数
            step_days: ウィンドウのステップ日数
            min_samples: 最小サンプル数

        Returns:
            ウォークフォワードウィンドウのリスト
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("データインデックスはDatetimeIndexである必要があります")

        data_start = data.index.min()
        data_end = data.index.max()
        total_days = (data_end - data_start).days

        if total_days < train_days + test_days:
            raise ValueError(
                f"データ期間が不足しています: {total_days}日 < {train_days + test_days}日"
            )

        windows = []
        window_id = 0
        current_start = data_start

        while True:
            train_end = current_start + timedelta(days=train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_days)

            if test_end > data_end:
                break

            # データ存在チェック
            train_data = data.loc[current_start:train_end]
            test_data = data.loc[test_start:test_end]

            if len(train_data) >= min_samples and len(test_data) >= min_samples // 3:
                window = WalkForwardWindow(
                    train_start=current_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    window_id=window_id,
                )
                windows.append(window)
                window_id += 1

            current_start += timedelta(days=step_days)

        self.logger.info(f"作成されたウィンドウ数: {len(windows)}")
        return windows

    def optimize_parameters(
        self,
        train_data: pd.DataFrame,
        strategy_func: Callable[[pd.DataFrame, ParameterSet], Dict[str, float]],
        parameter_sets: Optional[List[ParameterSet]] = None,
    ) -> Tuple[ParameterSet, Dict[str, float]]:
        """
        トレーニングデータでパラメータを最適化

        Args:
            train_data: トレーニングデータ
            strategy_func: 戦略評価関数
            parameter_sets: 評価するパラメータセット（Noneの場合は全セット）

        Returns:
            最適パラメータとその性能
        """
        if parameter_sets is None:
            # デフォルトで保守的なパラメータセットのみを使用（計算時間を短縮）
            parameter_sets = [self.parameter_space.get_conservative_defaults()]

        self.logger.info(
            f"パラメータ最適化開始: {len(parameter_sets)}個のパラメータセットを評価"
        )

        best_score = float("-inf")
        best_params = None
        best_performance = None

        for param_set in parameter_sets:
            try:
                self.logger.debug(f"パラメータセット評価: {param_set.name}")
                performance = strategy_func(train_data, param_set)

                # Sharpe Ratioを最適化指標として使用
                score = performance.get("sharpe_ratio", float("-inf"))
                self.logger.debug(f"スコア: {score:.3f}")

                if score > best_score:
                    best_score = score
                    best_params = param_set
                    best_performance = performance

            except Exception as e:
                self.logger.warning(
                    f"パラメータセット {param_set.name} の評価に失敗: {e}"
                )
                continue

        if best_params is None:
            # デフォルトパラメータを使用
            best_params = self.parameter_space.get_conservative_defaults()
            try:
                best_performance = strategy_func(train_data, best_params)
            except Exception as e:
                self.logger.error(f"デフォルトパラメータ評価に失敗: {e}")
                best_performance = {
                    "sharpe_ratio": 0.0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                }

        # best_performanceがNoneでないことを保証
        if best_performance is None:
            best_performance = {
                "sharpe_ratio": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
            }

        self.logger.info(
            f"最適パラメータ: {best_params.name}, Sharpe Ratio: {best_performance['sharpe_ratio']:.3f}"
        )
        return best_params, best_performance

    def evaluate_out_of_sample(
        self,
        test_data: pd.DataFrame,
        parameters: ParameterSet,
        strategy_func: Callable[[pd.DataFrame, ParameterSet], Dict[str, float]],
    ) -> Dict[str, float]:
        """
        アウトオブサンプル性能を評価

        Args:
            test_data: テストデータ
            parameters: 最適化されたパラメータ
            strategy_func: 戦略評価関数

        Returns:
            アウトオブサンプル性能
        """
        try:
            performance = strategy_func(test_data, parameters)
            self.logger.info(
                f"アウトオブサンプル評価完了: Sharpe Ratio = {performance.get('sharpe_ratio', 0):.3f}"
            )
            return performance
        except Exception as e:
            self.logger.error(f"アウトオブサンプル評価に失敗: {e}")
            return {
                "sharpe_ratio": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
            }

    @PerformanceProfiler.profile
    def walk_forward_optimization(
        self,
        data: pd.DataFrame,
        strategy_func: Callable[[pd.DataFrame, ParameterSet], Dict[str, float]],
        train_days: int = 90,
        test_days: int = 30,
        step_days: int = 15,
        parameter_sets: Optional[List[ParameterSet]] = None,
        min_samples: int = 30,
    ) -> List[OptimizationResult]:
        """
        ウォークフォワード最適化を実行

        Args:
            data: 市場データ
            strategy_func: 戦略評価関数
            train_days: トレーニング期間の日数
            test_days: テスト期間の日数
            step_days: ステップ日数
            parameter_sets: 評価するパラメータセット

        Returns:
            各ウィンドウの最適化結果
        """
        self.logger.info("ウォークフォワード最適化を開始します")

        # ウィンドウ作成
        windows = self.create_sliding_windows(
            data, train_days, test_days, step_days, min_samples
        )

        results = []

        for window in windows:
            self.logger.info(
                f"ウィンドウ {window.window_id} の処理を開始: {window.train_start.date()} - {window.test_end.date()}"
            )

            # トレーニングデータとテストデータを分割
            train_data = pd.DataFrame(data.loc[window.train_start : window.train_end])
            test_data = pd.DataFrame(data.loc[window.test_start : window.test_end])

            # パラメータ最適化
            best_params, in_sample_perf = self.optimize_parameters(
                train_data, strategy_func, parameter_sets
            )

            # アウトオブサンプル評価
            out_sample_perf = self.evaluate_out_of_sample(
                test_data, best_params, strategy_func
            )

            # 結果保存
            result = OptimizationResult(
                window=window,
                best_parameters=best_params,
                in_sample_performance=in_sample_perf,
                out_of_sample_performance=out_sample_perf,
                parameter_scores=[],  # 詳細スコアはオプション
            )

            results.append(result)

            # 過学習チェック
            if result.is_overfitted:
                self.logger.warning(
                    f"ウィンドウ {window.window_id}: 過学習の兆候あり (Sharpe Ratio改善: {result.sharpe_ratio_improvement:.3f})"
                )

        self.logger.info(
            f"ウォークフォワード最適化完了: {len(results)}個のウィンドウを処理"
        )
        return results

    def summarize_results(self, results: List[OptimizationResult]) -> Dict[str, Any]:
        """
        最適化結果を要約

        Args:
            results: 最適化結果のリスト

        Returns:
            要約統計
        """
        if not results:
            return {}

        # Sharpe Ratioの統計
        in_sample_sharpes = [
            r.in_sample_performance.get("sharpe_ratio", 0) for r in results
        ]
        out_sample_sharpes = [
            r.out_of_sample_performance.get("sharpe_ratio", 0) for r in results
        ]

        # 過学習の割合
        overfitted_count = sum(1 for r in results if r.is_overfitted)

        # パラメータの安定性分析
        param_stability = self._analyze_parameter_stability(results)

        summary = {
            "total_windows": len(results),
            "in_sample_sharpe": {
                "mean": np.mean(in_sample_sharpes),
                "std": np.std(in_sample_sharpes),
                "min": np.min(in_sample_sharpes),
                "max": np.max(in_sample_sharpes),
            },
            "out_sample_sharpe": {
                "mean": np.mean(out_sample_sharpes),
                "std": np.std(out_sample_sharpes),
                "min": np.min(out_sample_sharpes),
                "max": np.max(out_sample_sharpes),
            },
            "overfitting_ratio": overfitted_count / len(results),
            "sharpe_improvement_avg": np.mean(
                [r.sharpe_ratio_improvement for r in results]
            ),
            "parameter_stability": param_stability,
            "recommendations": self._generate_recommendations(results),
        }

        return summary

    def _analyze_parameter_stability(
        self, results: List[OptimizationResult]
    ) -> Dict[str, Any]:
        """パラメータの安定性を分析"""
        if not results:
            return {}

        # 各パラメータの出現頻度をカウント
        param_counts = {}
        for result in results:
            param_dict = result.best_parameters.to_dict()
            param_key = (
                param_dict["stop_loss_atr_multiplier"],
                param_dict["take_profit_risk_multiplier"],
                param_dict["position_size_kelly_fraction"],
                param_dict["confidence_threshold"],
                param_dict["max_positions"],
            )

            param_counts[param_key] = param_counts.get(param_key, 0) + 1

        # 最頻パラメータ
        most_common = max(param_counts.items(), key=lambda x: x[1])
        stability_score = most_common[1] / len(results)

        return {
            "most_common_params": most_common[0],
            "stability_score": stability_score,
            "unique_param_sets": len(param_counts),
        }

    def _generate_recommendations(self, results: List[OptimizationResult]) -> List[str]:
        """推奨事項を生成"""
        recommendations = []

        if not results:
            return recommendations

        # Sharpe Ratio改善の平均
        avg_improvement = np.mean([r.sharpe_ratio_improvement for r in results])

        if avg_improvement > 0.1:
            recommendations.append("良好なアウトオブサンプル性能を確認")
        elif avg_improvement < -0.1:
            recommendations.append("過学習の可能性あり - パラメータを簡素化検討")

        # 安定性のチェック
        stability = self._analyze_parameter_stability(results)
        if stability.get("stability_score", 0) > 0.7:
            recommendations.append("パラメータが安定 - 信頼性の高い最適化")
        else:
            recommendations.append("パラメータが不安定 - より広いパラメータ範囲を検討")

        return recommendations


# ===== 戦略評価関数テンプレート =====


class BaseStrategyEvaluator(ABC):
    """戦略評価の基底クラス"""

    @abstractmethod
    def evaluate_strategy(
        self, data: pd.DataFrame, parameters: ParameterSet
    ) -> Dict[str, float]:
        """
        戦略を評価

        Args:
            data: 市場データ
            parameters: パラメータセット

        Returns:
            性能指標
        """
        pass

    def calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """Sharpe Ratioを計算"""
        return sharpe_ratio(returns, rf=risk_free_rate, period_per_year=252)

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """最大ドローダウンを計算"""
        return max_drawdown(equity_curve)

    def calculate_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """勝率を計算"""
        if not trades:
            return 0.0

        winning_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
        return winning_trades / len(trades)


# ===== 使用例 =====

if __name__ == "__main__":
    """使用例"""

    # ウォークフォワード分析器の初期化
    analyzer = WalkForwardAnalyzer()

    # パラメータ空間の確認
    param_space = ParameterSpace()
    print(f"パラメータセット数: {len(param_space.generate_parameter_sets())}")

    # デフォルトパラメータの確認
    conservative = param_space.get_conservative_defaults()
    aggressive = param_space.get_aggressive_defaults()

    print(f"保守的デフォルト: {conservative.to_dict()}")
    print(f"積極的デフォルト: {aggressive.to_dict()}")
