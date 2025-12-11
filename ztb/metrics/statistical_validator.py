#!/usr/bin/env python3
"""
Statistical Validator for SAC v445
Phase 3: Risk Management & Statistical Validation

既存の統計ユーティリティを統合した統計的検証フレームワーク。
多重検定補正、信頼区間評価、統計的有意性検定を提供。
"""

from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ztb.metrics import sharpe_ratio
from ztb.metrics.metrics import kurtosis, skewness
from ztb.metrics.statistics import calculate_volatility, rolling_statistics
from ztb.metrics.technical import calculate_volatility_from_returns
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class StatisticalValidator:
    """
    統計的検証フレームワーク

    多重検定補正、信頼区間評価、統計的有意性検定を統合した
    包括的な統計検証システム。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 統計検証設定
        """
        self.config = config

        # 多重検定補正設定
        self.multiple_test_method = config.get("multiple_test_method", "bonferroni")
        self.alpha_level = config.get("alpha_level", 0.05)

        # 信頼区間設定
        self.confidence_level = config.get("confidence_level", 0.95)
        self.bootstrap_samples = config.get("bootstrap_samples", 10000)

        # 性能評価設定
        self.min_sample_size = config.get("min_sample_size", 100)
        self.stability_window = config.get("stability_window", 50)

        logger.info(
            f"StatisticalValidator initialized with alpha={self.alpha_level}, "
            f"confidence={self.confidence_level}"
        )

    def validate_performance_metrics(
        self,
        returns: List[float],
        benchmark_returns: Optional[List[float]] = None,
        multiple_metrics: bool = False,
    ) -> Dict[str, Any]:
        """
        性能指標の統計的検証

        Args:
            returns: 戦略のリターン系列
            benchmark_returns: ベンチマークのリターン系列
            multiple_metrics: 多重検定補正を適用するか

        Returns:
            統計的検証結果
        """
        if len(returns) < self.min_sample_size:
            return {
                "error": f"Insufficient sample size: {len(returns)} < {self.min_sample_size}",
                "valid": False,
            }

        results: Dict[str, Any] = {}

        try:
            # 基本統計量
            results["basic_stats"] = self._calculate_basic_statistics(returns)

            # Sharpe ratio分析
            # sharpe_ratio expects Series or ndarray
            returns_np = np.array(returns)
            sharpe = sharpe_ratio(returns_np)
            results["sharpe_ratio"] = {
                "value": sharpe,
                "confidence_interval": self._calculate_sharpe_confidence_interval(
                    returns
                ),
                "stability": self._assess_metric_stability(returns, "sharpe"),
            }

            # ベンチマーク比較
            if benchmark_returns:
                results["benchmark_comparison"] = self._compare_with_benchmark(
                    returns, benchmark_returns
                )

            # 多重検定補正（複数の指標を評価する場合）
            if multiple_metrics:
                results["multiple_testing"] = self._apply_multiple_testing_correction(
                    [
                        results["sharpe_ratio"]["value"],
                        results["basic_stats"]["mean"],
                        results["basic_stats"]["volatility"],
                    ]
                )

            # 安定性評価
            results["stability_analysis"] = self._analyze_performance_stability(returns)

            results["valid"] = True

        except Exception as e:
            logger.error(f"Performance metrics validation failed: {e}")
            results["error"] = str(e)
            results["valid"] = False

        return results

    def validate_multiple_strategies(
        self,
        strategy_returns: Dict[str, List[float]],
        benchmark_returns: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        複数戦略の統計的比較検証

        Args:
            strategy_returns: 戦略名 -> リターン系列の辞書
            benchmark_returns: ベンチマークのリターン系列

        Returns:
            複数戦略比較結果
        """
        if len(strategy_returns) < 2:
            return {
                "error": "Need at least 2 strategies for comparison",
                "valid": False,
            }

        results: Dict[str, Any] = {}

        try:
            # 個別戦略の検証
            individual_results: Dict[str, Any] = {}
            all_sharpes = []

            for strategy_name, returns in strategy_returns.items():
                if len(returns) >= self.min_sample_size:
                    validation = self.validate_performance_metrics(
                        returns, benchmark_returns
                    )
                    individual_results[strategy_name] = validation

                    if validation.get("valid"):
                        all_sharpes.append(validation["sharpe_ratio"]["value"])
                    else:
                        all_sharpes.append(0.0)  # 無効な場合は0として扱う
                else:
                    individual_results[strategy_name] = {
                        "error": f"Insufficient sample size: {len(returns)}",
                        "valid": False,
                    }
                    all_sharpes.append(0.0)

            results["individual_results"] = individual_results

            # 戦略間比較
            if len([s for s in all_sharpes if s != 0.0]) >= 2:
                results["strategy_comparison"] = self._compare_strategies_statistically(
                    strategy_returns
                )

            # 多重検定補正
            results[
                "multiple_testing_correction"
            ] = self._apply_multiple_testing_correction(all_sharpes)

            results["valid"] = True

        except Exception as e:
            logger.error(f"Multiple strategies validation failed: {e}")
            results["error"] = str(e)
            results["valid"] = False

        return results

    def validate_signal_quality(
        self,
        predictions: List[float],
        actual_returns: List[float],
        signal_thresholds: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        シグナル品質の統計的検証

        Args:
            predictions: 予測値系列
            actual_returns: 実際のリターン系列
            signal_thresholds: シグナル閾値リスト

        Returns:
            シグナル品質検証結果
        """
        if len(predictions) != len(actual_returns):
            return {"error": "Predictions and returns length mismatch", "valid": False}

        if len(predictions) < self.min_sample_size:
            return {
                "error": f"Insufficient sample size: {len(predictions)}",
                "valid": False,
            }

        results: Dict[str, Any] = {}

        try:
            # 基本相関分析
            correlation = np.corrcoef(predictions, actual_returns)[0, 1]
            results["correlation"] = {
                "value": correlation,
                "p_value": self._calculate_correlation_p_value(
                    predictions, actual_returns
                ),
                "confidence_interval": self._calculate_correlation_confidence_interval(
                    predictions, actual_returns
                ),
            }

            # シグナル閾値別分析
            if signal_thresholds:
                results["threshold_analysis"] = self._analyze_signal_thresholds(
                    predictions, actual_returns, signal_thresholds
                )

            # 予測精度分析
            results["prediction_accuracy"] = self._analyze_prediction_accuracy(
                predictions, actual_returns
            )

            results["valid"] = True

        except Exception as e:
            logger.error(f"Signal quality validation failed: {e}")
            results["error"] = str(e)
            results["valid"] = False

        return results

    def _calculate_basic_statistics(self, returns: List[float]) -> Dict[str, float]:
        """基本統計量計算"""
        returns_array = np.array(returns)

        return {
            "mean": float(np.mean(returns_array)),
            "std": float(np.std(returns_array, ddof=1)),
            "skewness": float(skewness(returns_array)),
            "kurtosis": float(kurtosis(returns_array)),
            "min": float(np.min(returns_array)),
            "max": float(np.max(returns_array)),
            "volatility": calculate_volatility_from_returns(
                returns_array, window=len(returns_array), annualize=True
            ),
            "sharpe_ratio": sharpe_ratio(returns_array),
        }

    def _calculate_sharpe_confidence_interval(
        self, returns: List[float]
    ) -> List[float]:
        """Sharpe ratioの信頼区間計算"""
        # ブートストラップ法による信頼区間推定
        sharpes = []
        n_samples = len(returns)

        for _ in range(self.bootstrap_samples):
            bootstrap_sample = np.random.choice(returns, size=n_samples, replace=True)
            sharpe = sharpe_ratio(bootstrap_sample)
            sharpes.append(sharpe)

        return [
            float(np.percentile(sharpes, (1 - self.confidence_level) * 100 / 2)),
            float(np.percentile(sharpes, (1 + self.confidence_level) * 100 / 2)),
        ]

    def _assess_metric_stability(
        self, returns: List[float], metric: str
    ) -> Dict[str, Any]:
        """指標の安定性評価"""
        if len(returns) < self.stability_window * 2:
            return {
                "stable": False,
                "reason": "Insufficient data for stability analysis",
            }

        # ローリング統計量計算
        rolling_stats = rolling_statistics(returns, self.stability_window)

        # 安定性指標
        mean_std = np.std(rolling_stats["mean"])
        std_std = np.std(rolling_stats["std"])

        # 安定性の判断
        stability_score = 1.0 / (1.0 + mean_std + std_std)  # 0-1のスコア

        return {
            "stable": stability_score > 0.7,
            "stability_score": float(stability_score),
            "rolling_mean_std": float(mean_std),
            "rolling_std_std": float(std_std),
        }

    def _compare_with_benchmark(
        self, strategy_returns: List[float], benchmark_returns: List[float]
    ) -> Dict[str, Any]:
        """ベンチマークとの統計的比較"""
        # t検定
        t_stat, p_value = stats.ttest_ind(strategy_returns, benchmark_returns)

        # 効果量（Cohen's d）
        mean_diff = np.mean(strategy_returns) - np.mean(benchmark_returns)
        pooled_std = np.sqrt(
            (
                np.std(strategy_returns, ddof=1) ** 2
                + np.std(benchmark_returns, ddof=1) ** 2
            )
            / 2
        )
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha_level,
            "effect_size": float(effect_size),
            "mean_difference": float(mean_diff),
        }

    def _compare_strategies_statistically(
        self, strategy_returns: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """戦略間の統計的比較（ANOVA）"""
        # 有効な戦略のみ抽出
        valid_strategies = {
            name: returns
            for name, returns in strategy_returns.items()
            if len(returns) >= self.min_sample_size
        }

        if len(valid_strategies) < 2:
            return {"error": "Need at least 2 valid strategies"}

        # ANOVA
        f_stat, p_value = stats.f_oneway(*valid_strategies.values())

        # Tukey HSD事後検定（簡易版）
        means = {name: np.mean(returns) for name, returns in valid_strategies.items()}
        best_strategy = max(means.items(), key=lambda x: x[1])

        return {
            "anova_f_stat": float(f_stat),
            "anova_p_value": float(p_value),
            "significant_difference": p_value < self.alpha_level,
            "best_strategy": best_strategy[0],
            "strategy_means": {name: float(mean) for name, mean in means.items()},
        }

    def _apply_multiple_testing_correction(
        self, p_values: List[float]
    ) -> Dict[str, Any]:
        """多重検定補正適用"""
        if not p_values:
            return {"error": "No p-values provided"}

        # statsmodelsのmultipletestsを使用
        rejected, p_adjusted, _, alpha_corrected = multipletests(
            p_values, alpha=self.alpha_level, method=self.multiple_test_method
        )

        return {
            "original_p_values": p_values,
            "adjusted_p_values": p_adjusted.tolist(),
            "rejected": rejected.tolist(),
            "method": self.multiple_test_method,
            "alpha_corrected": float(alpha_corrected),
        }

    def _analyze_performance_stability(self, returns: List[float]) -> Dict[str, Any]:
        """性能安定性分析"""
        # ドローダウン分析
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative

        max_drawdown = np.max(drawdowns)
        avg_drawdown = np.mean(drawdowns[drawdowns > 0])

        # ボラティリティ分析
        volatility = calculate_volatility(returns, window=min(20, len(returns) // 2))

        return {
            "max_drawdown": float(max_drawdown),
            "avg_drawdown": float(avg_drawdown)
            if len(drawdowns[drawdowns > 0]) > 0
            else 0.0,
            "volatility_trend": volatility,
            "volatility_stability": float(np.std(volatility)) if volatility else 0.0,
        }

    def _calculate_correlation_p_value(self, x: List[float], y: List[float]) -> float:
        """相関係数のp値計算"""
        correlation, p_value = stats.pearsonr(x, y)
        return float(p_value)

    def _calculate_correlation_confidence_interval(
        self, x: List[float], y: List[float]
    ) -> List[float]:
        """相関係数の信頼区間計算（Fisher変換使用）"""
        correlation, _ = stats.pearsonr(x, y)

        # Fisher変換
        z = np.arctanh(correlation)
        se = 1 / np.sqrt(len(x) - 3)

        z_critical = stats.norm.ppf((1 + self.confidence_level) / 2)
        z_lower = z - z_critical * se
        z_upper = z + z_critical * se

        # 逆Fisher変換
        r_lower = np.tanh(z_lower)
        r_upper = np.tanh(z_upper)

        return [float(r_lower), float(r_upper)]

    def _analyze_signal_thresholds(
        self,
        predictions: List[float],
        actual_returns: List[float],
        thresholds: List[float],
    ) -> Dict[str, Any]:
        """シグナル閾値別分析"""
        results = {}

        for threshold in thresholds:
            # 閾値以上のシグナル
            strong_signals = [
                r for p, r in zip(predictions, actual_returns) if p >= threshold
            ]
            weak_signals = [
                r for p, r in zip(predictions, actual_returns) if p < threshold
            ]

            if strong_signals and weak_signals:
                # t検定
                t_stat, p_value = stats.ttest_ind(strong_signals, weak_signals)

                results[f"threshold_{threshold}"] = {
                    "strong_signals_count": len(strong_signals),
                    "weak_signals_count": len(weak_signals),
                    "strong_mean_return": float(np.mean(strong_signals)),
                    "weak_mean_return": float(np.mean(weak_signals)),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < self.alpha_level,
                }

        return results

    def _analyze_prediction_accuracy(
        self, predictions: List[float], actual_returns: List[float]
    ) -> Dict[str, Any]:
        """予測精度分析"""
        # 方向予測精度
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actual_returns)

        direction_accuracy = np.mean(pred_direction == actual_direction)

        # MSE, MAE
        mse = np.mean((np.array(predictions) - np.array(actual_returns)) ** 2)
        mae = np.mean(np.abs(np.array(predictions) - np.array(actual_returns)))

        return {
            "direction_accuracy": float(direction_accuracy),
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(np.sqrt(mse)),
        }
