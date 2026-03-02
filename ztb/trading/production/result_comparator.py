"""
V433 Phase 5: Parallel Running Layer - Result Comparator

両システムの並行比較と統計的有意性検証を行う。
"""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Awaitable, Callable

import scipy.stats as stats
from ztb.trading.production.state_persistence import (
    read_state_payload,
    write_state_payload,
)

# Mock classes for testing

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class Position:
    symbol: str
    quantity: Decimal
    average_price: Decimal
    current_price: Decimal | None = None
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

class Order:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal | None = None
    average_price: Decimal
    current_price: Decimal | None = None
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

@dataclass
class Trade:
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    fee: Decimal = Decimal("0")

class ComparisonMetric(Enum):
    """比較指標"""

    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    AVG_WIN = "avg_win"
    AVG_LOSS = "avg_loss"
    TOTAL_TRADES = "total_trades"
    EXECUTION_LATENCY = "execution_latency"

class StatisticalTest(Enum):
    """統計テスト"""

    T_TEST = "t_test"  # t検定
    MANN_WHITNEY = "mann_whitney"  # Mann-Whitney U検定
    WILCOXON = "wilcoxon"  # Wilcoxon符号順位検定
    KS_TEST = "ks_test"  # Kolmogorov-Smirnov検定
    LEVENE = "levene"  # Levene検定（等分散性）

@dataclass
class SystemResult:
    """システム結果"""

    system_id: str
    timestamp: datetime
    total_return: Decimal
    sharpe_ratio: float
    max_drawdown: Decimal
    win_rate: float
    profit_factor: float
    avg_win: Decimal
    avg_loss: Decimal
    total_trades: int
    execution_latency_ms: float
    trades: list[Trade] = field(default_factory=list)
    positions: list[Position] = field(default_factory=list)

@dataclass
class ComparisonResult:
    """比較結果"""

    comparison_id: str
    timestamp: datetime
    system_a: str
    system_b: str
    metric: ComparisonMetric
    value_a: float
    value_b: float
    difference: float
    percent_difference: float
    statistical_tests: dict[str, dict[str, Any]] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    effect_size: float | None = None
    interpretation: str = ""

@dataclass
class ComparativeAnalysis:
    """比較分析"""

    analysis_id: str
    timestamp: datetime
    period_start: datetime
    period_end: datetime
    systems_compared: list[str]
    overall_winner: str | None
    confidence_level: float
    key_findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    comparison_results: list[ComparisonResult] = field(default_factory=list)

class ResultComparator:
    """
    結果比較器

    両システムの並行比較と統計的有意性検証を行い、
    パフォーマンスの優劣を定量的に評価する。
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        min_sample_size: int = 30,
        comparison_window_hours: int = 24,
    ):
        """
        初期化

        Args:
            confidence_level: 信頼水準
            min_sample_size: 最小サンプルサイズ
            comparison_window_hours: 比較ウィンドウ（時間）
        """
        self.confidence_level = confidence_level
        self.min_sample_size = min_sample_size
        self.comparison_window_hours = comparison_window_hours

        # システム結果履歴
        self.system_results: dict[str, list[SystemResult]] = {}

        # 比較分析履歴
        self.comparative_analyses: list[ComparativeAnalysis] = []

        # 比較設定
        self.enabled_metrics = {metric for metric in ComparisonMetric}
        self.enabled_tests = {test for test in StatisticalTest}

        # コールバック
        self.analysis_callbacks: list[
            Callable[[ComparativeAnalysis], Awaitable[None]]
        ] = []

        # ロギング
        self.logger = logging.getLogger(__name__)

        self.logger.info("Result Comparator initialized")

    def submit_system_result(self, result: SystemResult) -> None:
        """
        システム結果提出

        Args:
            result: システム結果
        """
        if result.system_id not in self.system_results:
            self.system_results[result.system_id] = []

        self.system_results[result.system_id].append(result)

        # 履歴サイズ制限（最新1000件）
        if len(self.system_results[result.system_id]) > 1000:
            self.system_results[result.system_id] = self.system_results[
                result.system_id
            ][-1000:]

        self.logger.debug(
            f"System result submitted: {result.system_id} at {result.timestamp}"
        )

    async def perform_comparison(
        self, system_a: str, system_b: str, analysis_period_hours: int | None = None
    ) -> ComparativeAnalysis | None:
        """
        比較実行

        Args:
            system_a: システムA ID
            system_b: システムB ID
            analysis_period_hours: 分析期間（時間）

        Returns:
            ComparativeAnalysis | None: 比較分析結果
        """
        period_hours = analysis_period_hours or self.comparison_window_hours
        period_start = datetime.now() - timedelta(hours=period_hours)

        try:
            # 期間内の結果取得
            results_a = self._get_results_in_period(system_a, period_start)
            results_b = self._get_results_in_period(system_b, period_start)

            if (
                len(results_a) < self.min_sample_size
                or len(results_b) < self.min_sample_size
            ):
                self.logger.warning(
                    f"Insufficient data for comparison: {system_a}={len(results_a)}, {system_b}={len(results_b)}"
                )
                return None

            # 比較分析実行
            analysis = await self._analyze_comparison(
                system_a, system_b, results_a, results_b, period_start
            )

            if analysis:
                self.comparative_analyses.append(analysis)

                # コールバック実行
                for callback in self.analysis_callbacks:
                    try:
                        await callback(analysis)
                    except Exception as e:
                        self.logger.error(f"Analysis callback error: {e}")

                self.logger.info(f"Comparison completed: {system_a} vs {system_b}")

            return analysis

        except Exception as e:
            self.logger.error(f"Comparison failed: {e}")
            return None

    def _get_results_in_period(
        self, system_id: str, period_start: datetime
    ) -> list[SystemResult]:
        """
        期間内の結果取得

        Args:
            system_id: システムID
            period_start: 期間開始

        Returns:
            list[SystemResult]: 期間内の結果
        """
        if system_id not in self.system_results:
            return []

        return [
            r for r in self.system_results[system_id] if r.timestamp >= period_start
        ]

    async def _analyze_comparison(
        self,
        system_a: str,
        system_b: str,
        results_a: list[SystemResult],
        results_b: list[SystemResult],
        period_start: datetime,
    ) -> ComparativeAnalysis:
        """
        比較分析

        Args:
            system_a: システムA ID
            system_b: システムB ID
            results_a: システムA結果
            results_b: システムB結果
            period_start: 期間開始

        Returns:
            ComparativeAnalysis: 比較分析
        """
        analysis = ComparativeAnalysis(
            analysis_id=f"CMP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            period_start=period_start,
            period_end=datetime.now(),
            systems_compared=[system_a, system_b],
            overall_winner=None,
            confidence_level=self.confidence_level,
        )

        # 各指標の比較
        for metric in self.enabled_metrics:
            comparison = await self._compare_metric(
                system_a, system_b, metric, results_a, results_b
            )
            if comparison:
                analysis.comparison_results.append(comparison)

        # 全体勝者決定
        analysis.overall_winner = self._determine_overall_winner(
            analysis.comparison_results
        )

        # 主要発見と推奨事項生成
        (
            analysis.key_findings,
            analysis.recommendations,
        ) = self._generate_findings_and_recommendations(analysis)

        return analysis

    async def _compare_metric(
        self,
        system_a: str,
        system_b: str,
        metric: ComparisonMetric,
        results_a: list[SystemResult],
        results_b: list[SystemResult],
    ) -> ComparisonResult | None:
        """
        指標比較

        Args:
            system_a: システムA ID
            system_b: システムB ID
            metric: 比較指標
            results_a: システムA結果
            results_b: システムB結果

        Returns:
            ComparisonResult | None: 比較結果
        """
        try:
            # 指標値抽出
            values_a = self._extract_metric_values(results_a, metric)
            values_b = self._extract_metric_values(results_b, metric)

            if not values_a or not values_b:
                return None

            # 平均値計算
            mean_a = statistics.mean(values_a)
            mean_b = statistics.mean(values_b)

            # 差分計算
            difference = mean_b - mean_a  # B - A
            percent_difference = (difference / mean_a * 100) if mean_a != 0 else 0

            comparison = ComparisonResult(
                comparison_id=f"CMP_{metric.value}_{datetime.now().strftime('%H%M%S')}",
                timestamp=datetime.now(),
                system_a=system_a,
                system_b=system_b,
                metric=metric,
                value_a=mean_a,
                value_b=mean_b,
                difference=difference,
                percent_difference=percent_difference,
            )

            # 統計テスト実行
            if (
                len(values_a) >= self.min_sample_size
                and len(values_b) >= self.min_sample_size
            ):
                comparison.statistical_tests = await self._run_statistical_tests(
                    values_a, values_b
                )
                comparison.confidence_intervals = self._calculate_confidence_intervals(
                    values_a, values_b
                )
                comparison.effect_size = self._calculate_effect_size(values_a, values_b)

            # 解釈生成
            comparison.interpretation = self._interpret_comparison(comparison)

            return comparison

        except Exception as e:
            self.logger.error(f"Metric comparison failed for {metric.value}: {e}")
            return None

    def _extract_metric_values(
        self, results: list[SystemResult], metric: ComparisonMetric
    ) -> list[float]:
        """
        指標値抽出

        Args:
            results: システム結果
            metric: 比較指標

        Returns:
            list[float]: 指標値リスト
        """
        values = []

        for result in results:
            if metric == ComparisonMetric.TOTAL_RETURN:
                values.append(float(result.total_return))
            elif metric == ComparisonMetric.SHARPE_RATIO:
                values.append(result.sharpe_ratio)
            elif metric == ComparisonMetric.MAX_DRAWDOWN:
                values.append(float(result.max_drawdown))
            elif metric == ComparisonMetric.WIN_RATE:
                values.append(result.win_rate)
            elif metric == ComparisonMetric.PROFIT_FACTOR:
                values.append(result.profit_factor)
            elif metric == ComparisonMetric.AVG_WIN:
                values.append(float(result.avg_win))
            elif metric == ComparisonMetric.AVG_LOSS:
                values.append(float(result.avg_loss))
            elif metric == ComparisonMetric.TOTAL_TRADES:
                values.append(result.total_trades)
            elif metric == ComparisonMetric.EXECUTION_LATENCY:
                values.append(result.execution_latency_ms)

        return values

    async def _run_statistical_tests(
        self, values_a: list[float], values_b: list[float]
    ) -> dict[str, dict[str, Any]]:
        """
        統計テスト実行

        Args:
            values_a: システムAの値
            values_b: システムBの値

        Returns:
            dict[str, dict[str, Any]]: テスト結果
        """
        results = {}

        try:
            # 等分散性テスト（Levene）
            if StatisticalTest.LEVENE in self.enabled_tests:
                stat, p_value = stats.levene(values_a, values_b)
                results["levene"] = {
                    "statistic": stat,
                    "p_value": p_value,
                    "equal_variance": p_value > (1 - self.confidence_level),
                }

            equal_var = results.get("levene", {}).get("equal_variance", True)

            # t検定
            if StatisticalTest.T_TEST in self.enabled_tests:
                stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=equal_var)
                results["t_test"] = {
                    "statistic": stat,
                    "p_value": p_value,
                    "significant": p_value < (1 - self.confidence_level),
                }

            # Mann-Whitney U検定
            if StatisticalTest.MANN_WHITNEY in self.enabled_tests:
                stat, p_value = stats.mannwhitneyu(
                    values_a, values_b, alternative="two-sided"
                )
                results["mann_whitney"] = {
                    "statistic": stat,
                    "p_value": p_value,
                    "significant": p_value < (1 - self.confidence_level),
                }

            # Kolmogorov-Smirnov検定
            if StatisticalTest.KS_TEST in self.enabled_tests:
                stat, p_value = stats.ks_2samp(values_a, values_b)
                results["ks_test"] = {
                    "statistic": stat,
                    "p_value": p_value,
                    "significant": p_value < (1 - self.confidence_level),
                }

        except Exception as e:
            self.logger.error(f"Statistical test error: {e}")

        return results

    def _calculate_confidence_intervals(
        self, values_a: list[float], values_b: list[float]
    ) -> dict[str, tuple[float, float]]:
        """
        信頼区間計算

        Args:
            values_a: システムAの値
            values_b: システムBの値

        Returns:
            dict[str, tuple[float, float]]: 信頼区間
        """
        intervals = {}

        try:
            # 平均の差の信頼区間
            mean_a = statistics.mean(values_a)
            mean_b = statistics.mean(values_b)
            std_a = statistics.stdev(values_a) if len(values_a) > 1 else 0
            std_b = statistics.stdev(values_b) if len(values_b) > 1 else 0

            n_a, n_b = len(values_a), len(values_b)
            se_diff = ((std_a**2 / n_a) + (std_b**2 / n_b)) ** 0.5

            if se_diff > 0:
                t_value = stats.t.ppf((1 + self.confidence_level) / 2, n_a + n_b - 2)
                margin = t_value * se_diff
                diff_mean = mean_b - mean_a

                intervals["mean_difference"] = (diff_mean - margin, diff_mean + margin)

        except Exception as e:
            self.logger.error(f"Confidence interval calculation error: {e}")

        return intervals

    def _calculate_effect_size(
        self, values_a: list[float], values_b: list[float]
    ) -> float | None:
        """
        効果量計算（Cohen's d）

        Args:
            values_a: システムAの値
            values_b: システムBの値

        Returns:
            float | None: 効果量
        """
        try:
            mean_a = statistics.mean(values_a)
            mean_b = statistics.mean(values_b)
            std_a = statistics.stdev(values_a) if len(values_a) > 1 else 0
            std_b = statistics.stdev(values_b) if len(values_b) > 1 else 0

            # プール標準偏差
            pooled_std = ((std_a**2 + std_b**2) / 2) ** 0.5

            if pooled_std > 0:
                return (mean_b - mean_a) / pooled_std

        except Exception as e:
            self.logger.error(f"Effect size calculation error: {e}")

        return None

    def _interpret_comparison(self, comparison: ComparisonResult) -> str:
        """
        比較解釈

        Args:
            comparison: 比較結果

        Returns:
            str: 解釈文
        """
        metric_name = comparison.metric.value.replace("_", " ").title()
        diff = comparison.difference
        percent = comparison.percent_difference

        # 有意性チェック
        significant = any(
            test.get("significant", False)
            for test in comparison.statistical_tests.values()
        )

        if abs(percent) < 1:
            return f"{metric_name}: No significant difference between systems"
        elif significant:
            if diff > 0:
                return f"{metric_name}: System B outperforms System A by {abs(percent):.1f}% (statistically significant)"
            else:
                return f"{metric_name}: System A outperforms System B by {abs(percent):.1f}% (statistically significant)"
        else:
            if diff > 0:
                return f"{metric_name}: System B shows {abs(percent):.1f}% advantage (not statistically significant)"
            else:
                return f"{metric_name}: System A shows {abs(percent):.1f}% advantage (not statistically significant)"

    def _determine_overall_winner(
        self, comparisons: list[ComparisonResult]
    ) -> str | None:
        """
        全体勝者決定

        Args:
            comparisons: 比較結果リスト

        Returns:
            str | None: 勝者システムID
        """
        if not comparisons:
            return None

        # 重要な指標に重み付け
        weights = {
            ComparisonMetric.TOTAL_RETURN: 3,
            ComparisonMetric.SHARPE_RATIO: 3,
            ComparisonMetric.MAX_DRAWDOWN: 2,
            ComparisonMetric.WIN_RATE: 2,
            ComparisonMetric.PROFIT_FACTOR: 2,
            ComparisonMetric.EXECUTION_LATENCY: 1,
        }

        score_a = 0
        score_b = 0

        for comparison in comparisons:
            if comparison.metric not in weights:
                continue

            weight = weights[comparison.metric]
            significant = any(
                test.get("significant", False)
                for test in comparison.statistical_tests.values()
            )

            if significant:
                if comparison.difference > 0:  # Bが優位
                    score_b += weight
                elif comparison.difference < 0:  # Aが優位
                    score_a += weight

        if score_a > score_b:
            return comparisons[0].system_a
        elif score_b > score_a:
            return comparisons[0].system_b
        else:
            return None  # 引き分け

    def _generate_findings_and_recommendations(
        self, analysis: ComparativeAnalysis
    ) -> tuple[list[str], list[str]]:
        """
        発見と推奨事項生成

        Args:
            analysis: 比較分析

        Returns:
            tuple[list[str], list[str]]: (発見, 推奨事項)
        """
        findings = []
        recommendations = []

        if not analysis.comparison_results:
            findings.append("Insufficient data for meaningful comparison")
            recommendations.append(
                "Collect more performance data before making decisions"
            )
            return findings, recommendations

        # 勝者分析
        if analysis.overall_winner:
            findings.append(
                f"{analysis.overall_winner} shows superior overall performance"
            )
            if analysis.overall_winner == analysis.systems_compared[1]:  # 新システム
                recommendations.append(
                    "Consider increasing traffic allocation to the better performing system"
                )
        else:
            findings.append("No clear winner in overall performance comparison")

        # 各指標の分析
        significant_improvements = []
        concerning_metrics = []

        for result in analysis.comparison_results:
            significant = any(
                test.get("significant", False)
                for test in result.statistical_tests.values()
            )

            if significant:
                if (
                    result.metric == ComparisonMetric.MAX_DRAWDOWN
                    and result.difference < 0
                ):
                    # ドローダウンが小さい方が良い
                    significant_improvements.append(
                        f"Lower maximum drawdown ({abs(result.percent_difference):.1f}% improvement)"
                    )
                elif (
                    result.metric == ComparisonMetric.EXECUTION_LATENCY
                    and result.difference < 0
                ):
                    # 遅延が小さい方が良い
                    significant_improvements.append(
                        f"Faster execution latency ({abs(result.percent_difference):.1f}% improvement)"
                    )
                elif result.difference > 0:
                    significant_improvements.append(
                        f"Better {result.metric.value} ({result.percent_difference:.1f}% improvement)"
                    )
                elif result.difference < 0:
                    concerning_metrics.append(
                        f"Worse {result.metric.value} ({abs(result.percent_difference):.1f}% decline)"
                    )

        findings.extend(significant_improvements)
        findings.extend(concerning_metrics)

        # 推奨事項生成
        if significant_improvements:
            recommendations.append("Monitor the improving metrics closely")
        if concerning_metrics:
            recommendations.append("Investigate causes of performance degradation")

        # 統計的有意性の分析
        significant_tests = sum(
            1
            for r in analysis.comparison_results
            for t in r.statistical_tests.values()
            if t.get("significant", False)
        )

        if significant_tests > len(analysis.comparison_results) * 0.5:
            findings.append("Strong statistical evidence of performance differences")
            recommendations.append(
                "Consider confidence in performance differences for decision making"
            )
        else:
            findings.append(
                "Limited statistical significance in performance differences"
            )
            recommendations.append("Continue monitoring and collect more data")

        return findings, recommendations

    def get_latest_analysis(
        self, system_a: str, system_b: str
    ) -> ComparativeAnalysis | None:
        """
        最新分析取得

        Args:
            system_a: システムA ID
            system_b: システムB ID

        Returns:
            ComparativeAnalysis | None: 最新比較分析
        """
        for analysis in reversed(self.comparative_analyses):
            if set(analysis.systems_compared) == {system_a, system_b}:
                return analysis
        return None

    def get_analysis_history(
        self, system_a: str, system_b: str, limit: int | None = None
    ) -> list[ComparativeAnalysis]:
        """
        分析履歴取得

        Args:
            system_a: システムA ID
            system_b: システムB ID
            limit: 取得件数制限

        Returns:
            list[ComparativeAnalysis]: 比較分析履歴
        """
        history = [
            a
            for a in self.comparative_analyses
            if set(a.systems_compared) == {system_a, system_b}
        ]

        if limit:
            history = history[-limit:]

        return history

    def get_performance_summary(
        self, system_id: str, hours: int = 24
    ) -> dict[str, Any] | None:
        """
        パフォーマンス要約取得

        Args:
            system_id: システムID
            hours: 集計期間（時間）

        Returns:
            dict[str, Any] | None: パフォーマンス要約
        """
        period_start = datetime.now() - timedelta(hours=hours)
        results = self._get_results_in_period(system_id, period_start)

        if not results:
            return None

        # 各指標の統計
        metrics_summary = {}
        for metric in ComparisonMetric:
            values = self._extract_metric_values(results, metric)
            if values:
                metrics_summary[metric.value] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        return {
            "system_id": system_id,
            "period_hours": hours,
            "total_results": len(results),
            "metrics": metrics_summary,
        }

    def add_analysis_callback(
        self, callback: Callable[[ComparativeAnalysis], Awaitable[None]]
    ) -> None:
        """
        分析コールバック追加

        Args:
            callback: コールバック関数
        """
        self.analysis_callbacks.append(callback)

    def save_state(self, filepath: str) -> None:
        """
        状態保存

        Args:
            filepath: 保存ファイルパス
        """
        state = {
            "confidence_level": self.confidence_level,
            "min_sample_size": self.min_sample_size,
            "comparison_window_hours": self.comparison_window_hours,
            "enabled_metrics": [m.value for m in self.enabled_metrics],
            "enabled_tests": [t.value for t in self.enabled_tests],
            "comparative_analyses": [
                {
                    "analysis_id": a.analysis_id,
                    "timestamp": a.timestamp.isoformat(),
                    "period_start": a.period_start.isoformat(),
                    "period_end": a.period_end.isoformat(),
                    "systems_compared": a.systems_compared,
                    "overall_winner": a.overall_winner,
                    "confidence_level": a.confidence_level,
                    "key_findings": a.key_findings,
                    "recommendations": a.recommendations,
                }
                for a in self.comparative_analyses[-50:]  # 最新50件
            ],
        }

        write_state_payload(filepath, state)

        self.logger.info(f"Comparator state saved to {filepath}")

    def load_state(self, filepath: str) -> bool:
        """
        状態読み込み

        Args:
            filepath: 読み込みファイルパス

        Returns:
            bool: 読み込み成功フラグ
        """
        try:
            state = read_state_payload(filepath)

            self.confidence_level = state.get("confidence_level", 0.95)
            self.min_sample_size = state.get("min_sample_size", 30)
            self.comparison_window_hours = state.get("comparison_window_hours", 24)

            self.enabled_metrics = {
                ComparisonMetric(m) for m in state.get("enabled_metrics", [])
            }
            self.enabled_tests = {
                StatisticalTest(t) for t in state.get("enabled_tests", [])
            }

            # 分析履歴復元（簡易版）
            self.comparative_analyses = []
            for a_data in state.get("comparative_analyses", []):
                analysis = ComparativeAnalysis(
                    analysis_id=a_data["analysis_id"],
                    timestamp=datetime.fromisoformat(a_data["timestamp"]),
                    period_start=datetime.fromisoformat(a_data["period_start"]),
                    period_end=datetime.fromisoformat(a_data["period_end"]),
                    systems_compared=a_data["systems_compared"],
                    overall_winner=a_data["overall_winner"],
                    confidence_level=a_data["confidence_level"],
                    key_findings=a_data.get("key_findings", []),
                    recommendations=a_data.get("recommendations", []),
                )
                self.comparative_analyses.append(analysis)

            self.logger.info(f"Comparator state loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load comparator state: {e}")
            return False
