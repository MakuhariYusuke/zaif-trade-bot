"""
V433 Phase 5: Paper Trading Layer - Performance Validator

仮想取引結果の統計的有意性検証とパフォーマンス評価を行う。
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import scipy.stats as stats

from ztb.trading.production.virtual_portfolio_manager import (
    PortfolioMetrics,
    VirtualTrade,
)


class ValidationResult(Enum):
    """検証結果"""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class StatisticalTest:
    """統計テスト結果"""

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    confidence_level: float = 0.95
    interpretation: str = ""


@dataclass
class RiskMetrics:
    """リスク指標"""

    volatility: float
    max_drawdown: Decimal
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    value_at_risk_95: Decimal
    expected_shortfall_95: Decimal
    beta: float = 0.0  # 市場に対するベータ
    correlation: float = 0.0  # 市場相関


@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""

    total_return: Decimal
    annualized_return: float
    win_rate: float
    profit_factor: float
    average_win: Decimal
    average_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    consecutive_wins: int
    consecutive_losses: int
    recovery_factor: float


@dataclass
class BenchmarkComparison:
    """ベンチマーク比較"""

    benchmark_return: Decimal
    excess_return: Decimal
    alpha: float
    information_ratio: float
    tracking_error: float
    r_squared: float


@dataclass
class ValidationReport:
    """検証レポート"""

    validation_timestamp: datetime
    total_trades: int
    evaluation_period_days: int
    overall_rating: ValidationResult

    statistical_tests: List[StatisticalTest] = field(default_factory=list)
    risk_metrics: RiskMetrics = None
    performance_metrics: PerformanceMetrics = None
    benchmark_comparison: BenchmarkComparison = None

    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)


class PerformanceValidator:
    """
    パフォーマンスバリデーター

    仮想取引結果の統計的有意性検証と包括的なパフォーマンス評価を行う。
    """

    def __init__(
        self,
        benchmark_returns: Optional[List[Decimal]] = None,
        risk_free_rate: float = 0.02,
        confidence_level: float = 0.95,
        min_trades_required: int = 30,
    ):
        """
        初期化

        Args:
            benchmark_returns: ベンチマークリターン（日次）
            risk_free_rate: 無リスク金利
            confidence_level: 信頼水準
            min_trades_required: 最小必要取引数
        """
        self.benchmark_returns = benchmark_returns or []
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.min_trades_required = min_trades_required

        # ロギング
        self.logger = logging.getLogger(__name__)

        self.logger.info("Performance Validator initialized")

    def validate_performance(
        self,
        portfolio_metrics: List[PortfolioMetrics],
        trades: List[VirtualTrade],
        evaluation_period_days: int,
    ) -> ValidationReport:
        """
        パフォーマンス検証

        Args:
            portfolio_metrics: ポートフォリオ指標履歴
            trades: 取引履歴
            evaluation_period_days: 評価期間（日）

        Returns:
            ValidationReport: 検証レポート
        """
        report = ValidationReport(
            validation_timestamp=datetime.now(),
            total_trades=len(trades),
            evaluation_period_days=evaluation_period_days,
            overall_rating=ValidationResult.UNACCEPTABLE,
        )

        try:
            # 基本チェック
            if len(trades) < self.min_trades_required:
                report.critical_issues.append(
                    f"Insufficient trades: {len(trades)} < {self.min_trades_required}"
                )
                return report

            if len(portfolio_metrics) < 2:
                report.critical_issues.append("Insufficient portfolio metrics history")
                return report

            # 統計テスト実行
            report.statistical_tests = self._run_statistical_tests(
                portfolio_metrics, trades
            )

            # リスク指標計算
            report.risk_metrics = self._calculate_risk_metrics(portfolio_metrics)

            # パフォーマンス指標計算
            report.performance_metrics = self._calculate_performance_metrics(
                trades, evaluation_period_days
            )

            # ベンチマーク比較
            if self.benchmark_returns:
                report.benchmark_comparison = self._calculate_benchmark_comparison(
                    portfolio_metrics, evaluation_period_days
                )

            # 全体評価
            report.overall_rating = self._calculate_overall_rating(report)

            # レコメンデーション生成
            (
                report.recommendations,
                report.warnings,
                report.critical_issues,
            ) = self._generate_recommendations(report)

            self.logger.info(
                f"Performance validation completed. Rating: {report.overall_rating.value}"
            )

        except Exception as e:
            self.logger.error(f"Performance validation failed: {e}")
            report.critical_issues.append(f"Validation error: {str(e)}")

        return report

    def _run_statistical_tests(
        self, portfolio_metrics: List[PortfolioMetrics], trades: List[VirtualTrade]
    ) -> List[StatisticalTest]:
        """
        統計テスト実行

        Args:
            portfolio_metrics: ポートフォリオ指標履歴
            trades: 取引履歴

        Returns:
            List[StatisticalTest]: 統計テスト結果
        """
        tests = []

        try:
            # PnLの正規性テスト（Shapiro-Wilk）
            pnl_values = [
                float(m.total_pnl) for m in portfolio_metrics if m.total_pnl != 0
            ]
            if len(pnl_values) >= 3:
                stat, p_value = stats.shapiro(pnl_values)
                significant = p_value < (1 - self.confidence_level)
                interpretation = (
                    "PnL is normally distributed"
                    if not significant
                    else "PnL is not normally distributed"
                )

                tests.append(
                    StatisticalTest(
                        test_name="Shapiro-Wilk Normality Test (PnL)",
                        statistic=stat,
                        p_value=p_value,
                        significant=significant,
                        interpretation=interpretation,
                    )
                )

            # 取引リターンの系列相関テスト（Ljung-Box）
            returns = self._calculate_returns(portfolio_metrics)
            if len(returns) >= 10:
                # Ljung-Boxテストの簡易実装
                autocorr = np.correlate(returns, returns, mode="full")
                autocorr = autocorr[autocorr.size // 2 :] / len(returns)
                q_stat = (
                    len(returns)
                    * (len(returns) + 2)
                    * sum(autocorr[1:11] ** 2 / (len(returns) - np.arange(1, 11)))
                )
                p_value = 1 - stats.chi2.cdf(q_stat, 10)

                significant = p_value < (1 - self.confidence_level)
                interpretation = (
                    "No significant autocorrelation"
                    if not significant
                    else "Significant autocorrelation detected"
                )

                tests.append(
                    StatisticalTest(
                        test_name="Ljung-Box Autocorrelation Test",
                        statistic=q_stat,
                        p_value=p_value,
                        significant=significant,
                        interpretation=interpretation,
                    )
                )

            # 勝率の二項検定
            winning_trades = sum(1 for t in trades if t.realized_pnl > 0)
            total_trades = len([t for t in trades if t.realized_pnl != 0])

            if total_trades >= 10:
                p_value = stats.binomtest(winning_trades, total_trades, 0.5).pvalue
                significant = p_value < (1 - self.confidence_level)
                interpretation = (
                    "Win rate is significantly different from 50%"
                    if significant
                    else "Win rate is not significantly different from 50%"
                )

                tests.append(
                    StatisticalTest(
                        test_name="Binomial Test (Win Rate vs 50%)",
                        statistic=winning_trades / total_trades
                        if total_trades > 0
                        else 0,
                        p_value=p_value,
                        significant=significant,
                        interpretation=interpretation,
                    )
                )

        except Exception as e:
            self.logger.error(f"Statistical test error: {e}")

        return tests

    def _calculate_risk_metrics(
        self, portfolio_metrics: List[PortfolioMetrics]
    ) -> RiskMetrics:
        """
        リスク指標計算

        Args:
            portfolio_metrics: ポートフォリオ指標履歴

        Returns:
            RiskMetrics: リスク指標
        """
        if len(portfolio_metrics) < 2:
            return RiskMetrics(
                volatility=0.0,
                max_drawdown=Decimal("0"),
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                value_at_risk_95=Decimal("0"),
                expected_shortfall_95=Decimal("0"),
            )

        # リターン計算
        returns = self._calculate_returns(portfolio_metrics)

        # ボラティリティ
        volatility = float(np.std(returns)) if returns else 0.0

        # 最大ドローダウン
        max_drawdown = max(
            (m.max_drawdown for m in portfolio_metrics), default=Decimal("0")
        )

        # シャープレシオ
        excess_returns = [
            r - self.risk_free_rate / 252 for r in returns
        ]  # 日次リスクフリーレート
        sharpe_ratio = (
            float(np.mean(excess_returns) / np.std(excess_returns))
            if excess_returns and np.std(excess_returns) > 0
            else 0.0
        )

        # ソルティノレシオ
        downside_returns = [r for r in returns if r < 0]
        sortino_ratio = (
            float(np.mean(excess_returns) / np.std(downside_returns))
            if downside_returns
            else 0.0
        )

        # カルマーレシオ
        calmar_ratio = (
            float(np.mean(returns) * 252 / float(max_drawdown))
            if max_drawdown > 0
            else 0.0
        )

        # VaRとES（95%信頼水準）
        if returns:
            value_at_risk_95 = Decimal(str(np.percentile(returns, 5)))
            expected_shortfall_95 = Decimal(
                str(np.mean([r for r in returns if r <= float(value_at_risk_95)]))
            )
        else:
            value_at_risk_95 = Decimal("0")
            expected_shortfall_95 = Decimal("0")

        return RiskMetrics(
            volatility=volatility,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            value_at_risk_95=value_at_risk_95,
            expected_shortfall_95=expected_shortfall_95,
        )

    def _calculate_performance_metrics(
        self, trades: List[VirtualTrade], evaluation_period_days: int
    ) -> PerformanceMetrics:
        """
        パフォーマンス指標計算

        Args:
            trades: 取引履歴
            evaluation_period_days: 評価期間（日）

        Returns:
            PerformanceMetrics: パフォーマンス指標
        """
        if not trades:
            return PerformanceMetrics(
                total_return=Decimal("0"),
                annualized_return=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                average_win=Decimal("0"),
                average_loss=Decimal("0"),
                largest_win=Decimal("0"),
                largest_loss=Decimal("0"),
                consecutive_wins=0,
                consecutive_losses=0,
                recovery_factor=0.0,
            )

        # 実現PnLのある取引のみ対象
        realized_trades = [t for t in trades if t.realized_pnl != 0]

        if not realized_trades:
            return PerformanceMetrics(
                total_return=Decimal("0"),
                annualized_return=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                average_win=Decimal("0"),
                average_loss=Decimal("0"),
                largest_win=Decimal("0"),
                largest_loss=Decimal("0"),
                consecutive_wins=0,
                consecutive_losses=0,
                recovery_factor=0.0,
            )

        # 勝ち取引と負け取引
        winning_trades = [t for t in realized_trades if t.realized_pnl > 0]
        losing_trades = [t for t in realized_trades if t.realized_pnl < 0]

        # 総リターン
        total_return = sum(t.realized_pnl for t in realized_trades)

        # 年率リターン
        years = evaluation_period_days / 365.25
        annualized_return = float(total_return) / years if years > 0 else 0.0

        # 勝率
        win_rate = (
            len(winning_trades) / len(realized_trades) if realized_trades else 0.0
        )

        # プロフィットファクター
        gross_profit = sum(t.realized_pnl for t in winning_trades)
        gross_loss = abs(sum(t.realized_pnl for t in losing_trades))
        profit_factor = (
            float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        )

        # 平均勝ち/負け
        average_win = (
            gross_profit / len(winning_trades) if winning_trades else Decimal("0")
        )
        average_loss = (
            gross_loss / len(losing_trades) if losing_trades else Decimal("0")
        )

        # 最大勝ち/負け
        largest_win = max(
            (t.realized_pnl for t in winning_trades), default=Decimal("0")
        )
        largest_loss = min(
            (t.realized_pnl for t in losing_trades), default=Decimal("0")
        )

        # 連続勝ち/負け
        consecutive_wins, consecutive_losses = self._calculate_consecutive_trades(
            realized_trades
        )

        # リカバリーファクター（総利益 / 最大ドローダウン）
        # 簡易計算のため、総利益をドローダウンの代用
        recovery_factor = (
            float(total_return / abs(largest_loss))
            if largest_loss < 0
            else float("inf")
        )

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            recovery_factor=recovery_factor,
        )

    def _calculate_benchmark_comparison(
        self, portfolio_metrics: List[PortfolioMetrics], evaluation_period_days: int
    ) -> BenchmarkComparison:
        """
        ベンチマーク比較計算

        Args:
            portfolio_metrics: ポートフォリオ指標履歴
            evaluation_period_days: 評価期間（日）

        Returns:
            BenchmarkComparison: ベンチマーク比較
        """
        if not self.benchmark_returns or len(portfolio_metrics) < 2:
            return BenchmarkComparison(
                benchmark_return=Decimal("0"),
                excess_return=Decimal("0"),
                alpha=0.0,
                information_ratio=0.0,
                tracking_error=0.0,
                r_squared=0.0,
            )

        # ポートフォリオリターン
        portfolio_returns = self._calculate_returns(portfolio_metrics)

        # ベンチマーク調整
        benchmark_len = min(len(portfolio_returns), len(self.benchmark_returns))
        portfolio_returns = portfolio_returns[-benchmark_len:]
        benchmark_returns = [float(r) for r in self.benchmark_returns[-benchmark_len:]]

        if not portfolio_returns or not benchmark_returns:
            return BenchmarkComparison(
                benchmark_return=Decimal("0"),
                excess_return=Decimal("0"),
                alpha=0.0,
                information_ratio=0.0,
                tracking_error=0.0,
                r_squared=0.0,
            )

        # ベンチマークリターン
        benchmark_return = Decimal(str(np.prod([1 + r for r in benchmark_returns]) - 1))

        # 超過リターン
        portfolio_total_return = np.prod([1 + r for r in portfolio_returns]) - 1
        benchmark_total_return = np.prod([1 + r for r in benchmark_returns]) - 1
        excess_return = Decimal(str(portfolio_total_return - benchmark_total_return))

        # CAPMアルファ（簡易計算）
        beta = (
            np.cov(portfolio_returns, benchmark_returns)[0, 1]
            / np.var(benchmark_returns)
            if np.var(benchmark_returns) > 0
            else 0.0
        )
        alpha = portfolio_total_return - beta * benchmark_total_return

        # インフォメーションレシオ
        tracking_error = np.std(
            [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
        )
        information_ratio = (
            (portfolio_total_return - benchmark_total_return) / tracking_error
            if tracking_error > 0
            else 0.0
        )

        # R-squared
        correlation_matrix = np.corrcoef(portfolio_returns, benchmark_returns)
        r_squared = (
            correlation_matrix[0, 1] ** 2 if correlation_matrix.shape == (2, 2) else 0.0
        )

        return BenchmarkComparison(
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            alpha=alpha,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            r_squared=r_squared,
        )

    def _calculate_returns(
        self, portfolio_metrics: List[PortfolioMetrics]
    ) -> List[float]:
        """
        リターン計算

        Args:
            portfolio_metrics: ポートフォリオ指標履歴

        Returns:
            List[float]: 日次リターン
        """
        if len(portfolio_metrics) < 2:
            return []

        returns = []
        prev_value = float(portfolio_metrics[0].total_value)

        for metric in portfolio_metrics[1:]:
            current_value = float(metric.total_value)
            if prev_value > 0:
                ret = (current_value - prev_value) / prev_value
                returns.append(ret)
            prev_value = current_value

        return returns

    def _calculate_consecutive_trades(
        self, trades: List[VirtualTrade]
    ) -> Tuple[int, int]:
        """
        連続勝敗計算

        Args:
            trades: 取引履歴

        Returns:
            Tuple[int, int]: (最大連続勝ち, 最大連続負け)
        """
        if not trades:
            return 0, 0

        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            if trade.realized_pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif trade.realized_pnl < 0:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0

        return max_consecutive_wins, max_consecutive_losses

    def _calculate_overall_rating(self, report: ValidationReport) -> ValidationResult:
        """
        全体評価計算

        Args:
            report: 検証レポート

        Returns:
            ValidationResult: 全体評価
        """
        score = 0
        max_score = 0

        # リスク指標評価
        if report.risk_metrics:
            max_score += 3
            if report.risk_metrics.sharpe_ratio > 1.0:
                score += 1
            if report.risk_metrics.max_drawdown < Decimal("0.1"):
                score += 1
            if report.risk_metrics.value_at_risk_95 > Decimal("-0.05"):
                score += 1

        # パフォーマンス指標評価
        if report.performance_metrics:
            max_score += 3
            if report.performance_metrics.win_rate > 0.55:
                score += 1
            if report.performance_metrics.profit_factor > 1.5:
                score += 1
            if report.performance_metrics.recovery_factor > 2.0:
                score += 1

        # 統計テスト評価
        if report.statistical_tests:
            max_score += 1
            significant_issues = sum(
                1
                for test in report.statistical_tests
                if test.significant and "not" in test.interpretation.lower()
            )
            if significant_issues == 0:
                score += 1

        # スコアに基づく評価
        if max_score == 0:
            return ValidationResult.UNACCEPTABLE

        ratio = score / max_score

        if ratio >= 0.9:
            return ValidationResult.EXCELLENT
        elif ratio >= 0.7:
            return ValidationResult.GOOD
        elif ratio >= 0.5:
            return ValidationResult.ACCEPTABLE
        elif ratio >= 0.3:
            return ValidationResult.POOR
        else:
            return ValidationResult.UNACCEPTABLE

    def _generate_recommendations(
        self, report: ValidationReport
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        レコメンデーション生成

        Args:
            report: 検証レポート

        Returns:
            Tuple[List[str], List[str], List[str]]: (レコメンデーション, 警告, 重大問題)
        """
        recommendations = []
        warnings = []
        critical_issues = []

        # リスク指標ベース
        if report.risk_metrics:
            if report.risk_metrics.sharpe_ratio < 0.5:
                warnings.append(
                    "Sharpe ratio is low. Consider risk management improvements."
                )
            if report.risk_metrics.max_drawdown > Decimal("0.2"):
                critical_issues.append(
                    "Maximum drawdown is too high. Immediate risk controls needed."
                )

        # パフォーマンス指標ベース
        if report.performance_metrics:
            if report.performance_metrics.win_rate < 0.5:
                warnings.append("Win rate is below 50%. Strategy may need refinement.")
            if report.performance_metrics.profit_factor < 1.2:
                warnings.append(
                    "Profit factor is low. Consider improving reward-to-risk ratio."
                )

        # 統計テストベース
        for test in report.statistical_tests:
            if test.significant:
                if "autocorrelation" in test.test_name.lower():
                    recommendations.append(
                        "Consider incorporating mean-reversion or momentum filters."
                    )
                elif "normality" in test.test_name.lower():
                    recommendations.append(
                        "Consider using distribution-robust performance measures."
                    )

        # 全体評価ベース
        if report.overall_rating in [
            ValidationResult.POOR,
            ValidationResult.UNACCEPTABLE,
        ]:
            critical_issues.append(
                "Overall performance rating is poor. Strategy requires significant changes."
            )
        elif report.overall_rating == ValidationResult.ACCEPTABLE:
            warnings.append("Performance is acceptable but could be improved.")

        return recommendations, warnings, critical_issues
