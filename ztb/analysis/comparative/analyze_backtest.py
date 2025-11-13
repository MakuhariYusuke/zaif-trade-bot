#!/usr/bin/env python3
"""
Generic Backtest Analysis Tool
汎用バックテスト分析ツール

This tool provides comprehensive analysis of trading backtest results,
including risk metrics, temporal analysis, and market condition analysis.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import glob
import os

import numpy as np
import pandas as pd

from ztb.core.base import BaseAnalyzer
from ztb.data.btc_data_augmentation import BTCBiasDetector
from ztb.metrics.metrics import (
    calculate_all_metrics,
    classify_market_regime,
    max_drawdown,
    multi_market_backtest_analysis,
    profit_factor,
    seasonality_analysis,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)
from ztb.utils.trading_metrics import action_distribution
from ztb.trading.constants import (
    TRADING_DAYS_PER_YEAR,  # = 252
    ACTION_BUY, ACTION_HOLD, ACTION_SELL,
)
from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.utils.logging_utils import get_logger
from ztb.utils.performance_utils import PerformanceMonitor

# Import type definitions
from .backtest_analysis_types import (
    ActionAveragesResult,
    AnalysisResult,
    AutocorrelationResult,
    CorrelationAnalysisResult,
    MarketConditionResult,
    NormalityTestResult,
    PerformanceMetricsResult,
    RiskAdjustedMetricsResult,
    RobustnessAnalysisResult,
    StatisticalTestResult,
    TemporalPatternsResult,
    TradingFrequencyResult,
    VolatilityClusteringResult,
)

logger = get_logger(__name__)


# Utility function for coefficient of variation calculation
def _calculate_coefficient_of_variation(values: np.ndarray) -> float:
    """変動係数（Coefficient of Variation）を計算"""
    if len(values) == 0:
        return 0.0
    mean_val = np.mean(values)
    if mean_val == 0:
        return 0.0
    return np.std(values) / mean_val


class BacktestAnalyzer(BaseAnalyzer):
    """汎用バックテスト分析クラス"""

    def __init__(
        self,
        results_path: str,
        training_report_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name="BacktestAnalyzer", config=config)
        self.results_path = Path(results_path)
        self.training_report_path = (
            Path(training_report_path) if training_report_path else None
        )
        self.data = self._load_data()
        self.is_unified = 'results' in self.data and isinstance(self.data.get('results'), list) and 'avg_return_pct' in self.data
        self.training_data = (
            self._load_training_data() if training_report_path else None
        )
        self._validate_data()
        self.performance_monitor = PerformanceMonitor("backtest_analyzer")
        self.bias_detector = BTCBiasDetector()

    def _load_data(self) -> Dict[str, Any]:
        """Load backtest results from JSON file."""
        try:
            with open(self.results_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {self.results_path}: {e}")

    def _load_training_data(self) -> Optional[Dict[str, Any]]:
        """Load training report from JSON file."""
        if not self.training_report_path:
            return None

        try:
            with open(self.training_report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(
                f"Training report file not found: {self.training_report_path}"
            )
            return None
        except json.JSONDecodeError as e:
            logger.warning(
                f"Invalid JSON format in training report {self.training_report_path}: {e}"
            )
            return None

    def _validate_data(self):
        """Validate that required data fields are present."""
        required_fields = ["total_steps", "initial_portfolio", "final_portfolio"]
        missing_fields = [field for field in required_fields if field not in self.data]

        # If primary fields are missing, check for alternative field names
        if "initial_portfolio" not in self.data and "initial_balance" in self.data:
            self.data["initial_portfolio"] = self.data["initial_balance"]
        if "final_portfolio" not in self.data and "final_portfolio_value" in self.data:
            self.data["final_portfolio"] = self.data["final_portfolio_value"]

        # BTC-related field mapping
        if "initial_btc" not in self.data and "initial_btc_balance" in self.data:
            self.data["initial_btc"] = self.data["initial_btc_balance"]
        if "final_btc" not in self.data and "final_btc_balance" in self.data:
            self.data["final_btc"] = self.data["final_btc_balance"]
        if "btc_holdings" not in self.data and "btc_history" in self.data:
            self.data["btc_holdings"] = self.data["btc_history"]

        # Re-check after mapping alternative fields
        missing_fields = [field for field in required_fields if field not in self.data]
        if missing_fields:
            raise ValueError(f"Missing required fields in results: {missing_fields}")

    def analyze(self, data: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """Perform comprehensive backtest analysis."""
        if data:
            self.data = data
            self._validate_data()

        results = {
            "risk_metrics": self.calculate_risk_metrics(),
            "temporal_patterns": self.analyze_temporal_patterns(),
            "market_conditions": self.analyze_market_conditions(),
            "trading_frequency": self.analyze_trading_frequency(),
            "btc_analysis": self.analyze_btc_performance(),
        }

        if self.is_unified:
            if self.data.get('enable_signal_guidance'):
                results['signal_guidance_analysis'] = self._analyze_signal_guidance()

        self.results = results
        return results

    def _analyze_signal_guidance(self) -> Dict[str, Any]:
        """SIGNAL_GUIDANCE分析を実行"""
        episodes = self.data['results']
        all_guidance_scores = []
        all_original_actions = []
        all_guidance_actions = []

        for ep in episodes:
            if 'guidance_signals' in ep:
                for signal in ep['guidance_signals']:
                    all_guidance_scores.append(signal['guidance_score'])
                    all_original_actions.append(signal['original_action'])
                    all_guidance_actions.append(signal['guidance_action'])

        if not all_guidance_scores:
            return {}

        # アクション分布
        orig_discrete = []
        guide_discrete = []
        for a in all_original_actions:
            if isinstance(a, list) and len(a) > 0:
                orig_discrete.append(continuous_to_discrete_action(a[0]))
            elif isinstance(a, (int, float)):
                orig_discrete.append(a)
            else:
                orig_discrete.append(0)

        for a in all_guidance_actions:
            if isinstance(a, list) and len(a) > 0:
                guide_discrete.append(continuous_to_discrete_action(a[0]))
            elif isinstance(a, (int, float)):
                guide_discrete.append(a)
            else:
                guide_discrete.append(0)

        differences = sum(1 for o, g in zip(orig_discrete, guide_discrete) if o != g)

        # スコア vs ポートフォリオ価値の相関
        portfolio_values = []
        for ep in episodes:
            if 'guidance_signals' in ep:
                portfolio_values.extend([s['portfolio_value'] for s in ep['guidance_signals']])

        correlation = np.corrcoef(all_guidance_scores, portfolio_values)[0,1] if len(portfolio_values) == len(all_guidance_scores) else 0

        return {
            'number_of_signals': len(all_guidance_scores),
            'average_score': float(np.mean(all_guidance_scores)),
            'score_std': float(np.std(all_guidance_scores)),
            'min_score': float(min(all_guidance_scores)),
            'max_score': float(max(all_guidance_scores)),
            'original_hold': orig_discrete.count(0),
            'original_buy': orig_discrete.count(1),
            'original_sell': orig_discrete.count(-1),
            'guidance_hold': guide_discrete.count(0),
            'guidance_buy': guide_discrete.count(1),
            'guidance_sell': guide_discrete.count(-1),
            'differences': differences,
            'total_actions': len(orig_discrete),
            'difference_pct': differences / len(orig_discrete) * 100 if orig_discrete else 0,
            'correlation': float(correlation)
        }

    def calculate_risk_metrics(self) -> Dict[str, float]:
        """リスク指標を計算"""
        if "portfolio_history" not in self.data:
            return {}

        portfolio_values = np.array(self.data["portfolio_history"])

        # 総リターン
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[
            0
        ]

        # 日次リターン（分足データを日次に変換）
        if "timestamps" in self.data:
            timestamps = pd.to_datetime(self.data["timestamps"])
            daily_returns = []
            current_day = None
            day_start_value = None

            for i, (ts, value) in enumerate(zip(timestamps, portfolio_values)):
                day = ts.date()
                if current_day != day:
                    if day_start_value is not None and i > 0:
                        daily_return = (
                            portfolio_values[i - 1] - day_start_value
                        ) / day_start_value
                        daily_returns.append(daily_return)
                    current_day = day
                    day_start_value = value
            if day_start_value is not None and len(portfolio_values) > 0:
                daily_return = (
                    portfolio_values[-1] - day_start_value
                ) / day_start_value
                daily_returns.append(daily_return)

            daily_returns = np.array(daily_returns)
        else:
            # タイムスタンプがない場合はステップごとのリターンを使用
            step_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            # 適当に日次にグループ化（仮定: 1日=1440分）
            steps_per_day = 1440
            daily_returns = []
            for i in range(0, len(step_returns), steps_per_day):
                day_return = np.prod(1 + step_returns[i : i + steps_per_day]) - 1
                daily_returns.append(day_return)
            daily_returns = np.array(daily_returns)

        if len(daily_returns) == 0:
            return {
                "total_return": total_return,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
            }

        # metrics.pyの関数を使用して指標を計算

        # シャープレシオ
        sharpe_ratio_value = sharpe_ratio(daily_returns)

        # 最大ドローダウン
        max_drawdown_value = max_drawdown(portfolio_values)

        # ボラティリティ（年率化）
        volatility = np.std(daily_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)

        # ソルティーノレシオ
        sortino_ratio_value = sortino_ratio(daily_returns)

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio_value,
            "max_drawdown": max_drawdown_value,
            "volatility": volatility,
            "sortino_ratio": sortino_ratio_value,
            "win_rate": (
                self.data.get("win_rate", 0) / 100.0
                if self.data.get("win_rate", 0) > 1
                else self.data.get("win_rate", 0)
            ),
            "profit_factor": self._calculate_profit_factor(),
        }

    def _calculate_profit_factor(self) -> float:
        """プロフィットファクターを計算"""
        if "trade_pnls" not in self.data:
            return 0.0

        pnls = np.array(self.data["trade_pnls"])
        return profit_factor(pnls)

    def calculate_enhanced_statistics(self) -> Dict[str, Any]:
        """改善された統計分析を実行"""
        if "portfolio_history" not in self.data:
            return {}

        portfolio_values = np.array(self.data["portfolio_history"])

        # 基本リターンの計算
        if len(portfolio_values) < 2:
            return {"error": "Insufficient data for enhanced statistics"}

        # リターンの計算
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # 分布分析
        distribution_stats = {
            "mean_return": float(np.mean(returns)),
            "median_return": float(np.median(returns)),
            "std_return": float(np.std(returns)),
            "skewness": float(self._calculate_skewness(returns)),
            "kurtosis": float(self._calculate_kurtosis(returns)),
            "return_percentiles": {
                "1%": float(np.percentile(returns, 1)),
                "5%": float(np.percentile(returns, 5)),
                "25%": float(np.percentile(returns, 25)),
                "75%": float(np.percentile(returns, 75)),
                "95%": float(np.percentile(returns, 95)),
                "99%": float(np.percentile(returns, 99)),
            },
        }

        # 正規性検定
        normality_tests = self._test_normality(returns)

        # 自己相関分析
        autocorrelation = self._calculate_autocorrelation(returns, lags=20)

        # ボラティリティ・クラスタリング分析
        volatility_clustering = self._analyze_volatility_clustering(returns)

        # リスク調整リターン指標
        risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(returns)

        # 統計的有意性検定
        statistical_tests = self._perform_statistical_tests(returns)

        return {
            "distribution_analysis": distribution_stats,
            "normality_tests": normality_tests,
            "autocorrelation": autocorrelation,
            "volatility_clustering": volatility_clustering,
            "risk_adjusted_metrics": risk_adjusted_metrics,
            "statistical_tests": statistical_tests,
        }

    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """歪度を計算"""
        if len(returns) < 3:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        return np.mean(((returns - mean) / std) ** 3)

    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """尖度を計算"""
        if len(returns) < 4:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        return np.mean(((returns - mean) / std) ** 4) - 3

    def _test_normality(self, returns: np.ndarray) -> NormalityTestResult:
        """正規性検定を実行"""
        try:
            from scipy import stats

            # Shapiro-Wilk検定
            if len(returns) >= 3 and len(returns) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(returns)
            else:
                shapiro_stat, shapiro_p = None, None

            # Kolmogorov-Smirnov検定
            ks_stat, ks_p = stats.kstest(
                returns, "norm", args=(np.mean(returns), np.std(returns))
            )

            # Jarque-Bera検定
            jb_stat, jb_p = stats.jarque_bera(returns)

            return {
                "shapiro_wilk": {
                    "statistic": float(shapiro_stat) if shapiro_stat else None,
                    "p_value": float(shapiro_p) if shapiro_p else None,
                    "is_normal": (shapiro_p or 0) > 0.05,
                },
                "kolmogorov_smirnov": {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_p),
                    "is_normal": ks_p > 0.05,
                },
                "jarque_bera": {
                    "statistic": float(jb_stat),
                    "p_value": float(jb_p),
                    "is_normal": jb_p > 0.05,
                },
            }
        except ImportError:
            return {"error": "scipy not available for normality tests"}
        except Exception as e:
            return {"error": f"Normality test failed: {str(e)}"}

    def _calculate_autocorrelation(
        self, returns: np.ndarray, lags: int = 20
    ) -> AutocorrelationResult:
        """自己相関を計算"""
        try:
            autocorr = {}
            for lag in range(1, min(lags + 1, len(returns))):
                corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                autocorr[f"lag_{lag}"] = float(corr) if not np.isnan(corr) else 0.0

            # Ljung-Box検定
            try:
                from scipy import stats

                lb_stat, lb_p = stats.acorr_ljungbox(
                    returns, lags=[lags], return_df=False
                )
                ljung_box = {
                    "statistic": float(lb_stat[0]),
                    "p_value": float(lb_p[0]),
                    "no_autocorrelation": lb_p[0] > 0.05,
                }
            except:
                ljung_box = {"error": "Ljung-Box test failed"}

            return {"autocorrelations": autocorr, "ljung_box_test": ljung_box}
        except Exception as e:
            return {"error": f"Autocorrelation calculation failed: {str(e)}"}

    def _analyze_volatility_clustering(
        self, returns: np.ndarray
    ) -> VolatilityClusteringResult:
        """ボラティリティ・クラスタリングを分析"""
        try:
            # 絶対リターンの自己相関
            abs_returns = np.abs(returns)
            autocorr_abs = {}
            for lag in range(1, min(11, len(abs_returns))):
                corr = np.corrcoef(abs_returns[:-lag], abs_returns[lag:])[0, 1]
                autocorr_abs[f"lag_{lag}"] = float(corr) if not np.isnan(corr) else 0.0

            # 条件付き分散の変化
            rolling_volatility = []
            window_size = min(50, len(returns) // 4)
            if window_size >= 10:
                for i in range(window_size, len(returns)):
                    window_returns = returns[i - window_size : i]
                    vol = np.std(window_returns)
                    rolling_volatility.append(float(vol))

            return {
                "absolute_return_autocorrelation": autocorr_abs,
                "rolling_volatility": rolling_volatility,
                "volatility_persistence": float(
                    np.mean(list(autocorr_abs.values())[:5])
                )
                if autocorr_abs
                else 0.0,
            }
        except Exception as e:
            return {"error": f"Volatility clustering analysis failed: {str(e)}"}

    def _calculate_risk_adjusted_metrics(
        self, returns: np.ndarray
    ) -> RiskAdjustedMetricsResult:
        """リスク調整リターン指標を計算"""
        if len(returns) == 0:
            return {}

        try:
            # Calmarレシオ（最大ドローダウンに対する年間リターン）
            cumulative = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - peak) / peak
            max_dd = np.min(drawdown) if len(drawdown) > 0 else 0

            # 年間リターンの推定（日次リターンを仮定）
            annual_return = np.mean(returns) * TRADING_DAYS_PER_YEAR  # 252取引日
            calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else 0

            # Omegaレシオ
            threshold = 0.0  # 無リスク金利
            gains = returns[returns > threshold]
            losses = returns[returns <= threshold]
            omega_ratio = (
                (np.sum(gains) / len(gains)) / (abs(np.sum(losses)) / len(losses))
                if len(losses) > 0 and len(gains) > 0
                else 0
            )

            # Kappaレシオ（kappa = 3）
            kappa = 3
            downside_deviation = np.sqrt(np.mean(np.minimum(returns, 0) ** 2))
            kappa_ratio = (
                np.mean(returns) / (downside_deviation**kappa)
                if downside_deviation > 0
                else 0
            )

            return {
                "calmar_ratio": float(calmar_ratio),
                "omega_ratio": float(omega_ratio),
                "kappa_ratio": float(kappa_ratio),
                "annual_return": float(annual_return),
                "max_drawdown": float(max_dd),
            }
        except Exception as e:
            return {"error": f"Risk-adjusted metrics calculation failed: {str(e)}"}

    def _perform_statistical_tests(self, returns: np.ndarray) -> StatisticalTestResult:
        """統計的有意性検定を実行"""
        try:
            from scipy import stats

            # t検定（平均リターンが0と異なるか）
            t_stat, t_p = stats.ttest_1samp(returns, 0)

            # Mann-Whitney U検定（中央値が0と異なるか）
            try:
                u_stat, u_p = stats.mannwhitneyu(
                    returns, np.zeros(len(returns)), alternative="two-sided"
                )
            except:
                u_stat, u_p = None, None

            # Levene検定（分散の等質性 - ここでは自己比較）
            # 代わりにBartlett検定を使用
            try:
                bartlett_stat, bartlett_p = stats.bartlett(
                    returns[: len(returns) // 2], returns[len(returns) // 2 :]
                )
            except:
                bartlett_stat, bartlett_p = None, None

            return {
                "t_test": {
                    "statistic": float(t_stat),
                    "p_value": float(t_p),
                    "mean_significantly_different_from_zero": t_p < 0.05,
                },
                "mann_whitney_u": {
                    "statistic": float(u_stat) if u_stat else None,
                    "p_value": float(u_p) if u_p else None,
                    "median_significantly_different_from_zero": (u_p or 1) < 0.05,
                },
                "bartlett_test": {
                    "statistic": float(bartlett_stat) if bartlett_stat else None,
                    "p_value": float(bartlett_p) if bartlett_p else None,
                    "variance_homogeneous": (bartlett_p or 1) > 0.05,
                },
            }
        except ImportError:
            return {"error": "scipy not available for statistical tests"}
        except Exception as e:
            return {"error": f"Statistical tests failed: {str(e)}"}

    def analyze_temporal_patterns(self) -> TemporalPatternsResult:
        """時間帯別の分析"""
        if "timestamps" not in self.data or "portfolio_history" not in self.data:
            return {}

        # timestampsが既にDatetimeIndexの場合はそのまま使用、そうでなければ変換
        if isinstance(self.data["timestamps"], pd.DatetimeIndex):
            timestamps = self.data["timestamps"]
        else:
            timestamps = pd.to_datetime(self.data["timestamps"])
        portfolio_values = np.array(self.data["portfolio_history"])

        # 時間帯別のリターン
        hourly_returns = {}
        for hour in range(24):
            hour_mask = timestamps.hour == hour
            if hour_mask.sum() > 1:
                hour_values = portfolio_values[hour_mask]
                hour_return = (hour_values[-1] - hour_values[0]) / hour_values[0]
                hourly_returns[f"{hour:02d}:00"] = hour_return

        # 曜日別のリターン
        weekday_returns = {}
        for weekday in range(7):
            weekday_mask = timestamps.weekday == weekday
            if weekday_mask.sum() > 1:
                weekday_values = portfolio_values[weekday_mask]
                weekday_return = (
                    weekday_values[-1] - weekday_values[0]
                ) / weekday_values[0]
                weekday_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][
                    weekday
                ]
                weekday_returns[weekday_name] = weekday_return

        return {"hourly_returns": hourly_returns, "weekday_returns": weekday_returns}

    def analyze_market_conditions(self) -> MarketConditionResult:
        """市場環境別の分析"""
        if "price_history" not in self.data or "portfolio_history" not in self.data:
            return MarketConditionResult(uptrend=None, downtrend=None, sideways=None)

        prices = np.array(self.data["price_history"])
        portfolio_values = np.array(self.data["portfolio_history"])

        # 長さが一致しない場合は最小長に合わせる
        min_length = min(len(prices), len(portfolio_values))
        prices = prices[:min_length]
        portfolio_values = portfolio_values[:min_length]

        # 価格トレンドの計算（移動平均）
        if len(prices) >= 20:
            short_ma = pd.Series(prices).rolling(10).mean()
            long_ma = pd.Series(prices).rolling(50).mean()

            # 市場環境の分類
            uptrend_mask = (short_ma > long_ma).fillna(False).values
            downtrend_mask = (short_ma < long_ma).fillna(False).values
            sideways_mask = ~(uptrend_mask | downtrend_mask)

            conditions = {
                "uptrend": {"mask": uptrend_mask, "name": "上昇トレンド"},
                "downtrend": {"mask": downtrend_mask, "name": "下降トレンド"},
                "sideways": {"mask": sideways_mask, "name": "横ばい"},
            }

            results = {}
            for condition_key, condition_data in conditions.items():
                mask = condition_data["mask"]
                if mask.sum() > 0:
                    condition_portfolio = portfolio_values[mask]
                    condition_return = (
                        condition_portfolio[-1] - condition_portfolio[0]
                    ) / condition_portfolio[0]
                    results[condition_key] = {
                        "return": condition_return,
                        "periods": int(mask.sum()),
                        "name": condition_data["name"],
                    }

            return results

        return MarketConditionResult(uptrend=None, downtrend=None, sideways=None)

    def analyze_trading_frequency(self) -> TradingFrequencyResult:
        """取引頻度分析"""
        if "actions" not in self.data:
            return TradingFrequencyResult()

        actions = np.array(self.data["actions"])
        total_steps = len(actions)

        # アクション分布
        action_dist = action_distribution(actions)

        # 取引頻度（BUY/SELLの割合）
        trade_actions = actions[(actions == ACTION_BUY) | (actions == ACTION_SELL)]
        trade_frequency = len(trade_actions) / total_steps if total_steps > 0 else 0

        # 平均取引間隔
        trade_indices = np.where((actions == ACTION_BUY) | (actions == ACTION_SELL))[0]
        if len(trade_indices) > 1:
            intervals = np.diff(trade_indices)
            avg_trade_interval = np.mean(intervals)
            min_trade_interval = np.min(intervals)
            max_trade_interval = np.max(intervals)
        else:
            avg_trade_interval = min_trade_interval = max_trade_interval = 0

        return {
            "action_distribution": action_dist,
            "trade_frequency": trade_frequency,
            "avg_trade_interval": avg_trade_interval,
            "min_trade_interval": min_trade_interval,
            "max_trade_interval": max_trade_interval,
            "total_trades": len(trade_actions),
        }

    def analyze_btc_performance(self) -> Dict[str, Any]:
        """BTCパフォーマンス分析"""
        btc_analysis = {}

        # BTC保有量の初期値と最終値
        initial_btc = self.data.get("initial_btc", 0.0)
        final_btc = self.data.get("final_btc", 0.0)

        btc_analysis["initial_btc"] = initial_btc
        btc_analysis["final_btc"] = final_btc
        btc_analysis["net_btc_gained"] = final_btc - initial_btc

        # BTCリターン計算
        if initial_btc > 0:
            btc_return = (final_btc - initial_btc) / initial_btc * 100
        else:
            btc_return = 0.0
        btc_analysis["btc_return_pct"] = btc_return

        # BTC保有履歴の分析
        if "btc_holdings" in self.data and len(self.data["btc_holdings"]) > 0:
            btc_history = np.array(self.data["btc_holdings"])

            # BTC保有量の統計
            btc_analysis["btc_mean_holding"] = np.mean(btc_history)
            btc_analysis["btc_max_holding"] = np.max(btc_history)
            btc_analysis["btc_min_holding"] = np.min(btc_history)
            btc_analysis["btc_holding_volatility"] = np.std(btc_history)

            # BTC保有量の変化分析
            if len(btc_history) > 1:
                btc_changes = np.diff(btc_history)
                btc_positive_changes = np.sum(btc_changes > 0)
                btc_negative_changes = np.sum(btc_changes < 0)
                btc_analysis["btc_positive_changes"] = int(btc_positive_changes)
                btc_analysis["btc_negative_changes"] = int(btc_negative_changes)

                # BTC取引頻度
                btc_trade_frequency = (
                    btc_positive_changes + btc_negative_changes
                ) / len(btc_history)
                btc_analysis["btc_trade_frequency"] = btc_trade_frequency

        # USD vs BTC パフォーマンス比較
        usd_return = self.data.get("total_return_pct", 0.0)
        btc_analysis["usd_return_pct"] = usd_return

        if abs(btc_return) > 0.01:  # ゼロ除算を避ける
            btc_vs_usd_ratio = (
                usd_return / btc_return if btc_return != 0 else float("inf")
            )
            btc_analysis["btc_vs_usd_performance_ratio"] = btc_vs_usd_ratio
        else:
            btc_analysis["btc_vs_usd_performance_ratio"] = 0.0

        # BTCポジションの安定性分析
        if "btc_holdings" in self.data and len(self.data["btc_holdings"]) > 1:
            btc_history = np.array(self.data["btc_holdings"])
            # ポジションの変化率
            btc_change_rates = np.abs(np.diff(btc_history)) / (btc_history[:-1] + 1e-8)
            btc_analysis["btc_position_stability"] = 1.0 - np.mean(
                btc_change_rates
            )  # 安定性指標（1=非常に安定、0=非常に不安定）

        return btc_analysis

    def analyze_action_averages(self) -> ActionAveragesResult:
        """アクション平均分析"""
        if "actions" not in self.data:
            return {}

        actions = np.array(self.data["actions"])
        if len(actions) == 0:
            return {}

        # 基本的なアクション統計
        action_mean = np.mean(actions)
        action_std = np.std(actions)
        action_median = np.median(actions)
        action_mode = (
            float(pd.Series(actions).mode().iloc[0]) if len(actions) > 0 else 0.0
        )

        # アクションの時間的変化（トレンド）
        if len(actions) > 10:
            # 移動平均でアクションのトレンドを分析
            window_size = min(50, len(actions) // 10)
            action_ma = pd.Series(actions).rolling(window=window_size).mean()
            action_trend = (
                action_ma.iloc[-1] - action_ma.iloc[0] if len(action_ma) > 1 else 0
            )
        else:
            action_trend = 0

        # アクションの安定性（変動係数）
        action_cv = action_std / action_mean if action_mean != 0 else 0

        # アクションの偏り分析
        from scipy import stats

        action_skewness = stats.skew(actions)
        action_kurtosis = stats.kurtosis(actions)

        # アクションの遷移確率
        transitions = {}
        for i in range(len(actions) - 1):
            current = int(actions[i])
            next_action = int(actions[i + 1])
            key = f"{current}->{next_action}"
            transitions[key] = transitions.get(key, 0) + 1

        # 最も頻繁な遷移
        most_common_transition = (
            max(transitions.items(), key=lambda x: x[1]) if transitions else ("N/A", 0)
        )

        return ActionAveragesResult(
            action_mean=float(action_mean),
            action_std=float(action_std),
            action_median=float(action_median),
            action_mode=float(action_mode),
            action_trend=float(action_trend),
            action_cv=float(action_cv),
            action_skewness=action_skewness,
            action_kurtosis=action_kurtosis,
            most_common_transition=most_common_transition[0],
            transition_frequency=most_common_transition[1],
            total_transitions=sum(transitions.values()),
        )

    def generate_comprehensive_report(self) -> str:
        """包括的な分析レポートを生成"""
        with self.performance_monitor:
            try:
                report_lines = []
                report_lines.append("=" * 80)
                report_lines.append("汎用バックテスト分析レポート")
                report_lines.append("=" * 80)
                report_lines.append(f"分析対象ファイル: {self.results_path.name}")
                report_lines.append(
                    f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                report_lines.append("")

                # 基本情報
                report_lines.append("=== 基本情報 ===")
                report_lines.append(
                    f"総ステップ数: {self.data.get('total_steps', 'N/A')}"
                )
                report_lines.append(
                    f"初期ポートフォリオ: {self.data.get('initial_portfolio', 0):,.0f} JPY"
                )
                report_lines.append(
                    f"最終ポートフォリオ: {self.data.get('final_portfolio', 0):,.0f} JPY"
                )
                total_return_pct = self.data.get("total_return_pct", 0)
                report_lines.append(f"総リターン: {total_return_pct:.2f}%")
                report_lines.append(f"総取引数: {self.data.get('total_trades', 0)}")
                report_lines.append(f"勝率: {self.data.get('win_rate', 0):.1f}%")
                report_lines.append("")

                # トレーニング比較分析（トレーニングレポートがある場合）
                if self.training_data:
                    report_lines.append("=== トレーニング vs バックテスト比較 ===")

                    # アクション分布比較
                    training_actions = self.training_data.get("training_stats", {}).get(
                        "action_distribution", {}
                    )
                    backtest_actions = self._analyze_backtest_action_distribution()

                    if training_actions and backtest_actions:
                        report_lines.append("アクション分布比較:")
                        for action in ["HOLD", "BUY", "SELL"]:
                            train_pct = training_actions.get(action.upper(), 0) * 100
                            backtest_pct = backtest_actions.get(action, 0) * 100
                            diff = backtest_pct - train_pct
                            report_lines.append(
                                f"  {action}: 訓練 {train_pct:.1f}% → テスト {backtest_pct:.1f}% (差: {diff:+.1f}%)"
                            )
                        report_lines.append("")

                    # トレーニング情報
                    training_stats = self.training_data.get("training_stats", {})
                    report_lines.append("トレーニング情報:")
                    report_lines.append(
                        f"  トレーニング時間: {training_stats.get('training_time', 0):.1f}秒"
                    )
                    report_lines.append(
                        f"  ステップ/秒: {training_stats.get('steps_per_second', 0):.2f}"
                    )
                    report_lines.append(
                        f"  総ステップ数: {training_stats.get('total_timesteps', 0)}"
                    )
                    report_lines.append("")

                    # 環境設定比較
                    env_config = self.training_data.get("configuration", {}).get(
                        "environment", {}
                    )
                    report_lines.append("環境設定:")
                    report_lines.append(
                        f"  初期残高: ¥{env_config.get('initial_balance', 0):,.0f}"
                    )
                    report_lines.append(
                        f"  取引コスト: {env_config.get('transaction_cost', 0):.2e}"
                    )
                    report_lines.append(
                        f"  最大ポジションサイズ: {env_config.get('max_position_size', 0)}"
                    )
                    report_lines.append(
                        f"  連続アクション: {env_config.get('use_continuous_actions', False)}"
                    )
                    report_lines.append(
                        f"  標準化観測: {env_config.get('use_standardized_observations', False)}"
                    )
                    report_lines.append("")

                # リスク指標（metrics.pyの関数を使用）
                pnl_returns = None
                if "pnls" in self.data and self.data["pnls"]:
                    pnl_returns = np.array(self.data["pnls"])
                elif (
                    "portfolio_history" in self.data
                    and len(self.data["portfolio_history"]) > 1
                ):
                    # portfolio_historyからpnlsを計算
                    portfolio_values = np.array(self.data["portfolio_history"])
                    pnl_returns = np.diff(portfolio_values) / portfolio_values[:-1]

                if pnl_returns is not None and len(pnl_returns) > 0:
                    metrics = calculate_all_metrics(pnl_returns)
                    report_lines.append("=== リスク指標 (metrics.py) ===")
                    report_lines.append(
                        f"総リターン: {metrics.get('total_return', 0):.2%} → 全体期間での総収益率"
                    )
                    report_lines.append(
                        f"年間リターン: {metrics.get('annual_return', 0):.2%} → 年間換算のリターン"
                    )
                    report_lines.append(
                        f"シャープレシオ: {metrics.get('sharpe_ratio', 0):.3f} → リスク1単位あたりの超過リターン（>1.0が良好）"
                    )
                    report_lines.append(
                        f"ソルティーノレシオ: {metrics.get('sortino_ratio', 0):.3f} → 下落リスクのみ考慮したシャープレシオ"
                    )
                    report_lines.append(
                        f"カルマーレシオ: {metrics.get('calmar_ratio', 0):.3f} → 最大ドローダウンに対する年間リターンの比率"
                    )
                    report_lines.append(
                        f"最大ドローダウン: {metrics.get('max_drawdown', 0):.2%} → ピークからの最大下落率（低いほど安定）"
                    )
                    report_lines.append(
                        f"勝率: {metrics.get('win_rate', 0):.1%} → 利益が出た取引の割合"
                    )
                    report_lines.append(
                        f"プロフィットファクター: {metrics.get('profit_factor', 0):.3f} → 総利益/総損失の比率（>1.0が利益）"
                    )
                    report_lines.append(
                        f"期待値: {metrics.get('expected_value', 0):.3f} → 1トレードあたりの平均期待収益"
                    )
                    report_lines.append(
                        f"回復力: {metrics.get('recovery_factor', 0):.3f} → ドローダウンからの回復能力"
                    )
                    report_lines.append(
                        f"ボラティリティ: {metrics.get('volatility', 0):.2%} → リターンの変動性（高いほどリスク大）"
                    )
                    report_lines.append("")

                    report_lines.append("")

                    # P平均法分析（幾何平均リターンで代用）
                    try:
                        # 幾何平均リターンを計算（P平均法の近似）
                        if len(pnl_returns) > 0:
                            geometric_mean = (
                                np.exp(np.mean(np.log(1 + pnl_returns))) - 1
                            )
                            report_lines.append("=== P平均法分析 (幾何平均) ===")
                            report_lines.append(
                                f"P平均リターン (幾何平均): {geometric_mean:.4f} → 複利効果を考慮した平均リターン率。値が大きいほど安定した収益性が高い"
                            )
                            report_lines.append("")
                    except Exception as e:
                        logger.warning(f"P平均法分析エラー: {e}")
                        report_lines.append(f"P平均法分析エラー: {e}")
                        report_lines.append("")

                    # 市場レジーム分類
                    try:
                        regime_result = classify_market_regime(pnl_returns)
                        if regime_result is not None:
                            # 最後のレジームを取得
                            if hasattr(regime_result, "iloc"):
                                current_regime = (
                                    regime_result.iloc[-1]
                                    if len(regime_result) > 0
                                    else "unknown"
                                )
                            else:
                                current_regime = regime_result
                            report_lines.append("=== 市場レジーム分類 ===")
                            report_lines.append(f"現在のレジーム: {current_regime}")
                            report_lines.append("")
                    except Exception as e:
                        logger.warning(f"市場レジーム分類エラー: {e}")
                        report_lines.append(f"市場レジーム分類エラー: {e}")
                        report_lines.append("")

                    # 統計的検定（単一データなので基本的なもののみ）
                    try:
                        # 基本統計量
                        report_lines.append("=== 統計的検定 ===")
                        report_lines.append(f"平均リターン: {np.mean(pnl_returns):.6f}")
                        report_lines.append(
                            f"リターン標準偏差: {np.std(pnl_returns):.6f}"
                        )
                        report_lines.append(
                            f"歪度: {pd.Series(pnl_returns).skew():.3f}"
                        )
                        report_lines.append(
                            f"尖度: {pd.Series(pnl_returns).kurtosis():.3f}"
                        )
                        report_lines.append("")
                    except Exception as e:
                        logger.warning(f"統計的検定エラー: {e}")
                        report_lines.append(f"統計的検定エラー: {e}")
                        report_lines.append("")

                    # 季節性分析
                    if "timestamps" in self.data and self.data["timestamps"]:
                        try:
                            seasonality_results = seasonality_analysis(
                                pnl_returns, self.data["timestamps"]
                            )
                            if seasonality_results:
                                report_lines.append("=== 季節性分析 ===")
                                report_lines.append(
                                    f"日次季節性: {seasonality_results.get('daily_seasonality', 'N/A')}"
                                )
                                report_lines.append(
                                    f"週次季節性: {seasonality_results.get('weekly_seasonality', 'N/A')}"
                                )
                                report_lines.append(
                                    f"月次季節性: {seasonality_results.get('monthly_seasonality', 'N/A')}"
                                )
                                report_lines.append(
                                    f"季節性強度: {seasonality_results.get('seasonality_strength', 0):.2%}"
                                )
                                report_lines.append("")
                        except Exception as e:
                            logger.warning(f"季節性分析エラー: {e}")
                            report_lines.append(f"季節性分析エラー: {e}")
                            report_lines.append("")

                    # 多市場分析
                    if (
                        "price_history" in self.data
                        and self.data["price_history"]
                        and len(self.data["price_history"]) > len(pnl_returns)
                    ):
                        try:
                            market_analysis = multi_market_backtest_analysis(
                                pnl_returns, self.data["price_history"]
                            )
                            if market_analysis:
                                report_lines.append("=== 多市場分析 ===")
                                report_lines.append(
                                    f"市場相関: {market_analysis.get('market_correlation', 0):.3f}"
                                )
                                report_lines.append(
                                    f"ベータ値: {market_analysis.get('beta', 0):.3f}"
                                )
                                report_lines.append(
                                    f"アルファ値: {market_analysis.get('alpha', 0):.4f}"
                                )
                                report_lines.append(
                                    f"情報比率: {market_analysis.get('information_ratio', 0):.3f}"
                                )
                                report_lines.append("")
                        except Exception as e:
                            logger.warning(f"多市場分析エラー: {e}")
                            report_lines.append(f"多市場分析エラー: {e}")
                            report_lines.append("")

                # 従来のリスク指標（フォールバック）
                risk_metrics = self.calculate_risk_metrics()
                if risk_metrics and not ("pnls" in self.data and self.data["pnls"]):
                    report_lines.append("=== リスク指標 (従来) ===")
                    report_lines.append(
                        f"シャープレシオ: {risk_metrics.get('sharpe_ratio', 0):.3f}"
                    )
                    report_lines.append(
                        f"ソルティーノレシオ: {risk_metrics.get('sortino_ratio', 0):.3f}"
                    )
                    report_lines.append(
                        f"最大ドローダウン: {risk_metrics.get('max_drawdown', 0):.2%}"
                    )
                    report_lines.append(
                        f"ボラティリティ: {risk_metrics.get('volatility', 0):.2%}"
                    )
                    report_lines.append(
                        f"プロフィットファクター: {risk_metrics.get('profit_factor', 0):.3f}"
                    )
                    report_lines.append("")

                # アクション分析
                if "action_distribution" in self.data:
                    report_lines.append("=== アクション分布 ===")
                    actions = self.data["action_distribution"]
                    total_actions = sum(actions.values())
                    action_names = {
                        ACTION_HOLD: "HOLD",
                        ACTION_BUY: "BUY",
                        ACTION_SELL: "SELL",
                    }

                    # action_distributionが辞書の場合（キーが'HOLD', 'BUY', 'SELL'の文字列）
                    if isinstance(actions, dict) and all(
                        isinstance(k, str) for k in actions.keys()
                    ):
                        for action_name, count in actions.items():
                            pct = (
                                count / total_actions * 100 if total_actions > 0 else 0
                            )
                            report_lines.append(
                                f"  {action_name}: {count}回 ({pct:.1f}%)"
                            )
                    else:
                        # 従来の数値キー形式
                        for action_id, count in actions.items():
                            pct = (
                                count / total_actions * 100 if total_actions > 0 else 0
                            )
                            action_name = action_names.get(
                                int(action_id), f"UNKNOWN({action_id})"
                            )
                            report_lines.append(
                                f"  {action_name}: {count}回 ({pct:.1f}%)"
                            )
                    report_lines.append("")

                # アクション平均分析
                action_averages = self.analyze_action_averages()
                if action_averages:
                    report_lines.append("=== アクション平均分析 ===")
                    report_lines.append(
                        f"アクション平均: {action_averages.get('action_mean', 0):.3f} → 平均アクション値（0=HOLD, 1=BUY, 2=SELL）"
                    )
                    report_lines.append(
                        f"アクション標準偏差: {action_averages.get('action_std', 0):.3f} → アクションの変動性"
                    )
                    report_lines.append(
                        f"アクション中央値: {action_averages.get('action_median', 0):.1f} → アクションの中央値"
                    )
                    report_lines.append(
                        f"アクション最頻値: {action_averages.get('action_mode', 0):.1f} → 最も頻繁なアクション"
                    )
                    report_lines.append(
                        f"アクション変動係数: {action_averages.get('action_cv', 0):.3f} → アクションの相対的変動性"
                    )
                    report_lines.append(
                        f"アクション歪度: {action_averages.get('action_skewness', 0):.3f} → アクション分布の歪み"
                    )
                    report_lines.append(
                        f"アクション尖度: {action_averages.get('action_kurtosis', 0):.3f} → アクション分布の尖り具合"
                    )
                    if action_averages.get("action_trend", 0) != 0:
                        trend_desc = (
                            "上昇傾向"
                            if action_averages.get("action_trend", 0) > 0
                            else "下降傾向"
                        )
                        report_lines.append(
                            f"アクショントレンド: {action_averages.get('action_trend', 0):.3f} ({trend_desc}) → アクションの時間的変化"
                        )
                    report_lines.append(
                        f"最も一般的な遷移: {action_averages.get('most_common_transition', 'N/A')} ({action_averages.get('transition_frequency', 0)}回)"
                    )
                    report_lines.append("")

                # 取引頻度分析
                trading_freq = self.analyze_trading_frequency()
                if trading_freq:
                    report_lines.append("=== 取引頻度分析 ===")
                    report_lines.append(
                        f"取引頻度: {trading_freq.get('trade_frequency', 0):.3f} (取引/ステップ)"
                    )
                    report_lines.append(
                        f"総取引数: {trading_freq.get('total_trades', 0)}"
                    )
                    if trading_freq.get("avg_trade_interval", 0) > 0:
                        report_lines.append(
                            f"平均取引間隔: {trading_freq.get('avg_trade_interval', 0):.1f}ステップ"
                        )
                        report_lines.append(
                            f"最小取引間隔: {trading_freq.get('min_trade_interval', 0)}ステップ"
                        )
                        report_lines.append(
                            f"最大取引間隔: {trading_freq.get('max_trade_interval', 0)}ステップ"
                        )
                    report_lines.append("")

                # BTCパフォーマンス分析
                btc_analysis = self.analyze_btc_performance()
                if btc_analysis:
                    report_lines.append("=== BTCパフォーマンス分析 ===")
                    report_lines.append(
                        f"初期BTC保有量: {btc_analysis.get('initial_btc', 0):.6f} BTC"
                    )
                    report_lines.append(
                        f"最終BTC保有量: {btc_analysis.get('final_btc', 0):.6f} BTC"
                    )
                    report_lines.append(
                        f"純BTC獲得量: {btc_analysis.get('net_btc_gained', 0):+.6f} BTC"
                    )
                    report_lines.append(
                        f"BTCリターン: {btc_analysis.get('btc_return_pct', 0):+.2f}%"
                    )
                    report_lines.append(
                        f"USDリターン: {btc_analysis.get('usd_return_pct', 0):+.2f}%"
                    )

                    btc_vs_usd_ratio = btc_analysis.get(
                        "btc_vs_usd_performance_ratio", 0
                    )
                    if btc_vs_usd_ratio != 0:
                        report_lines.append(
                            f"BTC/USDパフォーマンス比: {btc_vs_usd_ratio:.2f}"
                        )

                    # BTC保有統計（利用可能な場合）
                    if "btc_mean_holding" in btc_analysis:
                        report_lines.append(
                            f"平均BTC保有量: {btc_analysis.get('btc_mean_holding', 0):.6f} BTC"
                        )
                        report_lines.append(
                            f"最大BTC保有量: {btc_analysis.get('btc_max_holding', 0):.6f} BTC"
                        )
                        report_lines.append(
                            f"最小BTC保有量: {btc_analysis.get('btc_min_holding', 0):.6f} BTC"
                        )
                        report_lines.append(
                            f"BTC保有ボラティリティ: {btc_analysis.get('btc_holding_volatility', 0):.6f}"
                        )

                    # BTC取引分析
                    if "btc_trade_frequency" in btc_analysis:
                        report_lines.append(
                            f"BTC取引頻度: {btc_analysis.get('btc_trade_frequency', 0):.3f} (取引/ステップ)"
                        )
                        report_lines.append(
                            f"BTCポジション増加回数: {btc_analysis.get('btc_positive_changes', 0)}回"
                        )
                        report_lines.append(
                            f"BTCポジション減少回数: {btc_analysis.get('btc_negative_changes', 0)}回"
                        )

                    # BTCポジション安定性
                    if "btc_position_stability" in btc_analysis:
                        stability = btc_analysis.get("btc_position_stability", 0)
                        stability_desc = (
                            "非常に安定"
                            if stability > 0.8
                            else "安定"
                            if stability > 0.6
                            else "不安定"
                            if stability > 0.4
                            else "非常に不安定"
                        )
                        report_lines.append(
                            f"BTCポジション安定性: {stability:.2f} ({stability_desc})"
                        )

                    report_lines.append("")

                # 時間帯別分析
                temporal = self.analyze_temporal_patterns()
                if temporal and temporal.get("hourly_returns"):
                    report_lines.append("=== 時間帯別リターン (上位/下位3件) ===")
                    hourly = temporal.get("hourly_returns", {})
                    sorted_hourly = sorted(
                        hourly.items(), key=lambda x: x[1], reverse=True
                    )

                    report_lines.append("上位:")
                    for hour, ret in sorted_hourly[:3]:
                        report_lines.append(f"  {hour}: {ret:.2%}")

                    report_lines.append("下位:")
                    for hour, ret in sorted_hourly[-3:]:
                        report_lines.append(f"  {hour}: {ret:.2%}")
                    report_lines.append("")

                # 市場環境別分析
                market_cond = self.analyze_market_conditions()
                if market_cond:
                    report_lines.append("=== 市場環境別分析 ===")
                    for condition, data in market_cond.items():
                        if data is not None and isinstance(data, dict):
                            report_lines.append(f"{data.get('name', condition)}:")
                            report_lines.append(
                                f"  リターン: {data.get('return', 0):.2%}"
                            )
                            report_lines.append(f"  期間数: {data.get('periods', 0)}")
                    report_lines.append("")
                    report_lines.append("")
                    report_lines.append("")

                # 連続アクション分析
                if (
                    "continuous_action_stats" in self.data
                    and self.data["continuous_action_stats"]
                ):
                    stats = self.data["continuous_action_stats"]
                    if "action_streaks" in stats:
                        streaks = stats["action_streaks"]
                        report_lines.append("=== 連続アクション分析 ===")
                        report_lines.append("BUYストリーク:")
                        report_lines.append(
                            f"  最大連続: {streaks.get('max_buy_streak', 0)}回"
                        )
                        report_lines.append(
                            f"  平均連続: {streaks.get('avg_buy_streak', 0):.1f}回"
                        )
                        report_lines.append("SELLストリーク:")
                        report_lines.append(
                            f"  最大連続: {streaks.get('max_sell_streak', 0)}回"
                        )
                        report_lines.append(
                            f"  平均連続: {streaks.get('avg_sell_streak', 0):.1f}回"
                        )
                        report_lines.append("")

                # データバイアス分析
                try:
                    # 分析用のDataFrameを作成
                    analysis_df = pd.DataFrame()
                    if "timestamps" in self.data and self.data["timestamps"]:
                        analysis_df["timestamp"] = pd.to_datetime(
                            self.data["timestamps"]
                        )

                    # price_historyとportfolio_historyの長さを合わせる
                    price_data = self.data.get("price_history", [])
                    portfolio_data = self.data.get("portfolio_history", [])

                    if price_data and portfolio_data:
                        min_length = min(len(price_data), len(portfolio_data))
                        analysis_df["close"] = price_data[:min_length]
                        analysis_df["portfolio"] = portfolio_data[:min_length]

                        # ポートフォリオリターンを計算
                        analysis_df["returns"] = pd.Series(
                            analysis_df["portfolio"]
                        ).pct_change()

                    # データがある場合のみバイアス分析を実行
                    if len(analysis_df) > 0:
                        bias_report = self.bias_detector.detect_data_bias(analysis_df)
                        report_lines.append("=== データバイアス分析 ===")

                        # 時間周期バイアス
                        time_bias = bias_report.get("time_period_bias", {})
                        if time_bias.get("bias_detected", False):
                            report_lines.append(
                                f"⚠️  時間周期バイアス検出: {time_bias.get('bias_score', 0):.2f}"
                            )
                            report_lines.append(
                                f"   カバー期間: {time_bias.get('time_range_days', 0)}日"
                            )
                            report_lines.append(
                                f"   カバー月数: {time_bias.get('months_covered', 0)}ヶ月"
                            )
                        else:
                            report_lines.append("✅ 時間周期バイアス: なし")

                        # トレンドバイアス
                        trend_bias = bias_report.get("trend_bias", {})
                        if trend_bias.get("bias_detected", False):
                            report_lines.append(
                                f"⚠️  トレンドバイアス検出: {trend_bias.get('bias_score', 0):.2f}"
                            )
                            report_lines.append(
                                f"   陽線比率: {trend_bias.get('positive_ratio', 0):.1%}"
                            )
                        else:
                            report_lines.append("✅ トレンドバイアス: なし")

                        # ボラティリティバイアス
                        vol_bias = bias_report.get("volatility_bias", {})
                        if vol_bias.get("bias_detected", False):
                            report_lines.append(
                                f"⚠️  ボラティリティバイアス検出: {vol_bias.get('bias_score', 0):.2f}"
                            )
                            report_lines.append(
                                f"   ボラティリティ: {vol_bias.get('volatility', 0):.4f}"
                            )
                        else:
                            report_lines.append("✅ ボラティリティバイアス: なし")

                        # レジームバイアス
                        regime_bias = bias_report.get("regime_bias", {})
                        if regime_bias.get("bias_detected", False):
                            report_lines.append(
                                f"⚠️  市場レジームバイアス検出: {regime_bias.get('bias_score', 0):.2f}"
                            )
                            reg_dist = regime_bias.get("regime_distribution", {})
                            report_lines.append(
                                f"   強気相場: {reg_dist.get('bull_pct', 0):.1%}"
                            )
                            report_lines.append(
                                f"   弱気相場: {reg_dist.get('bear_pct', 0):.1%}"
                            )
                            report_lines.append(
                                f"   横ばい相場: {reg_dist.get('sideways_pct', 0):.1%}"
                            )
                        else:
                            report_lines.append("✅ 市場レジームバイアス: なし")

                        # BTC特有バイアス
                        btc_bias = bias_report.get("btc_specific_bias", {})
                        if btc_bias.get("bias_detected", False):
                            report_lines.append(
                                f"⚠️  BTC特有バイアス検出: {btc_bias.get('bias_score', 0):.2f}"
                            )
                            for issue in btc_bias.get("issues", []):
                                report_lines.append(f"   - {issue}")
                        else:
                            report_lines.append("✅ BTC特有バイアス: なし")

                        report_lines.append("")

                except Exception as e:
                    logger.warning(f"バイアス分析エラー: {e}")
                    report_lines.append(f"バイアス分析エラー: {e}")
                    report_lines.append("")

                # ロバストネス分析
                try:
                    robustness = self.analyze_robustness()
                    if "error" not in robustness:
                        report_lines.append("=== ロバストネス分析 ===")
                        report_lines.append(
                            f"総合ロバストネススコア: {robustness.get('robustness_score', 0):.3f}"
                        )

                        # ボラティリティレジーム分析
                        vol_analysis = robustness.get("volatility_analysis", {})
                        if vol_analysis:
                            report_lines.append("ボラティリティレジーム分析:")
                            high_vol = vol_analysis.get(
                                "high_volatility_performance", {}
                            )
                            low_vol = vol_analysis.get("low_volatility_performance", {})
                            report_lines.append(
                                f"  高ボラティリティ: シャープ {high_vol.get('sharpe_ratio', 0):.3f}, 勝率 {high_vol.get('win_rate', 0):.1%}"
                            )
                            report_lines.append(
                                f"  低ボラティリティ: シャープ {low_vol.get('sharpe_ratio', 0):.3f}, 勝率 {low_vol.get('win_rate', 0):.1%}"
                            )
                            report_lines.append(
                                f"  レジーム間一貫性: {vol_analysis.get('volatility_regime_consistency', 0):.3f}"
                            )

                        # トレンドレジーム分析
                        trend_analysis = robustness.get("trend_analysis", {})
                        if trend_analysis:
                            report_lines.append("トレンドレジーム分析:")
                            uptrend = trend_analysis.get("uptrend_performance", {})
                            downtrend = trend_analysis.get("downtrend_performance", {})
                            sideways = trend_analysis.get("sideways_performance", {})
                            report_lines.append(
                                f"  上昇トレンド: シャープ {uptrend.get('sharpe_ratio', 0):.3f}, 勝率 {uptrend.get('win_rate', 0):.1%}"
                            )
                            report_lines.append(
                                f"  下降トレンド: シャープ {downtrend.get('sharpe_ratio', 0):.3f}, 勝率 {downtrend.get('win_rate', 0):.1%}"
                            )
                            report_lines.append(
                                f"  横ばい: シャープ {sideways.get('sharpe_ratio', 0):.3f}, 勝率 {sideways.get('win_rate', 0):.1%}"
                            )
                            report_lines.append(
                                f"  レジーム間一貫性: {trend_analysis.get('trend_regime_consistency', 0):.3f}"
                            )

                        # 季節性分析
                        seasonal = robustness.get("seasonal_analysis", {})
                        if seasonal and "error" not in seasonal:
                            report_lines.append("季節性分析:")
                            report_lines.append(
                                f"  時間帯別一貫性: {seasonal.get('seasonal_consistency_score', 0):.3f}"
                            )

                            # 最も良い/悪い時間帯を表示
                            hourly_perf = seasonal.get("hourly_performance", {})
                            if hourly_perf:
                                best_hour = max(
                                    hourly_perf.items(),
                                    key=lambda x: x[1].get("sharpe_ratio", 0)
                                    if x[1]
                                    else 0,
                                )
                                worst_hour = min(
                                    hourly_perf.items(),
                                    key=lambda x: x[1].get("sharpe_ratio", 0)
                                    if x[1]
                                    else 0,
                                )
                                if best_hour[1] and isinstance(best_hour[1], dict):
                                    report_lines.append(
                                        f"  最適時間帯: {best_hour[0]}時 (シャープ {best_hour[1].get('sharpe_ratio', 0):.3f})"
                                    )
                                if worst_hour[1] and isinstance(worst_hour[1], dict):
                                    report_lines.append(
                                        f"  最悪時間帯: {worst_hour[0]}時 (シャープ {worst_hour[1].get('sharpe_ratio', 0):.3f})"
                                    )

                        report_lines.append("")
                    else:
                        report_lines.append(
                            f"ロバストネス分析エラー: {robustness['error']}"
                        )
                        report_lines.append("")

                except Exception as e:
                    logger.warning(f"ロバストネス分析エラー: {e}")
                    report_lines.append(f"ロバストネス分析エラー: {e}")
                    report_lines.append("")

                # 相関分析と依存関係
                try:
                    correlation_analysis = self.analyze_correlation_and_dependencies()
                    if correlation_analysis:
                        report_lines.append("=== 相関分析と依存関係 ===")
                        report_lines.append(
                            f"価格-ポートフォリオ相関: {correlation_analysis.get('price_portfolio_correlation', 0):.3f}"
                        )

                        # ラグ相関のトップ3
                        lag_corrs = correlation_analysis.get("lag_correlations", {})
                        if lag_corrs:
                            sorted_lags = sorted(
                                lag_corrs.items(), key=lambda x: abs(x[1]), reverse=True
                            )[:3]
                            report_lines.append("ラグ相関 (トップ3):")
                            for lag, corr in sorted_lags:
                                report_lines.append(f"  {lag}: {corr:.3f}")

                        report_lines.append(
                            f"ベータ値: {correlation_analysis.get('beta', 0):.3f}"
                        )

                        # アクションと価格変化の関係
                        action_rels = correlation_analysis.get(
                            "action_price_relationships", {}
                        )
                        if action_rels:
                            report_lines.append("アクション別価格変化:")
                            for action, change in action_rels.items():
                                report_lines.append(f"  {action}: {change:.4f}")
                        report_lines.append("")
                except Exception as e:
                    logger.warning(f"相関分析エラー: {e}")
                    report_lines.append(f"相関分析エラー: {e}")
                    report_lines.append("")

                # 取引コスト影響分析
                try:
                    cost_impact = self.analyze_transaction_cost_impact()
                    if cost_impact:
                        report_lines.append("=== 取引コスト影響分析 ===")
                        report_lines.append(
                            f"総取引コスト: ¥{cost_impact.get('total_transaction_cost', 0):,.0f}"
                        )
                        report_lines.append(
                            f"取引ごとの平均コスト: ¥{cost_impact.get('average_cost_per_trade', 0):,.2f}"
                        )
                        report_lines.append(
                            f"コスト対リターン比: {cost_impact.get('cost_to_return_ratio', 0):.4f}"
                        )
                        report_lines.append(
                            f"ステップあたり取引数: {cost_impact.get('trades_per_step', 0):.3f}"
                        )
                        report_lines.append(
                            f"コスト効率スコア: {cost_impact.get('cost_efficiency_score', 0):.3f}"
                        )
                        report_lines.append("")
                except Exception as e:
                    logger.warning(f"取引コスト分析エラー: {e}")
                    report_lines.append(f"取引コスト分析エラー: {e}")
                    report_lines.append("")

                # ストレステスト
                try:
                    stress_tests = self.perform_stress_tests()
                    if stress_tests:
                        report_lines.append("=== ストレステスト結果 ===")
                        for test_name, results in stress_tests.items():
                            report_lines.append(
                                f"{test_name.replace('_', ' ').title()}:"
                            )
                            report_lines.append(
                                f"  ストレス下リターン: {results.get('stressed_return', 0):.2%}"
                            )
                            if "survival_probability" in results:
                                report_lines.append(
                                    f"  生存確率: {results['survival_probability']:.1%}"
                                )
                            if "volatility_multiplier" in results:
                                report_lines.append(
                                    f"  ボラティリティ倍率: {results['volatility_multiplier']:.1f}"
                                )
                            if "cost_multiplier" in results:
                                report_lines.append(
                                    f"  コスト倍率: {results['cost_multiplier']:.1f}"
                                )
                            if "affected_periods" in results:
                                report_lines.append(
                                    f"  影響期間数: {results['affected_periods']}"
                                )
                        report_lines.append("")
                except Exception as e:
                    logger.warning(f"ストレステストエラー: {e}")
                    report_lines.append(f"ストレステストエラー: {e}")
                    report_lines.append("")

                # ウォークフォワード効率分析
                try:
                    wf_efficiency = self.analyze_walk_forward_efficiency()
                    if wf_efficiency:
                        report_lines.append("=== ウォークフォワード効率分析 ===")
                        for window_name, metrics in wf_efficiency.items():
                            if window_name.startswith("window_"):
                                report_lines.append(
                                    f"{window_name.replace('_', ' ').title()}:"
                                )
                                report_lines.append(
                                    f"  平均リターン: {metrics.get('mean_return', 0):.4f}"
                                )
                                report_lines.append(
                                    f"  ボラティリティ: {metrics.get('volatility', 0):.4f}"
                                )
                                report_lines.append(
                                    f"  シャープレシオ: {metrics.get('sharpe_ratio', 0):.3f}"
                                )
                                report_lines.append(
                                    f"  一貫性スコア: {metrics.get('consistency_score', 0):.3f}"
                                )
                            elif window_name == "adaptation_analysis":
                                report_lines.append("学習適応分析:")
                                report_lines.append(
                                    f"  前半リターン: {metrics.get('first_half_return', 0):.4f}"
                                )
                                report_lines.append(
                                    f"  後半リターン: {metrics.get('second_half_return', 0):.4f}"
                                )
                                report_lines.append(
                                    f"  適応比率: {metrics.get('adaptation_ratio', 0):.3f}"
                                )
                                report_lines.append(
                                    f"  学習効率: {metrics.get('learning_efficiency', 0):.3f}"
                                )
                        report_lines.append("")
                except Exception as e:
                    logger.warning(f"ウォークフォワード分析エラー: {e}")
                    report_lines.append(f"ウォークフォワード分析エラー: {e}")
                    report_lines.append("")

                # 市場マイクロストラクチャー分析
                try:
                    microstructure = self.analyze_microstructure_effects()
                    if microstructure:
                        report_lines.append("=== 市場マイクロストラクチャー分析 ===")

                        # 価格インパクト
                        price_impact = microstructure.get("price_impact", {})
                        if price_impact:
                            report_lines.append("価格インパクト:")
                            report_lines.append(
                                f"  平均インパクト: {price_impact.get('average_impact', 0):.4f}"
                            )
                            report_lines.append(
                                f"  インパクトボラティリティ: {price_impact.get('impact_volatility', 0):.4f}"
                            )
                            report_lines.append(
                                f"  逆選択コスト: {price_impact.get('adverse_selection_cost', 0):.4f}"
                            )

                        # 市場の深さ
                        market_depth = microstructure.get("market_depth", {})
                        if market_depth:
                            report_lines.append("市場の深さ:")
                            report_lines.append(
                                f"  価格ボラティリティ: {market_depth.get('price_volatility', 0):.4f}"
                            )
                            report_lines.append(
                                f"  流動性プロキシ: {market_depth.get('liquidity_proxy', 0):.3f}"
                            )
                            report_lines.append(
                                f"  取引効率: {market_depth.get('trading_efficiency', 0):.3f}"
                            )

                        # スプレッド分析
                        spread_analysis = microstructure.get("spread_analysis", {})
                        if spread_analysis:
                            report_lines.append("スプレッド分析:")
                            report_lines.append(
                                f"  推定スプレッド: ¥{spread_analysis.get('estimated_spread', 0):,.2f}"
                            )
                            report_lines.append(
                                f"  スプレッド/価格比: {spread_analysis.get('spread_to_price_ratio', 0):.4f}"
                            )
                            report_lines.append(
                                f"  スリッページリスク: ¥{spread_analysis.get('slippage_risk', 0):,.0f}"
                            )

                        # 行動パターン
                        behavioral = microstructure.get("behavioral_patterns", {})
                        if behavioral:
                            report_lines.append("行動パターン:")
                            report_lines.append(
                                f"  アクション自己相関: {behavioral.get('action_autocorrelation', 0):.3f}"
                            )
                            report_lines.append(
                                f"  モメンタム効果: {behavioral.get('momentum_effect', 0):.3f}"
                            )
                            report_lines.append(
                                f"  平均回帰傾向: {behavioral.get('mean_reversion_tendency', 0):.3f}"
                            )
                        report_lines.append("")
                except Exception as e:
                    logger.warning(f"マイクロストラクチャー分析エラー: {e}")
                    report_lines.append(f"マイクロストラクチャー分析エラー: {e}")
                    report_lines.append("")

                report_lines.append("=" * 80)
                return "\n".join(report_lines)

            except Exception as e:
                logger.error(f"レポート生成中にエラーが発生しました: {e}")
                return f"エラーが発生しました: {e}"

    def analyze_robustness(self) -> RobustnessAnalysisResult:
        """多様な市場条件下でのロバストネス分析"""
        with self.performance_monitor:
            robustness_metrics = {}

            if "pnls" not in self.data and "portfolio_history" not in self.data:
                # Return default robustness result for error case
                default_metrics = PerformanceMetricsResult(
                    total_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                    num_trades=0,
                )
                return RobustnessAnalysisResult(
                    overall_performance=default_metrics,
                    volatility_analysis=None,
                    trend_analysis=None,
                    drawdown_analysis=None,
                    seasonal_analysis=None,
                    robustness_score=0.0,
                )

            # PnLデータの準備
            if "pnls" in self.data:
                pnls = np.array(self.data["pnls"])
            elif "portfolio_history" in self.data:
                portfolio_values = np.array(self.data["portfolio_history"])
                pnls = np.diff(portfolio_values) / portfolio_values[:-1]
            else:
                # Return default robustness result for error case
                default_metrics = PerformanceMetricsResult(
                    total_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                    num_trades=0,
                )
                return RobustnessAnalysisResult(
                    overall_performance=default_metrics,
                    volatility_analysis=None,
                    trend_analysis=None,
                    drawdown_analysis=None,
                    seasonal_analysis=None,
                    robustness_score=0.0,
                )

            # 様々な市場条件下での分析
            robustness_metrics[
                "overall_performance"
            ] = self._calculate_performance_metrics(pnls)

            # ボラティリティ別の分析
            try:
                robustness_metrics[
                    "volatility_analysis"
                ] = self._analyze_by_volatility_regimes(pnls)
            except Exception as e:
                robustness_metrics["volatility_analysis"] = {
                    "error": f"Volatility analysis failed: {str(e)}"
                }

            # トレンド別の分析
            try:
                robustness_metrics["trend_analysis"] = self._analyze_by_trend_regimes(
                    pnls
                )
            except Exception as e:
                robustness_metrics["trend_analysis"] = {
                    "error": f"Trend analysis failed: {str(e)}"
                }

            # ドローダウン別の分析
            try:
                robustness_metrics[
                    "drawdown_analysis"
                ] = self._analyze_by_drawdown_periods(pnls)
            except Exception as e:
                robustness_metrics["drawdown_analysis"] = {
                    "error": f"Drawdown analysis failed: {str(e)}"
                }

            # 季節性分析
            try:
                robustness_metrics[
                    "seasonal_analysis"
                ] = self._analyze_seasonal_performance(pnls)
            except Exception as e:
                robustness_metrics["seasonal_analysis"] = {
                    "error": f"Seasonal analysis failed: {str(e)}"
                }

            # ロバストネススコアの計算
            robustness_score = self._calculate_robustness_score(robustness_metrics)

            return RobustnessAnalysisResult(
                overall_performance=robustness_metrics["overall_performance"],
                volatility_analysis=robustness_metrics.get("volatility_analysis"),
                trend_analysis=robustness_metrics.get("trend_analysis"),
                drawdown_analysis=robustness_metrics.get("drawdown_analysis"),
                seasonal_analysis=robustness_metrics.get("seasonal_analysis"),
                robustness_score=robustness_score,
            )

    def _calculate_performance_metrics(
        self, pnls: np.ndarray
    ) -> PerformanceMetricsResult:
        """基本的なパフォーマンス指標を計算"""
        if len(pnls) == 0:
            return PerformanceMetricsResult(
                total_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                num_trades=0,
            )

        # metrics.pyの関数を使用して指標を計算
        from ztb.metrics.metrics import max_drawdown, sharpe_ratio, win_rate

        total_return = np.prod(1 + pnls) - 1
        volatility = np.std(pnls)

        # 既存のmetrics関数を使用
        sharpe_ratio_value = sharpe_ratio(pnls)

        # returnsからequity_curveを計算してmax_drawdownを計算
        equity_curve = np.cumprod(1 + pnls)
        max_drawdown_value = max_drawdown(equity_curve)

        win_rate_value = win_rate(pnls)

        return {
            "total_return": total_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio_value,
            "max_drawdown": max_drawdown_value,
            "win_rate": win_rate_value,
            "num_trades": len(pnls),
        }

    def _analyze_by_volatility_regimes(self, pnls: np.ndarray) -> Dict[str, Any]:
        """ボラティリティレジーム別の分析"""
        rolling_vol = pd.Series(pnls).rolling(50).std()
        # NaNを除外した有効なボラティリティデータを使用
        valid_vol = rolling_vol.dropna()
        vol_median = valid_vol.median()

        high_vol_mask = valid_vol > vol_median
        low_vol_mask = valid_vol <= vol_median

        # 元のpnls配列から対応する期間のデータを取得（rolling windowのオフセットを考慮）
        high_vol_returns = pnls[50:][
            high_vol_mask.values
        ]  # 最初の50期間をスキップし、有効なマスクを適用
        low_vol_returns = pnls[50:][low_vol_mask.values]

        return {
            "high_volatility_performance": self._calculate_performance_metrics(
                high_vol_returns
            ),
            "low_volatility_performance": self._calculate_performance_metrics(
                low_vol_returns
            ),
            "volatility_regime_consistency": self._calculate_regime_consistency(
                self._calculate_performance_metrics(high_vol_returns),
                self._calculate_performance_metrics(low_vol_returns),
            ),
        }

    def _analyze_by_trend_regimes(self, pnls: np.ndarray) -> Dict[str, Any]:
        """トレンドレジーム別の分析"""
        # 移動平均でトレンドを判定
        ma_short = pd.Series(pnls).rolling(20).mean()
        ma_long = pd.Series(pnls).rolling(50).mean()

        # NaNを除外した有効な期間のみを使用
        valid_periods = ma_long.notna()
        uptrend = (ma_short > ma_long) & valid_periods
        downtrend = (ma_short < ma_long) & valid_periods
        sideways = (~(ma_short > ma_long) & ~(ma_short < ma_long)) & valid_periods

        uptrend_returns = pnls[uptrend.values]
        downtrend_returns = pnls[downtrend.values]
        sideways_returns = pnls[sideways.values]

        return {
            "uptrend_performance": self._calculate_performance_metrics(uptrend_returns),
            "downtrend_performance": self._calculate_performance_metrics(
                downtrend_returns
            ),
            "sideways_performance": self._calculate_performance_metrics(
                sideways_returns
            ),
            "trend_regime_consistency": self._calculate_regime_consistency(
                self._calculate_performance_metrics(uptrend_returns),
                self._calculate_performance_metrics(downtrend_returns),
                self._calculate_performance_metrics(sideways_returns),
            ),
        }

    def _analyze_by_drawdown_periods(self, pnls: np.ndarray) -> Dict[str, Any]:
        """ドローダウン期間別の分析"""
        cumulative = np.cumprod(1 + pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak

        # ドローダウン期間を特定
        in_drawdown = drawdown < 0
        drawdown_returns = pnls[in_drawdown]
        recovery_returns = pnls[~in_drawdown]

        return {
            "drawdown_performance": self._calculate_performance_metrics(
                drawdown_returns
            ),
            "recovery_performance": self._calculate_performance_metrics(
                recovery_returns
            ),
            "drawdown_recovery_consistency": self._calculate_regime_consistency(
                self._calculate_performance_metrics(drawdown_returns),
                self._calculate_performance_metrics(recovery_returns),
            ),
        }

    def _analyze_seasonal_performance(self, pnls: np.ndarray) -> Dict[str, Any]:
        """季節性パフォーマンス分析"""
        if "timestamps" not in self.data:
            return {"error": "No timestamp data available"}

        timestamps = pd.to_datetime(self.data["timestamps"])
        seasonal_returns = {}

        # 時間帯別分析
        hourly_returns = {}
        for hour in range(24):
            hour_mask = timestamps.dt.hour == hour
            if hour_mask.sum() > 0:
                hour_pnls = pnls[hour_mask]
                hourly_returns[hour] = self._calculate_performance_metrics(hour_pnls)

        # 曜日別分析
        weekday_returns = {}
        for day in range(7):
            day_mask = timestamps.dt.weekday == day
            if day_mask.sum() > 0:
                day_pnls = pnls[day_mask]
                weekday_returns[day] = self._calculate_performance_metrics(day_pnls)

        # 月別分析
        monthly_returns = {}
        for month in range(1, 13):
            month_mask = timestamps.dt.month == month
            if month_mask.sum() > 0:
                month_pnls = pnls[month_mask]
                monthly_returns[month] = self._calculate_performance_metrics(month_pnls)

        return {
            "hourly_performance": hourly_returns,
            "weekday_performance": weekday_returns,
            "monthly_performance": monthly_returns,
            "seasonal_consistency_score": self._calculate_seasonal_consistency(
                hourly_returns, weekday_returns, monthly_returns
            ),
        }

    def _calculate_regime_consistency(self, *regime_metrics) -> float:
        """レジーム間の一貫性を計算"""
        if len(regime_metrics) < 2:
            return 1.0

        sharpe_ratios = [m.get("sharpe_ratio", 0) for m in regime_metrics if m]
        win_rates = [m.get("win_rate", 0) for m in regime_metrics if m]

        if len(sharpe_ratios) < 2 or len(win_rates) < 2:
            return 0.0

        # シャープレシオと勝率の変動係数を計算
        sharpe_cv = _calculate_coefficient_of_variation(sharpe_ratios)
        win_rate_cv = _calculate_coefficient_of_variation(win_rates)

        # 一貫性スコア（低い変動係数 = 高い一貫性）
        consistency_score = 1 / (1 + sharpe_cv + win_rate_cv)
        return consistency_score

    def _calculate_seasonal_consistency(
        self, hourly: Dict, weekday: Dict, monthly: Dict
    ) -> float:
        """季節性の一貫性を計算"""
        seasonal_scores = []

        # 時間帯別一貫性
        if hourly:
            hourly_sharpes = [m.get("sharpe_ratio", 0) for m in hourly.values() if m]
            if len(hourly_sharpes) > 1:
                seasonal_scores.append(
                    1 / (1 + _calculate_coefficient_of_variation(hourly_sharpes))
                )

        # 曜日別一貫性
        if weekday:
            weekday_sharpes = [m.get("sharpe_ratio", 0) for m in weekday.values() if m]
            if len(weekday_sharpes) > 1:
                seasonal_scores.append(
                    1 / (1 + _calculate_coefficient_of_variation(weekday_sharpes))
                )

        # 月別一貫性
        if monthly:
            monthly_sharpes = [m.get("sharpe_ratio", 0) for m in monthly.values() if m]
            if len(monthly_sharpes) > 1:
                seasonal_scores.append(
                    1 / (1 + _calculate_coefficient_of_variation(monthly_sharpes))
                )

        return np.mean(seasonal_scores) if seasonal_scores else 0.0

    def _calculate_robustness_score(self, robustness_metrics: Dict) -> float:
        """総合的なロバストネススコアを計算"""
        scores = []

        # レジーム間一貫性スコア
        vol_consistency = robustness_metrics.get("volatility_analysis", {}).get(
            "volatility_regime_consistency", 0
        )
        trend_consistency = robustness_metrics.get("trend_analysis", {}).get(
            "trend_regime_consistency", 0
        )
        drawdown_consistency = robustness_metrics.get("drawdown_analysis", {}).get(
            "drawdown_recovery_consistency", 0
        )
        seasonal_consistency = robustness_metrics.get("seasonal_analysis", {}).get(
            "seasonal_consistency_score", 0
        )

        scores.extend(
            [
                vol_consistency,
                trend_consistency,
                drawdown_consistency,
                seasonal_consistency,
            ]
        )

        # 全体パフォーマンススコア
        overall_perf = robustness_metrics.get("overall_performance", {})
        sharpe = overall_perf.get("sharpe_ratio", 0)
        win_rate = overall_perf.get("win_rate", 0)

        # シャープレシオと勝率に基づくパフォーマンススコア
        perf_score = min(max(sharpe / 2, 0), 1) * 0.6 + win_rate * 0.4
        scores.append(perf_score)

        return np.mean(scores) if scores else 0.0

    def _analyze_backtest_action_distribution(self) -> Dict[str, float]:
        """バックテスト時のアクション分布を分析"""
        # action_distributionが既に計算済みの場合はそれを使用
        if "action_distribution" in self.data:
            action_dist = self.data["action_distribution"]
            if isinstance(action_dist, dict):
                # 文字列キーを統一
                return {
                    "HOLD": action_dist.get("HOLD", action_dist.get("hold", 0)),
                    "BUY": action_dist.get("BUY", action_dist.get("buy", 0)),
                    "SELL": action_dist.get("SELL", action_dist.get("sell", 0)),
                }

        # actions配列がある場合は計算
        if "actions" not in self.data:
            return {}

        actions = np.array(self.data["actions"])
        total_actions = len(actions)

        if total_actions == 0:
            return {}

        hold_threshold = 0.1  # HOLDと判定する閾値

        hold_count = np.sum(np.abs(actions) < hold_threshold)
        buy_count = np.sum(actions > hold_threshold)
        sell_count = np.sum(actions < -hold_threshold)

        return {
            "HOLD": hold_count / total_actions,
            "BUY": buy_count / total_actions,
            "SELL": sell_count / total_actions,
        }

    def analyze_correlation_and_dependencies(self) -> CorrelationAnalysisResult:
        """相関分析と依存関係の分析"""
        if "portfolio_history" not in self.data or "price_history" not in self.data:
            return CorrelationAnalysisResult(
                price_portfolio_correlation=0.0,
                lag_correlations={},
                beta=0.0,
                action_price_relationships={},
            )

        portfolio_values = np.array(self.data["portfolio_history"])
        price_values = np.array(self.data["price_history"])

        min_length = min(len(portfolio_values), len(price_values))
        portfolio_values = portfolio_values[:min_length]
        price_values = price_values[:min_length]

        # 価格とポートフォリオの相関
        price_returns = np.diff(price_values) / price_values[:-1]
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]

        if len(price_returns) > 0 and len(portfolio_returns) > 0:
            correlation = np.corrcoef(price_returns, portfolio_returns)[0, 1]
        else:
            correlation = 0.0

        # ラグ相関分析（価格変化がポートフォリオに与える遅延効果）
        lag_correlations = {}
        for lag in range(1, min(11, len(price_returns))):
            lagged_price = price_returns[:-lag] if lag > 0 else price_returns
            lagged_portfolio = portfolio_returns[lag:] if lag > 0 else portfolio_returns
            if len(lagged_price) == len(lagged_portfolio) and len(lagged_price) > 0:
                lag_corr = np.corrcoef(lagged_price, lagged_portfolio)[0, 1]
                lag_correlations[f"lag_{lag}"] = lag_corr

        # ベータ計算（市場リスクに対する感応度）
        if np.std(price_returns) > 0:
            beta = np.cov(portfolio_returns, price_returns)[0, 1] / np.var(
                price_returns
            )
        else:
            beta = 0.0

        # アクションと価格変化の関係
        action_price_correlations = {}
        if "actions" in self.data:
            actions = np.array(self.data["actions"])
            min_length = min(len(actions), len(price_returns))
            actions = actions[:min_length]
            price_returns_trimmed = price_returns[:min_length]

            # HOLD, BUY, SELL別の価格変化
            hold_mask = np.abs(actions) < 0.1
            buy_mask = actions > 0.1
            sell_mask = actions < -0.1

            if np.sum(hold_mask) > 0:
                action_price_correlations["hold_price_change"] = np.mean(
                    price_returns_trimmed[hold_mask]
                )
            if np.sum(buy_mask) > 0:
                action_price_correlations["buy_price_change"] = np.mean(
                    price_returns_trimmed[buy_mask]
                )
            if np.sum(sell_mask) > 0:
                action_price_correlations["sell_price_change"] = np.mean(
                    price_returns_trimmed[sell_mask]
                )

        return {
            "price_portfolio_correlation": correlation,
            "lag_correlations": lag_correlations,
            "beta": beta,
            "action_price_relationships": action_price_correlations,
        }

    def analyze_transaction_cost_impact(self) -> Dict[str, Any]:
        """取引コストの影響分析"""
        if "actions" not in self.data or "price_history" not in self.data:
            return {}

        actions = np.array(self.data["actions"])
        prices = np.array(self.data["price_history"])

        # 設定から取引コストを取得（デフォルトは0.001%）
        transaction_cost = (
            self.training_data.get("configuration", {})
            .get("environment", {})
            .get("transaction_cost", 0.00001)
            if self.training_data
            else 0.00001
        )

        # 取引が発生したステップを特定
        trade_mask = (actions == ACTION_BUY) | (actions == ACTION_SELL)
        trade_indices = np.where(trade_mask)[0]

        # 取引コストの累積計算
        total_transaction_cost = len(trade_indices) * transaction_cost

        # 取引ごとのコスト影響
        trade_costs = []
        for idx in trade_indices:
            if idx < len(prices):
                position_value = prices[idx]  # 簡易的に価格をポジション価値とする
                cost = position_value * transaction_cost
                trade_costs.append(cost)

        trade_costs = np.array(trade_costs)

        # コスト対パフォーマンス比
        if "portfolio_history" in self.data:
            final_portfolio = self.data["portfolio_history"][-1]
            initial_portfolio = self.data["portfolio_history"][0]
            gross_return = final_portfolio - initial_portfolio
            cost_ratio = (
                total_transaction_cost / abs(gross_return)
                if gross_return != 0
                else float("inf")
            )
        else:
            cost_ratio = 0.0

        # 取引頻度別のコスト効率
        trades_per_step = len(trade_indices) / len(actions) if len(actions) > 0 else 0

        return {
            "total_transaction_cost": total_transaction_cost,
            "average_cost_per_trade": np.mean(trade_costs)
            if len(trade_costs) > 0
            else 0,
            "cost_to_return_ratio": cost_ratio,
            "trades_per_step": trades_per_step,
            "cost_efficiency_score": 1.0 / (1.0 + cost_ratio)
            if cost_ratio != float("inf")
            else 0.0,
        }

    def perform_stress_tests(self) -> Dict[str, Any]:
        """ストレステストの実行"""
        if "portfolio_history" not in self.data:
            return {}

        portfolio_values = np.array(self.data["portfolio_history"])

        stress_tests = {}

        # 1. 急激な価格下落シナリオ（-10%, -20%, -30%）
        for drop_pct in [0.1, 0.2, 0.3]:
            stressed_portfolio = portfolio_values * (1 - drop_pct)
            stressed_return = (
                stressed_portfolio[-1] - stressed_portfolio[0]
            ) / stressed_portfolio[0]
            stress_tests[f"price_drop_{int(drop_pct*100)}pct"] = {
                "stressed_return": stressed_return,
                "survival_probability": 1.0
                if stressed_portfolio[-1] > stressed_portfolio[0] * 0.5
                else 0.5,
            }

        # 2. 高ボラティリティシナリオ
        if len(portfolio_values) > 20:
            # ボラティリティを2倍に増幅
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            stressed_returns = returns * 2
            stressed_portfolio = np.cumprod(1 + stressed_returns) * portfolio_values[0]
            stressed_return = (
                stressed_portfolio[-1] - portfolio_values[0]
            ) / portfolio_values[0]
            stress_tests["high_volatility"] = {
                "stressed_return": stressed_return,
                "volatility_multiplier": 2.0,
            }

        # 3. ゼロリターンスパンの影響
        if "timestamps" in self.data:
            try:
                timestamps = pd.to_datetime(self.data["timestamps"])
                # 週末のゼロリターン期間をシミュレート
                weekend_mask = timestamps.dt.weekday >= 5
                if np.sum(weekend_mask) > 0:
                    stressed_returns = np.diff(portfolio_values) / portfolio_values[:-1]
                    stressed_returns[weekend_mask[:-1]] = 0  # 週末のリターンをゼロに
                    stressed_portfolio = (
                        np.cumprod(1 + stressed_returns) * portfolio_values[0]
                    )
                    stressed_return = (
                        stressed_portfolio[-1] - portfolio_values[0]
                    ) / portfolio_values[0]
                    stress_tests["weekend_zero_returns"] = {
                        "stressed_return": stressed_return,
                        "affected_periods": np.sum(weekend_mask),
                    }
            except (ValueError, AttributeError):
                # timestampsが日時形式でない場合、週末テストをスキップ
                pass

        # 4. 取引コスト増大シナリオ
        transaction_impact = self.analyze_transaction_cost_impact()
        if transaction_impact:
            # 取引コストを5倍に
            increased_cost = transaction_impact.get("total_transaction_cost", 0) * 5
            if "portfolio_history" in self.data:
                final_value = self.data["portfolio_history"][-1]
                stressed_return = (
                    final_value - increased_cost - self.data["portfolio_history"][0]
                ) / self.data["portfolio_history"][0]
                stress_tests["increased_transaction_costs"] = {
                    "stressed_return": stressed_return,
                    "cost_multiplier": 5.0,
                }

        return stress_tests

    def analyze_walk_forward_efficiency(self) -> Dict[str, Any]:
        """ウォークフォワード分析の効率性評価"""
        if "portfolio_history" not in self.data:
            return {}

        portfolio_values = np.array(self.data["portfolio_history"])

        # 移動窓分析（簡易的なウォークフォワードシミュレーション）
        window_sizes = [100, 200, 500]
        walk_forward_metrics = {}

        for window_size in window_sizes:
            if len(portfolio_values) < window_size * 2:
                continue

            window_returns = []
            for i in range(window_size, len(portfolio_values), window_size // 2):
                window_data = portfolio_values[i - window_size : i]
                if len(window_data) >= window_size // 2:
                    window_return = (window_data[-1] - window_data[0]) / window_data[0]
                    window_returns.append(window_return)

            if window_returns:
                window_returns = np.array(window_returns)
                walk_forward_metrics[f"window_{window_size}"] = {
                    "mean_return": np.mean(window_returns),
                    "volatility": np.std(window_returns),
                    "sharpe_ratio": sharpe_ratio(window_returns),
                    "consistency_score": 1.0
                    / (1.0 + _calculate_coefficient_of_variation(window_returns)),
                }

        # アダプティブ能力の評価（後半のパフォーマンス vs 前半）
        if len(portfolio_values) > 10:
            midpoint = len(portfolio_values) // 2
            first_half = portfolio_values[:midpoint]
            second_half = portfolio_values[midpoint:]

            first_half_return = (first_half[-1] - first_half[0]) / first_half[0]
            second_half_return = (second_half[-1] - second_half[0]) / second_half[0]

            adaptation_ratio = (
                second_half_return / first_half_return if first_half_return != 0 else 0
            )

            walk_forward_metrics["adaptation_analysis"] = {
                "first_half_return": first_half_return,
                "second_half_return": second_half_return,
                "adaptation_ratio": adaptation_ratio,
                "learning_efficiency": max(
                    0, adaptation_ratio
                ),  # 学習効率：改善した場合のみ正
            }

        return walk_forward_metrics

    def analyze_microstructure_effects(self) -> Dict[str, Any]:
        """市場マイクロストラクチャーの影響分析"""
        if "price_history" not in self.data or "actions" not in self.data:
            return {}

        prices = np.array(self.data["price_history"])
        actions = np.array(self.data["actions"])

        microstructure_analysis = {}

        # 1. 価格インパクト分析（取引後の価格変化）
        trade_mask = (actions == 1) | (actions == 2)
        trade_indices = np.where(trade_mask)[0]

        price_impacts = []
        for idx in trade_indices:
            if idx + 5 < len(prices):  # 取引後5ステップの価格変化を分析
                pre_trade_price = prices[idx]
                post_trade_prices = prices[idx + 1 : idx + 6]
                avg_post_price = np.mean(post_trade_prices)
                impact = (avg_post_price - pre_trade_price) / pre_trade_price
                price_impacts.append(impact)

        if price_impacts:
            microstructure_analysis["price_impact"] = {
                "average_impact": np.mean(price_impacts),
                "impact_volatility": np.std(price_impacts),
                "adverse_selection_cost": abs(
                    np.mean(price_impacts)
                ),  # 逆選択コストの推定
            }

        # 2. 市場の深さ（liquidity）の影響
        # 価格の変動性で市場の深さを推定
        if len(prices) > 10:
            price_volatility = np.std(np.diff(prices) / prices[:-1])
            microstructure_analysis["market_depth"] = {
                "price_volatility": price_volatility,
                "liquidity_proxy": 1.0
                / (1.0 + price_volatility),  # ボラティリティが高いほどliquidityが低い
                "trading_efficiency": 1.0
                / (1.0 + price_volatility * len(trade_indices)),  # 取引頻度を考慮
            }

        # 3. スプレッドとスリッページの推定
        if len(prices) > 1:
            # Rollのモデルを使用してスプレッドを推定: S = 2 * sqrt(-cov(ΔP_t, ΔP_{t-1}))
            price_changes = np.diff(prices)
            if len(price_changes) > 1:
                # 価格変化の自己共分散を計算
                cov_delta_p = np.cov(price_changes[:-1], price_changes[1:])[0, 1]
                if cov_delta_p < 0:
                    spread_estimate = 2 * np.sqrt(-cov_delta_p)
                else:
                    # 共分散が正の場合は代替推定（絶対変化の平均を使用）
                    spread_estimate = np.mean(np.abs(price_changes)) * 0.1
            else:
                spread_estimate = (
                    np.mean(np.abs(price_changes)) * 0.1
                )  # データが少ない場合のフォールバック

            microstructure_analysis["spread_analysis"] = {
                "estimated_spread": spread_estimate,
                "spread_to_price_ratio": spread_estimate / np.mean(prices),
                "slippage_risk": spread_estimate
                * len(trade_indices),  # 総スリッページリスク
            }

        # 4. 市場参加者の行動パターン
        if len(actions) > 10:
            # アクションの自己相関（慣性効果）
            action_autocorr = np.corrcoef(actions[:-1], actions[1:])[0, 1]
            microstructure_analysis["behavioral_patterns"] = {
                "action_autocorrelation": action_autocorr,
                "momentum_effect": action_autocorr,  # 慣性の強さ
                "mean_reversion_tendency": 1 - abs(action_autocorr),  # 平均回帰の傾向
            }

        return microstructure_analysis

    def create_market_condition_balanced_dataset(
        self, output_path: str = "data/btc_balanced_dataset.csv"
    ):
        """市場条件バランスの取れたデータセットを作成"""
        try:
            # BTCDataAugmentorを活用して多様な市場条件を追加
            from ztb.data.btc_data_augmentation import BTCDataAugmentor

            augmentor = BTCDataAugmentor("data/btc_jpy_real_dataset.csv")

            # 既存データのバイアス分析
            bias_analysis = augmentor.analyze_data_bias()
            print("=== 既存データバイアス分析 ===")
            for key, value in bias_analysis.items():
                print(f"{key}: {value}")

            # 多様な市場条件を追加（特にSELLバイアス対策）
            balanced_data = augmentor.add_diverse_market_conditions(
                target_samples=50000
            )

            # 拡張データを保存
            augmentor.save_augmented_data(balanced_data, output_path)

            print(f"市場条件バランスデータセット作成完了: {output_path}")
            return balanced_data

        except Exception as e:
            print(f"データセット作成エラー: {e}")
            return None


def main():
    """メイン関数 - コマンドラインからバックテスト分析を実行"""
    import argparse

    parser = argparse.ArgumentParser(description="バックテスト分析ツール")
    parser.add_argument(
        "--results-path",
        type=str,
        required=True,
        help="バックテスト結果JSONファイルのパス",
    )
    parser.add_argument(
        "--training-report",
        type=str,
        default=None,
        help="トレーニングレポートJSONファイルのパス",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="分析結果出力ディレクトリ",
    )
    parser.add_argument(
        "--enhanced-stats", action="store_true", help="改善された統計機能を有効化"
    )

    args = parser.parse_args()

    if args.unified:
        # 統一されたバックテスト結果の解析
        analyzer = BacktestAnalyzer(results_path=args.results_path)
        results = analyzer.analyze()

        summary = analyzer.data
        print(f"=== Unified Backtest Analysis: {args.results_path} ===")
        print(f'Mode: {summary["mode"]}')
        print(f'Episodes: {summary["n_episodes"]}')
        print(f'Periods: {summary["n_periods"]}')
        print(f'Signal Guidance: {summary["enable_signal_guidance"]}')
        print()

        episodes = summary['results']
        final_balances = [ep['final_balance'] for ep in episodes]
        print(f'Final Portfolio Values: {final_balances}')
        print(f'Average Final Balance: {np.mean(final_balances):.2f}')
        print(f'Std Final Balance: {np.std(final_balances):.2f}')
        print(f'Average Return: {summary["avg_return_pct"]:.2f}%')
        print(f'Std Return: {summary["std_return_pct"]:.2f}%')
        print(f'Win Rate: {summary["win_rate"]:.1f}%')
        print(f'Sharpe Ratio: {summary["sharpe_ratio"]:.4f}')
        print()

        if 'signal_guidance_analysis' in results:
            sga = results['signal_guidance_analysis']
            print('=== SIGNAL_GUIDANCE Analysis ===')
            print(f'Number of signals: {sga["number_of_signals"]}')
            print(f'Average guidance score: {sga["average_score"]:.2f}')
            print(f'Score std: {sga["score_std"]:.2f}')
            print(f'Min score: {sga["min_score"]:.2f}')
            print(f'Max score: {sga["max_score"]:.2f}')
            print()

            print('=== Action Distribution ===')
            print(f'Original actions - Hold: {sga["original_hold"]}, Buy: {sga["original_buy"]}, Sell: {sga["original_sell"]}')
            print(f'Guidance actions - Hold: {sga["guidance_hold"]}, Buy: {sga["guidance_buy"]}, Sell: {sga["guidance_sell"]}')

            differences = sga["differences"]
            total = sga["total_actions"]
            print(f'Actions where guidance differed from original: {differences}/{total} ({sga["difference_pct"]:.1f}%)')
            print()

            print(f'Correlation between SIGNAL_GUIDANCE score and portfolio value: {sga["correlation"]:.3f}')
            print()

        return 0

    if not args.results_path:
        parser.error("--results-path is required unless --unified is specified")

    try:
        # 分析器の初期化
        analyzer = BacktestAnalyzer(
            results_path=args.results_path, training_report_path=args.training_report
        )

        print("=== バックテスト分析開始 ===")
        print(f"結果ファイル: {args.results_path}")
        if args.training_report:
            print(f"トレーニングレポート: {args.training_report}")

        # 基本分析実行
        results = analyzer.analyze()

        # 改善された統計機能（オプション）
        if args.enhanced_stats:
            print("\n=== 改善された統計分析実行 ===")

            # テンポラルパターン分析
            temporal_results = analyzer.analyze_temporal_patterns()
            results["temporal_analysis"] = temporal_results

            # 市場条件分析
            market_results = analyzer.analyze_market_conditions()
            results["market_condition_analysis"] = market_results

            # 取引頻度分析
            frequency_results = analyzer.analyze_trading_frequency()
            results["trading_frequency_analysis"] = frequency_results

            # ロバストネス分析
            robustness_results = analyzer.analyze_robustness()
            results["robustness_analysis"] = robustness_results

            # 相関・依存関係分析
            correlation_results = analyzer.analyze_correlation_and_dependencies()
            results["correlation_analysis"] = correlation_results

            # 取引コスト影響分析
            cost_results = analyzer.analyze_transaction_cost_impact()
            results["transaction_cost_analysis"] = cost_results

            # ウォークフォワード効率分析
            walk_forward_results = analyzer.analyze_walk_forward_efficiency()
            results["walk_forward_analysis"] = walk_forward_results

            # マイクロストラクチャ効果分析
            microstructure_results = analyzer.analyze_microstructure_effects()
            results["microstructure_analysis"] = microstructure_results

        # 出力ディレクトリの作成
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # タイムスタンプ付きファイル名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"backtest_analysis_{timestamp}.json"

        # 結果の保存
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("\n=== 分析完了 ===")
        print(f"結果ファイル: {output_file}")

        # 主要メトリクスの表示
        if "summary" in results:
            summary = results["summary"]
            print(
                """
=== 主要メトリクス ==="""
            )
            print(f"総リターン: {summary.get('total_return', 'N/A')}")
            print(f"年間リターン: {summary.get('annual_return', 'N/A')}")
            print(f"シャープレシオ: {summary.get('sharpe_ratio', 'N/A')}")
            print(f"最大ドローダウン: {summary.get('max_drawdown', 'N/A')}")
            print(f"勝率: {summary.get('win_rate', 'N/A')}")

    except Exception as e:
        print("エラーが発生しました。詳細は下記をご確認ください。")
        print(f"分析実行エラー: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
