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

import numpy as np
import pandas as pd

from ztb.core.base import BaseAnalyzer
from ztb.data.btc_data_augmentation import BTCBiasDetector
from ztb.metrics.metrics import (
    calculate_all_metrics,
    classify_market_regime,
    multi_market_backtest_analysis,
    seasonality_analysis,
)
from ztb.trading.constants import TRADING_DAYS_PER_YEAR  # = 252
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.utils.logging_utils import get_logger
from ztb.utils.performance_utils import PerformanceMonitor

logger = get_logger(__name__)


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
        if missing_fields:
            raise ValueError(f"Missing required fields in results: {missing_fields}")

    def analyze(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform comprehensive backtest analysis."""
        if data:
            self.data = data
            self._validate_data()

        results = {
            "risk_metrics": self.calculate_risk_metrics(),
            "temporal_patterns": self.analyze_temporal_patterns(),
            "market_conditions": self.analyze_market_conditions(),
            "trading_frequency": self.analyze_trading_frequency(),
        }

        self.results = results
        return results

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

            for ts, value in zip(timestamps, portfolio_values):
                day = ts.date()
                if current_day != day:
                    if day_start_value is not None:
                        daily_return = (
                            portfolio_values[i - 1] - day_start_value
                        ) / day_start_value
                        daily_returns.append(daily_return)
                    current_day = day
                    day_start_value = value
                i = len(daily_returns) + 1

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

        # シャープレシオ（無リスク金利を0%として）
        risk_free_rate = 0.0
        excess_returns = (
            daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
        )  # 日次無リスク金利
        if np.std(excess_returns) > 0:
            sharpe_ratio = (
                np.mean(excess_returns)
                / np.std(excess_returns)
                * np.sqrt(TRADING_DAYS_PER_YEAR)
            )
        else:
            sharpe_ratio = 0.0

        # 最大ドローダウン
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)

        # ボラティリティ（年率化）
        volatility = np.std(daily_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)

        # ソルティーノレシオ
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino_ratio = (
                np.mean(daily_returns)
                / np.std(downside_returns)
                * np.sqrt(TRADING_DAYS_PER_YEAR)
            )
        else:
            sortino_ratio = 0.0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "sortino_ratio": sortino_ratio,
            "win_rate": self.data.get("win_rate", 0) / 100.0,
            "profit_factor": self._calculate_profit_factor(),
        }

    def _calculate_profit_factor(self) -> float:
        """プロフィットファクターを計算"""
        if "trade_pnls" not in self.data:
            return 0.0

        pnls = np.array(self.data["trade_pnls"])
        winning_trades = pnls[pnls > 0]
        losing_trades = pnls[pnls < 0]

        if len(winning_trades) == 0:
            return 0.0
        if len(losing_trades) == 0:
            return float("inf")

        gross_profit = np.sum(winning_trades)
        gross_loss = abs(np.sum(losing_trades))

        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    def analyze_temporal_patterns(self) -> Dict[str, Any]:
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

    def analyze_market_conditions(self) -> Dict[str, Any]:
        """市場環境別の分析"""
        if "price_history" not in self.data or "portfolio_history" not in self.data:
            return {}

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

        return {}

    def analyze_trading_frequency(self) -> Dict[str, Any]:
        """取引頻度分析"""
        if "actions" not in self.data:
            return {}

        actions = np.array(self.data["actions"])
        total_steps = len(actions)

        # アクション分布
        unique, counts = np.unique(actions, return_counts=True)
        action_distribution = dict(zip(unique.astype(int), counts))

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
            "action_distribution": action_distribution,
            "trade_frequency": trade_frequency,
            "avg_trade_interval": avg_trade_interval,
            "min_trade_interval": min_trade_interval,
            "max_trade_interval": max_trade_interval,
            "total_trades": len(trade_actions),
        }

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
                                count * 100
                                if isinstance(count, float) and count <= 1.0
                                else count / total_actions * 100
                            )
                            report_lines.append(f"  {action_name}: {count:.1%}")
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
                            f"平均取引間隔: {trading_freq['avg_trade_interval']:.1f}ステップ"
                        )
                        report_lines.append(
                            f"最小取引間隔: {trading_freq['min_trade_interval']}ステップ"
                        )
                        report_lines.append(
                            f"最大取引間隔: {trading_freq['max_trade_interval']}ステップ"
                        )
                    report_lines.append("")

                # 時間帯別分析
                temporal = self.analyze_temporal_patterns()
                if temporal and temporal.get("hourly_returns"):
                    report_lines.append("=== 時間帯別リターン (上位/下位3件) ===")
                    hourly = temporal["hourly_returns"]
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
                        report_lines.append(f"{data['name']}:")
                        report_lines.append(f"  リターン: {data['return']:.2%}")
                        report_lines.append(f"  期間数: {data['periods']}")
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
                                report_lines.append(
                                    f"  最適時間帯: {best_hour[0]}時 (シャープ {best_hour[1].get('sharpe_ratio', 0):.3f})"
                                )
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

    def analyze_robustness(self) -> Dict[str, Any]:
        """多様な市場条件下でのロバストネス分析"""
        with self.performance_monitor:
            robustness_metrics = {}

            if "pnls" not in self.data and "portfolio_history" not in self.data:
                return {"error": "No PnL or portfolio data available"}

            # PnLデータの準備
            if "pnls" in self.data:
                pnls = np.array(self.data["pnls"])
            elif "portfolio_history" in self.data:
                portfolio_values = np.array(self.data["portfolio_history"])
                pnls = np.diff(portfolio_values) / portfolio_values[:-1]
            else:
                return {"error": "Unable to calculate PnL data"}

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
            robustness_metrics["robustness_score"] = self._calculate_robustness_score(
                robustness_metrics
            )

            return robustness_metrics

    def _calculate_performance_metrics(self, pnls: np.ndarray) -> Dict[str, float]:
        """基本的なパフォーマンス指標を計算"""
        if len(pnls) == 0:
            return {}

        total_return = np.prod(1 + pnls) - 1
        volatility = np.std(pnls)
        sharpe_ratio = total_return / volatility if volatility > 0 else 0
        max_drawdown = self._calculate_max_drawdown_from_returns(pnls)
        win_rate = np.sum(pnls > 0) / len(pnls)

        return {
            "total_return": total_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": len(pnls),
        }

    def _calculate_max_drawdown_from_returns(self, returns: np.ndarray) -> float:
        """リターン配列から最大ドローダウンを計算"""
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return np.min(drawdown)

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
        sharpe_cv = (
            np.std(sharpe_ratios) / np.mean(sharpe_ratios)
            if np.mean(sharpe_ratios) != 0
            else 1
        )
        win_rate_cv = (
            np.std(win_rates) / np.mean(win_rates) if np.mean(win_rates) != 0 else 1
        )

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
                    1
                    / (1 + np.std(hourly_sharpes) / max(np.mean(hourly_sharpes), 0.01))
                )

        # 曜日別一貫性
        if weekday:
            weekday_sharpes = [m.get("sharpe_ratio", 0) for m in weekday.values() if m]
            if len(weekday_sharpes) > 1:
                seasonal_scores.append(
                    1
                    / (
                        1
                        + np.std(weekday_sharpes) / max(np.mean(weekday_sharpes), 0.01)
                    )
                )

        # 月別一貫性
        if monthly:
            monthly_sharpes = [m.get("sharpe_ratio", 0) for m in monthly.values() if m]
            if len(monthly_sharpes) > 1:
                seasonal_scores.append(
                    1
                    / (
                        1
                        + np.std(monthly_sharpes) / max(np.mean(monthly_sharpes), 0.01)
                    )
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

    def analyze_correlation_and_dependencies(self) -> Dict[str, Any]:
        """相関分析と依存関係の分析"""
        if "portfolio_history" not in self.data or "price_history" not in self.data:
            return {}

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
                    "sharpe_ratio": np.mean(window_returns) / np.std(window_returns)
                    if np.std(window_returns) > 0
                    else 0,
                    "consistency_score": 1.0
                    / (
                        1.0
                        + np.std(window_returns)
                        / max(abs(np.mean(window_returns)), 0.001)
                    ),
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

    def implement_adaptive_reward_system(self):
        """適応的報酬システムの実装"""
        print("=== 適応的報酬システム設計 ===")

        # 既存のRewardCalculatorを拡張
        reward_suggestions = {
            "dynamic_penalty_adjustment": {
                "description": "アクション分布に基づく動的ペナルティ調整",
                "implementation": "SELL比率 > 50%の場合、transaction_penaltyを段階的に増加",
                "expected_impact": "SELLバイアス67% → 均衡分布へ是正",
            },
            "market_regime_aware_rewards": {
                "description": "市場レジームに応じた報酬調整",
                "implementation": "上昇トレンド時はBUY優位、下降トレンド時はSELL優位",
                "expected_impact": "市場適応性向上、レジーム間一貫性改善",
            },
            "correlation_based_bonuses": {
                "description": "価格相関に基づくボーナス/ペナルティ",
                "implementation": "価格変動とポジション変化の相関が低い場合ペナルティ",
                "expected_impact": "市場連動性改善（β値0.017 → 適切な値へ）",
            },
        }

        for key, details in reward_suggestions.items():
            print(f"\n{key}:")
            print(f"  説明: {details['description']}")
            print(f"  実装: {details['implementation']}")
            print(f"  期待効果: {details['expected_impact']}")

    def enhance_feature_engineering(self):
        """特徴量エンジニアリングの強化"""
        print("=== 特徴量エンジニアリング強化案 ===")

        feature_improvements = {
            "price_momentum_features": {
                "description": "価格モメンタム特徴量の追加",
                "features": ["roc_1", "roc_5", "roc_20", "momentum_1d", "momentum_1w"],
                "expected_impact": "短期価格変動の捕捉改善",
            },
            "volatility_regime_features": {
                "description": "ボラティリティレジーム特徴量",
                "features": [
                    "volatility_ratio",
                    "volatility_trend",
                    "regime_stability",
                ],
                "expected_impact": "ボラティリティ適応性の向上",
            },
            "market_microstructure_features": {
                "description": "市場マイクロストラクチャー特徴量",
                "features": ["spread_estimate", "liquidity_proxy", "order_flow"],
                "expected_impact": "取引コストと流動性の考慮",
            },
            "correlation_aware_features": {
                "description": "相関意識型特徴量",
                "features": [
                    "price_position_corr",
                    "action_price_corr",
                    "regime_alignment",
                ],
                "expected_impact": "市場連動性の直接的改善",
            },
        }

        for key, details in feature_improvements.items():
            print(f"\n{key}:")
            print(f"  説明: {details['description']}")
            print(f"  特徴量: {', '.join(details['features'])}")
            print(f"  期待効果: {details['expected_impact']}")

    def implement_curriculum_learning_v2(self):
        """カリキュラム学習V2の実装"""
        print("=== カリキュラム学習V2設計 ===")

        curriculum_stages = {
            "data_bias_awareness": {
                "description": "データバイアス意識段階",
                "focus": "BTCBiasDetectorを活用したバイアス検出と修正",
                "duration": "初期1000ステップ",
                "reward_structure": "バイアス是正を優先",
            },
            "market_regime_adaptation": {
                "description": "市場レジーム適応段階",
                "focus": "多様な市場条件でのロバストネス獲得",
                "duration": "次の2000ステップ",
                "reward_structure": "レジーム間一貫性重視",
            },
            "correlation_optimization": {
                "description": "相関最適化段階",
                "focus": "価格連動性の最大化",
                "duration": "次の3000ステップ",
                "reward_structure": "β値と相関係数ベースの報酬",
            },
            "scalping_fine_tuning": {
                "description": "スキャルピング微調整段階",
                "focus": "低レイテンシー・高頻度取引の最適化",
                "duration": "最終ステップ",
                "reward_structure": "取引効率とコスト意識",
            },
        }

        for stage, details in curriculum_stages.items():
            print(f"\n{stage}:")
            print(f"  説明: {details['description']}")
            print(f"  重点: {details['focus']}")
            print(f"  期間: {details['duration']}")
            print(f"  報酬構造: {details['reward_structure']}")

    def create_comprehensive_validation_suite(self):
        """包括的検証スイート作成"""
        print("=== 包括的検証スイート ===")

        validation_tests = {
            "bias_detection_validation": {
                "description": "バイアス検出の妥当性検証",
                "method": "BTCBiasDetectorの結果を複数データセットで検証",
                "thresholds": "SELLバイアス < 40%, 時間周期バイアス検出済み",
            },
            "correlation_validation": {
                "description": "相関性の実質的検証",
                "method": "価格変動に対するポジション変化のラグ相関分析",
                "thresholds": "β値 > 0.1, 価格相関 > 0.05",
            },
            "robustness_validation": {
                "description": "ロバストネスの包括的検証",
                "method": "複数市場条件でのバックテスト",
                "thresholds": "ロバストネススコア > 0.5, レジーム間一貫性 > 0.3",
            },
            "adaptation_validation": {
                "description": "適応能力の検証",
                "method": "ウォークフォワード分析と学習曲線評価",
                "thresholds": "学習効率 > 0.1, 適応比率 > -1.0",
            },
        }

        for test, details in validation_tests.items():
            print(f"\n{test}:")
            print(f"  説明: {details['description']}")
            print(f"  方法: {details['method']}")
            print(f"  閾値: {details['thresholds']}")

    def generate_v425_improvement_plan(self):
        """v425改善計画の生成"""
        print("\n" + "=" * 80)
        print("🎯 v425改善計画 - 既存システム最大活用版")
        print("=" * 80)

        improvement_plan = {
            "phase_1_data_foundation": {
                "title": "Phase 1: データ基盤強化（1-2日）",
                "actions": [
                    "BTCDataAugmentorで市場条件バランスデータセット作成",
                    "BTCBiasDetectorでバイアス分析の自動化",
                    "既存データに多様なレジームデータを追加（5万サンプル）",
                ],
                "expected_outcome": "時間周期バイアス解消、レジーム分布均衡化",
            },
            "phase_2_feature_engineering": {
                "title": "Phase 2: 特徴量エンジニアリング強化（2-3日）",
                "actions": [
                    "相関意識型特徴量の追加（price_position_corr, action_price_corr）",
                    "市場マイクロストラクチャー特徴量の実装",
                    "ボラティリティレジーム特徴量の統合",
                ],
                "expected_outcome": "価格相関0.019 → 0.1以上、β値適切化",
            },
            "phase_3_reward_system": {
                "title": "Phase 3: 適応的報酬システム（3-4日）",
                "actions": [
                    "RewardCalculator拡張：動的ペナルティ調整",
                    "市場レジーム対応報酬の実装",
                    "相関ベースボーナスの追加",
                ],
                "expected_outcome": "SELLバイアス67% → 均衡分布、ロバストネススコア向上",
            },
            "phase_4_curriculum_v2": {
                "title": "Phase 4: カリキュラム学習V2（2-3日）",
                "actions": [
                    "4段階カリキュラムの実装（バイアス意識→レジーム適応→相関最適化→スキャルピング）",
                    "段階的難易度上昇の自動化",
                    "進捗評価と自動遷移ロジック",
                ],
                "expected_outcome": "学習効率0.000 → 0.2以上、適応比率改善",
            },
            "phase_5_validation_integration": {
                "title": "Phase 5: 包括的検証統合（2-3日）",
                "actions": [
                    "analyze_backtest.pyの自動検証機能拡張",
                    "トレーニング中のリアルタイムバイアス監視",
                    "複数メトリクスでの早期停止判定",
                ],
                "expected_outcome": "問題の早期検知と是正、安定した学習プロセス",
            },
        }

        for phase, details in improvement_plan.items():
            print(f"\n{details['title']}")
            print("アクション:")
            for action in details["actions"]:
                print(f"  • {action}")
            print(f"期待成果: {details['expected_outcome']}")

        print("\n総工期: 10-15日")
        print(
            "既存活用率: 85%（BTCDataAugmentor, BTCBiasDetector, RewardCalculator, analyze_backtest.py）"
        )
        print("新規開発: 15%（拡張機能のみ）")
