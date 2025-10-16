#!/usr/bin/env python3
"""
Generic Backtest Analysis Tool
汎用バックテスト分析ツール

This tool provides comprehensive analysis of trading backtest results,
including risk metrics, temporal analysis, and market condition analysis.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class BacktestAnalyzer:
    """汎用バックテスト分析クラス"""

    def __init__(self, results_path: str):
        """Initialize analyzer with backtest results file."""
        self.results_path = Path(results_path)
        self.data = self._load_data()
        self._validate_data()

    def _load_data(self) -> Dict[str, Any]:
        """Load backtest results from JSON file."""
        try:
            with open(self.results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {self.results_path}: {e}")

    def _validate_data(self):
        """Validate that required data fields are present."""
        required_fields = ['total_steps', 'initial_portfolio', 'final_portfolio']
        missing_fields = [field for field in required_fields if field not in self.data]
        if missing_fields:
            raise ValueError(f"Missing required fields in results: {missing_fields}")

    def calculate_risk_metrics(self) -> Dict[str, float]:
        """リスク指標を計算"""
        if 'portfolio_history' not in self.data:
            return {}

        portfolio_values = np.array(self.data['portfolio_history'])

        # 総リターン
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

        # 日次リターン（分足データを日次に変換）
        if 'timestamps' in self.data:
            timestamps = pd.to_datetime(self.data['timestamps'])
            daily_returns = []
            current_day = None
            day_start_value = None

            for ts, value in zip(timestamps, portfolio_values):
                day = ts.date()
                if current_day != day:
                    if day_start_value is not None:
                        daily_return = (portfolio_values[i-1] - day_start_value) / day_start_value
                        daily_returns.append(daily_return)
                    current_day = day
                    day_start_value = value
                i = len(daily_returns) + 1

            if day_start_value is not None and len(portfolio_values) > 0:
                daily_return = (portfolio_values[-1] - day_start_value) / day_start_value
                daily_returns.append(daily_return)

            daily_returns = np.array(daily_returns)
        else:
            # タイムスタンプがない場合はステップごとのリターンを使用
            step_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            # 適当に日次にグループ化（仮定: 1日=1440分）
            steps_per_day = 1440
            daily_returns = []
            for i in range(0, len(step_returns), steps_per_day):
                day_return = np.prod(1 + step_returns[i:i+steps_per_day]) - 1
                daily_returns.append(day_return)
            daily_returns = np.array(daily_returns)

        if len(daily_returns) == 0:
            return {
                'total_return': total_return,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0
            }

        # シャープレシオ（無リスク金利を0%として）
        risk_free_rate = 0.0
        excess_returns = daily_returns - risk_free_rate / 252  # 日次無リスク金利
        if np.std(excess_returns) > 0:
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # 最大ドローダウン
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)

        # ボラティリティ（年率化）
        volatility = np.std(daily_returns) * np.sqrt(252)

        # ソルティーノレシオ
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino_ratio = np.mean(daily_returns) / np.std(downside_returns) * np.sqrt(252)
        else:
            sortino_ratio = 0.0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sortino_ratio': sortino_ratio,
            'win_rate': self.data.get('win_rate', 0) / 100.0,
            'profit_factor': self._calculate_profit_factor()
        }

    def _calculate_profit_factor(self) -> float:
        """プロフィットファクターを計算"""
        if 'trade_pnls' not in self.data:
            return 0.0

        pnls = np.array(self.data['trade_pnls'])
        winning_trades = pnls[pnls > 0]
        losing_trades = pnls[pnls < 0]

        if len(winning_trades) == 0:
            return 0.0
        if len(losing_trades) == 0:
            return float('inf')

        gross_profit = np.sum(winning_trades)
        gross_loss = abs(np.sum(losing_trades))

        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """時間帯別の分析"""
        if 'timestamps' not in self.data or 'portfolio_history' not in self.data:
            return {}

        # timestampsが既にDatetimeIndexの場合はそのまま使用、そうでなければ変換
        if isinstance(self.data['timestamps'], pd.DatetimeIndex):
            timestamps = self.data['timestamps']
        else:
            timestamps = pd.to_datetime(self.data['timestamps'])
        portfolio_values = np.array(self.data['portfolio_history'])

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
                weekday_return = (weekday_values[-1] - weekday_values[0]) / weekday_values[0]
                weekday_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][weekday]
                weekday_returns[weekday_name] = weekday_return

        return {
            'hourly_returns': hourly_returns,
            'weekday_returns': weekday_returns
        }

    def analyze_market_conditions(self) -> Dict[str, Any]:
        """市場環境別の分析"""
        if 'price_history' not in self.data or 'portfolio_history' not in self.data:
            return {}

        prices = np.array(self.data['price_history'])
        portfolio_values = np.array(self.data['portfolio_history'])

        # 価格トレンドの計算（移動平均）
        if len(prices) >= 20:
            short_ma = pd.Series(prices).rolling(10).mean().values
            long_ma = pd.Series(prices).rolling(50).mean().values

            # 市場環境の分類
            uptrend_mask = short_ma > long_ma
            downtrend_mask = short_ma < long_ma
            sideways_mask = ~(uptrend_mask | downtrend_mask)

            conditions = {
                'uptrend': {'mask': uptrend_mask, 'name': '上昇トレンド'},
                'downtrend': {'mask': downtrend_mask, 'name': '下降トレンド'},
                'sideways': {'mask': sideways_mask, 'name': '横ばい'}
            }

            results = {}
            for condition_key, condition_data in conditions.items():
                mask = condition_data['mask']
                if mask.sum() > 0:
                    condition_portfolio = portfolio_values[mask]
                    condition_return = (condition_portfolio[-1] - condition_portfolio[0]) / condition_portfolio[0]
                    results[condition_key] = {
                        'return': condition_return,
                        'periods': int(mask.sum()),
                        'name': condition_data['name']
                    }

            return results

        return {}

    def analyze_trading_frequency(self) -> Dict[str, Any]:
        """取引頻度分析"""
        if 'action_history' not in self.data:
            return {}

        actions = np.array(self.data['action_history'])
        total_steps = len(actions)

        # アクション分布
        unique, counts = np.unique(actions, return_counts=True)
        action_distribution = dict(zip(unique.astype(int), counts))

        # 取引頻度（BUY/SELLの割合）
        trade_actions = actions[(actions == 1) | (actions == 2)]  # BUY or SELL
        trade_frequency = len(trade_actions) / total_steps if total_steps > 0 else 0

        # 平均取引間隔
        trade_indices = np.where((actions == 1) | (actions == 2))[0]
        if len(trade_indices) > 1:
            intervals = np.diff(trade_indices)
            avg_trade_interval = np.mean(intervals)
            min_trade_interval = np.min(intervals)
            max_trade_interval = np.max(intervals)
        else:
            avg_trade_interval = min_trade_interval = max_trade_interval = 0

        return {
            'action_distribution': action_distribution,
            'trade_frequency': trade_frequency,
            'avg_trade_interval': avg_trade_interval,
            'min_trade_interval': min_trade_interval,
            'max_trade_interval': max_trade_interval,
            'total_trades': len(trade_actions)
        }

    def generate_comprehensive_report(self) -> str:
        """包括的な分析レポートを生成"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("汎用バックテスト分析レポート")
        report_lines.append("=" * 80)
        report_lines.append(f"分析対象ファイル: {self.results_path.name}")
        report_lines.append(f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # 基本情報
        report_lines.append("=== 基本情報 ===")
        report_lines.append(f"総ステップ数: {self.data.get('total_steps', 'N/A')}")
        report_lines.append(f"初期ポートフォリオ: {self.data.get('initial_portfolio', 0):,.0f} JPY")
        report_lines.append(f"最終ポートフォリオ: {self.data.get('final_portfolio', 0):,.0f} JPY")
        total_return_pct = self.data.get('total_return_pct', 0)
        report_lines.append(f"総リターン: {total_return_pct:.2f}%")
        report_lines.append(f"総取引数: {self.data.get('total_trades', 0)}")
        report_lines.append(f"勝率: {self.data.get('win_rate', 0):.1f}%")
        report_lines.append("")

        # リスク指標
        risk_metrics = self.calculate_risk_metrics()
        if risk_metrics:
            report_lines.append("=== リスク指標 ===")
            report_lines.append(f"シャープレシオ: {risk_metrics.get('sharpe_ratio', 0):.3f}")
            report_lines.append(f"ソルティーノレシオ: {risk_metrics.get('sortino_ratio', 0):.3f}")
            report_lines.append(f"最大ドローダウン: {risk_metrics.get('max_drawdown', 0):.2%}")
            report_lines.append(f"ボラティリティ: {risk_metrics.get('volatility', 0):.2%}")
            report_lines.append(f"プロフィットファクター: {risk_metrics.get('profit_factor', 0):.3f}")
            report_lines.append("")

        # アクション分析
        if 'action_distribution' in self.data:
            report_lines.append("=== アクション分布 ===")
            actions = self.data['action_distribution']
            total_actions = sum(actions.values())
            action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            for action_id, count in actions.items():
                pct = count / total_actions * 100 if total_actions > 0 else 0
                action_name = action_names.get(int(action_id), f'UNKNOWN({action_id})')
                report_lines.append(f"  {action_name}: {count}回 ({pct:.1f}%)")
            report_lines.append("")

        # 取引頻度分析
        trading_freq = self.analyze_trading_frequency()
        if trading_freq:
            report_lines.append("=== 取引頻度分析 ===")
            report_lines.append(f"取引頻度: {trading_freq.get('trade_frequency', 0):.3f} (取引/ステップ)")
            report_lines.append(f"総取引数: {trading_freq.get('total_trades', 0)}")
            if trading_freq.get('avg_trade_interval', 0) > 0:
                report_lines.append(f"平均取引間隔: {trading_freq['avg_trade_interval']:.1f}ステップ")
                report_lines.append(f"最小取引間隔: {trading_freq['min_trade_interval']}ステップ")
                report_lines.append(f"最大取引間隔: {trading_freq['max_trade_interval']}ステップ")
            report_lines.append("")

        # 時間帯別分析
        temporal = self.analyze_temporal_patterns()
        if temporal and temporal.get('hourly_returns'):
            report_lines.append("=== 時間帯別リターン (上位/下位3件) ===")
            hourly = temporal['hourly_returns']
            sorted_hourly = sorted(hourly.items(), key=lambda x: x[1], reverse=True)

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
        if 'continuous_action_stats' in self.data and self.data['continuous_action_stats']:
            stats = self.data['continuous_action_stats']
            if 'action_streaks' in stats:
                streaks = stats['action_streaks']
                report_lines.append("=== 連続アクション分析 ===")
                report_lines.append(f"BUYストリーク:")
                report_lines.append(f"  最大連続: {streaks.get('max_buy_streak', 0)}回")
                report_lines.append(f"  平均連続: {streaks.get('avg_buy_streak', 0):.1f}回")
                report_lines.append(f"SELLストリーク:")
                report_lines.append(f"  最大連続: {streaks.get('max_sell_streak', 0)}回")
                report_lines.append(f"  平均連続: {streaks.get('avg_sell_streak', 0):.1f}回")
                report_lines.append("")

        report_lines.append("=" * 80)
        return "\n".join(report_lines)


def main():
    """メイン関数"""
    if len(sys.argv) != 2:
        print("Usage: python analyze_backtest.py <results_json_path>")
        sys.exit(1)

    results_path = sys.argv[1]

    try:
        analyzer = BacktestAnalyzer(results_path)
        report = analyzer.generate_comprehensive_report()
        print(report)

    except Exception as e:
        print(f"Error analyzing backtest results: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()