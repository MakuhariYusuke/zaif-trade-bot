#!/usr/bin/env python3
"""
Phase 3 Enhanced vs Baseline Backtest Comparison Script

このスクリプトはPhase 3の強化されたリスク管理と統計的検証機能を
ベースライン（標準）バックテストと比較します。
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.metrics import sharpe_ratio

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Phase3ComparisonAnalyzer:
    """Phase 3比較分析クラス"""


    def run_baseline_backtest(
        self, data_path: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        ベースラインバックテストを実行（簡易版）

        Args:
            data_path: テストデータのパス
            config: バックテスト設定

        Returns:
            バックテスト結果
        """
        logger.info("🏃 Running baseline backtest (simplified)")

        try:
            # 簡易的なバックテスト実装
            import pandas as pd

            # データ読み込み
            data = pd.read_csv(data_path)
            if "close" not in data.columns and "open" in data.columns:
                data["close"] = data["open"]  # closeカラムがない場合、openを使用
            if "timestamp" not in data.columns:
                data["timestamp"] = pd.date_range(
                    start="2023-01-01", periods=len(data), freq="1H"
                )

            # 簡易的な取引シグナル生成（ランダム）
            np.random.seed(42)  # 再現性のため
            signals = np.random.choice([-1, 0, 1], size=len(data), p=[0.3, 0.4, 0.3])

            # 簡易バックテスト実行
            capital = config.get("initial_balance", 1000000)
            position = 0
            entry_price = 0
            trades = []
            equity_curve = [capital]

            for i, (idx, row) in enumerate(data.iterrows()):
                signal = signals[i]
                price = row["close"]

                # ポジション変更
                if signal != position:
                    if position != 0:  # ポジションクローズ
                        pnl = (
                            (price - entry_price) / entry_price
                            if position == 1
                            else (entry_price - price) / entry_price
                        )
                        capital *= 1 + pnl * position
                        trades.append(
                            {
                                "entry_price": entry_price,
                                "exit_price": price,
                                "pnl": pnl,
                                "type": "long" if position == 1 else "short",
                            }
                        )

                    if signal != 0:  # 新規ポジション
                        entry_price = price
                        position = signal

                equity_curve.append(capital)

            # メトリクス計算
            returns = pd.Series(equity_curve).pct_change().dropna()
            total_return = (
                capital - config.get("initial_balance", 1000000)
            ) / config.get("initial_balance", 1000000)
            sharpe = sharpe_ratio(returns.values) if len(returns) > 0 else 0

            # 最大ドローダウン計算
            peak = pd.Series(equity_curve).expanding().max()
            drawdown = (pd.Series(equity_curve) - peak) / peak
            max_dd = drawdown.min() if len(drawdown) > 0 else 0

            # 勝率計算
            winning_trades = len([t for t in trades if t["pnl"] > 0])
            win_rate = winning_trades / len(trades) if trades else 0

            result = {
                "metrics": {
                    "total_return": total_return,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_dd,
                    "win_rate": win_rate,
                    "total_trades": len(trades),
                },
                "trades": trades,
                "equity_curve": equity_curve,
                "returns": returns.tolist(),
            }

            logger.info("✅ Baseline backtest completed")
            return result

        except Exception as e:
            logger.error(f"❌ Baseline backtest failed: {e}")
            return {"error": str(e)}

    def run_enhanced_backtest(
        self, data_path: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
    ) -> Dict[str, Any]:
        """
        Phase 3強化バックテストを実行（改善版）

        Args:
            data_path: テストデータのパス
            config: バックテスト設定

        Returns:
            バックテスト結果
        """
        logger.info("🚀 Running Phase 3 enhanced backtest (improved version)")

        try:
            # データ読み込み
            data = pd.read_csv(data_path)
            if "close" not in data.columns and "open" in data.columns:
                data["close"] = data["open"]  # closeカラムがない場合、openを使用
            if "timestamp" not in data.columns:
                data["timestamp"] = pd.date_range(
                    start="2023-01-01", periods=len(data), freq="h"
                )

            # 改善版：より賢いシグナル生成（複数指標の組み合わせ）
            prices = data["close"].values
            signals = np.zeros(len(data))

            # 移動平均とRSIを組み合わせたシグナル生成
            for i in range(20, len(data)):
                # 短期・長期移動平均
                short_ma = np.mean(prices[i - 5 : i])
                long_ma = np.mean(prices[i - 20 : i])

                # RSI計算（簡易版）
                gains = []
                losses = []
                for j in range(max(0, i - 14), i):
                    change = prices[j] - prices[j - 1] if j > 0 else 0
                    if change > 0:
                        gains.append(change)
                    else:
                        losses.append(abs(change))

                avg_gain = np.mean(gains) if gains else 0
                avg_loss = np.mean(losses) if losses else 0
                rsi = 100 - (100 / (1 + (avg_gain / avg_loss if avg_loss != 0 else 1)))

                # 改善版エントリー条件
                trend_up = short_ma > long_ma * 1.005  # トレンド強度を緩和
                oversold = rsi < 35  # RSIオーバーソールド

                trend_down = short_ma < long_ma * 0.995  # トレンド強度を緩和
                overbought = rsi > 65  # RSIオーバーボート

                if trend_up and oversold:
                    signals[i] = 1  # 強気シグナル
                elif trend_down and overbought:
                    signals[i] = -1  # 弱気シグナル

            # 改善版バックテスト実行（より積極的なパラメータ）
            capital = config.get("initial_balance", 1000000)
            position = 0
            entry_price = 0
            trades = []
            equity_curve = [capital]

            # 改善版リスク管理パラメータ（より積極的）
            max_position_size = 0.08  # 8% max position (4倍に増加)
            stop_loss = 0.04  # 4% stop loss (2倍に緩和)
            take_profit = 0.10  # 10% take profit (2倍に増加)

            consecutive_losses = 0  # 連敗カウンター
            max_consecutive_losses = 3  # 最大連敗許容数

            for i, (idx, row) in enumerate(data.iterrows()):
                signal = signals[i]
                price = row["close"]

                # ポジション変更（改善版ロジック）
                if signal != position and signal != 0:
                    # 連敗チェック（リスク管理）
                    if consecutive_losses >= max_consecutive_losses:
                        continue  # 連敗時は新規エントリーを控える

                    if position != 0:  # ポジションクローズ
                        pnl = (
                            (price - entry_price) / entry_price
                            if position == 1
                            else (entry_price - price) / entry_price
                        )
                        capital *= 1 + pnl * position * max_position_size

                        # 連敗カウンター更新
                        if pnl < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0

                        trades.append(
                            {
                                "entry_price": entry_price,
                                "exit_price": price,
                                "pnl": pnl,
                                "type": "long" if position == 1 else "short",
                            }
                        )

                    # 新規ポジション（改善版リスク管理適用）
                    entry_price = price
                    position = signal

                # ストップロス/テイクプロフィットチェック（改善版）
                if position != 0:
                    current_pnl = (
                        (price - entry_price) / entry_price
                        if position == 1
                        else (entry_price - price) / entry_price
                    )

                    if current_pnl <= -stop_loss or current_pnl >= take_profit:
                        capital *= 1 + current_pnl * position * max_position_size

                        # 連敗カウンター更新
                        if current_pnl < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0

                        trades.append(
                            {
                                "entry_price": entry_price,
                                "exit_price": price,
                                "pnl": current_pnl,
                                "type": "long" if position == 1 else "short",
                                "exit_reason": "stop_loss"
                                if current_pnl <= -stop_loss
                                else "take_profit",
                            }
                        )
                        position = 0

                equity_curve.append(capital)

            # メトリクス計算
            returns = pd.Series(equity_curve).pct_change().dropna()
            total_return = (
                capital - config.get("initial_balance", 1000000)
            ) / config.get("initial_balance", 1000000)
            sharpe = sharpe_ratio(returns.values) if len(returns) > 0 else 0

            # 最大ドローダウン計算
            peak = pd.Series(equity_curve).expanding().max()
            drawdown = (pd.Series(equity_curve) - peak) / peak
            max_dd = drawdown.min() if len(drawdown) > 0 else 0

            # 勝率計算
            winning_trades = len([t for t in trades if t["pnl"] > 0])
            win_rate = winning_trades / len(trades) if trades else 0

            result = {
                "metrics": {
                    "total_return": total_return,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_dd,
                    "win_rate": win_rate,
                    "total_trades": len(trades),
                    "profit_factor": sum(t["pnl"] for t in trades if t["pnl"] > 0)
                    / abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
                    if any(t["pnl"] < 0 for t in trades)
                    else float("inf"),
                    "avg_trade_pnl": np.mean([t["pnl"] for t in trades])
                    if trades
                    else 0,
                    "max_consecutive_losses": consecutive_losses,
                },
                "trades": trades,
                "equity_curve": equity_curve,
                "returns": returns.tolist(),
                "parameters": {
                    "max_position_size": max_position_size,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "max_consecutive_losses": max_consecutive_losses,
                },
            }

        except Exception as e:
            logger.error(f"❌ Baseline backtest failed: {e}")
            return {"error": str(e)}

    def run_enhanced_backtest_aggressive(
        self, data_path: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Phase 3強化バックテストを実行（積極改善版）

        Args:
            data_path: テストデータのパス
            config: バックテスト設定

        Returns:
            バックテスト結果
        """
        logger.info("🚀 Running Phase 3 enhanced backtest (aggressive improvement)")

        try:
            # データ読み込み
            data = pd.read_csv(data_path)
            if "close" not in data.columns and "open" in data.columns:
                data["close"] = data["open"]  # closeカラムがない場合、openを使用
            if "timestamp" not in data.columns:
                data["timestamp"] = pd.date_range(
                    start="2023-01-01", periods=len(data), freq="h"
                )

            # 積極改善版：より緩いシグナル生成（より多くの取引機会）
            prices = data["close"].values
            signals = np.zeros(len(data))

            # 移動平均とRSIを組み合わせたシグナル生成（緩和版）
            for i in range(20, len(data)):
                # 短期・長期移動平均
                short_ma = np.mean(prices[i - 5 : i])
                long_ma = np.mean(prices[i - 20 : i])

                # RSI計算（簡易版）
                gains = []
                losses = []
                for j in range(max(0, i - 14), i):
                    change = prices[j] - prices[j - 1] if j > 0 else 0
                    if change > 0:
                        gains.append(change)
                    else:
                        losses.append(abs(change))

                avg_gain = np.mean(gains) if gains else 0
                avg_loss = np.mean(losses) if losses else 0
                rsi = 100 - (100 / (1 + (avg_gain / avg_loss if avg_loss != 0 else 1)))

                # 積極改善版エントリー条件（大幅緩和）
                trend_up = short_ma > long_ma * 1.002  # トレンド強度を大幅緩和
                oversold = rsi < 40  # RSIオーバーソールドを緩和

                trend_down = short_ma < long_ma * 0.998  # トレンド強度を大幅緩和
                overbought = rsi > 60  # RSIオーバーボートを緩和

                if trend_up and oversold:
                    signals[i] = 1  # 強気シグナル
                elif trend_down and overbought:
                    signals[i] = -1  # 弱気シグナル

            # 積極改善版バックテスト実行（より積極的なパラメータ）
            capital = config.get("initial_balance", 1000000)
            position = 0
            entry_price = 0
            trades = []
            equity_curve = [capital]

            # 積極改善版リスク管理パラメータ（より積極的）
            max_position_size = 0.10  # 10% max position (5倍に増加)
            stop_loss = 0.05  # 5% stop loss (2.5倍に緩和)
            take_profit = 0.12  # 12% take profit (2.4倍に増加)

            consecutive_losses = 0  # 連敗カウンター
            max_consecutive_losses = 5  # 最大連敗許容数を増加

            for i, (idx, row) in enumerate(data.iterrows()):
                signal = signals[i]
                price = row["close"]

                # ポジション変更（積極改善版ロジック）
                if signal != position and signal != 0:
                    # 連敗チェック（より緩和）
                    if consecutive_losses >= max_consecutive_losses:
                        continue  # 連敗時は新規エントリーを控える

                    if position != 0:  # ポジションクローズ
                        pnl = (
                            (price - entry_price) / entry_price
                            if position == 1
                            else (entry_price - price) / entry_price
                        )
                        capital *= 1 + pnl * position * max_position_size

                        # 連敗カウンター更新
                        if pnl < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0

                        trades.append(
                            {
                                "entry_price": entry_price,
                                "exit_price": price,
                                "pnl": pnl,
                                "type": "long" if position == 1 else "short",
                            }
                        )

                    # 新規ポジション（積極改善版リスク管理適用）
                    entry_price = price
                    position = signal

                # ストップロス/テイクプロフィットチェック（積極改善版）
                if position != 0:
                    current_pnl = (
                        (price - entry_price) / entry_price
                        if position == 1
                        else (entry_price - price) / entry_price
                    )

                    if current_pnl <= -stop_loss or current_pnl >= take_profit:
                        capital *= 1 + current_pnl * position * max_position_size

                        # 連敗カウンター更新
                        if current_pnl < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0

                        trades.append(
                            {
                                "entry_price": entry_price,
                                "exit_price": price,
                                "pnl": current_pnl,
                                "type": "long" if position == 1 else "short",
                                "exit_reason": "stop_loss"
                                if current_pnl <= -stop_loss
                                else "take_profit",
                            }
                        )
                        position = 0

                equity_curve.append(capital)

            # メトリクス計算
            returns = pd.Series(equity_curve).pct_change().dropna()
            total_return = (
                capital - config.get("initial_balance", 1000000)
            ) / config.get("initial_balance", 1000000)
            sharpe = sharpe_ratio(returns.values) if len(returns) > 0 else 0

            # 最大ドローダウン計算
            peak = pd.Series(equity_curve).expanding().max()
            drawdown = (pd.Series(equity_curve) - peak) / peak
            max_dd = drawdown.min() if len(drawdown) > 0 else 0

            # 勝率計算
            winning_trades = len([t for t in trades if t["pnl"] > 0])
            win_rate = winning_trades / len(trades) if trades else 0

            result = {
                "metrics": {
                    "total_return": total_return,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_dd,
                    "win_rate": win_rate,
                    "total_trades": len(trades),
                    "profit_factor": sum(t["pnl"] for t in trades if t["pnl"] > 0)
                    / abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
                    if any(t["pnl"] < 0 for t in trades)
                    else float("inf"),
                    "avg_trade_pnl": np.mean([t["pnl"] for t in trades])
                    if trades
                    else 0,
                    "max_consecutive_losses": consecutive_losses,
                },
                "trades": trades,
                "equity_curve": equity_curve,
                "returns": returns.tolist(),
                "parameters": {
                    "max_position_size": max_position_size,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "max_consecutive_losses": max_consecutive_losses,
                    "rsi_oversold_threshold": 40,
                    "rsi_overbought_threshold": 60,
                    "trend_up_threshold": 1.002,
                    "trend_down_threshold": 0.998,
                },
            }

            logger.info("✅ Phase 3 enhanced backtest (aggressive) completed")
            return result

        except Exception as e:
            logger.error(f"❌ Phase 3 enhanced backtest (aggressive) failed: {e}")
            return {"error": str(e)}

    def compare_results(
        self, baseline_results: Dict[str, Any], enhanced_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        バックテスト結果を比較

        Args:
            baseline_results: ベースライン結果
            enhanced_results: Phase 3強化結果

        Returns:
            比較分析結果
        """
        logger.info("📊 Comparing backtest results")

        comparison = {
            "summary": {},
            "metrics_comparison": {},
            "risk_metrics": {},
            "statistical_significance": {},
            "recommendations": [],
        }

        try:
            # エラーチェック
            if "error" in baseline_results:
                comparison["summary"]["baseline_error"] = baseline_results["error"]
                return comparison

            if "error" in enhanced_results:
                comparison["summary"]["enhanced_error"] = enhanced_results["error"]
                return comparison

            # 基本メトリクスの比較
            baseline_metrics = baseline_results.get("metrics", {})
            enhanced_metrics = enhanced_results.get("metrics", {})

            metrics_to_compare = [
                "total_return",
                "sharpe_ratio",
                "max_drawdown",
                "win_rate",
                "profit_factor",
                "total_trades",
            ]

            for metric in metrics_to_compare:
                baseline_val = baseline_metrics.get(metric, 0)
                enhanced_val = enhanced_metrics.get(metric, 0)

                comparison["metrics_comparison"][metric] = {
                    "baseline": baseline_val,
                    "enhanced": enhanced_val,
                    "improvement": enhanced_val - baseline_val
                    if isinstance(enhanced_val, (int, float))
                    and isinstance(baseline_val, (int, float))
                    else None,
                    "improvement_pct": (
                        (enhanced_val - baseline_val) / abs(baseline_val)
                    )
                    * 100
                    if baseline_val != 0
                    and isinstance(enhanced_val, (int, float))
                    and isinstance(baseline_val, (int, float))
                    else None,
                }

            # リスクメトリクスの比較
            risk_metrics = ["max_drawdown", "volatility", "var_95"]
            for metric in risk_metrics:
                if metric in baseline_metrics and metric in enhanced_metrics:
                    comparison["risk_metrics"][metric] = {
                        "baseline": baseline_metrics[metric],
                        "enhanced": enhanced_metrics[metric],
                        "reduction": baseline_metrics[metric] - enhanced_metrics[metric]
                        if metric in ["max_drawdown", "volatility"]
                        else None,
                    }

            # 統計的有意性の評価（簡易版）
            comparison[
                "statistical_significance"
            ] = self._assess_statistical_significance(
                baseline_results, enhanced_results
            )

            # 推奨事項の生成
            comparison["recommendations"] = self._generate_recommendations(comparison)

            logger.info("✅ Results comparison completed")
            return comparison

        except Exception as e:
            logger.error(f"❌ Results comparison failed: {e}")
            comparison["error"] = str(e)
            return comparison

    def _assess_statistical_significance(
        self, baseline: Dict[str, Any], enhanced: Dict[str, Any]
    ) -> Dict[str, Any]:
        """統計的有意性の評価（簡易版）"""
        significance = {}

        try:
            baseline_returns = baseline.get("returns", [])
            enhanced_returns = enhanced.get("returns", [])

            if len(baseline_returns) > 10 and len(enhanced_returns) > 10:
                # t-testの簡易実装
                baseline_mean = np.mean(baseline_returns)
                enhanced_mean = np.mean(enhanced_returns)
                baseline_std = np.std(baseline_returns, ddof=1)
                enhanced_std = np.std(enhanced_returns, ddof=1)

                n1, n2 = len(baseline_returns), len(enhanced_returns)

                # プールされた標準偏差
                pooled_std = np.sqrt(
                    ((n1 - 1) * baseline_std**2 + (n2 - 1) * enhanced_std**2)
                    / (n1 + n2 - 2)
                )

                # t統計量
                t_stat = (enhanced_mean - baseline_mean) / (
                    pooled_std * np.sqrt(1 / n1 + 1 / n2)
                )

                # 自由度
                df = n1 + n2 - 2

                # p値の近似（簡易版）
                p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))

                significance["return_improvement"] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "effect_size": (enhanced_mean - baseline_mean) / baseline_std
                    if baseline_std > 0
                    else 0,
                }

        except Exception as e:
            logger.warning(f"Statistical significance assessment failed: {e}")

        return significance

                improvement = values.get("improvement", 0)
                improvement_pct = values.get("improvement_pct")

                report += f"  {metric.replace('_', ' ').title()}:\n"
                report += f"    Baseline: {baseline_val}\n"
                report += f"    Phase 3:  {enhanced_val}\n"
                if improvement_pct is not None:
                    report += f"    Change:  {improvement_pct:+.1f}%\n"
                report += "\n"

            report += "🎯 RISK METRICS:\n"
            for metric, values in comparison.get("risk_metrics", {}).items():
                baseline_val = values.get("baseline", 0)
                enhanced_val = values.get("enhanced", 0)
                reduction = values.get("reduction")

                report += f"  {metric.replace('_', ' ').title()}:\n"
                report += f"    Baseline: {baseline_val}\n"
                report += f"    Phase 3:  {enhanced_val}\n"
                if reduction is not None and metric in ["max_drawdown", "volatility"]:
                    report += f"    Reduction: {reduction:.3f}\n"
                report += "\n"

            # 統計的有意性
            significance = comparison.get("statistical_significance", {})
            if significance:
                report += "📊 STATISTICAL SIGNIFICANCE:\n"
                for test_name, results in significance.items():
                    if results.get("significant", False):
                        report += f"  ✅ {test_name.replace('_', ' ').title()}: Significant improvement (p={results.get('p_value', 1):.3f})\n"
                    else:
                        report += f"  ⚠️  {test_name.replace('_', ' ').title()}: Not statistically significant\n"
                report += "\n"

        # 推奨事項
        recommendations = comparison.get("recommendations", [])
        if recommendations:
            report += "💡 RECOMMENDATIONS:\n"
            for i, rec in enumerate(recommendations, 1):
                report += f"  {i}. {rec}\n"
            report += "\n"

        report += "=" * 80 + "\n"

        # レポートをファイルに保存
        report_file = self.results_dir / f"phase3_comparison_report_{timestamp}.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"📁 Detailed report saved to: {report_file}")

        # JSON結果も保存
        json_file = self.results_dir / f"phase3_comparison_results_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "baseline_results": baseline_results,
                    "enhanced_results": enhanced_results,
                    "comparison": comparison,
                    "timestamp": timestamp,
                },
                f,
                indent=2,
                default=str,
            )

        logger.info(f"📁 JSON results saved to: {json_file}")

        return report

    def generate_improved_report(
        self,
        baseline_results: Dict[str, Any],
        enhanced_results: Dict[str, Any],
        comparison: Dict[str, Any],
    ) -> str:
        """
        改善版比較レポートを生成

        Args:
            baseline_results: ベースライン結果
            enhanced_results: Phase 3強化結果（改善版）
            comparison: 比較分析結果

        Returns:
            レポート文字列
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = f"""
================================================================================
PHASE 3 ENHANCED VS BASELINE BACKTEST COMPARISON (IMPROVED VERSION)
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
================================================================================

"""

        if "error" in baseline_results:
            report += f"❌ Baseline Error: {baseline_results['error']}\n\n"
        if "error" in enhanced_results:
            report += f"❌ Phase 3 Enhanced Error: {enhanced_results['error']}\n\n"

        if "error" not in baseline_results and "error" not in enhanced_results:
            report += "📈 METRICS COMPARISON (IMPROVED):\n"
            for metric, values in comparison.get("metrics_comparison", {}).items():
                baseline_val = values.get("baseline", 0)
                enhanced_val = values.get("enhanced", 0)
                improvement = values.get("improvement", 0)
                improvement_pct = values.get("improvement_pct")

                report += f"  {metric.replace('_', ' ').title()}:\n"
                report += f"    Baseline: {baseline_val}\n"
                report += f"    Phase 3 Improved:  {enhanced_val}\n"
                if improvement_pct is not None:
                    report += f"    Change:  {improvement_pct:+.1f}%\n"
                report += "\n"

            # 改善版パラメータ表示
            params = enhanced_results.get("parameters", {})
            if params:
                report += "🔧 IMPROVED PARAMETERS:\n"
                report += (
                    f"  Max Position Size: {params.get('max_position_size', 0):.1%}\n"
                )
                report += f"  Stop Loss: {params.get('stop_loss', 0):.1%}\n"
                report += f"  Take Profit: {params.get('take_profit', 0):.1%}\n"
                report += f"  Max Consecutive Losses: {params.get('max_consecutive_losses', 0)}\n\n"

            # 追加メトリクス
            enhanced_metrics = enhanced_results.get("metrics", {})
            report += "📊 ADDITIONAL METRICS:\n"
            report += (
                f"  Average Trade P&L: {enhanced_metrics.get('avg_trade_pnl', 0):.4f}\n"
            )
            report += f"  Max Consecutive Losses: {enhanced_metrics.get('max_consecutive_losses', 0)}\n\n"

        # 改善点のハイライト
        report += "💡 IMPROVEMENT HIGHLIGHTS:\n"
        metrics_comp = comparison.get("metrics_comparison", {})

        # リターンの改善を確認
        return_comp = metrics_comp.get("total_return", {})
        if return_comp.get("improvement_pct", 0) > 10:
            report += "  ✅ Significant return improvement achieved\n"
        elif return_comp.get("improvement_pct", 0) > 0:
            report += "  ⚠️  Moderate return improvement\n"
        else:
            report += "  ❌ Return still below baseline - further optimization needed\n"

        # 取引数の改善を確認
        trades_comp = metrics_comp.get("total_trades", {})
        if trades_comp.get("improvement_pct", 0) > -50:  # 50%以上の削減は改善
            report += "  ✅ Better trade frequency maintained\n"
        else:
            report += "  ⚠️  Trade frequency still low - consider further relaxation\n"

        # 勝率の改善を確認
        winrate_comp = metrics_comp.get("win_rate", {})
        if winrate_comp.get("improvement_pct", 0) > 5:
            report += "  ✅ Improved win rate\n"
        else:
            report += "  ⚠️  Win rate needs attention\n"

        report += "\n" + "=" * 80 + "\n"

        # レポートをファイルに保存
        report_file = (
            self.results_dir / f"phase3_improved_comparison_report_{timestamp}.txt"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"📁 Improved report saved to: {report_file}")

        return report

    def generate_aggressive_report(
        self,
        baseline_results: Dict[str, Any],
        enhanced_results: Dict[str, Any],
        comparison: Dict[str, Any],
    ) -> str:
        """
        積極改善版比較レポートを生成

        Args:
            baseline_results: ベースライン結果
            enhanced_results: Phase 3強化結果（積極改善版）
            comparison: 比較分析結果

        Returns:
            レポート文字列
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = f"""
================================================================================
PHASE 3 ENHANCED VS BASELINE BACKTEST COMPARISON (AGGRESSIVE VERSION)
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
================================================================================

"""

        if "error" in baseline_results:
            report += f"❌ Baseline Error: {baseline_results['error']}\n\n"
        if "error" in enhanced_results:
            report += f"❌ Phase 3 Enhanced Error: {enhanced_results['error']}\n\n"

        if "error" not in baseline_results and "error" not in enhanced_results:
            report += "📈 METRICS COMPARISON (AGGRESSIVE):\n"
            for metric, values in comparison.get("metrics_comparison", {}).items():
                baseline_val = values.get("baseline", 0)
                enhanced_val = values.get("enhanced", 0)
                improvement = values.get("improvement", 0)
                improvement_pct = values.get("improvement_pct")

                report += f"  {metric.replace('_', ' ').title()}:\n"
                report += f"    Baseline: {baseline_val}\n"
                report += f"    Phase 3 Aggressive:  {enhanced_val}\n"
                if improvement_pct is not None:
                    report += f"    Change:  {improvement_pct:+.1f}%\n"
                report += "\n"

            # 積極改善版パラメータ表示
            params = enhanced_results.get("parameters", {})
            if params:
                report += "🔧 AGGRESSIVE PARAMETERS:\n"
                report += (
                    f"  Max Position Size: {params.get('max_position_size', 0):.1%}\n"
                )
                report += f"  Stop Loss: {params.get('stop_loss', 0):.1%}\n"
                report += f"  Take Profit: {params.get('take_profit', 0):.1%}\n"
                report += f"  Max Consecutive Losses: {params.get('max_consecutive_losses', 0)}\n\n"

            # 追加メトリクス
            enhanced_metrics = enhanced_results.get("metrics", {})
            report += "📊 ADDITIONAL METRICS:\n"
            report += (
                f"  Average Trade P&L: {enhanced_metrics.get('avg_trade_pnl', 0):.4f}\n"
            )
            report += f"  Max Consecutive Losses: {enhanced_metrics.get('max_consecutive_losses', 0)}\n\n"

        # 改善点のハイライト
        report += "💡 AGGRESSIVE IMPROVEMENT HIGHLIGHTS:\n"
        metrics_comp = comparison.get("metrics_comparison", {})

        # リターンの改善を確認
        return_comp = metrics_comp.get("total_return", {})
        if return_comp.get("improvement_pct", 0) > 10:
            report += "  ✅ Significant return improvement achieved\n"
        elif return_comp.get("improvement_pct", 0) > 0:
            report += "  ⚠️  Moderate return improvement\n"
        else:
            report += "  ❌ Return still below baseline - further optimization needed\n"

        # 取引数の改善を確認
        trades_comp = metrics_comp.get("total_trades", {})
        if trades_comp.get("improvement_pct", 0) > -50:  # 50%以上の削減は改善
            report += "  ✅ Better trade frequency maintained\n"
        else:
            report += "  ⚠️  Trade frequency still low - consider further relaxation\n"

        # 勝率の改善を確認
        winrate_comp = metrics_comp.get("win_rate", {})
        if winrate_comp.get("improvement_pct", 0) > 5:
            report += "  ✅ Improved win rate\n"
        else:
            report += "  ⚠️  Win rate needs attention\n"

        report += "\n" + "=" * 80 + "\n"

        # レポートをファイルに保存
        report_file = (
            self.results_dir / f"phase3_aggressive_comparison_report_{timestamp}.txt"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"📁 Aggressive report saved to: {report_file}")

        return report




if __name__ == "__main__":
    main()
