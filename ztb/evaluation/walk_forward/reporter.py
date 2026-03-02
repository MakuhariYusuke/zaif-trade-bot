"""
Walk-Forward 分析結果のレポート機能
"""

import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ztb.io.json_io import write_json

from .types import WalkForwardResult

logger = logging.getLogger(__name__)

def classify_trade_type(position_before: float, position_after: float) -> str:
    """
    Doc04仕様: 8種類の詳細Trade Type分類
    
    Args:
        position_before: 取引前ポジション（正=ロング、負=ショート）
        position_after: 取引後ポジション
    
    Returns:
        Trade Type: "long_open", "long_close", "long_add", "long_reduce",
                    "short_open", "short_close", "short_add", "short_reduce",
                    "reverse", "hold"
    """
    if np.isclose(position_before, position_after, atol=1e-8):
        return "hold"
    
    # Long側の判定
    if position_before >= 0 and position_after >= 0:
        if np.isclose(position_before, 0.0, atol=1e-8) and not np.isclose(position_after, 0.0, atol=1e-8):
            return "long_open"
        elif not np.isclose(position_before, 0.0, atol=1e-8) and np.isclose(position_after, 0.0, atol=1e-8):
            return "long_close"
        elif position_after > position_before + 1e-8:
            return "long_add"
        elif position_after < position_before - 1e-8:
            return "long_reduce"
    
    # Short側の判定
    if position_before <= 0 and position_after <= 0:
        if np.isclose(position_before, 0.0, atol=1e-8) and not np.isclose(position_after, 0.0, atol=1e-8):
            return "short_open"
        elif not np.isclose(position_before, 0.0, atol=1e-8) and np.isclose(position_after, 0.0, atol=1e-8):
            return "short_close"
        elif position_after < position_before - 1e-8:
            return "short_add"
        elif position_after > position_before + 1e-8:
            return "short_reduce"
    
    # Long→Short または Short→Long の反転
    if (position_before > 0 and position_after < 0) or (position_before < 0 and position_after > 0):
        return "reverse"
    
    return "hold"

def decompose_reverse_trade(
    position_before: float,
    position_after: float,
    price: float,
    timestamp: pd.Timestamp
) -> list[dict[str, Any]]:
    """
    Doc04仕様: 反転取引を決済+新規エントリーに分解
    
    Args:
        position_before: 取引前ポジション
        position_after: 取引後ポジション
        price: 執行価格
        timestamp: タイムスタンプ
    
    Returns:
        分解された取引リスト [close_trade, open_trade]
    """
    trades = []
    
    # 1. 既存ポジションの全決済
    if position_before > 0:
        trades.append({
            "type": "long_close",
            "position_before": position_before,
            "position_after": 0.0,
            "price": price,
            "size": abs(position_before),
            "timestamp": timestamp
        })
    elif position_before < 0:
        trades.append({
            "type": "short_close",
            "position_before": position_before,
            "position_after": 0.0,
            "price": price,
            "size": abs(position_before),
            "timestamp": timestamp
        })
    
    # 2. 新規ポジションの開設
    if position_after > 0:
        trades.append({
            "type": "long_open",
            "position_before": 0.0,
            "position_after": position_after,
            "price": price,
            "size": abs(position_after),
            "timestamp": timestamp
        })
    elif position_after < 0:
        trades.append({
            "type": "short_open",
            "position_before": 0.0,
            "position_after": position_after,
            "price": price,
            "size": abs(position_after),
            "timestamp": timestamp
        })
    
    return trades

def to_serializable(obj):
    """JSON serializable に変換"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(item) for item in obj]
    else:
        return obj

class WalkForwardReporter:
    """結果集約と報告"""

    def __init__(self, result: WalkForwardResult) -> None:
        """初期化
        
        Args:
            result: Walk-Forward 分析結果
        """
        self.result: WalkForwardResult = result

    def report(self) -> None:
        """コンソール上に結果報告"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 Walk-Forward Analysis Results")
        logger.info("=" * 70)
        
        # ウィンドウ別結果
        logger.info(f"\nWindow-by-Window Performance:")
        for i, perf in enumerate(self.result.performances):
            reporter = self.result.reporters[i] if i < len(self.result.reporters) else None
            action_dist = reporter.stats.get("action_distribution", {}) if reporter else {}
            logger.info(
                f"  Window {perf.window_id}: "
                f"Val ROI {perf.val_roi:.4f} | "
                f"Test ROI {perf.test_roi:.4f} | "
                f"Sharpe {perf.sharpe_ratio:.4f} | "
                f"Actions: {action_dist}"
            )
        
        # 平均性能
        logger.info(f"\nAggregate Performance:")
        logger.info(f"  Average Val ROI: {self.result.average_val_roi:.4f}")
        logger.info(f"  Average Test ROI: {self.result.average_test_roi:.4f}")
        logger.info(f"  Test ROI Std Dev: {self.result.test_roi_std:.4f}")
        logger.info(f"  Average Sharpe: {self.result.average_sharpe:.4f}")
        logger.info(f"  Sharpe Consistency: {self.result.sharpe_consistency:.4f}")
        logger.info(f"  Average Win Rate: {self.result.average_win_rate:.4f}")
        logger.info(f"  Overfitting Ratio: {self.result.overfitting_ratio:.4f}")
        logger.info(f"  Average Profit Factor: {self.result.profit_factor:.4f}")
        logger.info(f"  Average Expectancy: {self.result.expectancy:.4f}")
        logger.info(f"  Average Avg Win: {self.result.avg_win:.4f}")
        logger.info(f"  Average Avg Loss: {self.result.avg_loss:.4f}")
        
        # 堅牢性判定
        status: str = "✅ ROBUST" if self.result.is_robust_model() else "⚠️ WATCH"
        logger.info(f"\n  Status: {status}")
        
        logger.info("=" * 70)

    def save_results(self, output_path: Path) -> None:
        """結果をJSON保存
        
        Args:
            output_path: 出力ファイルパス
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result_dict: dict[str, Any] = {
            "windows": len(self.result.windows),
            "average_val_roi": self.result.average_val_roi,
            "average_test_roi": self.result.average_test_roi,
            "test_roi_std": self.result.test_roi_std,
            "average_sharpe": self.result.average_sharpe,
            "sharpe_consistency": self.result.sharpe_consistency,
            "average_win_rate": self.result.average_win_rate,
            "overfitting_ratio": self.result.overfitting_ratio,
            "profit_factor": self.result.profit_factor,
            "expectancy": self.result.expectancy,
            "avg_win": self.result.avg_win,
            "avg_loss": self.result.avg_loss,
            "is_robust": self.result.is_robust_model(),
            "performances": [
                {
                    "window_id": p.window_id,
                    "val_roi": p.val_roi,
                    "test_roi": p.test_roi,
                    "sharpe_ratio": p.sharpe_ratio,
                    "win_rate": p.win_rate,
                    "overfitting_ratio": p.overfitting_ratio,
                    "profit_factor": p.profit_factor,
                    "expectancy": p.expectancy,
                    "avg_win": p.avg_win,
                    "avg_loss": p.avg_loss,
                }
                for p in self.result.performances
            ],
        }
        
        # JSON serializable に変換
        result_dict = to_serializable(result_dict)
        
        write_json(output_path, result_dict, indent=2)
        
        logger.info(f"✓ Results saved to {output_path}")

class BacktestReporter:
    """バックテストの統計情報を管理"""

    def __init__(self):
        self.stats = {
            "total_steps": 0,
            "total_trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "total_fees": 0.0,
            "total_slippage": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_percent": 0.0,
            "sharpe_ratio": 0.0,
            "action_distribution": {},
            "ttl_action_distribution": {},
            "raw_action_sum": 0.0,
            "raw_action_count": 0,
            "abs_action_sum": 0.0,
            "ttl_action_sum": 0.0,
            "ttl_action_count": 0,
            "avg_abs_action": 0.0,
            "avg_ttl_action": 0.0,
            "profit_factor": 0.0,
            "ttl_forced_exits": 0,
            "cooldown_triggers": 0,
            "ttl_enabled": None,
            "start_index": None,
            "seed": None,
            "baseline_mode": None,
            "reward_scale": None,
            "reward_clip": None,
        }
        self.portfolio_history = []
        self.trade_history = []

    @staticmethod
    def _bucket_ttl_action(ttl_value: float) -> str:
        ttl_value = max(0.0, min(ttl_value, 1.0))
        bucket_index = min(int(ttl_value * 5), 4)
        low = bucket_index * 0.2
        high = low + 0.2
        return f"{low:.1f}-{high:.1f}"

    def update_step(self, step, portfolio_value, action, env_info=None):
        self.stats["total_steps"] += 1
        self.portfolio_history.append(portfolio_value)

        # Action Distribution
        # action は [target_position, ttl] の2次元
        act_key = "hold"
        if isinstance(action, np.ndarray):
            action_val = float(action[0]) if len(action) > 0 else 0.0
        else:
            action_val = float(action)

        if action_val > 0.3:
            act_key = "buy"
        elif action_val < -0.3:
            act_key = "sell"

        self.stats["action_distribution"][act_key] = (
            self.stats["action_distribution"].get(act_key, 0) + 1
        )

        if isinstance(action, np.ndarray) and action.size > 1:
            ttl_val = float(action[1])
            ttl_key = self._bucket_ttl_action(ttl_val)
            self.stats["ttl_action_distribution"][ttl_key] = (
                self.stats["ttl_action_distribution"].get(ttl_key, 0) + 1
            )
            self.stats["ttl_action_sum"] += ttl_val
            self.stats["ttl_action_count"] += 1

        # Action Strength Stats
        self.stats["raw_action_sum"] += action_val
        self.stats["abs_action_sum"] += abs(action_val)
        self.stats["raw_action_count"] += 1

        if isinstance(env_info, dict):
            ttl_forced = env_info.get("ttl_forced_exits")
            cooldown_triggers = env_info.get("cooldown_triggers")
            if ttl_forced is not None:
                self.stats["ttl_forced_exits"] = int(ttl_forced)
            if cooldown_triggers is not None:
                self.stats["cooldown_triggers"] = int(cooldown_triggers)
            if "ttl_enabled" in env_info:
                self.stats["ttl_enabled"] = bool(env_info.get("ttl_enabled"))

    def record_trade(
        self,
        position_before: float,
        position_after: float,
        pnl: float,
        entry_price: float,
        exit_price: float,
        size: float,
        fee: float,
        slippage: float,
        timestamp: pd.Timestamp | None = None,
        close_reason: str | None = None,  # ★ P1-1: Phase 2追加
    ):
        """
        Doc04仕様 + P0-3規約: 詳細Trade Type分類で取引記録
        
        ★ P0-3 PnL規約:
        - pnl: NET PnL（コスト控除済み）
        - env.step()のinfo['trade_pnl']から受け取る値は既にnet
        - fee/slippageは検証・統計目的でのみ記録（二重控除しない）
        
        ★ P1-1 close_reason:
        - close_reasonはポジション決済時（*_close, reverse）のみ設定
        - 値: "tp" (利確), "sl" (損切), "reversal" (反転), "manual" (手動)
        - Noneの場合は記録しない（後方互換性）
        
        Args:
            position_before: 取引前ポジション
            position_after: 取引後ポジション
            pnl: Net PnL（コスト控除済み） - envから提供
            entry_price: エントリー価格
            exit_price: エグジット価格
            size: 取引サイズ
            fee: 手数料（検証用）
            slippage: スリッページ（検証用）
            timestamp: タイムスタンプ
            close_reason: 決済理由（Phase 2追加、オプション）
        """
        # Trade Type分類
        trade_type = classify_trade_type(position_before, position_after)
        
        # Hold は記録しない
        if trade_type == "hold":
            return
        
        # 反転取引の場合は分解
        if trade_type == "reverse":
            trades = decompose_reverse_trade(position_before, position_after, exit_price, timestamp)
            # ★ Doc21指摘[Major]: PnL配賦修正 - クローズ側に全PnL、新規側はコストのみ
            # 反転 = クローズ(index 0) + 新規オープン(index 1)
            
            for i, trade_info in enumerate(trades):
                # ★ P1-1: 反転時はclose_reasonを"reversal"に固定
                trade_close_reason = "reversal" if close_reason == "reversal" else close_reason
                
                if i == 0:  # クローズ側: 全PnL配賦
                    trade_pnl = pnl  # realized PnLすべて
                    trade_fee = fee  # 全手数料
                    trade_slippage = slippage  # 全スリッページ
                else:  # 新規側: エントリーコストのみ（PnL=0）
                    trade_pnl = 0.0  # エントリーなのでPnLなし
                    trade_fee = 0.0  # コストはクローズ側に含める
                    trade_slippage = 0.0
                
                self._record_single_trade(
                    trade_type=trade_info["type"],
                    pnl=trade_pnl,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=trade_info["size"],
                    fee=trade_fee,
                    slippage=trade_slippage,
                    timestamp=timestamp,
                    close_reason=trade_close_reason,
                )
        else:
            self._record_single_trade(
                trade_type=trade_type,
                pnl=pnl,
                entry_price=entry_price,
                exit_price=exit_price,
                size=size,
                fee=fee,
                slippage=slippage,
                timestamp=timestamp,
                close_reason=close_reason,
            )
    
    def _record_single_trade(
        self,
        trade_type: str,
        pnl: float,
        entry_price: float,
        exit_price: float,
        size: float,
        fee: float,
        slippage: float,
        timestamp: pd.Timestamp | None = None,
        close_reason: str | None = None,  # ★ P1-1: Phase 2追加
    ):
        """
        単一取引の記録（内部用）
        
        Args:
            close_reason: 決済理由（"tp", "sl", "reversal", "manual"）
        """
        # ★ Z1 P1-1: 全 trade type を total_trades にカウント (close/reduce 含む)
        # 旧コード: close/reduce を除外 → total_trades < winning+losing → win_rate > 1.0
        self.stats["total_trades"] += 1
        
        if "long" in trade_type:
            self.stats["long_trades"] += 1
        elif "short" in trade_type:
            self.stats["short_trades"] += 1
        
        # Note: PnL here is Net PnL (already includes costs).
        net_pnl = pnl
        
        if net_pnl > 0:
            self.stats["winning_trades"] += 1
        elif net_pnl < 0:
            self.stats["losing_trades"] += 1
        # net_pnl == 0 はカウントしない
        
        # ★ Z1 P0-2: gross_pnl = net_pnl + fee + slippage (コスト戻し)
        # pnl は NET (コスト控除済) なので、gross は fee/slippage を加算して復元
        self.stats["gross_pnl"] += pnl + fee + slippage
        self.stats["net_pnl"] += net_pnl
        self.stats["total_fees"] += fee
        self.stats["total_slippage"] += slippage
        
        trade_record = {
            "type": trade_type,
            "gross_pnl": pnl + fee + slippage,  # Z1 P0-2: NET→GROSS 復元
            "net_pnl": net_pnl,
            "entry": entry_price,
            "exit": exit_price,
            "size": size,
            "fee": fee,
            "slippage": slippage,
            "timestamp": timestamp,
        }
        
        # ★ P1-1: close_reasonを記録（close/reverseの場合のみ）
        if close_reason is not None and ("close" in trade_type or "reverse" in trade_type):
            trade_record["close_reason"] = close_reason
        
        self.trade_history.append(trade_record)

    def finalize_stats(self):
        """最終統計の計算"""
        # Drawdown
        peak = -np.inf
        max_dd = 0.0
        max_dd_pct = 0.0

        for val in self.portfolio_history:
            if val > peak:
                peak = val
            dd = peak - val
            dd_pct = dd / peak if peak > 0 else 0.0

            if dd > max_dd:
                max_dd = dd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        self.stats["max_drawdown"] = max_dd
        self.stats["max_drawdown_percent"] = max_dd_pct

        # Sharpe (Doc04仕様: 1分足→日次集約)
        sharpe = self._calculate_sharpe_ratio()
        self.stats["sharpe_ratio"] = sharpe if sharpe is not None else 0.0

        # Action Strength
        if self.stats["raw_action_count"] > 0:
            self.stats["avg_abs_action"] = self.stats["abs_action_sum"] / self.stats["raw_action_count"]
        else:
            self.stats["avg_abs_action"] = 0.0

        if self.stats["ttl_action_count"] > 0:
            self.stats["avg_ttl_action"] = self.stats["ttl_action_sum"] / self.stats["ttl_action_count"]
        else:
            self.stats["avg_ttl_action"] = 0.0

        # Profit Factor (Net PnL) - Doc04仕様: ゼロ除算対応
        gross_profit = sum(t['net_pnl'] for t in self.trade_history if t['net_pnl'] > 0)
        gross_loss = sum(abs(t['net_pnl']) for t in self.trade_history if t['net_pnl'] < 0)
        
        if gross_loss > 0:
            self.stats["profit_factor"] = gross_profit / gross_loss
        elif gross_profit > 0:
            self.stats["profit_factor"] = float('inf')
        else:
            self.stats["profit_factor"] = 0.0  # 取引なしまたは全てゼロ

        # Additional metrics - Doc04仕様: 例外処理厳密化
        winning_trades = [t for t in self.trade_history if t['net_pnl'] > 0]
        losing_trades = [t for t in self.trade_history if t['net_pnl'] < 0]
        
        self.stats["avg_win"] = np.mean([t['net_pnl'] for t in winning_trades]) if winning_trades else 0.0
        self.stats["avg_loss"] = np.mean([abs(t['net_pnl']) for t in losing_trades]) if losing_trades else 0.0
        
        # Expectancy計算
        if self.trade_history:
            win_rate = len(winning_trades) / len(self.trade_history)
            loss_rate = len(losing_trades) / len(self.trade_history)
            self.stats["expectancy"] = self.stats["avg_win"] * win_rate - self.stats["avg_loss"] * loss_rate
        else:
            self.stats["expectancy"] = 0.0
        
        # Trades per day (assuming 1m data, 1440 min/day)
        total_days = len(self.portfolio_history) / 1440
        self.stats["trades_per_day"] = self.stats["total_trades"] / total_days if total_days > 0 else 0.0
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float | None:
        """
        Doc04仕様: 日次集約Sharpe Ratio計算
        
        Args:
            risk_free_rate: 年率無リスク金利（デフォルト0）
        
        Returns:
            Sharpe Ratio（年率換算）、計算不可の場合None
        """
        if len(self.portfolio_history) < 1440:  # 1日未満
            return None
        
        # 1分足→日次残高に集約（1440分 = 1日）
        minutes_per_day = 1440
        daily_balances = [
            self.portfolio_history[i]
            for i in range(0, len(self.portfolio_history), minutes_per_day)
            if i < len(self.portfolio_history)
        ]
        
        if len(daily_balances) < 2:
            return None
        
        # 日次リターン計算
        daily_returns = [
            (daily_balances[i] - daily_balances[i-1]) / daily_balances[i-1]
            for i in range(1, len(daily_balances))
            if daily_balances[i-1] > 0  # ゼロ除算防止
        ]
        
        if len(daily_returns) < 2:
            return None
        
        # 標準偏差チェック（Doc04仕様）
        std_dev = np.std(daily_returns, ddof=1)
        if std_dev == 0 or np.isnan(std_dev) or np.isinf(std_dev):
            return None
        
        mean_return = np.mean(daily_returns)
        
        # 年率換算（252営業日）
        sharpe = (mean_return - risk_free_rate / 252) / std_dev * np.sqrt(252)
        
        # NaN/inf最終チェック
        if np.isnan(sharpe) or np.isinf(sharpe):
            return None
        
        return float(sharpe)
