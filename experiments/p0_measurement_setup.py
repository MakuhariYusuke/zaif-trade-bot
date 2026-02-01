#!/usr/bin/env python3
"""
P0: 計測基盤整備スクリプト

89#に基づき、環境メトリクス（gross_pnl/net_pnl/total_fees等）の
取得・検証機能を整備する。
"""

from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ztb.environments.fast_intraday_env_v456 import FastIntradayEnv
from ztb.rl.common.evaluate_policy import evaluate_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EnvironmentMetrics:
    """環境から抽出したメトリクス"""
    
    # 基本値
    balance: float
    initial_balance: float
    gross_pnl: float
    net_pnl: float
    total_fees: float
    total_slippage: float
    total_trades: int
    
    # 派生指標
    @property
    def gross_roi(self) -> float:
        """取引自体のROI（%）"""
        if self.initial_balance <= 0:
            return 0.0
        return (self.gross_pnl / self.initial_balance) * 100
    
    @property
    def net_roi(self) -> float:
        """コスト控除後のROI（%）"""
        if self.initial_balance <= 0:
            return 0.0
        return (self.net_pnl / self.initial_balance) * 100
    
    @property
    def balance_roi(self) -> float:
        """最終残高ベースのROI（%）"""
        if self.initial_balance <= 0:
            return 0.0
        return ((self.balance - self.initial_balance) / self.initial_balance) * 100
    
    @property
    def cost_ratio(self) -> float:
        """コスト負担率（%）"""
        if self.initial_balance <= 0:
            return 0.0
        return ((self.total_fees + self.total_slippage) / self.initial_balance) * 100
    
    @property
    def fee_ratio(self) -> float:
        """手数料負担率（%）"""
        if self.initial_balance <= 0:
            return 0.0
        return (self.total_fees / self.initial_balance) * 100
    
    def validate(self) -> list[str]:
        """整合性チェック"""
        errors = []
        
        # net_pnl = gross_pnl - fees - slippage
        expected_net = self.gross_pnl - self.total_fees - self.total_slippage
        if abs(self.net_pnl - expected_net) > 1.0:
            errors.append(
                f"PnL不整合: net_pnl={self.net_pnl:.2f}, "
                f"expected={expected_net:.2f} (gross={self.gross_pnl:.2f}, "
                f"fees={self.total_fees:.2f}, slip={self.total_slippage:.2f})"
            )
        
        # balance = initial + net_pnl
        expected_balance = self.initial_balance + self.net_pnl
        if abs(self.balance - expected_balance) > 1.0:
            errors.append(
                f"Balance不整合: balance={self.balance:.2f}, "
                f"expected={expected_balance:.2f} (initial={self.initial_balance:.2f}, "
                f"net_pnl={self.net_pnl:.2f})"
            )
        
        return errors
    
    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "total_fees": self.total_fees,
            "total_slippage": self.total_slippage,
            "total_trades": self.total_trades,
            "gross_roi": self.gross_roi,
            "net_roi": self.net_roi,
            "balance_roi": self.balance_roi,
            "cost_ratio": self.cost_ratio,
            "fee_ratio": self.fee_ratio,
        }
    
    def summary(self) -> str:
        """サマリー文字列"""
        lines = [
            "=" * 50,
            "Environment Metrics Summary",
            "=" * 50,
            f"Initial Balance: {self.initial_balance:,.2f} JPY",
            f"Final Balance:   {self.balance:,.2f} JPY",
            f"Total Trades:    {self.total_trades}",
            "-" * 50,
            "PnL Breakdown:",
            f"  Gross PnL:     {self.gross_pnl:+,.2f} JPY ({self.gross_roi:+.2f}%)",
            f"  Total Fees:    {self.total_fees:,.2f} JPY ({self.fee_ratio:.2f}%)",
            f"  Slippage:      {self.total_slippage:,.2f} JPY",
            f"  Net PnL:       {self.net_pnl:+,.2f} JPY ({self.net_roi:+.2f}%)",
            "-" * 50,
            "ROI Summary:",
            f"  Gross ROI:     {self.gross_roi:+.2f}%",
            f"  Net ROI:       {self.net_roi:+.2f}%",
            f"  Balance ROI:   {self.balance_roi:+.2f}%",
            f"  Cost Ratio:    {self.cost_ratio:.2f}%",
            "=" * 50,
        ]
        return "\n".join(lines)


def extract_environment_metrics(env: Any) -> EnvironmentMetrics:
    """
    環境オブジェクトから詳細メトリクスを抽出
    
    VecEnv、Monitor等でラップされていても対応
    """
    # VecEnvをunwrap
    actual_env = env
    if hasattr(env, 'envs') and len(env.envs) > 0:
        actual_env = env.envs[0]
    
    # さらにMonitor等をunwrap
    unwrapped = actual_env
    for _ in range(10):
        if hasattr(unwrapped, 'env'):
            unwrapped = unwrapped.env
        elif hasattr(unwrapped, 'unwrapped'):
            unwrapped = unwrapped.unwrapped
        else:
            break
    
    # メトリクス取得
    def safe_get(attr: str, default: float = 0.0) -> float:
        val = getattr(unwrapped, attr, default)
        return float(val) if val is not None else default
    
    return EnvironmentMetrics(
        balance=safe_get('balance', 100000.0),
        initial_balance=safe_get('initial_balance', 100000.0),
        gross_pnl=safe_get('gross_pnl'),
        net_pnl=safe_get('net_pnl'),
        total_fees=safe_get('total_fees'),
        total_slippage=safe_get('total_slippage'),
        total_trades=int(safe_get('total_trades')),
    )


def run_p0_validation() -> None:
    """P0: 計測基盤の検証を実行"""
    logger.info("=" * 60)
    logger.info("P0: 計測基盤整備 - 検証開始")
    logger.info("=" * 60)
    
    # 環境構築
    logger.info("環境を構築中...")
    env = FastIntradayEnv(
        data_dir=str(project_root / "data"),
        initial_balance=100000.0,
        max_steps=1000,
        transaction_cost=0.001,  # 0.1%
    )
    
    # 初期状態確認
    obs, info = env.reset()
    metrics_initial = extract_environment_metrics(env)
    logger.info("\n初期状態:")
    logger.info(f"  Balance: {metrics_initial.balance:,.2f}")
    logger.info(f"  Gross PnL: {metrics_initial.gross_pnl:.2f}")
    logger.info(f"  Net PnL: {metrics_initial.net_pnl:.2f}")
    
    # ランダム取引でテスト
    logger.info("\nランダム取引を実行中（1000ステップ）...")
    total_reward = 0.0
    
    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    # 最終状態
    metrics_final = extract_environment_metrics(env)
    
    logger.info("\n" + metrics_final.summary())
    
    # 整合性チェック
    errors = metrics_final.validate()
    if errors:
        logger.error("整合性エラー検出:")
        for err in errors:
            logger.error(f"  - {err}")
    else:
        logger.info("✅ 整合性チェック: OK")
    
    # info dictからの取得もテスト
    logger.info("\ninfo dictからのメトリクス:")
    for key in ['balance', 'gross_pnl', 'net_pnl', 'total_fees', 'fee_paid', 'slippage_paid']:
        if key in info:
            logger.info(f"  {key}: {info[key]}")
    
    env.close()
    
    logger.info("\n" + "=" * 60)
    logger.info("P0 検証完了")
    logger.info("=" * 60)


def run_cost_breakdown_analysis() -> None:
    """取引コストの内訳分析"""
    logger.info("\n" + "=" * 60)
    logger.info("取引コスト内訳分析")
    logger.info("=" * 60)
    
    # 異なる取引コスト率でテスト
    cost_rates = [0.0, 0.0005, 0.001, 0.002]
    results = []
    
    for cost_rate in cost_rates:
        env = FastIntradayEnv(
            data_dir=str(project_root / "data"),
            initial_balance=100000.0,
            max_steps=1000,
            transaction_cost=cost_rate,
        )
        
        env.reset()
        
        # 固定パターンで取引（BUY-SELL繰り返し）
        for step in range(500):
            # 交互にBUY/SELL
            action = np.array([1.0 if step % 20 < 10 else -1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        metrics = extract_environment_metrics(env)
        results.append({
            'cost_rate': cost_rate * 100,  # %表示
            'gross_pnl': metrics.gross_pnl,
            'total_fees': metrics.total_fees,
            'net_pnl': metrics.net_pnl,
            'total_trades': metrics.total_trades,
            'gross_roi': metrics.gross_roi,
            'net_roi': metrics.net_roi,
        })
        
        env.close()
    
    # 結果表示
    logger.info("\n取引コスト影響分析:")
    logger.info("-" * 80)
    logger.info(f"{'Cost Rate':>10} | {'Trades':>7} | {'Gross PnL':>12} | {'Fees':>12} | {'Net PnL':>12} | {'Gross ROI':>10} | {'Net ROI':>10}")
    logger.info("-" * 80)
    
    for r in results:
        logger.info(
            f"{r['cost_rate']:>9.2f}% | {r['total_trades']:>7} | {r['gross_pnl']:>+12,.2f} | "
            f"{r['total_fees']:>12,.2f} | {r['net_pnl']:>+12,.2f} | "
            f"{r['gross_roi']:>+9.2f}% | {r['net_roi']:>+9.2f}%"
        )
    
    logger.info("-" * 80)


if __name__ == "__main__":
    run_p0_validation()
    run_cost_breakdown_analysis()
