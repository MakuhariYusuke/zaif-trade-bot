#!/usr/bin/env python3
"""
P0: 計測基盤整備スクリプト

89#に基づき、環境メトリクス（gross_pnl/net_pnl/total_fees等）の
取得・検証機能を整備する。

Day11 run_day11_verification.pyのアプローチを踏襲。
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

from ztb.training.unified_trainer.algorithms.sac_trainer import SACTrainer
from ztb.training.utils.env_metrics import (
    extract_trainer_env_metrics,
    compute_balance_roi,
)

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


def extract_environment_metrics(trainer: SACTrainer) -> EnvironmentMetrics:
    """
    トレーナーから環境メトリクスを抽出
    
    extract_trainer_env_metrics() を使用してメトリクスを取得し、
    EnvironmentMetricsデータクラスに変換。
    """
    metrics_dict = extract_trainer_env_metrics(trainer, include_optional=True)
    
    return EnvironmentMetrics(
        balance=metrics_dict.get('final_balance', metrics_dict.get('balance', 100000.0)),
        initial_balance=metrics_dict.get('initial_balance', 100000.0),
        gross_pnl=metrics_dict.get('gross_pnl', 0.0),
        net_pnl=metrics_dict.get('net_pnl', 0.0),
        total_fees=metrics_dict.get('total_fees', 0.0),
        total_slippage=metrics_dict.get('total_slippage', 0.0),
        total_trades=int(metrics_dict.get('total_trades', 0)),
    )


def create_minimal_config(seed: int = 42) -> dict:
    """最小限の設定を作成（Day11ベース）"""
    return {
        "experiment_name": "p0_measurement_test",
        "data": {
            "data_file": str(project_root / "data" / "btc_jpy_1m_v451_optimized_features.parquet"),
        },
        "environment": {
            "initial_balance": 100000.0,
            "max_steps": None,
            "use_continuous_actions": True,
            "action_space_type": "continuous",
        },
        "training": {
            "total_timesteps": 1000,  # 短時間テスト
            "seed": seed,
            "walk_forward": {
                "enabled": False,
            },
        },
        "sac": {},  # デフォルト値使用
        "reward": {},  # デフォルト値使用
    }


def run_p0_validation() -> None:
    """P0: 計測基盤の検証を実行"""
    logger.info("=" * 60)
    logger.info("P0: 計測基盤整備 - 検証開始")
    logger.info("=" * 60)
    
    # SACTrainerを使用して学習実行
    logger.info("短時間学習を実行中（1000ステップ）...")
    config = create_minimal_config(seed=42)
    
    trainer = SACTrainer(config=config, logger=logger)
    trainer.train()
    
    # 学習後にメトリクスを取得
    logger.info("\n学習後のメトリクス取得...")
    metrics = extract_environment_metrics(trainer)
    
    logger.info("\n" + metrics.summary())
    
    # 整合性チェック
    errors = metrics.validate()
    if errors:
        logger.error("整合性エラー検出:")
        for err in errors:
            logger.error(f"  - {err}")
    else:
        logger.info("✅ 整合性チェック: OK")
    
    # ROI計算
    roi_value = compute_balance_roi({
        'final_balance': metrics.balance,
        'initial_balance': metrics.initial_balance,
    })
    if roi_value is not None:
        logger.info(f"\nBalance ROI: {roi_value:+.2f}%")
    
    logger.info("\n" + "=" * 60)
    logger.info("P0 検証完了")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_p0_validation()
