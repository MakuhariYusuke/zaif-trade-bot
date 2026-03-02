from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, TypedDict

from ztb.types.common import ConfigDict

class MarketState(TypedDict, total=False):
    """
    Represents the current state of the market for a single step.
    Used for execution simulation and signal evaluation.
    """
    high: float
    low: float
    close: float
    atr: float
    volume: float
    timestamp: Any | None

@dataclass
class PositionManagementConfig:
    """ポジション管理設定"""

    def __init__(self, **kwargs):
        """dict から設定を初期化"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    # 資本利用設定
    full_capital_utilization: bool = True  # 100%資本利用
    capital_buffer_pct: float = 0.05  # 資本バッファ (5%)

    # ポジション制限
    max_positions_per_symbol: int = 1  # シンボルあたり最大ポジション数
    max_total_positions: int = 5  # 総最大ポジション数
    max_correlation_positions: int = 2  # 相関ポジションの最大数

    # リスク管理
    max_portfolio_risk_pct: float = 0.10  # ポートフォリオ最大リスク (10%)
    max_single_position_risk_pct: float = 0.05  # 単一ポジション最大リスク (5%)
    stop_loss_pct: float = 0.02  # ストップロス (2%)
    take_profit_pct: float = 0.05  # テイクプロフィット (5%)

    # 再バランス設定
    enable_rebalancing: bool = True
    rebalance_interval_hours: int = 24  # 再バランス間隔 (24時間)
    rebalance_threshold_pct: float = 0.10  # 再バランス閾値 (10%)

    # 最小取引単位管理
    strict_min_unit_enforcement: bool = True
    round_to_min_unit: bool = True  # 最小単位への丸め
    min_unit_buffer_pct: float = 0.001  # 最小単位バッファ (0.1%)

    # 追加のリスク管理パラメータ
    min_signal_strength: float = 0.5  # 最小信号強度
    max_volatility_threshold: float = 0.05  # 最大ボラティリティ閾値
    default_position_size_pct: float = 0.02  # デフォルトポジションサイズ (%)
    atr_stop_loss_multiplier: float = 2.0  # ATRストップロス乗数
    atr_take_profit_multiplier: float = 4.0  # ATRテイクプロフィット乗数

    # ダイナミックサイジング
    enable_dynamic_sizing: bool = True
    sizing_update_interval: int = 300  # 5分ごと
    market_condition_adjustment: bool = True  # 市場状況に応じた調整

@dataclass
class PortfolioState:
    """ポートフォリオ状態"""

    total_capital: float  # 総資本
    available_capital: float  # 利用可能資本
    used_capital: float  # 使用資本
    total_value: float  # 総資産価値
    unrealized_pnl: float  # 未実現損益
    realized_pnl: float  # 実現損益
    total_risk: float  # 総リスク
    positions: dict[str, Any] = field(default_factory=dict)  # Position 型は循環を避けるため Any
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_pnl(self) -> float:
        """総損益"""
        return self.unrealized_pnl + self.realized_pnl

    @property
    def portfolio_return(self) -> float:
        """ポートフォリオリターン"""
        if self.total_capital > 0:
            return self.total_pnl / self.total_capital
        return 0.0

    @property
    def risk_adjusted_return(self) -> float:
        """リスク調整リターン"""
        if self.total_risk > 0:
            return self.portfolio_return / self.total_risk
        return 0.0

@dataclass
class PositionSignal:
    """ポジションシグナル"""

    symbol: str  # シンボル名
    action: str  # "open_long", "open_short", "close_long", "close_short", "adjust"
    strength: float  # シグナル強度 (0-1)
    target_quantity: float  # 目標数量
    confidence: float  # シグナル信頼度 (0-1)
    reason: str  # シグナル理由
    timestamp: datetime = field(default_factory=datetime.now)
