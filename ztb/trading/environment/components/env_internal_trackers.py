from __future__ import annotations

import numpy as np

class EnvInternalTracker:
    """
    379# P3-A: Env-internal State Features Tracker for SAC.
    Tracks environment-internal states that cannot be captured from OHLCV alone.
    - InventoryTracker (162#/228#): Cumulative signed volume with time decay.
    - LossBoostTracker (226#): Loss event decay state for dynamic risk modulation.
    """
    def __init__(self, inventory_decay: float = 0.99, loss_decay: float = 0.95):
        self.inventory_decay = inventory_decay
        self.loss_decay = loss_decay
        self.inventory_pressure = 0.0
        self.loss_risk = 0.0
        self.time_in_market = 0.0

    def reset(self) -> None:
        self.inventory_pressure = 0.0
        self.loss_risk = 0.0
        self.time_in_market = 0.0

    def get_feature_vector(self) -> dict[str, float]:
        """観測空間に追加する特徴量ベクトルを返す"""
        return {
            "inventory_pressure": float(np.clip(self.inventory_pressure, -5.0, 5.0)),
            "loss_risk": float(np.clip(self.loss_risk, 0.0, 5.0)),
            "time_in_market": float(np.clip(self.time_in_market / 100.0, 0.0, 5.0)),
        }

    def update_step(self, has_position: bool) -> None:
        """毎ステップ呼ばれる減衰処理"""
        self.inventory_pressure *= self.inventory_decay
        self.loss_risk *= self.loss_decay
        
        if has_position:
            self.time_in_market += 1.0
        else:
            self.time_in_market = 0.0

    def on_trade(self, direction: int, amount: float) -> None:
        """
        取引（約定）時に呼ばれる
        Args:
           direction: 1 (BUY), -1 (SELL)
           amount: BTC枚数など (e.g. 0.01)
        """
        # inventory_pressure を更新 (amount=0.01 を想定しスケール調整)
        self.inventory_pressure += direction * amount * 50.0

    def on_trade_close(self, realized_pnl_pct: float) -> None:
        """
        ポジションクローズ時に呼ばれる
        Args:
           realized_pnl_pct: 決済損益率 (e.g. -0.01)
        """
        if realized_pnl_pct < 0:
            # 損失の大きさに応じて risk が跳ね上がる (e.g. 1% loss = +1.0)
            self.loss_risk += abs(realized_pnl_pct) * 100.0
