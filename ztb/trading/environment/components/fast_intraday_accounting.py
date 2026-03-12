from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class AccountingSnapshot:
    gross_pnl: float
    net_pnl: float
    total_fees: float
    total_slippage: float
    portfolio_value: float

class FastIntradayAccounting:
    """Track gross/net PnL and costs for FastIntradayEnvV456."""

    def __init__(self, initial_balance: float) -> None:
        self.initial_balance = float(initial_balance)
        self.reset()

    def reset(self) -> None:
        self.gross_pnl = 0.0
        self.net_pnl = 0.0
        self.total_fees = 0.0
        self.total_slippage = 0.0

    def update(self, step_pnl: float, fee_paid: float, slippage_paid: float) -> None:
        self.gross_pnl += float(step_pnl)
        self.total_fees += float(fee_paid)
        self.total_slippage += float(slippage_paid)
        self.net_pnl = self.gross_pnl - self.total_fees - self.total_slippage

    def portfolio_value(self) -> float:
        return self.initial_balance + self.net_pnl

    def snapshot(self) -> AccountingSnapshot:
        return AccountingSnapshot(
            gross_pnl=self.gross_pnl,
            net_pnl=self.net_pnl,
            total_fees=self.total_fees,
            total_slippage=self.total_slippage,
            portfolio_value=self.portfolio_value(),
        )
