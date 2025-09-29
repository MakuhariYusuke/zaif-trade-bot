"""
Position sizing module with volatility targeting.

Provides risk-based position sizing to achieve target portfolio volatility.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import json
from .circuit_breakers import get_global_kill_switch, KillSwitchActivatedError


class SizingMethod(Enum):
    """Position sizing methods."""
    EQUAL_WEIGHT = "equal_weight"
    VOL_TARGETING = "vol_targeting"
    KELLY_CRITERION = "kelly_criterion"


@dataclass
class PositionSize:
    """Position sizing result."""
    symbol: str
    quantity: float
    sizing_reason: str
    target_vol_contribution: float
    current_portfolio_vol: float
    sizing_chain: Dict[str, Any]  # Chain of sizing calculations


class PositionSizer:
    """Handles position sizing with volatility targeting."""

    def __init__(self, target_volatility: float = 0.10, method: SizingMethod = SizingMethod.VOL_TARGETING):
        """
        Initialize position sizer.

        Args:
            target_volatility: Annualized target portfolio volatility (e.g., 0.10 for 10%)
            method: Sizing method to use
        """
        self.target_volatility = target_volatility
        self.method = method
        self.rng = np.random.RandomState(42)  # Fixed seed for determinism

    def calculate_position_sizes(
        self,
        signals: Dict[str, float],  # symbol -> signal strength (-1 to 1)
        current_prices: Dict[str, float],
        portfolio_value: float,
        asset_volatilities: Dict[str, float],  # annualized volatilities
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> List[PositionSize]:
        """
        Calculate position sizes for given signals.

        Args:
            signals: Trading signals per symbol
            current_prices: Current prices per symbol
            portfolio_value: Current portfolio value
            asset_volatilities: Annualized volatilities per symbol
            correlation_matrix: Asset correlation matrix (optional)

        Returns:
            List of position sizes with sizing reasons
        """
        if not signals:
            return []

        if self.method == SizingMethod.EQUAL_WEIGHT:
            return self._equal_weight_sizing(signals, current_prices, portfolio_value)
        elif self.method == SizingMethod.VOL_TARGETING:
            return self._vol_targeting_sizing(
                signals, current_prices, portfolio_value,
                asset_volatilities, correlation_matrix
            )
        elif self.method == SizingMethod.KELLY_CRITERION:
            return self._kelly_sizing(signals, current_prices, portfolio_value, asset_volatilities)
        else:
            raise ValueError(f"Unknown sizing method: {self.method}")

    def _equal_weight_sizing(
        self,
        signals: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float
    ) -> List[PositionSize]:
        """Equal weight position sizing."""
        positions = []
        num_signals = len(signals)
        if num_signals == 0:
            return positions

        allocation_per_signal = portfolio_value / num_signals

        for symbol, signal in signals.items():
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]
            quantity = (allocation_per_signal * signal) / price

            positions.append(PositionSize(
                symbol=symbol,
                quantity=quantity,
                sizing_reason=f"Equal weight: {allocation_per_signal:.0f} JPY allocation",
                target_vol_contribution=0.0,  # Not applicable
                current_portfolio_vol=0.0
            ))

        return positions

    def _vol_targeting_sizing(
        self,
        signals: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float,
        asset_volatilities: Dict[str, float],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> List[PositionSize]:
        """Volatility targeting position sizing."""
        positions = []

        # Estimate portfolio volatility
        portfolio_vol = self._estimate_portfolio_volatility(
            signals, asset_volatilities, correlation_matrix
        )

        if portfolio_vol <= 0:
            # Fallback to equal weight
            return self._equal_weight_sizing(signals, current_prices, portfolio_value)

        # Scale positions to achieve target volatility
        vol_scaling = self.target_volatility / portfolio_vol

        for symbol, signal in signals.items():
            if symbol not in current_prices or symbol not in asset_volatilities:
                continue

            price = current_prices[symbol]
            vol = asset_volatilities[symbol]

            # Base allocation (risk parity style)
            base_allocation = portfolio_value * (self.target_volatility / vol) / len(signals)

            # Apply signal and volatility scaling
            allocation = base_allocation * signal * vol_scaling

            # Build sizing chain
            sizing_chain = {
                'raw_size': allocation,
                'vol_target': allocation,  # Already applied
                'kelly': allocation * 0.5,  # Half Kelly
                'decimal_rounded': None,
                'validated': None,
                'circuit_breaker_veto': None,
                'final_quantity': None,
                'skip_reason': None
            }

            # Kelly adjustment (0.5)
            allocation = sizing_chain['kelly']

            # Decimal rounding (price/qty)
            price_dec = Decimal(str(price))
            allocation_dec = Decimal(str(allocation))
            quantity_dec = allocation_dec / price_dec
            # Round down to avoid over-sizing
            quantity_rounded = float(quantity_dec.quantize(Decimal('0.00000001'), rounding=ROUND_DOWN))
            sizing_chain['decimal_rounded'] = quantity_rounded

            # Min notional/step validation (placeholder - need symbol config)
            # Assume min_order_size = 0.0001, min_price = 1, etc.
            min_order_size = 0.0001
            if quantity_rounded < min_order_size:
                sizing_chain['skip_reason'] = f"Quantity {quantity_rounded} below minimum {min_order_size}"
                quantity_rounded = 0

            sizing_chain['validated'] = quantity_rounded

            # Circuit breaker veto
            try:
                kill_switch = get_global_kill_switch()
                if kill_switch.is_active():
                    sizing_chain['circuit_breaker_veto'] = 0
                    sizing_chain['skip_reason'] = "Circuit breaker active"
                    quantity_rounded = 0
            except KillSwitchActivatedError:
                sizing_chain['circuit_breaker_veto'] = 0
                sizing_chain['skip_reason'] = "Kill switch activated"
                quantity_rounded = 0

            sizing_chain['final_quantity'] = quantity_rounded

            positions.append(PositionSize(
                symbol=symbol,
                quantity=quantity_rounded,
                sizing_reason=f"Vol targeting: target {self.target_volatility:.1%}, current {portfolio_vol:.1%}" + (f" - Skipped: {sizing_chain['skip_reason']}" if sizing_chain['skip_reason'] else ""),
                target_vol_contribution=self.target_volatility / len(signals),
                current_portfolio_vol=portfolio_vol,
                sizing_chain=sizing_chain
            ))

        return positions

    def _kelly_sizing(
        self,
        signals: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float,
        asset_volatilities: Dict[str, float]
    ) -> List[PositionSize]:
        """Kelly criterion position sizing."""
        positions = []

        for symbol, signal in signals.items():
            if symbol not in current_prices or symbol not in asset_volatilities:
                continue

            price = current_prices[symbol]
            vol = asset_volatilities[symbol]

            # Simplified Kelly: assume edge = signal strength, odds = 1
            # In practice, this would use historical win rate and payoff ratio
            kelly_fraction = signal  # Simplified

            # Risk-adjust Kelly
            risk_adjusted_kelly = kelly_fraction * (self.target_volatility / vol)

            allocation = portfolio_value * risk_adjusted_kelly
            quantity = allocation / price

            positions.append(PositionSize(
                symbol=symbol,
                quantity=quantity,
                sizing_reason=f"Kelly criterion: {kelly_fraction:.2f} fraction",
                target_vol_contribution=self.target_volatility,
                current_portfolio_vol=vol
            ))

        return positions

    def _estimate_portfolio_volatility(
        self,
        signals: Dict[str, float],
        asset_volatilities: Dict[str, float],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> float:
        """Estimate portfolio volatility from signals and asset data."""
        if not correlation_matrix:
            # Assume independence if no correlation matrix
            weights = np.array(list(signals.values()))
            vols = np.array([asset_volatilities.get(s, 0.5) for s in signals.keys()])

            # Normalize weights
            weights = np.abs(weights) / np.sum(np.abs(weights))

            portfolio_vol = np.sqrt(np.sum(weights**2 * vols**2))
        else:
            # Use correlation matrix
            symbols = list(signals.keys())
            weights = np.array([signals[s] for s in symbols])
            weights = np.abs(weights) / np.sum(np.abs(weights))

            vols = np.array([asset_volatilities.get(s, 0.5) for s in symbols])
            corr = correlation_matrix.loc[symbols, symbols].values

            # Portfolio variance: w' * Sigma * w where Sigma = diag(vols) * corr * diag(vols)
            vol_matrix = np.diag(vols)
            cov_matrix = vol_matrix @ corr @ vol_matrix
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)

        return portfolio_vol

    def estimate_asset_volatilities(self, price_history: Dict[str, pd.Series], window: int = 30) -> Dict[str, float]:
        """
        Estimate asset volatilities from price history.

        Args:
            price_history: Symbol -> price series
            window: Rolling window for volatility calculation

        Returns:
            Annualized volatilities per symbol
        """
        volatilities = {}

        for symbol, prices in price_history.items():
            if len(prices) < window:
                volatilities[symbol] = 0.5  # Default assumption
                continue

            # Calculate returns
            returns = prices.pct_change().dropna()

            # Rolling volatility (annualized)
            vol = returns.rolling(window).std() * np.sqrt(252)
            volatilities[symbol] = vol.iloc[-1] if not vol.empty else 0.5

        return volatilities