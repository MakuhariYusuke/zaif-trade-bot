"""
Fee model abstraction for trading costs
取引コスト用の手数料モデル抽象化
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path
import json

class FeeModel(ABC):
    """Abstract base class for fee models"""

    @abstractmethod
    def calculate_fee(self, trade_value: float, trade_type: str = 'buy') -> float:
        """
        Calculate trading fee for a given trade value

        Args:
            trade_value: Value of the trade
            trade_type: Type of trade ('buy' or 'sell')

        Returns:
            Fee amount
        """
        pass

    @abstractmethod
    def get_fee_rate(self, trade_type: str = 'buy') -> float:
        """
        Get fee rate for trade type

        Args:
            trade_type: Type of trade

        Returns:
            Fee rate as decimal
        """
        pass

class FixedFeeModel(FeeModel):
    """Fixed fee model with constant rates"""

    def __init__(self, buy_fee_rate: float = 0.001, sell_fee_rate: float = 0.001):
        """
        Initialize fixed fee model

        Args:
            buy_fee_rate: Fee rate for buy trades
            sell_fee_rate: Fee rate for sell trades
        """
        self.buy_fee_rate = buy_fee_rate
        self.sell_fee_rate = sell_fee_rate

    def calculate_fee(self, trade_value: float, trade_type: str = 'buy') -> float:
        """Calculate fee using fixed rate"""
        rate = self.get_fee_rate(trade_type)
        return abs(trade_value) * rate

    def get_fee_rate(self, trade_type: str = 'buy') -> float:
        """Get fee rate"""
        if trade_type.lower() == 'sell':
            return self.sell_fee_rate
        return self.buy_fee_rate

class TieredFeeModel(FeeModel):
    """Tiered fee model with volume-based rates"""

    def __init__(self, tiers: Optional[Dict[str, list]] = None):
        """
        Initialize tiered fee model

        Args:
            tiers: Dictionary with 'buy_tiers' and 'sell_tiers'
                  Each tier is a list of [volume_threshold, fee_rate] pairs
        """
        if tiers is None:
            # Default tiers
            tiers = {
                'buy_tiers': [
                    [0, 0.001],      # 0+ volume: 0.1%
                    [10000, 0.0008], # 10k+ volume: 0.08%
                    [50000, 0.0005], # 50k+ volume: 0.05%
                    [100000, 0.0003] # 100k+ volume: 0.03%
                ],
                'sell_tiers': [
                    [0, 0.001],
                    [10000, 0.0008],
                    [50000, 0.0005],
                    [100000, 0.0003]
                ]
            }

        self.buy_tiers = sorted(tiers['buy_tiers'], key=lambda x: x[0])
        self.sell_tiers = sorted(tiers['sell_tiers'], key=lambda x: x[0])

    def calculate_fee(self, trade_value: float, trade_type: str = 'buy') -> float:
        """Calculate fee using tiered rates"""
        rate = self.get_fee_rate(trade_type)
        return abs(trade_value) * rate

    def get_fee_rate(self, trade_type: str = 'buy') -> float:
        """
        Get fee rate based on volume tiers.

        Note:
            Currently always returns the lowest tier rate (base rate).
            In a future implementation, this method should accept a volume argument
            and return the appropriate rate based on trading volume.

        Args:
            trade_type: Type of trade ('buy' or 'sell')

        Returns:
            Fee rate as decimal
        """
        # For now, return the base rate.
        # In a real implementation, this would track volume.
        if trade_type.lower() == 'sell':
            return self.sell_tiers[0][1]
        return self.buy_tiers[0][1]

class ExchangeFeeModel(FeeModel):
    """Exchange-specific fee model"""

    def __init__(self, exchange_fees: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize exchange fee model

        Args:
            exchange_fees: Dictionary of exchange -> {'buy': rate, 'sell': rate}
        """
        if exchange_fees is None:
            exchange_fees = {
                'coincheck': {'buy': 0.0, 'sell': 0.0},
                'bitflyer': {'buy': 0.001, 'sell': 0.001},
                'binance': {'buy': 0.001, 'sell': 0.001}
            }

        self.exchange_fees = exchange_fees
        self.current_exchange = 'binance'  # Default

    def set_exchange(self, exchange: str):
        """Set current exchange"""
        if exchange in self.exchange_fees:
            self.current_exchange = exchange
        else:
            logging.warning(f"Exchange {exchange} not found, using default 'binance'")

    def calculate_fee(self, trade_value: float, trade_type: str = 'buy') -> float:
        """Calculate fee for current exchange"""
        rate = self.get_fee_rate(trade_type)
        return abs(trade_value) * rate

    def get_fee_rate(self, trade_type: str = 'buy') -> float:
        """Get fee rate for current exchange"""
        exchange_rates = self.exchange_fees.get(self.current_exchange, {'buy': 0.001, 'sell': 0.001})
        return exchange_rates.get(trade_type.lower(), 0.001)

class FeeModelFactory:
    """Factory for creating fee models"""

    @staticmethod
    def create_fee_model(model_type: str = 'fixed',
                        config: Optional[Dict[str, Any]] = None) -> FeeModel:
        """
        Create fee model instance

        Args:
            model_type: Type of fee model ('fixed', 'tiered', 'exchange')
            config: Configuration for the model

        Returns:
            FeeModel instance
        """
        if config is None:
            config = {}

        if model_type.lower() == 'fixed':
            return FixedFeeModel(
                buy_fee_rate=config.get('buy_fee_rate', 0.001),
                sell_fee_rate=config.get('sell_fee_rate', 0.001)
            )

        elif model_type.lower() == 'tiered':
            return TieredFeeModel(
                tiers=config.get('tiers')
            )

        elif model_type.lower() == 'exchange':
            return ExchangeFeeModel(
                exchange_fees=config.get('exchange_fees')
            )

        else:
            logging.warning(f"Unknown fee model type: {model_type}, using fixed")
            return FixedFeeModel()

def load_fee_model_from_config(config_path: str) -> Optional[FeeModel]:
    """Load fee model from configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        fee_config = config.get('fee_model', {})
        model_type = fee_config.get('type', 'fixed')

        return FeeModelFactory.create_fee_model(model_type, fee_config)

    except Exception as e:
        logging.error(f"Failed to load fee model from config file '{config_path}': {e}")
        return None