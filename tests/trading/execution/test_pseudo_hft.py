import pytest
from ztb.trading.execution.pseudo_hft import PseudoHFTExecutionModel
from ztb.trading.types import MarketState

class TestPseudoHFTExecutionModel:
    @pytest.fixture
    def model(self):
        config = {
            'c_spread': 0.5,
            'c_vol': 0.2,
            'c_imp': 0.5,
            'gamma': 0.5,
            'min_volume': 1.0,
            'latency_sec': 1.0
        }
        return PseudoHFTExecutionModel(config)

    def test_interface_compatibility(self, model):
        """Test calling simulate_execution with the base class signature."""
        # Base signature: action_type, requested_price, requested_size, current_atr, current_volume, market_regime
        result = model.simulate_execution(
            action_type='buy',
            requested_price=1000000.0,
            requested_size=0.1,
            current_atr=1000.0,
            current_volume=100.0,
            market_regime='trending'
        )
        
        assert result.executed_size == 0.1
        assert result.executed_price > 1000000.0 # Buy should have positive slippage (higher price)
        assert result.slippage_rate > 0.0

    def test_market_state_compatibility(self, model):
        """Test calling simulate_execution with MarketState dict."""
        market_data: MarketState = {
            'high': 1001000.0,
            'low': 999000.0,
            'close': 1000000.0,
            'atr': 1000.0,
            'volume': 100.0,
            'timestamp': 1234567890
        }
        
        result = model.simulate_execution(
            action_type='sell',
            requested_price=1000000.0,
            requested_size=0.1,
            market_data=market_data
        )
        
        assert result.executed_price < 1000000.0 # Sell should have negative slippage (lower price)
        assert result.timestamp == 1234567890

    def test_invalid_action_type(self, model):
        """Test that invalid action type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid action_type"):
            model.simulate_execution(
                action_type='hold',
                requested_price=1000000.0,
                requested_size=0.1
            )

    def test_slippage_components(self, model):
        """Test that slippage includes spread, vol, and impact."""
        # 1. Spread only (zero vol, zero impact via huge volume/small size)
        # Actually hard to zero out impact completely due to formula, but can minimize.
        # Let's calculate expected manually.
        
        market_data: MarketState = {
            'high': 100.0,
            'low': 90.0, # Spread = 10
            'atr': 5.0,
            'volume': 10000.0,
            'close': 95.0,
            'timestamp': None
        }
        
        # Config: c_spread=0.5, c_vol=0.2, c_imp=0.5, gamma=0.5, latency=1.0
        # Spread Proxy = 0.5 * (100 - 90) = 5.0
        # Vol Risk = 0.2 * 5.0 * sqrt(1/60) = 1.0 * 0.129 = 0.129
        # Impact = 0.5 * 5.0 * (0.1 / 10000)^0.5 = 2.5 * sqrt(0.00001) = 2.5 * 0.00316 = 0.0079
        
        expected_slippage = 5.0 + (0.2 * 5.0 * (1.0/60.0)**0.5) + (0.5 * 5.0 * (0.1/10000.0)**0.5)
        
        actual_slippage = model.calculate_slippage_one_way(market_data, 0.1)
        
        assert abs(actual_slippage - expected_slippage) < 1e-6

    def test_missing_market_data_proxy(self, model):
        """Test that missing market_data uses ATR to proxy spread."""
        # If we pass current_atr but no market_data dict
        atr = 10.0
        res = model.simulate_execution(
            action_type='buy',
            requested_price=100.0,
            requested_size=0.1,
            current_atr=atr,
            current_volume=1000.0
        )
        
        # Spread proxy should be c_spread * (high - low)
        # We set high = price + 0.5*atr, low = price - 0.5*atr
        # So high - low = atr
        # Spread proxy = c_spread * atr = 0.5 * 10.0 = 5.0
        
        # We can't easily inspect the internal spread component, but we can check total slippage.
        # Total = Spread + Vol + Impact
        # Spread = 5.0
        # Vol = 0.2 * 10.0 * sqrt(1/60) = 2.0 * 0.129 = 0.258
        # Impact = 0.5 * 10.0 * (0.1/1000)^0.5 = 5.0 * 0.01 = 0.05
        
        expected_slippage = 5.0 + (0.2 * 10.0 * (1.0/60.0)**0.5) + (0.5 * 10.0 * (0.1/1000.0)**0.5)
        
        actual_slippage = res.executed_price - 100.0
        
        assert abs(actual_slippage - expected_slippage) < 1e-6

    def test_data_glitch_high_low(self, model):
        """Test that high < low (data glitch) is handled by clamping spread to 0."""
        market_data: MarketState = {
            'high': 90.0, # Glitch: high < low
            'low': 100.0,
            'atr': 5.0,
            'volume': 10000.0,
            'close': 95.0,
            'timestamp': None
        }
        
        # Spread proxy should be 0, not negative
        slippage = model.calculate_slippage_one_way(market_data, 0.1)
        
        # Expected: Spread(0) + Vol + Impact
        expected_slippage = 0.0 + (0.2 * 5.0 * (1.0/60.0)**0.5) + (0.5 * 5.0 * (0.1/10000.0)**0.5)
        
        assert abs(slippage - expected_slippage) < 1e-6

    def test_negative_order_size(self, model):
        """Test that negative order size (sell) uses abs() for impact calculation."""
        market_data: MarketState = {
            'high': 100.0, 'low': 90.0, 'atr': 5.0, 'volume': 10000.0, 'close': 95.0, 'timestamp': None
        }
        
        # Impact should be same for +0.1 and -0.1
        slip_pos = model.calculate_slippage_one_way(market_data, 0.1)
        slip_neg = model.calculate_slippage_one_way(market_data, -0.1)
        
        assert slip_pos == slip_neg

    def test_missing_market_data_zero_atr(self, model):
        """Test fallback when market_data is missing AND atr is 0."""
        # Should use fallback ATR (e.g. 0.05% of price)
        price = 10000.0
        res = model.simulate_execution(
            action_type='buy',
            requested_price=price,
            requested_size=0.1,
            current_atr=0.0, # Missing ATR
            current_volume=1000.0
        )
        
        # Fallback ATR = 10000 * 0.0005 = 5.0
        # Spread proxy = 0.5 * 5.0 = 2.5
        # Vol risk = 0.2 * 5.0 * sqrt(1/60) = 1.0 * 0.129 = 0.129
        # Impact = 0.5 * 5.0 * (0.1/1000)^0.5 = 2.5 * 0.01 = 0.025
        
        expected_slippage = 2.5 + 0.129 + 0.025
        actual_slippage = res.executed_price - price
        
        # Allow some tolerance as fallback logic might change slightly
        assert actual_slippage > 0.0
        assert abs(actual_slippage - expected_slippage) < 0.1
