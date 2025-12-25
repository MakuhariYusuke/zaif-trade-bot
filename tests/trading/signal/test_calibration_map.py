import pytest
import math
from ztb.trading.signal.calibration_map import CalibrationMap, CalibrationGate
from ztb.trading.signal.types import FusedSignal
from ztb.trading.types import MarketState

class TestCalibrationMap:
    @pytest.fixture
    def calibration_map(self):
        config = {
            'ewma_tau': 10.0,
            'n_min': 5.0
        }
        return CalibrationMap(config)

    def test_update_and_decay(self, calibration_map):
        """Test that stats are updated and decayed correctly."""
        regime = "trending"
        action = 0.8 # Strong Buy
        
        # Initial update
        calibration_map.update(regime, action, gross_pnl=100.0, step=1)
        stats = calibration_map.get_stats(regime, action)['l1']
        
        assert stats['n_eff'] > 0
        assert stats['avg_win'] == pytest.approx(100.0, rel=1e-5)
        
        # Second update after some time (decay)
        # tau=10. dt=10 -> decay = exp(-1) approx 0.368
        calibration_map.update(regime, action, gross_pnl=-50.0, step=11)
        stats_new = calibration_map.get_stats(regime, action)['l1']
        
        # Check if n_eff increased but less than 2 (due to decay)
        # First weight decayed to ~0.368, new weight 1.0. Sum ~1.368.
        assert 1.0 < stats_new['n_eff'] < 2.0
        assert stats_new['avg_loss'] == pytest.approx(50.0, rel=1e-5)

    def test_fallback_logic(self, calibration_map):
        """Test fallback from L1 to L2/L3."""
        regime = "trending"
        action = 0.8 # Strong Buy
        
        # No data yet -> n_eff = 0 -> should use fallback (L3 global)
        bundle = calibration_map.get_stats(regime, action)
        assert bundle['l1']['n_eff'] == 0
        # L3 is also empty initially
        
        # Add data to L3 (Global) via another regime/action
        calibration_map.update("ranging", -0.8, 100.0, step=1)
        
        bundle = calibration_map.get_stats(regime, action)
        # L1 empty, L2 empty, L3 has data
        # Fallback logic: if L2 n_eff < n_min, use L3.
        assert bundle['fallback']['n_eff'] > 0 # Should pick up global stats

class TestCalibrationGate:
    @pytest.fixture
    def gate(self):
        map_config = {'ewma_tau': 10.0, 'n_min': 5.0}
        cmap = CalibrationMap(map_config)
        
        gate_config = {
            'fee_rate': 0.001,
            'c_spread': 0.0, # Zero spread for easier calc
            'c_vol': 0.0,
            'c_imp': 0.0,
            'order_size_btc': 0.01
        }
        return CalibrationGate(gate_config, cmap)

    def test_evaluate_with_order_size(self, gate):
        """Test evaluate uses passed order size."""
        # Setup stats to be positive
        # Update map to have high win rate
        for i in range(10):
            gate.calibration_map.update("trending", 0.8, 1000.0, step=i)
            
        fused: FusedSignal = {
            'rl_action': 0.8,
            'regime': 'trending',
            'pattern_score': 0.0
        }
        
        market: MarketState = {
            'high': 100.0, 'low': 100.0, 'close': 100.0,
            'atr': 1.0, 'volume': 1000.0, 'timestamp': None
        }
        
        # Case 1: Default size (0.01) -> Cost small
        res1 = gate.evaluate(fused, market)
        
        # Case 2: Huge size -> Cost large (if impact enabled, but here c_imp=0)
        # Let's enable impact for this test
        gate.c_imp = 1000.0
        gate.gamma = 1.0
        
        res2 = gate.evaluate(fused, market, order_size=100.0)
        
        assert res2['cost'] > res1['cost']

    def test_fail_closed_missing_data(self, gate):
        """Test that missing market data prevents entry."""
        fused: FusedSignal = {
            'rl_action': 0.8,
            'regime': 'trending',
            'pattern_score': 0.0
        }
        
        # Missing high/low/atr/volume (zeros)
        market: MarketState = {
            'high': 0.0, 'low': 0.0, 'close': 100.0,
            'atr': 0.0, 'volume': 0.0, 'timestamp': None
        }
        
        res = gate.evaluate(fused, market)
        
        assert res['should_enter'] is False
        assert res['cost'] == float('inf')

        # Test missing close
        market_no_close: MarketState = {
            'high': 100.0, 'low': 90.0, 'close': 0.0,
            'atr': 10.0, 'volume': 1000.0, 'timestamp': None
        }
        res_no_close = gate.evaluate(fused, market_no_close)
        assert res_no_close['cost'] == float('inf')

    def test_fail_closed_nan_inf(self, gate):
        """Test that NaN or Inf values trigger fail-closed."""
        fused: FusedSignal = {'rl_action': 0.8, 'regime': 'trending', 'pattern_score': 0.0}
        
        # NaN in ATR
        market_nan: MarketState = {
            'high': 100.0, 'low': 90.0, 'close': 95.0,
            'atr': float('nan'), 'volume': 1000.0, 'timestamp': None
        }
        assert gate.evaluate(fused, market_nan)['cost'] == float('inf')
        
        # Inf in High
        market_inf: MarketState = {
            'high': float('inf'), 'low': 90.0, 'close': 95.0,
            'atr': 10.0, 'volume': 1000.0, 'timestamp': None
        }
        assert gate.evaluate(fused, market_inf)['cost'] == float('inf')

    def test_n_min_zero_guard(self, gate):
        """Test that n_min=0 does not cause division by zero."""
        gate.calibration_map.n_min = 0.0
        
        fused: FusedSignal = {
            'rl_action': 0.8,
            'regime': 'trending',
            'pattern_score': 0.0
        }
        market: MarketState = {
            'high': 100.0, 'low': 90.0, 'close': 95.0,
            'atr': 10.0, 'volume': 1000.0, 'timestamp': None
        }
        
        # Should not raise ZeroDivisionError
        res = gate.evaluate(fused, market)
        assert res['lambda_val'] == 1.0

    def test_blending_logic(self, gate):
        """Test EV blending between L1 and Fallback."""
        # Setup: L1 has n_eff = 2.5, n_min = 5.0 -> lambda = 0.5
        gate.calibration_map.n_min = 5.0
        
        # Mock get_stats to return controlled values
        # We can't easily mock the internal method without patching, 
        # so let's just inject stats into the map manually or trust the integration.
        # Let's use the real map but force n_eff via internal manipulation for precise test.
        
        regime = "trending"
        action = 0.8
        bin_key = f"{regime}_Strong_Buy"
        
        # Inject L1 stats
        gate.calibration_map._init_stats(bin_key)
        s = gate.calibration_map.stats[bin_key]
        # To get n_eff = 2.5: n_eff = sum_w^2 / sum_w_sq.
        # If all w=1, n_eff = N. So let's add 2.5 weights? No, integer steps.
        # Let's just set the values directly since we are testing Gate logic, not Map logic here.
        # But Gate calls Map.get_stats.
        
        # Let's just use the fact that we can update.
        # Update 3 times. n_eff approx 3 (if no decay).
        for i in range(3):
            gate.calibration_map.update(regime, action, 100.0, i)
            
        # Check n_eff
        stats = gate.calibration_map.get_stats(regime, action)['l1']
        n_eff = stats['n_eff']
        # n_eff should be around 3.
        
        # Set n_min to 2 * n_eff to get lambda ~ 0.5
        gate.calibration_map.n_min = n_eff * 2.0
        
        fused: FusedSignal = {'rl_action': action, 'regime': regime, 'pattern_score': 0.0}
        market: MarketState = {
            'high': 100.0, 'low': 90.0, 'close': 95.0,
            'atr': 10.0, 'volume': 1000.0, 'timestamp': None
        }
        
        res = gate.evaluate(fused, market)
        
        assert 0.4 < res['lambda_val'] < 0.6
        assert res['ev'] == res['lambda_val'] * res['ev_l1'] + (1.0 - res['lambda_val']) * res['ev_fb']
        assert 'stats_fallback' in res
