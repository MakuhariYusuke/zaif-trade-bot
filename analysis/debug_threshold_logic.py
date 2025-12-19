
import logging
from ztb.trading.environment.components.threshold_manager import ThresholdManager
from ztb.types.common import ConfigDict

# Mock Config
class MockConfig:
    def __init__(self):
        self.continuous_to_discrete_threshold = 0.01
        self.adaptive_threshold_mode = True
        self.threshold_volatility_multiplier = 1.0
        self.min_action_threshold = 0.001
        self.max_action_threshold = 1.0
        self.regime_detection_window = 50
        self.performance_memory_size = 100
        self.trend_detection_threshold = 0.001
        self.volatility_detection_threshold = 0.02
        self.dynamic_threshold_mode = "fixed"

def test_threshold_behavior():
    config = MockConfig()
    tm = ThresholdManager(config)
    
    # Base case
    base_threshold = 0.01
    volatility = 0.01
    price = 100.0
    
    print("=== ThresholdManager Investigation ===")
    
    # Case 1: Regime is None (The "Buggy" State)
    t_none = tm.get_threshold(volatility, price, regime=None)
    print(f"Regime=None: Threshold={t_none:.6f} (Multiplier ~1.0)")
    
    # Case 2: Regime is "Unknown" (Explicit Unknown)
    t_unknown = tm.get_threshold(volatility, price, regime="Unknown")
    print(f"Regime='Unknown': Threshold={t_unknown:.6f} (Multiplier ~1.0)")
    
    # Case 3: Regime is "CONSOLIDATION" (The "Fixed" State - Before 1.0 fix)
    # Note: We need to simulate the logic BEFORE my latest 1.0 fix.
    # Since I already applied the fix in the file, this test will show the FIXED behavior (1.0).
    # But we can infer the previous behavior (it was 10.0).
    t_consolidation = tm.get_threshold(volatility, price, regime="CONSOLIDATION")
    print(f"Regime='CONSOLIDATION': Threshold={t_consolidation:.6f}")
    
    # Case 4: Regime is "STRONG_BULL_TREND"
    t_bull = tm.get_threshold(volatility, price, regime="STRONG_BULL_TREND")
    print(f"Regime='STRONG_BULL_TREND': Threshold={t_bull:.6f}")

    # Case 5: Regime is "BREAKDOWN" (New Sell Favorable)
    # Buy Threshold (positive base)
    t_breakdown_buy = tm.get_threshold(volatility, price, regime="BREAKDOWN", base_value=0.01)
    print(f"Regime='BREAKDOWN' (Buy): Threshold={t_breakdown_buy:.6f}")
    
    # Sell Threshold (negative base)
    t_breakdown_sell = tm.get_threshold(volatility, price, regime="BREAKDOWN", base_value=-0.01)
    print(f"Regime='BREAKDOWN' (Sell): Threshold={t_breakdown_sell:.6f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_threshold_behavior()
