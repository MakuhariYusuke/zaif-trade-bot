"""
Simple Phase 2 Test

Basic functionality test for Phase 2 components.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from ztb.trading.signal.guidance.enhanced_system import EnhancedSignalGuidanceSystem
from ztb.trading.signal.common.base_classes import SignalContext

def test_basic_functionality():
    """Test basic functionality of Phase 2 system"""
    print("Testing basic Phase 2 functionality...")

    # Create simple test data
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(95, 105, 100),
        'high': np.random.uniform(100, 110, 100),
        'low': np.random.uniform(90, 100, 100),
        'close': np.random.uniform(95, 105, 100),
        'volume': np.random.uniform(1000, 2000, 100)
    })
    data.set_index('timestamp', inplace=True)

    # Create system
    system = EnhancedSignalGuidanceSystem()

    # Create context
    context = SignalContext(
        market_data=data,
        position_context={'size': 0.0, 'entry_price': None},
        portfolio_state={'cash': 10000.0, 'total_value': 10000.0},
        timestamp=data.index[-1]
    )

    try:
        # Test regime detection
        regime_result = system.regime_classifier.process_signal(context)
        print(f"✓ Regime detection: {regime_result.metadata.get('regime_type', 'unknown')}")

        # Test quality scoring
        quality_result = system.quality_scorer.process_signal(context)
        print(f"✓ Quality scoring: {quality_result.quality_score:.1f}")

        # Test enhanced system
        result = system.process_signal(context)
        print(f"✓ Enhanced system: action={result.discrete_action}, score={result.quality_score:.1f}")

        print("✓ All basic tests passed!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1)