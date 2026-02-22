# Action Constants and Continuous Action Mapping

## Overview

This document describes the action constants used in the trading system and the continuous action mapping implementation.

## Action Constants

The trading system uses discrete action constants for BUY, HOLD, and SELL operations:

```python
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = -1  # Changed from 2 to -1 for continuous mapping consistency
```

## Continuous Action Mapping

The system supports continuous action mapping where actions are represented as floating-point values:

- **BUY**: 1.0 → ACTION_BUY (1)
- **HOLD**: 0.0 → ACTION_HOLD (0)
- **SELL**: -1.0 → ACTION_SELL (-1)

This mapping provides:
- Symmetric action space around zero
- Intuitive directionality (positive = buy, negative = sell)
- Better gradient flow for reinforcement learning

## Legacy Compatibility

For backward compatibility with existing models and configurations that use ACTION_SELL=2:

### normalize_action() Function

```python
def normalize_action(action: int) -> int:
    """Convert legacy ACTION_SELL=2 to current ACTION_SELL=-1."""
    if action == 2:
        return -1
    return action
```

### get_action_name() Function

```python
def get_action_name(action: int) -> str:
    """Get human-readable action name with legacy support."""
    normalized = normalize_action(action)
    return ACTION_NAMES.get(normalized, "UNKNOWN")
```

## Implementation Details

### Files Updated

1. **`ztb/trading/constants.py`**
   - ACTION_SELL changed from 2 to -1
   - Added normalize_action() function
   - Updated ACTION_NAMES dictionary
   - Added get_action_name() function

2. **`ztb/trading/live_trader/live_trader.py`**
   - Imports normalize_action from constants
   - Applies normalization in _predict_action() method

3. **`ztb/trading/environment/components/reward_calculator.py`**
   - Imports normalize_action from constants
   - Applies normalization in _convert_continuous_to_discrete_action()

4. **Test Files**
   - Updated unit tests to expect ACTION_SELL=-1
   - Verified legacy compatibility

## Migration Guide

### For Existing Models
- Models trained with ACTION_SELL=2 will continue to work
- The normalize_action() function automatically converts legacy values
- No manual intervention required

### For New Development
- Use ACTION_SELL = -1 in all new code
- Prefer continuous action mapping (-1.0, 0.0, 1.0) for RL training
- Use normalize_action() when interfacing with legacy components

## Testing

Run the following to verify the implementation:

```bash
# Test action constants
python -c "from ztb.trading.constants import *; print(f'ACTION_SELL: {ACTION_SELL}')"

# Test legacy compatibility
python -c "from ztb.trading.constants import normalize_action; print(f'normalize_action(2): {normalize_action(2)}')"

# Run unit tests
python -m pytest tests/unit/trading/live/test_live_trade.py::TestBugFixDocumentation::test_action_constants_defined -v
```

## Benefits

1. **Consistency**: Action mapping now aligns with continuous action space
2. **Backward Compatibility**: Existing models continue to work seamlessly
3. **Maintainability**: Clear separation between legacy and current implementations
4. **Future-Proof**: Foundation for advanced continuous action policies</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\docs\action_constants.md
