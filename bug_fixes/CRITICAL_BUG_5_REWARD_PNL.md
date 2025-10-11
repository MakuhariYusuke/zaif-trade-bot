# 🚨 CRITICAL BUG #5: Reward Calculation Uses Wrong PnL

## 📋 Summary
The reward calculation was using **total unrealized PnL** instead of **trade-specific PnL**, causing the agent to receive rewards based on market movements rather than trading decisions.

## 🔍 Root Cause
In `environment.py` line 817:
```python
# WRONG: Using total unrealized PnL
pnl = unrealized_pnl
```

This means:
- Agent gets rewarded/penalized for market movements, not for trading skill
- Even HOLD actions could generate large positive/negative rewards based on open positions
- Reward signal completely disconnected from action consequences

## ⚠️ Impact
**CRITICAL - This completely breaks reinforcement learning:**
- Agent cannot learn cause-and-effect between actions and rewards
- Holding a winning position generates continuous positive rewards without any action
- The reward function teaches the wrong behavior (market timing instead of trading skill)

## ✅ Fix Applied

### 1. Modified `PositionManager.execute_action()` to return trade PnL
**File:** `ztb/trading/environment/components/position_manager.py`

**Before:**
```python
def execute_action(self, action: int, current_step: int, min_holding_period: int = 0) -> None:
    if action == 0:  # HOLD
        self._consecutive_trade_steps = 0
        return
    
    if action == 1:  # BUY
        if self.position < 0:
            self.close_position()  # No return value captured
```

**After:**
```python
def execute_action(self, action: int, current_step: int, min_holding_period: int = 0) -> float:
    """Returns: trade_pnl - PnL from closing position in this action"""
    if action == 0:  # HOLD
        self._consecutive_trade_steps = 0
        return 0.0
    
    trade_pnl = 0.0
    
    if action == 1:  # BUY
        if self.position < 0:
            trade_pnl = self.close_position()  # Capture realized PnL
    
    return trade_pnl
```

### 2. Captured trade_pnl in Environment
**File:** `ztb/trading/environment/environment.py`

**Before:**
```python
self.position_manager.execute_action(action, self.current_step, min_holding_period)
# ...later...
pnl = unrealized_pnl  # WRONG
```

**After:**
```python
trade_pnl = self.position_manager.execute_action(action, self.current_step, min_holding_period)
# ...later...
pnl = trade_pnl  # CORRECT - only reward for actual trades
```

## 📊 Correct Behavior Now
- **HOLD action**: `pnl = 0.0` (no trade = no reward from PnL component)
- **BUY closing Short**: `pnl = realized_pnl_from_close` (actual profit/loss)
- **SELL closing Long**: `pnl = realized_pnl_from_close` (actual profit/loss)
- **BUY/SELL opening new position**: `pnl = 0.0` (position not yet closed)

The reward calculator still receives portfolio_value which includes unrealized PnL for context, but the primary `pnl` parameter now correctly represents **only the PnL from the current action's trade**.

## 🧪 Testing Required
This fix changes the fundamental reward signal. Need to:
1. ✅ Verify `execute_action()` returns correct PnL for all action types
2. ✅ Verify HOLD actions receive `pnl=0.0`
3. ✅ Verify position-closing actions receive realized PnL
4. ⚠️ **RETRAIN ALL MODELS** - existing models were trained with wrong reward signal
5. ⚠️ Compare new training behavior vs old (expect less erratic reward curves)

## 📝 Related Issues
This is the **5th critical bug** found in deep investigation after user's "石橋を叩いて渡る" warning.

Previous bugs:
1. MaskablePPO action_masks ignored in core training
2. Ensemble missing mask_provider enforcement  
3. min_holding_period + allow_reverse interaction
4. 4 evaluation scripts missing predict_with_masks

All these bugs suggest systematic issues from incomplete PPO→MaskablePPO migration and architectural debt.

## 🎯 Lesson Learned
**Reward function correctness is CRITICAL for RL.** Always verify:
- Reward signal matches intended behavior
- PnL attribution is accurate (realized vs unrealized)
- Actions have proper causal connection to rewards

This bug would have been caught earlier with:
- Unit tests for reward calculation scenarios
- Reward curve analysis during training
- Ablation studies on reward components
