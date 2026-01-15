#!/usr/bin/env python3
"""
簡素化されたご褒美関数をテストするスクリプト
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ztb.trading.rewards.fast_intraday import compute_hft_reward

# テストケース1: 買いで利益
print("=" * 70)
print("Test 1: BUY profitable (position increased, price up)")
print("=" * 70)
price_prev = 1000000.0
price_now = 1001000.0  # +1,000 JPY
position_prev = 0.0
position_now = 0.1  # 新規買い
atr = 5000.0
fee_paid = 100.0
slippage_paid = 50.0
max_position = 1.0

reward, info = compute_hft_reward(
    price_prev=price_prev,
    price_now=price_now,
    position_prev=position_prev,
    position_now=position_now,
    atr=atr,
    fee_paid=fee_paid,
    slippage_paid=slippage_paid,
    holding_steps=1,
    max_position=max_position,
)
print(f"Reward: {reward:.4f}")
print(f"Info: {info}")

# テストケース2: 売りで利益
print("\n" + "=" * 70)
print("Test 2: SELL profitable (position decreased, price down)")
print("=" * 70)
price_prev = 1000000.0
price_now = 999000.0  # -1,000 JPY
position_prev = 0.1  # ロング保有
position_now = 0.0  # 売却
atr = 5000.0
fee_paid = 100.0
slippage_paid = 50.0

reward, info = compute_hft_reward(
    price_prev=price_prev,
    price_now=price_now,
    position_prev=position_prev,
    position_now=position_now,
    atr=atr,
    fee_paid=fee_paid,
    slippage_paid=slippage_paid,
    holding_steps=5,
    max_position=max_position,
)
print(f"Reward: {reward:.4f}")
print(f"Info: {info}")

# テストケース3: ホールド（何もしない）
print("\n" + "=" * 70)
print("Test 3: HOLD (no position change)")
print("=" * 70)
price_prev = 1000000.0
price_now = 1001000.0  # +1,000 JPY (関係ない)
position_prev = 0.0
position_now = 0.0  # ポジションなし
atr = 5000.0
fee_paid = 0.0
slippage_paid = 0.0

reward, info = compute_hft_reward(
    price_prev=price_prev,
    price_now=price_now,
    position_prev=position_prev,
    position_now=position_now,
    atr=atr,
    fee_paid=fee_paid,
    slippage_paid=slippage_paid,
    holding_steps=0,
    max_position=max_position,
)
print(f"Reward: {reward:.4f}")
print(f"Info: {info}")

print("\n✅ すべてのテストが成功しました")
