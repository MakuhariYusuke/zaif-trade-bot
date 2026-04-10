# 725# Task 3: compat.py 移行シム削除

## 背景
`ztb/trading/live/orders/compat.py` は `ztb.trading.orders.state_machine` への
移行シムとして作成された。TODO コメントで「移行完了後に削除」と明記されている。

## 現状

### compat.py (全文)
```python
"""
Compatibility shim for order_state module.
Re-exports symbols from the canonical ztb.trading.order_state_machine module.
This shim allows gradual migration from old imports.
TODO: Remove this shim after migration to ztb.trading.order_state_machine is complete.
"""
from ztb.trading.orders.state_machine import (
    IdempotencyManager, OrderData, OrderEvent, OrderState,
    OrderStateMachine, get_idempotency_manager,
)
__all__ = [...]
```

### 残存インポート（1箇所のみ）
```
ztb/trading/live/orders/state.py:15: from ztb.trading.live.orders.compat import OrderData
```

## 要件

### Step 1: インポート先を正規モジュールに変更
```python
# ztb/trading/live/orders/state.py line 15
# Before
from ztb.trading.live.orders.compat import OrderData
# After
from ztb.trading.orders.state_machine import OrderData
```

### Step 2: compat.py 削除
`ztb/trading/live/orders/compat.py` を削除。

### Step 3: 他のインポート確認
以下のコマンドで残存インポートがないことを確認:
```bash
grep -r "ztb.trading.live.orders.compat" --include="*.py" .
grep -r "from ztb.trading.live.orders import compat" --include="*.py" .
```

## テスト対象ファイル
- `ztb/trading/live/orders/state.py`
- `ztb/trading/live/orders/compat.py` (削除)

## 検証
```bash
python -m pytest tests/ -x --tb=short -q -k "order"
mypy ztb/trading/live/orders/ --config-file mypy.ini
python -c "from ztb.trading.orders.state_machine import OrderData; print('OK')"
```

## 制約
- `ztb.trading.orders.state_machine` の正規モジュールは一切変更不可
- 削除前に全テストパスを確認すること（grep + pytest）
- `__init__.py` に compat re-export がある場合はそれも確認すること
