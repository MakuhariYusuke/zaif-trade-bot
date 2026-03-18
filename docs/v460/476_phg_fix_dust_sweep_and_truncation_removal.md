# 476# Dust sweep 修正 + 0.001 単位切り捨て廃止 + 残高連動ロット

## 概要
- Coincheck は satoshi 精度 (1e-8) を許容する設計 → 0.001 BTC 単位の切り捨てを廃止
- dust sweep が lot_scale チェーンに上書きされる不具合を修正
- 残高連動の動的ロットサイジング（buy 側: JPY 残高ベース）

## 根本原因

### dust sweep が機能しない問題
1. `_maybe_dust_sweep` が `current_lot = btc_free` (全額) に設定
2. `fill_cycle_executor.py` の `246# cooldown_release` lot_scale=0.30 が乗算
3. `0.00197546 × 0.30 = 0.000593 → max(0.001) = 0.001` — 全額売却が 1mBTC に縮小
4. 残余 (0.00097546) が micro-dust として永続ループを形成

### 0.001 単位切り捨ての問題
- `int(x / min_order_btc) * min_order_btc` パターンが 5 箇所に存在
- Coincheck の `quantity_precision: 8` (satoshi) を活かせていなかった

## 変更点

### balance_checker.py
- **sell lot 縮小**: `int(max_base / min_order) * min_order` → `round(max_base, 8)`
- **buy lot 縮小**: 同上パターン → `round(affordable_base, 8)`
- **apply_lot_floor**: `int` 切り捨て → `round(, 8)` + min guard
- **_maybe_dust_sweep**: regime_mult 対応。実効ロット (base × mult) で比較
  - btc_free > effective_lot: 全額売却ロットに拡張
  - effective_lot ≈ btc_free かつ lot ≠ order_quantity: 保護モード発動
- **sell 側 balance_lot 拡大** 削除: dust sweep と重複するため
- **buy 側 balance_lot 拡大** 追加: JPY 残高 → max_lot まで動的拡大

### fill_cycle_executor.py
- lot_scale チェーン全体を `if not _dust_active:` でガード
- dust_sweep_active 時は lot_scale (alert_mode, Recovery, DD soft, cooldown_release) をバイパス

### order_monitor.py
- 再価格設定時の lot 切り捨て: `int` → `round(, 8)`

## テスト更新
- `test_dust_sweep.py`: 新ロジックに合わせ期待値更新 (22 tests ✓)
- `test_145_structural_fixes.py`: regime_mult テストに `dust_sweep_enabled=False` 追加
  - buy 側 lot 期待値: 0.003 → 0.00396040 (satoshi 精度)

## 追加修正 (476# 第2コミット)

### _scale_lot フロア修正
- `fill_cycle_executor.py`: `_min_lot = config.order_quantity` → `config.min_order_btc`
- **問題**: order_quantity と min_order_btc が同値の場合、DD soft / alert_mode / cooldown の
  スケールダウンが `max(order_quantity, lot * scale)` で切り捨てられ実質無効化
- **修正**: 取引所最小 (min_order_btc) をフロアにすることで、order_quantity を引き上げた際に
  スケールダウンが正しく機能する

### 残高連動ロットの全体像
| 側面 | 拡大機構 | フロア |
|------|----------|--------|
| sell | `_maybe_dust_sweep` (btc_free > effective_lot → 全額売却) | min_order_btc |
| buy  | 476# balance_lot (JPY → max_lot まで動的拡大) | min_order_btc |
| 共通 | `_scale_lot` チェーン (DD soft, alert, cooldown) | min_order_btc (修正後) |
