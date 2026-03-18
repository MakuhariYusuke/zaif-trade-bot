# 474# Triple Bugfix + Orphan Process Cleanup

## 概要
473# 検証結果に基づき、3つのバグを修正し、PC重い問題（retrain_scheduler 8重起動）を解決。

## 修正一覧

### P0: Sell パラメータ修正 (`configs/v460/fill_test.yaml`)
- `offset_ceiling_ratio_sell`: 0.50 → **0.20**
- `sell_guard.offset_floor`: 0.30 → **0.05**

**根拠**: ratio=0.50 では `sell_price = mid + spread*(0.5 - 0.50) = mid+0` で mid 上に発注。
936 JPY の平均逆行に対し buffer=0。ratio=0.20 では `mid + spread*0.30 = mid+718 JPY` の buffer を確保。
floor=0.05 でパイプラインが自然な offset 分布を生成可能に。

### P1: `_recalc_price_with_new_offset` half-spread バグ修正 (`pre_order_adjustments.py`)
旧: `mid_est = price ∓ spread * ratio / 2` (half-spread)
新: `delta = spread * (old_ratio - new_ratio)` → 直接差分計算

**根拠**: ベース公式は `sell = best_ask - spread * ratio` = `mid + spread*(0.5 - ratio)` で full-spread。
旧式は half-spread を使い、mid 推定にエラーを含んでいた（sell: +278 JPY protective, buy: +360 JPY aggressive）。
新式は mid 推定を介さず、offset 差分のみで正確に再計算。

### P2: Micro-timeout re-quote 公式修正 (`fill_cycle_executor.py`)
旧: `order_price = mid * (1 ± effective_offset_ratio)`
新: `order_price = mid + spread * (0.5 - ratio)` (sell), `mid + spread * (ratio - 0.5)` (buy)

**根拠**: 旧式は ratio (~0.20) を価格割合と誤解釈 → 11.8M × 0.20 = 2.36M JPY のズレ。
実際には ratio は spread に対する割合。現時点で 0 firings のため dormant だが、今後 micro-timeout 有効化時の地雷を除去。

### P3: retrain_scheduler 多重起動防止 (`retrain_scheduler.py`, `hot_swap_restart.ps1`)
- `retrain_scheduler.py` に lockfile 機構追加（`logs/retrain_scheduler.lock`）
- PID 生存チェックによる stale lock 自動回収
- `hot_swap_restart.ps1` に `retrain_scheduler.lock` の除去ステップ追加
- **発見**: retrain_scheduler が 8 プロセス重複起動 → 全 kill で PC 負荷解消

## テスト結果
- 2212 passed, 128 skipped, 0 failed (220.51s)
- `test_346_pre_order_adjustments.py` の期待値を新公式に更新

## 影響範囲
| 修正 | 即効性 | 既存 fill への影響 |
|------|--------|-------------------|
| P0 sell params | **高** | 次回 sell から mid+718 JPY buffer |
| P1 _recalc | 中 | final clamp 通過時の再計算が正確に |
| P2 micro-timeout | 低 | 現在 dormant (0 firings) |
| P3 orphan cleanup | **高** | PC 負荷即時改善 |
