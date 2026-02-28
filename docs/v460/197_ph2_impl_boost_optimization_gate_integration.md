# 197# boost 最適化 + balance_forced offset + Gate 8-9 統合

## 概要

196# の「今後の課題」5 項目を一括実装。fill_records 5,102 件の実データ分析、CycleGateAggregator の Gate 7→9 拡張、balance_forced の設計ギャップ修正を含む。

### A. velocity_offset_boost_factor 2.0→1.5 (データ駆動最適化)

fill_records 16 日分 (2/13–2/28) を分析し、boost 帯別 PnL を検証。

| boost 帯 | 件数 | 60s PnL (bps) | win_rate |
|---|---|---|---|
| 1.0–1.5 | 155 | **+0.47** | 51.6% |
| 1.5–2.0 | 207 | -0.13 | 47.3% |
| 2.0+ | 90 | **-0.37** | 44.6% |

**結論**: boost が大きいほど PnL が悪化。1.0–1.5 帯が唯一の正 PnL → default を 2.0→1.5 に最適化。

### B. trending_sell_offset_boost_factor 3.0→2.0 (regime 累積修正)

trending boost (×3.0) は regime_trending_offset_boost (×1.8) と累積するため、total = 3.0×1.8 = **5.4x** となり、spread 10,000 JPY 時に offset > 5,400 JPY で実質的に約定不能。

| 設定 | trending boost | regime boost | 合計 | spread 10000 時の offset |
|---|---|---|---|---|
| 旧 | 3.0 | 1.8 | 5.4x | 5,400 JPY (非現実的) |
| 新 | 2.0 | 1.8 | 3.6x | 3,600 JPY (実用的) |

### C. balance_forced + trending offset 適用 (設計ギャップ修正)

**問題**: `_check_trending_sell()` の条件 `not balance_forced` により、forced sell は trending ゲートを完全バイパス。結果、forced sell は trending_up でも offset 保護なしで発注され、AS (adverse selection) リスクに晒される。

**修正**: `balance_forced_apply_trending_offset=True` 時、forced sell に trending offset を適用 (block はしない)。

```
before:  balance_forced=True → gate bypass → offset_mult=None → 通常価格
after:   balance_forced=True → offset_mult=2.0 → 保守的価格 (block しない)
```

### D. Gate 8: narrow_spread_pause Gate 統合 (旧 B3)

fill_cycle_executor 内の narrow_spread_pause 判定 (B3) を CycleGateAggregator に移管。cached spread を使い、spread < threshold で cycle skip。

- executor 側は defense-in-depth として残留 (リアルタイム spread 使用)
- Gate 判定 → orchestrator で `asyncio.sleep(pause_sec)` + continue

### E. Gate 9: maker_price 事前チェック (D1-D3)

maker_price.compute() が ValueError を raise するケースを Gate で事前検出。

| チェック | 条件 | blocking_reason |
|---|---|---|
| D1 | spread < min_spread_jpy | `spread_too_narrow` |
| D3 | sell + spread > sell_max_spread_jpy | `sell_guard_reject` |

実際の判定は executor の try/except が最終防衛線。Gate は cached spread で早期検出。

> **修正**: blocked=True だとフィードバックループ（Gate→compute未実行→キャッシュ未更新→永久デッドロック）が発生。advisory-only (blocked=False) に変更済み。

### F. maker_price._last_spread キャッシュ + public API

`compute()` 内で算出された spread を `_last_spread` にキャッシュ。
public property `last_spread` / `last_mid_price` を追加し、orchestrator から安全にアクセス。

## 変更ファイル

### 1. `scripts/v460/lib/fill_config.py`
- `velocity_offset_boost_factor: float = 1.5` (2.0→1.5)
- `trending_sell_offset_boost_factor: float = 2.0` (3.0→2.0)
- `balance_forced_apply_trending_offset: bool = True` (新フィールド)
- YAML マッピング追加 (loss_control セクション)

### 2. `scripts/v460/lib/cycle_gate_aggregator.py`
- `_check_trending_sell()`: balance_forced + trending → offset 適用ブロック追加
- `_check_narrow_spread()`: Gate 8 新規メソッド
- `_check_maker_price_precheck()`: Gate 9 新規メソッド
- `evaluate()`: Gate 8-9 呼び出し追加
- `_GATE_TO_CANCEL_REASON`: 3 エントリ追加

### 3. `scripts/v460/lib/fill_loop_orchestrator.py`
- `evaluate()` 呼び出しに `spread_jpy` / `mid_price` パラメータ追加
- narrow_spread_pause 時の `asyncio.sleep()` 処理追加

### 4. `scripts/v460/lib/maker_price.py`
- `__slots__` に `_last_spread` 追加
- `compute()` 内で `self._last_spread = spread` キャッシュ
- `last_spread` / `last_mid_price` public property 追加

### 5. `configs/v460/fill_test.yaml`
- `velocity_offset_boost_factor: 1.5` (2.0→1.5)
- `trending_sell_offset_boost_factor: 2.0` (3.0→2.0)
- `balance_forced_apply_trending_offset: true` (新規)

### 6. テストファイル修正
- `test_197_boost_optimization_gate_integration.py` (新規, 45 テスト)
- `test_194_cycle_gate.py`: gate count 7→9
- `test_195_velocity_b1_soft.py`: velocity default 2.0→1.5
- `test_196_velocity_proportional_trending_soft.py`: YAML trending 3.0→2.0
- `test_155_hindsight_review.py`: source scan range 400→1200

## テスト結果

```
2652 passed, 0 failed
  - 新規 45 テスト (test_197)
  - 既存テスト 全通過 (2607 テスト)
```

## 今後の課題

1. **narrow_spread_pause 有効化**: Gate 8 統合済みだが YAML `enabled: false` → AB テスト後に有効化
2. **balance_forced 在庫管理**: 977 回/runtime の forced sell → 在庫管理の根本改善が必要
3. **trending_up_sell 固有の問題**: データ分析で avg -2.77 bps → regime 別 boost 差分が有効か検証
4. **VG (Volatility Guard) の効果確認**: VG triggered +0.07 bps vs NOT triggered -0.87 bps → VG 強化の余地
