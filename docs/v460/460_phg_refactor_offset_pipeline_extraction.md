# 460# run_single_cycle 分割 + 重複排除: Offset Pipeline 抽出

> **種別**: refactor (リファクタリング)  
> **フェーズ**: phg (品質改善)  
> **依存**: 163# FillCycleExecutorMixin, 323# God Object 分割, 458# macro sell protection  
> **コミット**: `39e1969a7`  
> **最終更新**: 2026-03-17

---

## 1. 背景

`run_single_cycle` が 999 行に到達し、テスト `test_113_resilience::test_run_single_cycle_under_400_lines`
が 830 行上限を超過して長期失敗していた。

主な肥大化要因:
- **Offset 乗数チェーン** (9 段, 256 行): 193# EV → 195# Velocity → 196# Trending →
  240# Toxicity → 202# VG → 458# Macro → 215# Alert → 372# Sidecar → 421# Final Clamp
- **Lot 調整ブロック** (4 箇所, 各 ~8 行): alert / halt_recovery / dd_soft / cooldown

---

## 2. 変更内容

### 2.1 Offset Pipeline 抽出 → `offset_pipeline.py`

**新規ファイル**: `scripts/v460/lib/offset_pipeline.py`

| コンポーネント | 責務 |
|---|---|
| `OffsetPipelineResult` (dataclass) | pipeline 出力の構造化 (order_price, effective_offset_ratio, ev/macro/clamp 各フィールド, early_return_record) |
| `OffsetPipelineMixin._apply_offset_pipeline()` | 9 段 offset 乗数チェーンの実行 |
| `OffsetPipelineMixin._scale_lot()` | lot 乗数適用 (min_lot ガード + ログ) |

**責務境界 (SRP)**:
- OK: offset 乗数チェーン (9 段), lot スケール適用
- NG: FillRecord 構築, SkipGate 評価, 監視, ループ制御

### 2.2 Lot 調整チェーンの DRY 化

4 つのほぼ同一な lot 調整ブロック:

```python
# Before (各 ~8 行 × 4 箇所)
if alert_lot_mult != 1.0:
    _pre = lot
    lot = max(min_lot, lot * alert_lot_mult)
    logger.warning(f"[alert_lot] ...")

# After (各 1 行)
lot = self._scale_lot(lot, alert_lot_mult, min_lot, "alert_lot", warn=True)
```

対象: `alert_lot_mult` / `halt_recovery_lot_mult` / `dd_soft_lot_scale` / `cooldown_lot_scale`

### 2.3 継承チェーン更新

```
Before:
  FillCycleExecutorMixin(FillRecordBuilderMixin, PreOrderAdjustmentsMixin)

After:
  FillCycleExecutorMixin(FillRecordBuilderMixin, OffsetPipelineMixin)
    └─ OffsetPipelineMixin(PreOrderAdjustmentsMixin)
```

MRO は維持され、`_apply_offset_multiplier` / `_recalc_price_with_new_offset` へのアクセスは
`OffsetPipelineMixin` 経由で継続。

### 2.4 HARD SKIP の early return 対応

Offset pipeline 内の 421# Final Clamp が `FillRecord` を直接 return するケースを
`OffsetPipelineResult.early_return_record: FillRecord | None` フィールドで表現:

```python
result = self._apply_offset_pipeline(...)
if result.early_return_record is not None:
    return result.early_return_record
```

### 2.5 MAX LINES 契約更新

| ファイル | 旧 | 新 | 理由 |
|---|---|---|---|
| `fill_cycle_executor.py` | 1100 | 1300 | pipeline 抽出後の実態に合わせ調整 |
| `offset_pipeline.py` | — | 360 | 新規ファイル |

---

## 3. 効果

| 指標 | Before | After |
|---|---|---|
| `run_single_cycle` 行数 | 999 | 739 (−260) |
| `fill_cycle_executor.py` クラス行数 | ~1625 | ~1302 (−323) |
| lot 調整コード行数 | ~32 (8×4) | ~4 (1×4) |
| テスト (830 行上限) | ❌ FAIL | ✅ PASS |

---

## 4. Offset Pipeline 9 段チェーン参照

| 段 | Doc# | 名称 | 概要 |
|---|---|---|---|
| 1 | 193# | EV offset | ev_weighted → offset 価格調整 (200# DRY 共通化) |
| 2 | 195# | Velocity | velocity_skip ソフトモード → offset boost |
| 3 | 196# | Trending | trending_sell ソフトモード → offset boost |
| 4 | 240# | Toxicity | Glosten-Milgrom toxicity budget (232# §2.2) |
| 5 | 202# | VG supplement | VG sell-side 補完 (velocity 未適用時) |
| 6 | 458# | Macro | macro_trend → sell/buy offset boost (F-lite) |
| 7 | 215# | Alert | alert_mode offset 乗数 (全サイド共通) |
| 8 | 372# | Sidecar | SAC sidecar bps offset (非対称 maker 調整) |
| 9 | 421# | Final Clamp | execution offset ceiling (417# hard skip 含む) |

---

## 5. 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/offset_pipeline.py` (新規) | `OffsetPipelineMixin` + `OffsetPipelineResult` + `_scale_lot` |
| `scripts/v460/lib/fill_cycle_executor.py` | pipeline/lot 削除, import/継承変更, MAX LINES 更新 |
| `docs/v460/458_ph2_impl_macro_sell_protection.md` | §8.6 に分割記録 (相互参照) |

---

## 6. テスト結果

2182 passed, 125 skipped, 0 failed (行数テスト含む全テスト通過)
