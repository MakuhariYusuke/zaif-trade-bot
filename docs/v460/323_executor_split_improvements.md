# 323# God Object 分割 — fill_cycle_executor + 改善

## 概要

322# で `maker_price.py` を 3 Mixin に分割 (1,692 → 996 行)。
本ターンでは `fill_cycle_executor.py` の God Object 分割と、付随する改善を実施。

## 1. Executor Mixin 分割

### 分割結果

| ファイル | 行数 | 責務 |
|---|---|---|
| `fill_cycle_executor.py` | 1,502 → **1,090** (-27.4%) | run_single_cycle + OB/SkipGate/Fill/PnL |
| `fill_record_builder.py` **(NEW)** | **394** | FillRecord 構築, EV 加重計算, decision path 導出 |
| `pre_order_adjustments.py` **(NEW)** | **93** | offset 倍率適用, offset 変更後の price 再計算 |

### 抽出メソッド

**FillRecordBuilderMixin** (fill_record_builder.py):
- `_resolve_fill_cancel_reason` — cancel_reason 一元解決
- `_compute_fill_spread_bps` — spread_bps 安全算出
- `_build_fill_measurement_fields` — 約定/計測系フィールド構築
- `_build_fill_market_fields` — 市場観測/実行メタ系フィールド構築
- `_build_fill_strategy_fields` — strategy/macro 系フィールド構築
- `_compute_ev_weighted` (static) — 30s/120s PnL 加重平均
- `_build_fill_record` — FillRecord 組立
- `_derive_decision_path` (static) — decision_path 導出

**PreOrderAdjustmentsMixin** (pre_order_adjustments.py):
- `_recalc_price_with_new_offset` (static) — offset 変更後の price 再計算
- `_apply_offset_multiplier` (static) — offset 倍率適用

### 継承チェーン

```
FillCycleExecutorMixin(FillRecordBuilderMixin, PreOrderAdjustmentsMixin)
  └── FillTestRunner(FillRecordHelpersMixin, FillCycleExecutorMixin, ...)
```

MRO により全メソッドが `FillTestRunner` インスタンス上で解決。

## 2. 重複排除分析

### 調査結果

| パターン | 実装数 | 統合可能性 |
|---|---|---|
| **Parkinson σ** | 5 | 不可 — 入力形態・ユースケースが根本的に異なる |
| **VPIN** | 3 | コア算出 `|buy-sell|/total` は共通だが DataFrame vs NumPy array でデータアクセスパターンが異なる。純粋関数化は ~3行で投資対効果が低い |
| **Regime 検知** | 10+ | D+E+F (ztb/analysis/regime/ 12状態分類器) は統合可能 (~500行削減見込み) だが SAC/ML 訓練用で live bot に影響なし |

### 結論

現時点では live trading に直接影響する重複は発見されなかった。`ztb/analysis/regime/` の統合は将来の ML パイプライン整理時に実施する価値がある。

## 3. その他の改善

| 項目 | ファイル | 内容 |
|---|---|---|
| Dead import 除去 | `maker_price.py` | 未使用 `Sequence` を import から除去 |
| Dead import 除去 | `fill_cycle_executor.py` | 未使用 `CircuitState` (TYPE_CHECKING) を除去 |
| ソースリスト更新 | `_fill_test_source.py` | 新2ファイルを `_FILL_TEST_RUNNER_SOURCES` に追加 (6ファイル構成) |
| ドキュメント修正 | `_fill_test_source.py` | docstring 更新 (4→6ファイル) |
| テスト上限更新 | `test_253` | executor line count 上限 1510 → 1100 |

## 4. テスト結果

- **3971 passed, 33 skipped, 0 failed** (v460 テスト全量)
- 他テストディレクトリの既知 pre-existing failures は本変更と無関係

## 変更ファイル一覧

| ファイル | 操作 |
|---|---|
| `scripts/v460/lib/fill_record_builder.py` | NEW (394行) |
| `scripts/v460/lib/pre_order_adjustments.py` | NEW (93行) |
| `scripts/v460/lib/fill_cycle_executor.py` | MODIFIED (1502→1090行) |
| `scripts/v460/lib/maker_price.py` | MODIFIED (dead import 除去) |
| `tests/unit/v460/_fill_test_source.py` | MODIFIED (ソースリスト拡張) |
| `tests/unit/v460/test_253_hot_reload_dead_config_getattr_bare_except.py` | MODIFIED (line count 上限更新) |
