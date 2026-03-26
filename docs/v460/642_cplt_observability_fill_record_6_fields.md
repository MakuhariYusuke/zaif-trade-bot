# 642# 可観測性改善: FillRecord 6フィールド追加

## 概要

637#-640# の分析過程で、forced fill 件数・CV widen 方向・残高状態が
手動集計でしか得られず効率を著しく低下させていた。
今後の before/after 分析を自動化するため FillRecord に6フィールドを追加した。

**コミット**: `95101ca44` (2026-03-27)
**テスト**: 4130+ passed / 0 failed

## 背景: 637#-640# 分析で直面した課題

| 分析課題 | 手動集計の内容 | 所要時間 |
|----------|-------------|---------|
| forced fill 特定 | `skip_gate_reason` 文字列から `skip_rate_limit` を grep | 中 |
| skip 率閾値接近度 | skip_gate 内部変数で外部から不可視 | 高 |
| CV widen/tighten 判定 | pre/post offset 差分を手計算 | 中 |
| hard_skip mult トレース | 641# regime override 導入後、使用 mult の確認不可 | 高 |
| 残高状態時系列 | 別ソース (balance log) との突合が必要 | 高 |

## 追加フィールド一覧

| # | フィールド | 型 | 目的 |
|---|-----------|-----|------|
| 1 | `skip_gate_forced_pass` | `bool \| None` | rate_limit が skip を override → forced fill を直接識別 |
| 2 | `skip_gate_side_skip_rate` | `float \| None` | 判定時の side 別 skip 率 → 閾値接近度の可視化 |
| 3 | `execution_hard_skip_mult_used` | `float \| None` | hard skip 時に使用した mult 値 → 641# regime override のトレース |
| 4 | `cv_offset_action` | `str \| None` | `"widen"` / `"tighten"` / `None` → CV 適用方向の直接記録 |
| 5 | `balance_jpy_at_order` | `float \| None` | 発注時 JPY 残高 → freeze loop / state pollution 即時診断 |
| 6 | `balance_btc_at_order` | `float \| None` | 発注時 BTC 残高 → inventory skew 分析の自動化 |

## 実装: データフローチェーン

```
SkipGate.evaluate()
  └→ SkipDecision(forced_pass, side_skip_rate)    # ztb/ml/skip_gate.py
      └→ SkipDecisionLike protocol                # ztb/ml/skip_gate_contracts.py
          └→ SkipDecisionResultFields              # ztb/ml/skip_gate_result_fields.py
              └→ SkipGateResult                    # scripts/v460/lib/fill_config_results.py
                  └→ _assign_result_fields()       # scripts/v460/lib/skip_gate_evaluator.py

MultiplicativePipeline._apply_hard_skip()
  └→ OffsetPipelineResult.execution_hard_skip_mult_used  # scripts/v460/lib/offset_pipeline.py

FillCycleExecutorMixin._pre_order_phase()
  └→ _PreOrderPhaseResult(forced_pass, side_skip_rate, hard_skip_mult_used)
      └→ FillRecordBuilderMixin.build_fill_record(...)
          └→ FillRecord に 5 フィールド格納
          └→ cv_offset_action は _build_fill_cv_fields() 内で pre/post offset から計算

BalanceChecker.last_jpy_free / last_btc_free
  └→ FillCycleExecutorMixin が直接 build_fill_record() に渡す
```

## 変更ファイル (13 files, +122/-1 lines)

### コア層 (ztb/)

| ファイル | 変更内容 |
|----------|----------|
| `ztb/metrics/fill_quality.py` | `FillRecord` に6フィールド追加 (L242-248) |
| `ztb/ml/skip_gate.py` | `SkipDecision` に `forced_pass: bool`, `side_skip_rate: float \| None` 追加。`evaluate()` 内で `_forced_pass` フラグ設定 + `recent_rate` を `side_skip_rate` に渡す |
| `ztb/ml/skip_gate_contracts.py` | `SkipDecisionLike` protocol に `forced_pass`, `side_skip_rate` 追加 |
| `ztb/ml/skip_gate_result_fields.py` | `SkipDecisionResultFields` に `forced_pass`, `side_skip_rate` 追加して伝播 |

### スクリプト層 (scripts/v460/lib/)

| ファイル | 変更内容 |
|----------|----------|
| `fill_config_results.py` | `SkipGateResult` に `forced_pass`, `side_skip_rate` 追加 |
| `skip_gate_evaluator.py` | `_assign_result_fields()` 経由で伝播 |
| `offset_pipeline.py` | `OffsetPipelineResult` に `execution_hard_skip_mult_used` 追加 |
| `multiplicative_pipeline.py` | hard skip 発動時に `_hs_mult` を結果に含める |
| `fill_record_builder.py` | `build_fill_record()` に 5 パラメータ追加 + `cv_offset_action` を `_build_fill_cv_fields()` 内で pre/post offset 差分から計算 |
| `fill_cycle_executor.py` | `_PreOrderPhaseResult` に 3 フィールド追加 + `build_fill_record()` 呼び出しで balance snapshot を渡す |

### テスト

| ファイル | 変更内容 |
|----------|----------|
| `test_253_hot_reload_dead_config_getattr_bare_except.py` | 行数ガード 1535→1545 |
| `test_516_skip_gate_result_fields_migration.py` | stub に `forced_pass`, `side_skip_rate` 追加 |

## 設計判断

### `cv_offset_action` の計算方式

`cv_offset_action` は skip_gate 経由ではなく `fill_record_builder.py` 内で直接計算:

```python
_pre = self._maker_price._cross_venue_lead_lag_pre_offset
_post = self._maker_price._cross_venue_lead_lag_post_offset
if _pre is not None and _post is not None and _pre != _post:
    fields["cv_offset_action"] = "widen" if _post > _pre else "tighten"
```

理由: CV offset は skip_gate チェーンとは独立したパスで計算されるため、
builder 内で MakerPrice の内部状態を参照する方が自然。

### `skip_gate_forced_pass` の None 変換

`build_fill_record()` で `sg_forced_pass or None` として格納:
- `False` → `None` (JSONL 容量削減: 大多数は非 forced)
- `True` → `True`

### balance snapshot のソース

`BalanceChecker.last_jpy_free` / `last_btc_free` を直接参照。
`_pre_order_phase()` 実行時点の値で、発注直前の残高を反映。

## 分析効率の改善例

### forced fill 集計

**Before (637#)**:
```python
# skip_gate_reason 文字列からパターンマッチ
forced = [r for r in records if 'skip_rate_limit' in (r.get('skip_gate_reason') or '')]
by_regime = Counter((r['side'], r['regime']) for r in forced)
```

**After (642#)**:
```python
forced = [r for r in records if r.get('skip_gate_forced_pass')]
```

### skip 率閾値接近分析

**Before**: 不可能 (skip_gate 内部変数)
**After**:
```python
near_limit = [r for r in records
              if (r.get('skip_gate_side_skip_rate') or 0) > 0.35]
```

### CV widen 影響分析

**Before**: pre/post offset 差分を毎回手計算
**After**:
```python
widen_records = [r for r in records if r.get('cv_offset_action') == 'widen']
```

## 641# との関係

本ドキュメントは 641# (P0-A/B/C + P1-A) と同一作業セッションで実装。
641# の YAML 変更と本 642# のコード変更の違い:

| 項目 | 641# | 642# |
|------|------|------|
| 変更種別 | YAML 設定値 | Python コード |
| デプロイ方式 | hot reload (~120s) | プロセス再起動 |
| 効果発現 | 次回 YAML ポーリング後 | 再起動後の fill から |

641# ドキュメント内にも 642# セクションがあるが、
本ドキュメントが 642# の正式リファレンスとなる。

## 検証ポイント (3/29 中間確認時)

After 期間 (SHA `95101ca44`) のレコードで確認:
- `skip_gate_forced_pass` が `true`/`null` で正しく記録されているか
- `skip_gate_side_skip_rate` が `0.0`-`1.0` の範囲の数値か
- `cv_offset_action` が `null` (P0-A で widen 無効化済みのため)
- `execution_hard_skip_mult_used` が hard skip 発動時に非 null か
- `balance_jpy_at_order` / `balance_btc_at_order` が正の数値か
