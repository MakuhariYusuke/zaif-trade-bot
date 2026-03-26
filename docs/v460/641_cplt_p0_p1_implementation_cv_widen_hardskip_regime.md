# 641# P0-A/B/C + P1-A 実装: CV widen 無効化 / skip_rate 緩和 / freeze 短縮 / regime別 hard_skip

## 概要

640# で確立したアクションプラン (P0-A/B/C, P1-A) を実装。
7日間のログ再検証 (20260320-20260326, 1027 fills) で全主張を確認した上で変更を適用。

## 再検証結果サマリ (7日間: 20260320-20260326)

| 指標 | 値 | 640# 主張 |
|------|-----|-----------|
| buy/ranging avg pnl | -0.39bps (n=490) | ✅ 依然マイナス |
| buy/trending_down avg pnl | +2.03bps (n=40) | ✅ 唯一の黒字 buy regime |
| CV-widen buy uncapped | -0.56bps (n=165) | ✅ widen は有害 |
| CV-widen sell | -1.56bps (n=39) | ✅ sell 側も有害 |
| skip_rate_limit forced | -0.92bps (n=169, total=-154.65bps) | ✅ 強制fill損失大 |
| sell/ranging forced | n=85, total=-147.60bps | ✅ forced の主損失源 |
| final_clamp_hard_skip buy/trending_down | 14件 (16.7%) | ✅ 有益fill の過剰抑制 |

## 変更一覧

### P0-A: CV widen 全面無効化 (`offset_boost: 1.25→1.0`)

**ファイル**: `configs/v460/fill_test.yaml` L348
**根拠**: buy uncapped=-0.56bps, sell=-1.56bps → 両側とも有害。side別制御の複雑さを避け `offset_boost: 1.0` で全面無効化。

### P0-B: max_skip_rate 緩和 (`0.3→0.4`)

**ファイル**: `configs/v460/fill_test.yaml` L453
**根拠**: forced fill 169件=-154.65bps (sell/ranging=-147.60bps)。skip上限を緩和して強制fill頻度を下げる。

### P0-C: balance_freeze_cycles 短縮 (`3→1`)

**ファイル**: `configs/v460/fill_test.yaml` L985
**根拠**: freeze_side 汚染軽減。638#/640# 検証済。

### P1-A: buy/trending_down hard_skip 緩和 (regime別 override)

**変更ファイル**:
| ファイル | 変更内容 |
|----------|----------|
| `configs/v460/fill_test.yaml` | `execution_final_clamp_hard_skip_mult_overrides: {buy/trending_down: 4.0}` 追加 |
| `scripts/v460/lib/fill_config.py` | `execution_final_clamp_hard_skip_mult_overrides` フィールド + `resolve_hard_skip_mult()` メソッド追加 |
| `scripts/v460/lib/fill_config_parser.py` | YAML パース処理追加 |
| `scripts/v460/lib/multiplicative_pipeline.py` | `resolve_hard_skip_mult(side, regime)` 呼び出しに変更 |
| `scripts/v460/lib/config_hot_reload.py` | hot-reload allowlist に追加 |
| `tests/unit/v460/test_585_multiplicative_pipeline.py` | 既存テスト修正 + 新規6テスト追加 |
| `tests/unit/v460/test_336_yaml_code_drift_prevention.py` | KNOWN_YAML_OVERRIDES に4フィールド追加 |

**設計**: `resolve_hard_skip_mult(side, regime)` が `"side/regime"` キーで overrides dict を検索し、
ヒットすればその値、なければ `execution_final_clamp_hard_skip_mult` (デフォルト) を返す。
`_current_regime_value()` は `hasattr` ガードで mixin 互換性を維持。

**効果**: buy/trending_down で ceiling×4.0 に緩和 → ceiling=0.35 の場合、offset 1.40 超で初めて hard skip。
buy/trending_down は avg +2.03bps の唯一の黒字 buy regime であり、過剰スキップ (14件/16.7%) を軽減。

## テスト結果

- **既存テスト**: 48 passed (元42 + stub修正分)
- **新規テスト (6件)**:
  - `test_regime_aware_hard_skip_relaxed`: override 適用で hard skip 回避
  - `test_regime_aware_hard_skip_default_still_skips`: デフォルト mult で引き続き skip
  - `test_no_override_returns_default`: override なし → デフォルト値
  - `test_override_returns_override_value`: override ヒット → override 値
  - `test_override_miss_returns_default`: key 不一致 → デフォルト値
  - `test_regime_none_returns_default`: regime=None → デフォルト値
- **全体リグレッション**: 4129+ passed, 0 failed

## 期待効果

| 項目 | 期待 |
|------|------|
| CV widen 無効化 | buy -0.56bps / sell -1.56bps の損失源除去 |
| max_skip_rate 0.4 | forced fill 削減 → -154.65bps の損失縮小 |
| balance_freeze 1 | freeze_side 汚染軽減、反応速度向上 |
| hard_skip regime override | buy/trending_down +2.03bps の fill 機会回復 |

---

## 642# 可観測性改善 (637#-640# 分析の教訓)

### 背景

637#-640# の分析過程で、以下のデータが手動集計でしか得られず、
分析の効率を大幅に下げていた。次回以降の分析を自動化するため
FillRecord に6フィールドを追加。

### 追加フィールド

| フィールド | 型 | 目的 | 解決する分析課題 |
|-----------|-----|------|----------------|
| `skip_gate_forced_pass` | bool | rate_limit が skip を override したか | 637# 強制fill 169件の手動集計不要に |
| `skip_gate_side_skip_rate` | float | 判定時の side 別 skip 率 | skip率の閾値接近度を直接可視化 |
| `execution_hard_skip_mult_used` | float | hard skip 時に使用した mult 値 | 641# regime override のトレース |
| `cv_offset_action` | str | "widen"/"tighten" (CV 方向) | 638# CV widen 損失の手動計算不要に |
| `balance_jpy_at_order` | float | 発注時 JPY 残高 | freeze loop / state pollution 即時診断 |
| `balance_btc_at_order` | float | 発注時 BTC 残高 | inventory skew 分析の自動化 |

### 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `ztb/metrics/fill_quality.py` | FillRecord に6フィールド追加 |
| `ztb/ml/skip_gate.py` | SkipDecision に `forced_pass`, `side_skip_rate` 追加 |
| `ztb/ml/skip_gate_contracts.py` | SkipDecisionLike protocol 更新 |
| `ztb/ml/skip_gate_result_fields.py` | SkipDecisionResultFields 伝播 |
| `scripts/v460/lib/fill_config_results.py` | SkipGateResult に2フィールド追加 |
| `scripts/v460/lib/skip_gate_evaluator.py` | _assign_result_fields 経由で伝播 |
| `scripts/v460/lib/offset_pipeline.py` | OffsetPipelineResult に hard_skip_mult_used 追加 |
| `scripts/v460/lib/multiplicative_pipeline.py` | hard skip 時に _hs_mult を結果に含める |
| `scripts/v460/lib/fill_record_builder.py` | 5フィールド受け入れ + cv_offset_action 計算 |
| `scripts/v460/lib/fill_cycle_executor.py` | _PreOrderPhaseResult + balance snapshot 伝播 |

### 分析効率の改善例

**Before (637#):**
```python
# 手動で169件のforced fillを集計
forced = [r for r in records if 'skip_rate_limit' in (r.get('skip_gate_reason') or '')]
by_regime = Counter((r['side'], r['regime']) for r in forced)
```

**After (642#):**
```python
# 直接フィルタ
forced = [r for r in records if r.get('skip_gate_forced_pass')]
# side_skip_rate で閾値接近度も即座に確認
```
