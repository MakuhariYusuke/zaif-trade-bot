# 589# レビュー残課題修正

> **Date**: 2026-03-24  
> **Scope**: 585#/586#/588# レビュープロセスで検出された残課題の対応

---

## 1. eDRC 指数クリップ (HIGH)

**問題**: `resolve_offset_ceiling()` の `exp()` に入力制限がなく、極端な sigma/adverse_ofi で数値オーバーフローの危険。  
**対応**: 指数 (`alpha * sigma + beta * adverse_ofi`) を `min(exponent, 10.0)` でクリップ。  
- 入力値ではなく指数自体をクリップすることで、alpha/beta の組み合わせに依存しない安全性を確保  
- `exp(10) ≈ 22026` であり、`hard_cap` で最終制限される  
- 通常運用 (alpha=0.020, sigma=15): exponent=0.3 → クリップなし  

**ファイル**: `scripts/v460/lib/fill_config.py` L383-385

## 2. entry_gate_enabled YAML 定義 (MEDIUM)

**問題**: `entry_gate_enabled: bool = False` がコードに存在するが、YAML に未定義。hot-reload で切替不能。  
**対応**: `configs/v460/fill_test.yaml` に `entry_gate_enabled: false` を追加。

## 3. Sidecar Signal 分布分析セクション (HIGH)

**問題**: `sidecar_signal_status` (fresh/stale/missing/error) が FillRecord に記録されるが、analyze_fill_logs に集計セクションがなく、SAC sidecar の品質を測定不可。  
**対応**: `section_sidecar_signal()` を `analyze_fill_logs.py` に追加。  
- ステータス別件数・比率  
- ステータス別 PnL30s 平均・AS コスト平均  

## 4. 588# Post-587# 注記追加 (LOW)

**問題**: 588# は 587# 完了前に書かれた評価文書。§1.3 の三重切断は 587# で修復済みだが、文書にその記載がない。  
**対応**: 588# ヘッダに Post-587# Note を追加。

## 5. テスト修正

- `test_edrc_hard_cap_clamps_output`: 指数クリップ導入に伴い、alpha=1.0/sigma=100 でテスト (exponent_based)
- `test_resolve_offset_ceiling_edrc_exponent_clip`: 589# 新テスト (極端入力 → 指数10クリップ)
- `test_resolve_offset_ceiling_edrc_exponent_no_clip`: 589# 新テスト (通常入力 → クリップなし)
- `test_execution_telemetry_fields_roundtrip`: 587# で削除された `execution_sigma`/`execution_adverse_ofi` を除去
- `KNOWN_YAML_OVERRIDES` 整理: 値が一致していた 3 フィールドを除去

## 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/lib/fill_config.py` | eDRC 指数クリップ追加 |
| `configs/v460/fill_test.yaml` | `entry_gate_enabled: false` 追加 |
| `scripts/v460/analysis/analyze_fill_logs.py` | `section_sidecar_signal()` 追加 |
| `docs/v460/588_eval_585_586_review_deep_dive.md` | Post-587# Note 追加 |
| `tests/unit/v460/test_467_remaining_issues.py` | 589# exponent clip テスト追加 |
| `tests/unit/v460/test_421_final_clamp_deadlock.py` | 削除済みフィールド修正 |
| `tests/unit/v460/test_336_yaml_code_drift_prevention.py` | stale allowlist 整理 |
