# 505# PHG: 504# レビュー反映と Phase 0 着手

## 目的

504# のレビューを受けて、502# の `scripts/v460/lib` → `ztb` 移行計画を実コードベースに合わせて補正し、
そのまま Phase 0 の最初の実装に着手する。

## 504# の指摘で妥当だったもの

### 1. `cancel_reasons.py` の優先度が低すぎた

妥当。実際に `ztb/metrics/fill_quality.py` が `scripts.v460.lib.cancel_reasons` を import しており、
`ztb -> scripts` の逆依存が発生していた。

### 2. `fast_fill_defense.py` / `regime_detector.py` を「低リスク移行」と見ていた

妥当。実測の被参照数を踏まえると、これらは façade なしの直接移行には向かない。
特に `fast_fill_defense.py` は `tests/unit/v460/conftest.py` にも効く。

### 3. `fill_config.py` を split-first に置いていた

妥当。329# 時点で

- `fill_config.py`
- `fill_config_parser.py`
- `fill_config_validation.py`
- `fill_config_results.py`

へ分割済みで、追加分割の優先度は低い。

### 4. `ab_judgment.py` など大型未分類ファイルの明示分類不足

妥当。今の時点で少なくとも分類は固定しておく必要がある。

## 今回の軌道修正

### 1. 502# を改訂

反映内容:

- Phase 0 に `cancel_reasons.py` canonical 化を追加
- `fast_fill_defense.py` / `regime_detector.py` を Phase 1.5 の façade 必須へ格上げ
- `fill_config.py` を split-first から除外
- `ab_judgment.py` / `cycle_gate_aggregator.py` / `stopgap_health.py` / `daily_drawdown_guard.py` などの位置づけを明示
- `tests/unit/v460/conftest.py` 影響をテスト方針へ追記
- `fast_fill_defense` の移行先を `ztb/trading/risk/` に修正

### 2. Phase 0 を先に実装

最初の着手対象は `cancel_reasons.py` とした。

理由:

- ロジックが薄く canonical 化しやすい
- `ztb -> scripts` 違反を即解消できる
- 後続の `param_adapter` / `lot_sizer` / `sac_common` より先に片付ける価値がある

## 実装した Phase 0

### canonical module

- [cancel_reasons.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/ztb/trading/common/cancel_reasons.py)

### compatibility shim

- [cancel_reasons.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/cancel_reasons.py)

### 逆依存解消

- [fill_quality.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/ztb/metrics/fill_quality.py)
  - `AUDIT_CANCEL_REASONS` の import を canonical path に変更

### 型参照の追随

- [fill_record_helpers.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/fill_record_helpers.py)
  - `CancelReason` の TYPE_CHECKING import を canonical path に変更

## テスト方針

### focused

- canonical module と shim の整合
- `fill_quality` が canonical path を使っていること
- 既存 `cancel_reasons` 構造テストが落ちないこと

### broad

- `tests/unit/v460/` filtered broad

## 次の着手順

1. `param_adapter.py` → `ztb/trading/sizing/param_adapter.py`
2. `lot_sizer.py` → `ztb/trading/sizing/lot_sizer.py`
3. `sac_common.py` → `ztb/training/sac/runtime.py`
4. `fast_fill_defense.py` façade 移行
5. `regime_detector.py` / `bayesian_regime_filter.py` façade 移行
