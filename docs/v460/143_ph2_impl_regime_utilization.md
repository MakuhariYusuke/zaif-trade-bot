# 143# レジーム活用 R-1 実装 + 140#/141# レビュー対応

**作成日**: 2026-02-22  
**前提**: 142# ph2_plan_regime_utilization (R-1 計画)  
**テスト**: 1218 → 1244 (+26)

---

## §1 レビュー対応 (140# §7 / 141# Appendix A)

### §1.1 141# A.1-#1 (HIGH): side hot-reload 到達不能修正

**問題**: `_check_and_reload_model()` 内で unified model の hash 変更なし → early return が、その下の `_check_and_reload_side_models()` を到達不能にしていた。

**対策**: `_check_and_reload_side_models()` を unified hash チェック **前** に移動。side 別モデルの hot-reload が unified model の更新有無に依存しなくなった。

**変更**: `scripts/v460/lib/skip_gate_evaluator.py` L247

### §1.2 141# A.1-#2 (MEDIUM): unified 不在時の side model 読込

**問題**: `__init__` で unified model が存在しない場合、即 return → side model も読み込まれない。`evaluate()` のガードも `_skip_gate is None → return` のため side-only 動作が不可。

**対策**:
1. `__init__`: unified 不在時も `_load_side_models()` を呼び出し
2. `evaluate()`: ガード条件を `_skip_gate is None AND _gate_buy is None AND _gate_sell is None` に拡張

### §1.3 141# A.1-#3 (MEDIUM): online_monitor 未約定 audit 混入

**問題**: `records.tail(cfg.window)` が未約定 audit レコード (preflight_pause, circuit_breaker 等) を含み、有効 window を圧迫。

**対策**: `tail(window)` 前に `skip_gate_skipped=True OR filled=True` で pre-filter。audit レコードは window から除外。

**変更**: `ztb/ml/online_monitor.py`

### §1.4 140# §7-#1 (MEDIUM): quarantine bypass for cancel_reason

**問題**: skip/audit 用 FillRecord (`order_price=0`, `side="none"`) が `_quarantine_reason()` で `invalid_side` / `invalid_order_price` に引っかかり、Gate 分析で不可視。

**対策**: `cancel_reason` が設定されている場合、side/price/quantity バリデーションをバイパス。

**変更**: `ztb/metrics/fill_quality.py`

### §1.5 140# §7-#2 (LOW): preflight_pause cycle_id 一意性

**対策**: cycle_id に `_{int(time.time())}` サフィックスを追加。

### §1.6 141# A.1-#4 (LOW): テスト数ドキュメント不整合

**対策**: 141# ドキュメントのテスト数を "1213 (+42)" → "1218 (+47, 142# 修正含む)" に更新。

---

## §2 R-1a: レジーム別 offset 適応

### §2.1 概要

142# 計画の R-1a 施策。`FillTestRegimeDetector` の出力 (4値) に連動して offset_ratio を動的に調整。

### §2.2 新規 config フィールド

| フィールド | デフォルト | 説明 |
|---|---|---|
| `regime_high_vol_offset_boost` | 1.2 | high_vol 時に offset × 1.2 (+20% 拡張) |
| `regime_ranging_offset_discount` | 1.0 | ranging 時に offset × N (1.0=無効) |

YAML マッピング:
```yaml
regime:
  high_vol_offset_boost: 1.3
  ranging_offset_discount: 0.85
```

### §2.3 実装位置

`scripts/v460/lib/maker_price.py` — `compute()` メソッド内、既存の `regime_trending_offset_boost` (052#) の直後:

1. **high_vol**: `effective_offset_ratio × boost` → `min(result, max_offset_ratio)` でクランプ
2. **ranging**: `effective_offset_ratio × discount` → `max(result, min_offset_ratio)` でクランプ

### §2.4 レジーム別 offset 適用順序

```
base_offset → side別 → sell_floor → trending boost (052#)
→ high_vol boost (143#) → ranging discount (143#)
→ unknown_buy boost (130#) → spread adaptive (054#)
→ sell_floor 再適用 → volatility guard (107#)
```

---

## §3 R-1b: レジーム別 lot 適応

### §3.1 概要

142# 計画の R-1b 施策。レジーム別にロットサイズを一時的に調整。

### §3.2 新規 config フィールド

| フィールド | デフォルト | 説明 |
|---|---|---|
| `regime_lot_multipliers` | `{}` (空 dict) | レジーム名 → 倍率のマッピング |

YAML マッピング:
```yaml
regime:
  lot_multipliers:
    high_vol: 0.7
    trending: 1.2
    ranging: 1.0
```

### §3.3 実装方式

**メソッド**: `FillTestRunner._regime_adjusted_lot()` を新設。

**適用箇所**: `_process_cycle()` 内、`apply_lot_floor()` 後 → `place_order` 前 で 1 回計算。以降の FillRecord にも同じ調整済み lot を記録。

**安全制約**:
- `min_lot = 0.001` (Coincheck BTC 最小) を下回らない
- `max_lot` を超えない
- `multipliers` 空 or `regime_detector` None → base lot をそのまま返す
- `multiplier = 1.0` → 計算をスキップして base lot

---

## §4 テスト (26 件追加)

| カテゴリ | 件数 | 内容 |
|---|---|---|
| Config defaults | 3 | high_vol_boost, ranging_discount, lot_multipliers デフォルト |
| Source inspection | 2 | maker_price.py にキーワード存在確認 |
| Functional offset | 6 | boost/discount/clamp/disabled/none の動作 |
| YAML mapping | 2 | offset + lot multipliers の YAML → Config |
| Lot adaptation | 8 | shrink/expand/min_lot/max_lot/neutral/unknown |
| Review fixes | 5 | quarantine bypass, monitor pre-filter, side reload, evaluate guard |

**テストファイル**: `tests/unit/v460/test_143_regime_utilization.py`

---

## §5 変更ファイル一覧

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `scripts/v460/lib/fill_config.py` | 機能追加 | R-1a offset fields + R-1b lot field + YAML mapping |
| `scripts/v460/lib/maker_price.py` | 機能追加 | high_vol/ranging offset adaptation blocks |
| `scripts/v460/run_fill_test.py` | 機能追加 | `_regime_adjusted_lot()` + place_order での使用 |
| `scripts/v460/lib/skip_gate_evaluator.py` | バグ修正 | side hot-reload independence, unified-absent loading, evaluate guard |
| `ztb/ml/online_monitor.py` | バグ修正 | evaluable records pre-filter |
| `ztb/metrics/fill_quality.py` | バグ修正 | quarantine bypass for cancel_reason |
| `tests/unit/v460/test_143_regime_utilization.py` | 新規 | 26 テスト |
| `docs/v460/143_ph2_impl_regime_utilization.md` | 新規 | 本ドキュメント |

---

## §6 134# ロードマップ位置確認

```
Phase A (Data Infra)      : ✅ 135#
Phase B (Observability)   : ✅ 135#
Phase C (Re-measurement)  : ⬜ Operational (24h run 未実施)
Phase D (Retrain restart) : ✅ 136#
Phase E (P1 group)        : ✅ 137#-141# (全 9 項目完了)
142# Self-check           : ✅ C-1/M-1/M-3 修正
143# R-1 regime util      : ✅ 本セッション (R-1a + R-1b)
```

**次ステップ**: R-1c (reprice), R-1d (timeout), P2 グループ

---

## §7 Codexレビュー追記 (2026-02-22)

### §7.1 検証サマリ

- 実装照合対象: `scripts/v460/lib/fill_config.py`, `scripts/v460/lib/maker_price.py`, `scripts/v460/run_fill_test.py`, `scripts/v460/lib/skip_gate_evaluator.py`, `ztb/ml/online_monitor.py`, `ztb/metrics/fill_quality.py`, `tests/unit/v460/test_143_regime_utilization.py`
- テスト実行結果: `tests/unit/v460/test_143_regime_utilization.py` は **26 passed**、`tests/unit/v460` 全体は **1244 passed**。

### §7.2 指摘事項 (重大度順)

| # | 重大度 | 対象 | 指摘 | 推奨対応 |
|---|---|---|---|---|
| 1 | HIGH | `scripts/v460/run_fill_test.py:1293`, `scripts/v460/run_fill_test.py:805` | preflight 残高チェックは `_current_lot` 基準だが、実発注量は後段で `_regime_adjusted_lot()` により増減される。`trending` 増量時に preflight をすり抜け、`insufficient_funds` を誘発しやすい。 | `_regime_adjusted_lot()` を preflight 前に計算し、`BalanceChecker.check()` へ「今回発注量」を渡す設計へ変更する。 |
| 2 | MEDIUM | `scripts/v460/run_fill_test.py:293`, `scripts/v460/lib/fill_config.py:239` | `_regime_adjusted_lot()` が `min_lot=0.001` をハードコードしており、`config.min_order_btc` と二重管理。 | `min_lot = self.config.min_order_btc` に統一し、最小数量の単一ソース化を行う。 |
| 3 | MEDIUM | `ztb/metrics/fill_quality.py:902` | `cancel_reason` があれば side/price/quantity の妥当性チェックを広くバイパスするため、監査用途以外の壊れたレコードも clean 扱いになり得る。 | バイパス条件を監査系 reason (`preflight_pause` など) かつ `side=\"none\"` のみに限定する。 |
| 4 | MEDIUM | `tests/unit/v460/test_143_regime_utilization.py:425`, `tests/unit/v460/test_143_regime_utilization.py:436`, `tests/unit/v460/test_143_regime_utilization.py:452` | 一部の重要修正が「ソース文字列検査」に寄っており、動作回帰を十分に捕捉できない。 | side-only evaluate 実行、online_monitor の窓圧迫防止、hot-reload 実挙動をモックI/O付きで追加検証する。 |
| 5 | LOW | `docs/v460/143_ph2_impl_regime_utilization.md:3` | 作成日が `2025-02-24` となっており、142/141 系の時系列と不整合。 | 文書メタデータ日付を系列に合わせて補正する。 |

### §7.3 優先修正順

1. #1 (preflight と regime lot の整合) を最優先で修正。
2. #2 (min lot 一元化) と #4 (挙動テスト追加) を同一PRで実施。
3. #3 (quarantine バイパス条件の限定) を監査 reason リスト定義とセットで適用。
