# 461# skip_gate_evaluator Mixin 分割 + MAX LINES + Fill Test ログ分析

> **種別**: refactor / rpt  
> **フェーズ**: phg (フェーズ横断)  
> **日付**: 2026-03-17  
> **前提**: 460# offset pipeline 抽出完了後

---

## 概要

3 つの作業を実施:

1. **skip_gate_evaluator.py Mixin 分割** (1362→866 行)
2. **MAX LINES 宣言追加** (3 ファイル)
3. **Fill Test 5 日間ログ分析**

---

## 1. skip_gate_evaluator.py 分割

### 背景

`SkipGateEvaluator` は 1362 行に達し、model loading/hot-reload (~300 行) と ev_weighted 判定ロジック (~210 行) が evaluate() 本体と混在。既存の Mixin パターン (322#/323#/325#/332#/460#) に倣い分割。

### 抽出結果

| 新ファイル | クラス | 行数 | 責務 |
|---|---|---|---|
| `skip_gate_model_loader.py` | `SkipGateModelLoaderMixin` | ~300 | モデルパス解決, ロード, config overrides, warm_start, calibrator注入, side/alt モデル, hot-reload |
| `skip_gate_ev_weighted.py` | `SkipGateEvWeightedMixin` | ~210 | ev_weighted 統合判定 (188#/190#/193#), offset 変換 |

### クラス継承

```python
class SkipGateEvaluator(SkipGateModelLoaderMixin, SkipGateEvWeightedMixin):
    """MAX LINES: 900"""
```

### 移動メソッド一覧

**SkipGateModelLoaderMixin**:
- `_resolve_model_path`, `_read_model_hash`, `_load_gate_from_path`
- `_apply_config_overrides`, `_apply_warm_start`, `_inject_calibrator`
- `_load_side_models`, `_load_alt_models`
- `_check_and_reload_model`, `_check_and_reload_side_models`

**SkipGateEvWeightedMixin**:
- `_try_ev_weighted_decision`, `_ev_weighted_as_offset`

### テスト修正

5 テストファイルでソース読み取りパスを更新:
- `_fill_test_source.py`: `SKIP_GATE_MODEL_LOADER` パス定数追加
- `test_141_side_specific_models.py`: logger パッチパス変更
- `test_143_regime_utilization.py`: ソース読み取り先変更
- `test_139_review_fixes.py`: hot_reload ソース読み取り先変更
- `test_255_getattr_bare_except_cleanup.py`: hot_reload ソース読み取り先変更

---

## 2. MAX LINES 宣言追加

| ファイル | 現在行数 | MAX LINES |
|---|---|---|
| `skip_gate_evaluator.py` | 866 | 900 |
| `cycle_gate_aggregator.py` | 733 | 800 |
| `fill_config_parser.py` | 1024 | 1100 |

---

## 3. Fill Test 5 日間ログ分析 (2026-03-13 〜 2026-03-17)

### 3.1 基本統計 (3/17 最新)

| 指標 | 値 |
|---|---|
| 総レコード | 506 |
| Fill (約定) | 83 (16.4%) |
| Cancel | 423 (83.6%) |

### 3.2 Fill Rate 推移

```
3/13: 28% → 3/14: 26% → 3/15: 17% → 3/16: 14% → 3/17: 16%
```

低下傾向。`ranging_low_vol_skip` が日々増加し 3/17 で 51.3% を占有。

### 3.3 Cancel Reason 内訳 (3/17)

| Reason | 割合 |
|---|---|
| ranging_low_vol_skip | 51.3% |
| no_feasible_quote | 13.9% |
| spread_too_narrow | 9.2% |
| timeout | 9.0% |
| skip_gate | 4.3% |
| sell_dynamic_kill | 3.5% |
| その他 | 8.8% |

### 3.4 Adverse Selection

| 指標 | Buy | Sell | 全体 |
|---|---|---|---|
| AS Rate (processed) | 24% | 39% | 31.3% |
| AS Rate (raw 50.6%) | — | — | 50.6% |

Sell 側が buy の 1.6 倍。

### 3.5 PnL (bps)

| Horizon | Mean | Median |
|---|---|---|
| 30s | -1.12 | -0.34 |
| 60s | -1.91 | -0.74 |
| 120s | -1.82 | -0.59 |

全ホライズンで平均マイナス。

### 3.6 異常検出

- **route_to_kill_deadlock**: 0→43→13→53→0 (5 日間)。3/14-3/16 で PID 特有に発生、3/17 で解消済み (421# final clamp 修正後 SHA)
- **status_unknown_fast**: 6→0→0→3→8。増加傾向。`pending_reconciliation` と相関
- **early_exit**: 全 5 日間で 0 件。実質トリガーされない設定

### 3.7 EV Score と逆選択の関係

| グループ | Mean EV Score |
|---|---|
| 逆選択あり | 0.82 |
| 逆選択なし | 1.44 |

EV score が低いトレードで逆選択が集中。

### 3.8 改善提案

| 優先度 | 施策 |
|---|---|
| P0 | `status_unknown_fast` 増加の原因調査 + reconciliation 改善 |
| P1 | `ranging_low_vol_skip` 閾値の見直し (fill rate 低下の主因) |
| P1 | Sell 側 AS 防御の強化 (39% は高い) |
| P2 | EV score 低スコア帯 (< 0.8) でのスキップ強化 |
| P2 | early_exit の閾値調整 (現状発火せず) |

---

## 4. 品質監査 (付帯)

### Bare except 監査

4 箇所すべて適切にハンドリング済み確認:
- `fill_probability_model.py:322` — logger.exception
- `maker_risk_guards.py:482` — exc_info=True
- `sidecar_signal_io.py:72` — pass + fallback (意図的)
- `skip_gate_evaluator.py:633` — logger.warning + exc_info

### asyncio.sleep 直接呼出し

13 箇所。大半は計測/監視モジュール(pnl_measurer, order_monitor)でレジーム対応不要。変更なし。

### 低優先: ab_judgment.py

986 行。オフライン分析モジュールのため分割は低優先。

---

## テスト結果

```
v460 全テスト: 4470 passed, 9 skipped, 0 failed
skip_gate 関連: 648 passed, 28 skipped, 0 failed
```

---

## 変更ファイル一覧

### 新規

| ファイル | 行数 |
|---|---|
| `scripts/v460/lib/skip_gate_model_loader.py` | ~300 |
| `scripts/v460/lib/skip_gate_ev_weighted.py` | ~210 |

### 修正

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/skip_gate_evaluator.py` | Mixin 化 + MAX LINES: 900 (1362→866) |
| `scripts/v460/lib/cycle_gate_aggregator.py` | MAX LINES: 800 追加 |
| `scripts/v460/lib/fill_config_parser.py` | MAX LINES: 1100 追加 |
| `tests/unit/v460/_fill_test_source.py` | SKIP_GATE_MODEL_LOADER 定数追加 |
| `tests/unit/v460/test_141_side_specific_models.py` | logger パッチパス修正 |
| `tests/unit/v460/test_143_regime_utilization.py` | ソース読み取り先修正 |
| `tests/unit/v460/test_139_review_fixes.py` | ソース読み取り先修正 |
| `tests/unit/v460/test_255_getattr_bare_except_cleanup.py` | ソース読み取り先修正 |
