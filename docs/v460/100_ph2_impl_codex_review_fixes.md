# 100# ph2 098#/099# Review Implementation — Post-Review Fix Bundle

## 概要
098# Deep Analysis + 099# Codex Review で特定された全問題への対応実装。  
God Object 分割・重複排除・side-aware 状態管理の改善を含む。

## 変更一覧

### P0 (即時対応)

| # | 問題 | 修正内容 | ファイル |
|---|------|----------|----------|
| P0-1 | warm_start が閾値を復元しない (6+ cycle 収束ラグ) | `_calibrate_threshold` を warm_start 末尾で呼出 | `ml/skip_gate.py` |
| P0-2 | YAML 閾値 0.65/0.60 が P(AS) 分布 [0.42,0.56] に対し過大 | `as_threshold: 0.52`, `as_threshold_buy: 0.52`, `as_threshold_sell: 0.50`, `adaptive_step: 0.05` | `fill_test.yaml`, `run_fill_test.py`, `skip_gate.py` |
| P0-3 | `has_negative_edge` が sell fast-fill AS の 50% を見逃す | Two-layer 検出: L1=fill_price vs mid, L2=post_fill_pnl<0 | `fast_fill_defense.py` |
| P0-4 | `threshold_sec_buy: null` → buy defense 実質無効 | `threshold_sec_buy: 10.0` | `fill_test.yaml` |
| P0-5 | `_fast_fill_boost_active` が side-unaware (sell→buy 伝播) | `FastFillDefense` クラス抽出、per-side 状態管理 | `fast_fill_defense.py`, `run_fill_test.py` |
| P0-6 | stale reprice が SkipGate を完全 bypass | reprice 前に SkipGate 評価を追加 | `run_fill_test.py` |

### P1 (構造改善)

| # | 問題 | 修正内容 | ファイル |
|---|------|----------|----------|
| P1-1 | `max_skip_rate` が buy+sell 混在 (cross-side 干渉) | `_recent_skips_buy`/`_recent_skips_sell` に分離 | `ml/skip_gate.py` |
| P1-2 | boost cap が common `_base_offset_ratio` (0.05) で計算 | side 別 base_offset_ratio で cap 算出 (sell=0.12→cap=2.5) | `fast_fill_defense.py` |
| P1-3 | time_filter の side 別リストが global を無視 | union (global ∪ side) で判定 | `run_fill_test.py` |
| P1-4 | early_exit 時の `mid_30s_after` が実際は <30s (ラベルノイズ) | `actual_measurement_sec` を FillRecord に追加、実経過時間を記録 | `fill_quality.py`, `run_fill_test.py` |
| P1-6 | regime detector が unfilled 時に `order_price` (offset込み) を使用 | `_prev_mid_price` fallback、データ不足時は update スキップ | `run_fill_test.py` |

### God Object 分割

| 対象 | 旧 | 新 |
|------|-----|-----|
| fast_fill_defense | `run_fill_test.py` inline (~60行) | `scripts/v460/lib/fast_fill_defense.py` (190行) |
| `run_fill_test.py` | 2871行 | ~2910行 (inline 60行削除、import+委譲 20行追加) |

## テスト

- **既存**: 794 passed (4テスト修正: default 値/ソース検査の更新)
- **新規**: 13 passed (`test_100_fast_fill_defense.py`)
  - side isolation, two-layer neg_edge, side-specific cap, threshold, reset, deactivation, disabled, base offset sync

## 設定変更 (fill_test.yaml)

```yaml
# Before → After
skip_gate.as_threshold:      0.65 → 0.52
skip_gate.as_threshold_buy:  null → 0.52
skip_gate.as_threshold_sell:  0.60 → 0.50
skip_gate.adaptive_step:      0.02 → 0.05
fast_fill_defense.threshold_sec_buy: null → 10.0
```

## 影響範囲

- **SkipGate**: warm_start 後の初期閾値が较正済みになり、起動直後の skip 判定精度が向上
- **fast_fill_defense**: sell boost が buy に伝播しなくなり、buy fill rate 低下リスクが解消
- **stale reprice**: SkipGate による AS ガード追加で、AS 高確率時の無防備な再発注を抑止
- **time_filter**: global + side の union で、意図しないフィルター漏れを防止
- **regime**: unfilled 時の order_price ノイズが regime 判定に影響しなくなる
