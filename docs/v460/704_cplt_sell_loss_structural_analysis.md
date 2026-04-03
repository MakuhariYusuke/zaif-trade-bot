# 704# Sell 損失構造分析 + 即時改善 + Codex タスク計画

## 概要

703# 実装後のライブデータ (Apr 1-3, 429 fills) を多角的に分析。
**sell 側の構造的損失** (-167.90 bps / 3日) を特定し、即時修正 + Codex タスク群を策定。

## 分析結果サマリー

### Side×Regime PnL (3日間 post_fill_30s_pnl)

| Side×Regime | Count | Avg PnL (bps) | Total (bps) |
|-------------|-------|---------------|-------------|
| sell+trending_down | 64 | -1.17 | -74.84 |
| sell+ranging | 113 | -0.55 | -61.65 |
| sell+trending_up | 36 | -0.87 | -31.42 |
| buy+trending_up | 41 | -0.56 | -22.96 |
| buy+trending_down | 65 | +0.36 | +23.58 |
| buy+ranging | 110 | +0.53 | +58.80 |

**結論**: sell が全レジームでマイナス。buy は trending_up 以外でプラス。

### 時間帯別 Sell 損失 TOP5 (UTC)

| UTC | Count | Avg PnL | Total | 状態 |
|-----|-------|---------|-------|------|
| 02h | 10 | -4.15 | -41.48 | 既存 boost=5.0 |
| 12h | 11 | -3.23 | -35.57 | **704# 2.0→3.0 強化** |
| 07h | 8 | -3.72 | -29.78 | **704# boost=3.0 新規** |
| 16h | 7 | -4.18 | -29.24 | 既存 boost=2.5 |
| 14h | 5 | -4.66 | -23.28 | 既存 boost=2.5 |

### 重大発見

| # | 発見 | 影響 | 対応 |
|---|------|------|------|
| F1 | spread_as_guard が 100% 未起動 | EV penalty 0.5bps が全サイクルで適用されず | **即時修正済** |
| F2 | entry_gate EV が 100% 負値 | guard auto-disable → 全通過 | **パラメータ調整済** |
| F3 | sell+trending_down が最大損失 | -74.84 bps / 3日 | **offset 追加済** |
| F4 | UTC 7h/19h/20h に防御なし | 合計 -63.33 bps | **hour boost 追加済** |
| F5 | sell 全レジームで負 | offset (spread capture) 構造不足 | Codex T2/T3 |

---

## 即時実装 (cplt 直接修正)

### Fix 1: spread_as_guard staleness bug
**ファイル**: `scripts/v460/lib/orchestrator_mid_cycle.py`

`last_spread` (60s staleness guard 付き) → `last_spread_raw` (staleness guard なし) に変更。
entry gate の spread_as_guard は独立判定のため、Gate 8-9 向けの staleness guard は不要。

**root cause**: `MakerPriceCalculator.last_spread` は `compute()` 後 60秒で None を返す。
entry gate 評価は cycle 冒頭で行われるが、前回の `compute()` から 60秒以上経過している
ケースが多く、spread_bps が常に None → spread_as_guard_triggered が常に False。

### Fix 2: sell_trending_down_offset 追加
- `configs/v460/fill_test.yaml`: `sell_trending_down_offset: 0.5`
- `scripts/v460/lib/fill_config.py`: `skip_gate_sell_trending_down_offset: float = 0.0`
- `scripts/v460/lib/fill_config_parser.py`: parser mapping 追加
- `scripts/v460/lib/fill_config_validation.py`: [0, 2] 範囲検証追加
- `scripts/v460/lib/config_hot_reload.py`: allowlist 追加
- `scripts/v460/lib/skip_gate_evaluator.py`: `sell + trending_down` 条件で offset 適用

### Fix 3: sell_hour_offset_boost 拡張
| UTC (JST) | 変更 | 根拠 |
|-----------|------|------|
| 7h (16h) | 新規 3.0 | sell n=8 avg=-3.72bps total=-29.78bps |
| 12h (21h) | 2.0→3.0 | sell n=11 avg=-3.23bps total=-35.57bps |
| 19h (04h) | 新規 2.0 | sell n=9 avg=-1.78bps total=-16.03bps |
| 20h (05h) | 新規 3.0 | sell n=3 avg=-5.84bps total=-17.52bps |

### Fix 4: entry_gate_guard パラメータ調整
- `max_consecutive_blocks`: 15 → 50 (EV 常時負のため 15 では即 auto-disable)
- `max_block_rate`: 0.6 → 0.95 (100% 負 EV 環境で実質全通過を防止)

### Fix 5: regime_guard_overrides trending_down 有効化
- `trending_down.ev_threshold_premium_bps`: 0.0 → 0.3
- `trending_down.spread_as_guard_penalty_multiplier`: 1.0 → 1.5
- 全レジーム同等の防御水準に統一

---

## Codex タスク計画

### Task 1: ユニットテスト追加 (P0)
- 704# 変更のテスト: sell_trending_down_offset, spread_as_guard raw, regime_guard trending_down
- 詳細: `prompts/codex_704_task1_sell_trending_down_tests.md`

### Task 2: Entry Gate side-aware blocking (P1)
- buy 側は軽度負 EV で通過許可、sell 側は block 維持
- 詳細: `prompts/codex_704_task2_entry_gate_side_aware.md`

### Task 3: Sell offset pipeline 構造分析 (P2)
- spread_capture が構造的にマイナスの原因を offset pipeline stages から分析
- 詳細: `prompts/codex_704_task3_sell_offset_analysis.md`

## Codex 投入順序

```
Phase 1 (同時投入):
  Task 1 (ユニットテスト) ← 704# 変更の品質保証
  Task 2 (side-aware blocking) ← sell ブロック効果

Phase 2:
  Task 3 (offset 分析) ← データドリブンな次ステップ策定
```

---

## 全変更ファイル一覧

| ファイル | 種別 | 変更内容 |
|---------|------|---------|
| `scripts/v460/lib/orchestrator_mid_cycle.py` | bug fix | last_spread → last_spread_raw |
| `configs/v460/fill_test.yaml` | config | sell_trending_down_offset, hour boost, guard params, regime_guard |
| `scripts/v460/lib/fill_config.py` | 型定義 | skip_gate_sell_trending_down_offset |
| `scripts/v460/lib/fill_config_parser.py` | parser | sell_trending_down_offset mapping |
| `scripts/v460/lib/fill_config_validation.py` | validation | [0, 2] 範囲チェック |
| `scripts/v460/lib/config_hot_reload.py` | hot-reload | allowlist 追加 |
| `scripts/v460/lib/skip_gate_evaluator.py` | logic | sell+trending_down offset 適用 |

---

*生成: 2026-04-04 by cplt (704#)*
*入力: 703# 後のライブデータ 3日間分析 (429 fills)*

---

## 実装レビュー結果

**ステータス: Task 1-3 実装済み（current runtime に整合する形へ補正）**

### prompt 検証メモ

- `Task 1` は runtime 実装が先に入っていたため、実装追加ではなく
  - `sell_trending_down_offset`
  - `spread_as_guard` の raw spread 参照
  - `trending_down` regime guard
  の回帰固定が本体だった
- `Task 2` は prompt の方向性は妥当だったが、current runtime では
  - live YAML がすでに `50 / 0.95`
  - code default はまだ `15 / 0.6`
  のまま残っていた
  - そのため side-aware suppression 追加と同時に default も live posture に揃えた
- `Task 3` は新規 script 追加で正しい
  - ただし既存 `analysis_common` を使えば
    - filter
    - output
    - empty-dir graceful handling
    を再利用できるため、prompt の再実装は避けた

### hidden task

- `entry_gate_buy_suppress_ev_threshold` は top-level YAML と nested `entry_gate:` の両方を受けられるようにした
- `run_fill_test.py` の `EntryGateGuardConfig(...)` 構築 2 箇所にも新フィールドを通した
- `test_690_entry_gate_guard.py` は buy mild-negative 前提の既存ケースが新 side-aware suppress と衝突するため、
  旧ロジックだけを見たいケースは sell 側へ寄せて責務を分離した
- `skip_gate_sell_trending_down_offset` は YAML drift allowlist へ追随した

### 横展開

- `entry_gate` の side-aware 追加は
  - parser
  - validation
  - hot-reload
  - runner wiring
  - live YAML
  まで一気に通した
- `sell offset analysis` は単発 script にせず、以後の analysis task でも再利用しやすい
  stdlib-only パターンとして残した

### 回帰

- focused 704/config subset:
  - `145 passed in 3.51s`
- broader 704 + fill/PPO/SAC subset:
  - `471 passed, 1 skipped, 5 warnings in 6.42s`
