# 694# Codex タスク計画

## 概要

693# で Codex 納品レビューと staleness fix を完了。次の Codex タスクを優先度順に策定。
686# の4日間分析結果を基盤として「収益性直結」のタスクを優先。

## 優先度マトリクス

| 優先度 | タスク | 期待効果 | 工数 | 依存 |
|--------|--------|----------|------|------|
| **P0-1** | AS-aware skip filter | fill-rate +10-20%, PnL +0.5bps | 4-5h | なし |
| **P0-2** | Buy-side cross-venue protection | PnL +3bps (buy AS↓) | 4-5h | なし |
| **P1-1** | Protocol 688 type safety + threshold config化 | 品質改善 | 2-3h | なし |
| **P1-2** | Offset pipeline test math validation | テスト信頼性 | 2-3h | なし |
| **P2-1** | AS burst autocorrelation analysis | リスク軽減 | 4-5h | データ蓄積 |

## Codex タスク詳細

---

### Task 1: AS-aware skip filter（P0-1）

**背景**: 686# の4日間分析で SkipGate ML モデルの予測力が MI≈0 であることが判明。
SkipGate スコア Q4（高スコア）が最悪 PnL（-1.24bps）を示す逆相関。
18.1% の cancel rate が skip_gate 由来で、fill-rate を不必要に抑制している。

**目的**: ML モデル依存の skip_gate を、trailing AS rate ベースの deterministic filter に置換する observe モードを追加。

**実装方針**:
- `_apply_trend_5s_sell_guard()` パターンを踏襲
- rolling AS rate（直近100 fill、regime×spread_bucket 別）をトラッキング
- AS rate > threshold → offset boost（spread を広げる）
- AS rate > hard_veto_threshold → skip（cancel reason: `as_trailing_gate_veto`）
- `enabled: false` でデプロイ、observe モードで AS rate ログを蓄積

---

### Task 2: Buy-side cross-venue protection（P0-2）

**背景**: `cross_venue_lead_lag.py` は sell-side veto のみ実装。
buy-side にリファレンス venue からの AS 検出がない（501# で非対称性を指摘済み）。

**目的**: buy-side でもリファレンス venue の価格急落検出時に veto/offset boost を適用。

**実装方針**:
- 既存 `compute_cross_venue_lead_lag_hint()` の `adverse_side` を活用
- buy-side 用閾値を config に追加（sell と独立）
- `skip_gate_evaluator.py` に buy-side cross-venue guard を追加

---

### Task 3: Protocol 688 type safety（P1-1）

**背景**: Codex レビューで protocol_688.py の `.get()` パターンに型安全性の問題あり。
hard-coded threshold もメンテナンス性を下げている。

---

### Task 4: Offset pipeline test validation（P1-2）

**背景**: multiplicative_pipeline.py のテストで数値検証が不十分。
stage disable flags のテストカバレッジ向上。

---

## 実行計画

- Task 1, 2 は独立かつ最高優先度 → 並列で Codex に投入可能
- Task 3, 4 は軽量 → Task 1, 2 完了後に投入

---

## 2026-04-02 実施結果

### 完了

- Task 1: AS-aware trailing skip filter
  - `as_trailing_tracker.py` を追加
  - `skip_gate_evaluator` に pre-ML gate と post-fill trailing record を接続
  - additive / multiplicative offset pipeline に `as_trailing_guard` stage を追加
  - FillRecord / early skip record observability を追加
- Task 2: Buy-side cross-venue protection
  - prompt 記載の `skip_gate_evaluator.py` ではなく、live path の正本である
    `maker_risk_guards.py` / `maker_price.py` に実装
  - buy-side 専用 veto / boost threshold を追加
  - `cross_venue_buy_offset_mult` を FillRecord に保存
- Task 3: Protocol 688 type safety
  - `Protocol688Config` を追加
  - spread bucket / adverse-selection severity を config 化
  - `run_protocol.py` に `--days > 0` validation と protocol execute error handling を追加
- Task 4: Offset pipeline math validation
  - public pipeline API を通す math validation test を追加
  - stage-disable flag の実挙動を current runtime semantics に合わせて固定

### hidden task / 横展開

- `cross_venue_buy_protect` は generic adverse retreat と競合させず、独立 threshold と telemetry だけ追加
- `as_trailing_gate` は evaluator だけで止めず、
  - `fill_cycle_executor`
  - `fill_record_builder`
  - `offset_pipeline`
  - `multiplicative_pipeline`
  まで通して drift を防止
- `test_skip_gate_v3.py` / `test_516_skip_gate_result_fields_migration.py` / `test_439_cross_venue_lead_lag.py`
  の stub / migration test も追随

### 回帰

- focused:
  - `test_694_as_trailing_tracker.py`
  - `test_694_cross_venue_buy_protect.py`
  - `test_694_protocol_688_type_safety.py`
  - `test_694_pipeline_math_validation.py`
  - `33 passed, 3 skipped in 2.88s`
- broader:
  - parser / validation / hot-reload / YAML drift / skip-gate / cross-venue /
    analysis protocol / offset pipeline / fill quality / migration / 694 tests
  - `476 passed, 3 skipped, 5 warnings in 10.42s`

### 残課題

1. `fill_quality` の report shaping 残分割
2. PPO warm-start の weight/state continuity 次段
3. heavy test setup の次 batch
   - `test_enricher_skip_gate.py`
   - `test_680_ppo_retrain_scheduler.py`
   - `test_sac_retrain_scheduler.py`
