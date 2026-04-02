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
