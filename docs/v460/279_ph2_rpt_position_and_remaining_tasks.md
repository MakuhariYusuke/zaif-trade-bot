# 279# rpt: 0番ドキュメント立ち位置確認 + 残課題浚い上げ

**日付**: 2026-03-04
**種別**: rpt (調査・分析レポート)
**前提**: 278# (`04d9590eb`), docs commit `ce6d09d03`

---

## 1. 0番ドキュメント (000#) での現在地

### 1.1 フェーズ進捗

| フェーズ | Gate | 目的 | 状態 | 根拠 |
|---|---|---|---|---|
| **ph0** | — | プロジェクト提案・技術仕様 | ✅ 完了 | 000#/001# |
| **ph1** | G1-info | 情報ゲート: データ品質・特徴量 | ✅ PASS | 064#/065# |
| **ph2** | G1.1-exec | 執行ゲート: fill quality 実測 | **🔄 進行中** | 278# SHA で計測中 |
| **ph3** | G2-train | SAC 4-seed 訓練 | ⏳ 待機 | 一部先行 (015#/063#) |
| **ph4** | G3-pnl | コスト込み実収益性 | ⏳ 待機 | — |
| **ph5** | G4-live | Paper trading 本番前検証 | ⏳ 未着手 | — |
| **phg** | — | フェーズ横断品質改善 | 🔄 継続中 | 259#–278# |

**現在地**: ph2 G1.1-exec — 執行品質をリアル市場で実測検証するフェーズ。

### 1.2 Gate 通過状況

#### G1.1-quick (72h Kill Gate) — 過去に通過済み

000# §3.3 の Kill 指標 K1–K6 は過去判定で通過。
ただしコード変更が多く（234#–278# で 40+ コミット）、130# E.5 #3 の前提条件
「同一 Git SHA・同一 YAML で連続 72h」は度々リセットされている。

#### G1.2-full (168h Qualification Gate) — 未達

168h (7暦日) の連続クリーンデータ蓄積が必要。
278# (`04d9590eb`) デプロイ後、新たな計測期間が開始。

| 指標 | 閾値 | 現状 | 備考 |
|---|---|---|---|
| F1 attempted_fill_rate | ≥ 70% | 🟡 変動あり | 運用中に概ね達成圏内 |
| F2 attempted_cancel_ratio | ≤ 30% | 🟡 要確認 | F1 の裏 |
| F3 queue_wait_median | ≤ 60 sec | 🟢 概ね良好 | offset 最適化済み |
| F4 PnL30 | p ≥ 0.05 | 🟡 要計測 | 168h 分データ必要 |
| F5 AS_ratio | ≤ 30% | 🟡 Guard 制御中 | toxic_veto, inv_skew 等 |
| F6 skip_gate_ratio | ≤ 20% | 🟡 要確認 | SkipGate 閾値次第 |
| F7 calendar_coverage | ≥ 7 暦日 | ❌ 未達 | 278# から再計測開始 |
| F8 n_attempted | ≥ 500 | ❌ 未達 | 蓄積中 |

### 1.3 §3.9 中止ルールとの距離

| 条件 | 閾値 | 現状評価 | 判定 |
|---|---|---|---|
| fill_rate < 70% (n≥200) | **中止** | 運用中に変動するが概ね達成圏内 | 🟡 要監視 |
| AS_ratio > spread/2 が継続 (n≥500) | **中止** | Guard layer で制御中 | 🟢 管理下 |
| G1 再検証で全9ターゲット FAIL | **v461 移行** | ph1 PASS 済み | 🟢 該当なし |
| 累積実損 > 10,000 JPY | **一時停止** | 未到達 | 🟢 OK |

**結論**: 即時中止条件に該当するものはない。計測継続に問題なし。

### 1.4 §3.10 Gate 例外ルール

Exception-1 (Oracle Gap 継続例外) の発動は現時点で不要。
G1.1-quick は過去に PASS しており、Kill Gate での FAIL はない。

### 1.5 G2/G3/G4 への前進条件

| 次 Gate | 条件 | 現時点の距離 |
|---|---|---|
| **G2-train** (§3.4) | G1.2-full PASS → SAC 4-seed × 50K steps | G1.2 待ち。SAC 基盤は先行整備済み (015#/063#) |
| **G3-pnl** (§3.5) | G2 PASS → PF > 1.05, Sharpe > 0.8 | G2 未着手 |
| **G4-live** (§3.6) | G3 PASS → Paper trading 7日連続 | G3 未着手 |

**ボトルネック**: G1.2-full の 168h 計測完了が全後続フェーズの前提。

---

## 2. 残課題の浚い上げ

### 出典ドキュメント

| ソース | 内容 | 最終更新 |
|---|---|---|
| 000# §3.3–§3.10 | Gate 定義・中止ルール・例外ルール | 170# |
| index.md 重点課題 | 最優先〜低優先の 15 項目 | 277# |
| 259# Sweep | 型安全・複雑性・市場理論・改善提案 P1/P2/P3 | 2026-03-04 (v2) |
| 277# セルフレビュー | C1/S3 type safety deferred | 277# |

### 2.1 🔴 最優先 — Gate 進捗を直接ブロック

| ID | 課題 | 出典 | 状態 | 期待完了 |
|---|---|---|---|---|
| **R-1** | fill_test 168h 再実測完了 — 278# SHA でクリーン7暦日蓄積 | 000# §3.3 F7/F8 | 計測中 | ~2026-03-11 |
| **R-2** | G1.1-exec Gate 正式判定 — F1–F8 全指標算出・記録 | index.md 最優先 | R-1 待ち | R-1 直後 |

### 2.2 🟠 高優先 — 収益性直結

| ID | 課題 | 出典 | 状態 | Gate並行 |
|---|---|---|---|---|
| **R-3** | SkipGate 再訓練 — n_attempted≥500 でpreorder features再訓練 | 097#/index.md | データ蓄積待ち | ✅ 可 |
| **R-4** | spread_adaptive AB テスト — narrow_spread_bps 探索 | 093#/index.md | 未着手 | ⚠️ YAML変更→計測リセットリスク |
| **R-5** | Volatility Guard 動的ゲーティング | 107#/index.md | 設計済み・未実装 | ⚠️ コード変更→リセットリスク |

> **注意**: R-4/R-5 はコード/YAML 変更を伴うため、G1.2 計測中に実施するとSHA/YAML連続性がリセットされる。**R-1 完了後に実施が望ましい**。

### 2.3 🟡 中期 — ph5 本番前に必須

| ID | 課題 | 出典 | 備考 |
|---|---|---|---|
| **R-6** | `OrderManager.execute_trade()` — 実取引パス実装 | 013# D-1 | ph5 blocker |
| **R-7** | `post_only` 対応 — maker 保証 | 013# D-3 | ph5 blocker |
| **R-8** | `asyncio.to_thread` 残 5 メソッド | 013# C-4 | 非同期化完了 |
| **R-9** | Tier-2/3 統合: PnL MC, RiskRuleEngine, Reconciliation | 113# | ph5 品質保証 |

### 2.4 🔵 コード品質 — 259# Sweep 残

#### P1 (高優先改善)

| ID | 課題 | 259# ID | 現状 |
|---|---|---|---|
| **R-10** | `run_continuous` 分割 — 265# で 1694→1221行、まだ巨大 | P1-1 | 部分完了。目標 <500行 |
| **R-11** | adapter Protocol化 徹底 | P1-2 | 261#/262# で部分完了。残あり |
| **R-12** | `order_monitor` except narrow化 (12箇所) | P1-3 | 未着手 |

#### P2 (中優先改善)

| ID | 課題 | 259# ID | 現状 |
|---|---|---|---|
| **R-13** | OrderBook Protocol (OB) | P2-1 | 261# で一部実装 |
| **R-14** | SkipDecision 必須化 | P2-2 | 未着手 |
| — | ~~Kelly Criterion~~ | ~~P2-3~~ | ✅ **264# で完了** |
| **R-15** | `run_single_cycle` 分離 | P2-4 | 265# で部分着手 |
| **R-16** | evaluate 分離 | P2-5 | 未着手 |

#### P3 (低優先改善)

| ID | 課題 | 259# ID | 現状 |
|---|---|---|---|
| **R-17** | Optional 統一 | P3-1 | ✅ **263# で87箇所完了** |
| **R-18** | except 変数なし修正 | P3-2 | 未着手 |
| **R-19** | config_access docstring | P3-3 | 未着手 |

#### 型安全残数 (259# 集計)

| カテゴリ | 残数 | 削減トレンド |
|---|---|---|
| `getattr` | 29 箇所 | 253#–266# で段階的に削減中 |
| `hasattr` | 6 箇所 | 228#/230# で大半排除、残少 |
| `:object` 型 | ~20 箇所 | 277# C1/S3 deferred |
| `type:ignore` | 12 箇所 | 266# で 4 箇所排除、残 12 |
| `except Exception` | 78 箇所 | order_monitor 12, skip_gate_evaluator 12 が主 |
| `Any` 型 | 0 ✅ | 109# で完全撤去済み |
| `bare except` | 0 ✅ | 253# で完全撤去済み |

### 2.5 ⚪ 低優先 — v461+

| ID | 課題 | 出典 |
|---|---|---|
| **R-20** | SkipGate 単体テスト拡充 | 106# R3 |
| **R-21** | lib → ztb 移動 (残 4 モジュール) | 106# R5 |
| **R-22** | utils 70+ ファイル分割 | 106# R6 |
| **R-23** | config/ vs configs/ 重複ディレクトリ整理 | 106# R7 |
| **R-24** | UnifiedTrainer God Object (2835行) | 109# DUP3 |

### 2.6 277# セルフレビュー残 (Deferred)

| ID | 課題 | リスク | 理由 |
|---|---|---|---|
| **R-25** | C1: `_mcb: object | None` → Protocol 型 | LOW | TYPE_CHECKING リファクタ必要、機能影響なし |
| **R-26** | S3: `_regime_detector: object | None` → Protocol 型 | LOW | 同上 |

---

## 3. 優先度マトリクス

```
                    収益インパクト大
                         │
            R-1,R-2      │      R-3
           (Gate判定)     │   (SkipGate再訓練)
                         │
   緊急度高 ─────────────┼──────────────── 緊急度低
                         │
            R-6,R-7      │      R-10,R-12
           (ph5 blocker) │   (品質改善)
                         │
                    収益インパクト小
```

### 推奨アクション順序

```
Phase A: 計測凍結期間 (~7暦日)
  R-1  fill_test 168h 蓄積 (コード変更凍結)
  ↓
Phase B: Gate 判定
  R-2  G1.2-full F1–F8 判定
  ↓
  ┌─ PASS → Phase C へ
  └─ FAIL → §3.10 例外適用 or 対策実装 → Phase A に戻る
  ↓
Phase C: 収益性改善 (Gate PASS 後)
  R-3  SkipGate 再訓練
  R-4  spread_adaptive AB テスト
  R-5  Volatility Guard
  ↓
Phase D: ph3 移行準備
  R-6,R-7  OrderManager + post_only
  R-10     run_continuous 更なる分割
```

---

## 4. 総合所感

### プロジェクトの核心的ボトルネック

**G1.2-full 168h の安定した計測完了**。000# から ph5 まで5つの Gate が直列で、G1.2 が全体の律速段階になっている。278# まで 40+ コミットの修正を重ねた結果、計測期間の連続性が破れ続けている。

### 肯定的要因

- ph1 (G1-info) は PASS 済み — 情報量の存在は検証済み
- §3.9 中止条件に該当なし — 戦略の根本的不成立は見られない
- コード品質: `Any` 型 0、`bare except` 0、78回のテスト通過 (3827 passed)
- 市場理論の実装: Kelly (264#), GLFT τ (266#), AS reservation (258#), 10+ 理論を統合済み
- Guard layer: 20+ の防御施策が段階的に構築・検証済み

### リスク要因

- 278# は1行修正だが、`degraded_liquidation` パスの到達条件次第で同種の未テストパスが残存する可能性
- `run_continuous` が依然 1221行 — メンテナンス性の技術的負債
- `except Exception` 78箇所 — エラー原因の特定を遅らせる構造
- Gate 計測とコード改善の二律背反 — 改善するとリセット、放置すると品質低下

### 戦略的推奨

**今後7暦日はコード変更を完全に凍結し、R-1 (168h計測) に集中する。** これがプロジェクト全体のスループットを最大化する唯一の道筋。

---

## 関連

- 000# [000_ph0_plan_project_proposal.md](000_ph0_plan_project_proposal.md) — Gate 定義、§3.9/§3.10
- 259# [259_phg_rpt_codebase_sweep.md](259_phg_rpt_codebase_sweep.md) — 型安全・品質 Sweep
- 277# [277_magic_number_grounding.md](277_magic_number_grounding.md) — セルフレビュー・deferred 項目
- 278# [278_ph2_fix_degraded_liquidation_min_lot.md](278_ph2_fix_degraded_liquidation_min_lot.md) — 直前の修正
