# Codex / Gemini レビュー依頼プロンプト: 234# 〜 246#

## レビュアーへの依頼

以下のドキュメントと実装変更 (234# 〜 246#) をレビューしてください。
前回レビュー (232# Codex / 233# Gemini) で合意された構造的欠陥 — **balance_forced
による Kill Gate バイパスの全廃** — に対する対応と、その後の展開を評価してください。

---

## レビュー対象コミット (時系列順)

| # | commit | 概要 |
|---|--------|------|
| 234# | `b53483fc4` | **[P0] Gate bypass 廃止・縮退清算・片側エスカレーション** — 232#/233# で要求された `balance_forced` Gate 突破の全削除。代わりに Degraded Liquidation Mode (lot×0.2 + offset×3.0 + duty_cycle 1/3) を新設 |
| 235# | `6a70736e9` | 234# セルフレビュー: `duty_cycle=1` 永久 skip バグ修正、freeze 後即再発動ループ修正、dead parameter 削除 |
| 236# | `7559560e0` | State persistence 漏れ、CQS 分離、`hasattr` 排除、`consecutive_no_feasible` per-side 化 |
| 237# | `430e07c34` | **PhantomPositionGuard**: `status_unknown` 時の幽霊ポジション検知 — quarantine + 次サイクル残高差分再照合 |
| 238# | `b959a7dea` | 237# セルフレビュー: Protocol 型安全、残高スナップショット配線、TTL パージ、AS 準拠サイドベトー |
| 239# | `f7e1ba91c` | **Feasible Quote Proactive**: 制約前方移動 + `InfeasibleQuoteError` 型安全分類 |
| 240# | `30ddc3009` | **Toxicity Budget**: dynamic_kill 二値制御 → 4 段階 Glosten-Milgrom 連続応答 |
| 241# | `c141be4a2` | 240# セルフレビュー: graded response dead code バグ修正、評価順序不整合修正 |
| 242# | `014a0474c` | **Liveness Constraint Relaxation (233# P1)**: anti-stagnation probe 廃止、No Trade = 正常 |
| 243# | `2e322fd7f` | YAML Wiring Fix: 242# 新設定 (`quiescence_*`, `toxic_stale_multiplier`) 配線漏れ修正 |
| 244# | `be16dbc4f` | **Guard Reason Classification (232# P2-2)**: guard_fire_counts を MARKET/SYSTEM/RECOVERY 3 カテゴリ分類 |
| 245# | (doc) | **本番ログ分析**: 241# sha で 18 日間稼働した本番データ分析 (2,280 fills, cumPnL=-792 JPY) |
| 246# | `678a0fbc4` | **DD Halt Cooldown Release + Sell Defence Hardening**: halt 2h 後 lot30% 部分再開、sell offset/kill/veto 強化 |

---

## 前回レビュー (232#/233#) で合意された P0 要件の対応状況

| P0 要件 | 対応 # | 状態 | 備考 |
|---------|--------|------|------|
| Gate bypass (`and not balance_forced`) 全廃 | 234# | ✅ 完了 | `cycle_gate_aggregator.py` から全削除。Degraded Liquidation Mode で代替 |
| Liveness 制約への固執を捨てる | 242# | ✅ 完了 | anti-stagnation probe 廃止、`quiescence` 単純 sleep 化 |
| 「No Trade = 正常」を運用上許容 | 242# | ✅ 完了 | `quiescence_guard_fire` metric でサイレント停滞を可視化 |

---

## 本番実績 (245# 分析 — sha: c141be4a2947 / 241#)

### 数字の要約
- **期間**: 2026-02-13 〜 2026-03-03 (18 日間)
- **サイクル**: 5,726 / **約定**: 2,280 (FR=39.8%)
- **累積 PnL**: **-792 JPY** (avg -0.341bps/fill)
- **DD Halt**: 4/6 日発動 (15h+ idle/日)
- **残高**: JPY 1,100 + BTC 0.002 (buy 不可)

### サイド別パフォーマンス
- **Buy**: pass_pnl = -0.152bps (ほぼ損益分岐点)
- **Sell**: pass_pnl = **-1.316bps** (win_rate=34.4%, DEGRADED)

### レジーム別パフォーマンス
| regime | avg_pnl_bps | fill_rate |
|--------|------------|-----------|
| ranging | -0.075 | 41.5% |
| trending_down | **+0.385** | 37.5% |
| trending_up | **-0.919** | 27.7% |
| high_vol | -0.198 | 44.1% |

---

## 🔴 重要な視点転換: 在庫中立バイアスの是正

### 開発者からの指摘
245# の分析では「JPY 残高枯渇 → death spiral」と結論したが、**これは在庫中立
(inventory-neutral) への過度な固執に基づく誤った分析だった可能性が高い**。

**事実**: BTC 価格は分析期間中に上昇トレンドにあった。Bot が JPY を使い切って
BTC を購入したのは、**トレンドに順張りした正しいポジショニング**である。

**問題の本質**: JPY 枯渇それ自体ではなく、**上昇トレンド中に Bot が BTC を
売らされること**（forced sell、balance_forced、inventory skewing による中立化圧力）
こそが損失の真因。

233# で Gemini が明確に指摘していた:
> **「在庫が偏った ＋ 逆側が Dynamic Kill されている」状態は「今すぐ在庫を
> 戻さなければいけない」ではなく、「相場が落ち着くまでポジションを塩漬けに
> してでも絶対に動いてはいけない」が統計的最適解である。**

### レビュアーに問う
1. **Sell Defence Hardening (246#) は正しい方向か？** — offset_floor 引き上げは
   「売るけど広い offset で保護する」アプローチだが、そもそも **上昇トレンドで
   売ること自体を止めるべき** ではないか？

2. **在庫中立の前提は本 Bot に適切か？** — Avellaneda-Stoikov の在庫リスクモデルは
   mean-reverting 市場を前提とする。トレンド市場で在庫中立を強制すると、
   **トレンドに逆張りする（利益を捨てる）ことと等価** になる。

3. **BTC 保有の含み益を活用すべきではないか？** — 現在 BTC 0.002 保有。
   価格上昇による含み益は計上されていないが、これは **Bot の正しい判断の結果**
   であり、JPY 残高枯渇を「失敗」と見做すべきではない。

4. **トレンド方向のポジション保持は利殖機会** — 価格上昇中の BTC 保有は
   MM (マーケットメイキング) とは別の **directional alpha** であり、
   sell 強制による alpha 破壊を防ぐ設計が必要ではないか。

---

## レビュー観点 (具体的な質問)

### A. アーキテクチャ評価
1. 234# の Gate bypass 廃止 + Degraded Liquidation Mode は 232#/233# の要求を
   満たしているか？残存リスクはないか？
2. 237#-238# PhantomPositionGuard の設計は堅牢か？quarantine → 再照合のフローに
   誤検知/見逃しのリスクはないか？
3. 240#-241# Toxicity Budget の 4 段階 Glosten-Milgrom 応答は理論的に妥当か？
   二値 → 連続への拡張で失われた simplicity のコストは許容できるか？

### B. 市場構造への適応
4. **在庫偏重は「異常」ではなく「トレンドの結果」** — この認識に立った場合、
   以下の既存設計のどこを変えるべきか？
   - `inventory_skewing` (中立化圧力)
   - `one_sided_escalation` (片側取引制限)
   - `balance_forced` 残留ロジック
   - `sell_guard` / `trending_sell_skip`
5. trending_up レジームで sell を完全封鎖し、BTC 保有継続すべきか？
   仮にそうする場合、反転検知の信頼性は十分か？

### C. 収益性
6. 18 日間の累積 PnL が **-792 JPY** — 根本原因は何か？
   - (a) sell 側モデル劣化 (skip_gate DEGRADED)
   - (b) trending_up での逆張り sell 強制
   - (c) DD halt の機会損失 (15h+ idle)
   - (d) その他
7. 246# の cooldown release (2h 後 lot 30% 再開) は DD halt の機会損失軽減に
   有効か？lot 30% は保守的すぎるか／適切か？
8. **1 万円増やすために何が必要か？** — 現在の残高 (JPY 1,100 + BTC 0.002) から
   10,000 JPY の利益を出すための最も確実なパスは？

### D. コード品質・技術的負債
9. 234# 〜 246# で導入された新概念の数 (Degraded Liquidation, PhantomPositionGuard,
   Toxicity Budget, Quiescence, Guard Classification, Cooldown Release) —
   **複雑性は制御可能な範囲か？** 設計パターンの一貫性は保たれているか？
10. テスト: 3420 passed は十分か？カバレッジの盲点はないか？

### E. 次の優先施策
11. 上記の分析を踏まえ、**次にやるべき P0 施策** を提案してください。
    特に以下の軸で:
    - 在庫偏重許容 (directional position) の設計
    - sell 側モデルの緊急対応 (retrain? hard skip? offset only?)
    - DD halt 代替策 (cooldown release 以外)
    - 資金効率 (残高 ~1,100 JPY + BTC 0.002 での最適運用)

---

## 参考ドキュメント

- [232# Codex Pre-deployment Risk Review](docs/v460/232_ph2_rev_222_231_predeployment_risk_review.md)
- [233# Gemini Final Judgement](docs/v460/233_ph2_gemini_31_pro_final_judgement_and_breakthrough.md)
- [234# Gate bypass 廃止](docs/v460/234_ph2_fix_gate_bypass_degraded_liquidation.md)
- [237# PhantomPositionGuard](docs/v460/237_ph2_fix_phantom_position_guard.md)
- [240# Toxicity Budget](docs/v460/240_ph2_fix_toxicity_budget.md)
- [242# Liveness Constraint Relaxation](docs/v460/242_ph2_fix_liveness_constraint_relaxation.md)
- [244# Guard Reason Classification](docs/v460/244_ph2_impl_guard_reason_classification.md)
- [245# 本番ログ分析](docs/v460/245_ph2_production_log_analysis_mar03.md)
- [246# DD Cooldown + Sell Defence](docs/v460/246_dd_cooldown_release_sell_defense.md)

---

*Date: 2026-03-03*
*Review scope: 234# 〜 246# (13 commits, 3420 tests)*
*Previous review: 232# Codex / 233# Gemini (対象: 222# 〜 231#)*
