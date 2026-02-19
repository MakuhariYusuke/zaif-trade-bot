# 118# 残課題・未検討提案の深掘り考察

**Date**: 2026-02-19  
**Updated**: 2026-02-20 (追記: §5/§8 disposition, §9 実行, Appendix B/C/D2/F)  
**Phase**: phg (フェーズ横断)  
**前提文書**: 000#–117# 全 96 文書 + fill_test 144.8h 実測データ

> **凡例**: 本文書内の「追記」「§9-A1」等のタグは、初版 (2/19) 以降のセッションで
> 本文書に直接追加・更新した内容を指す。対応するコミットハッシュは Appendix B に記載。

---

## 目次

- [§0 エグゼクティブサマリ](#0-エグゼクティブサマリ)
- [§1 現在地の客観評価](#1-現在地の客観評価)
- [§2 Gate 判定直結の課題 (P0)](#2-gate-判定直結の課題-p0)
- [§3 収益性直結の未修正問題 (P0–P1)](#3-収益性直結の未修正問題-p0p1)
- [§4 SkipGate / ML の構造的限界](#4-skipgate--ml-の構造的限界)
- [§5 未検討のまま棚上げされた提案](#5-未検討のまま棚上げされた提案)
- [§6 本番前 (ph5) に必須のインフラ課題](#6-本番前-ph5-に必須のインフラ課題)
- [§7 コード品質・技術的負債 (低優先)](#7-コード品質技術的負債-低優先)
- [§8 新規考察: ドキュメントから見えない盲点](#8-新規考察-ドキュメントから見えない盲点)
- [§9 提案: 次の行動計画](#9-提案-次の行動計画)
- [Appendix A: 53 OPEN items 全件一覧](#appendix-a-53-open-items-全件一覧)
- [Appendix B: 42 RESOLVED items](#appendix-b-22--42-resolved-items)
- [Appendix C: §5 未検討提案 Disposition サマリ (追記)](#appendix-c-未検討提案-5-全件-disposition-サマリ-追記)
- [Appendix D2: §8 Disposition サマリ (追記)](#appendix-d2-8-全件-disposition-サマリ-追記)
- [Appendix F: SkipGate 再訓練計画 (特徴量統合)](#appendix-f-skipgate-再訓練計画-特徴量統合)

---

## §0 エグゼクティブサマリ

### 大義への距離

本プロジェクトの大義は **「短期間での高収益性システム」** (000# §0)。
現在地は ph2 (G1.1-exec gate) の 168h fill test 終盤 (144.8h/168h 経過)。

**結論**: Gate 判定指標の大半は閾値付近またはギリギリ PASS 水準にあるが、
**PnL30 が -0.172bps (有意に負の可能性)** であり、
F4 (PnL30 p ≥ 0.05) を PASS できるかが最大の焦点。

### 数字で見る現状 (144.8h / 1099 cycles)

| 指標 | 実測値 | G1.1-quick (K) | G1.2-full (F) | 判定 |
|------|--------|----------------|---------------|------|
| attempted_fill_rate | **73.9%** | ≥60% ✅ | ≥70% ✅ (ギリギリ) | ⚠️ |
| overall_fill_rate | **67.3%** | — | ≥62% ✅ | ✅ |
| queue_wait_median | **12.7s** | ≤120s ✅ | ≤60s ✅ | ✅ |
| PnL30 mean | **-0.172bps** | mean>-0.8 ✅ | p≥0.05 (?) | **⚠️ 要検定** |
| AS_ratio | **30.8%** | — | ≤30% ❌ | **❌** |
| skip_gate_ratio | **8.8%** | ≤25% ✅ | ≤20% ✅ | ✅ |
| calendar_coverage | **~6日** | — | ≥7日 ❌ | ❌ (プロセス停止) |
| n_attempted | **1002** | — | ≥500 ✅ | ✅ |

**F4 (PnL30)** と **F5 (AS_ratio ≤30%)** が Gate PASS の鍵。
特に AS_ratio 30.8% は閾値 30% を僅か 0.8pt 超過しており、
改善施策の実効性次第で PASS/FAIL が分かれる。

### 全体像: 53 OPEN / 22 RESOLVED / 24 提案未検討 → 追記: **42 RESOLVED / 11 残留**

| カテゴリ | OPEN | RESOLVED | 要点 |
|---------|------|----------|------|
| A. Gate 判定関連 | 7 | 3→5 | F4/F5 Gate待ち。Holm/auto-judgment ✅ |
| B. 収益性直結 | 10 | 4→8 | warm_start/sell SG/L2/window/regime warm-up ✅ |
| C. 本番前必須 | 10 | 6 | ph3 ブロッカー: WF/Oracle/execute |
| D. SkipGate/ML | 8 | 3→7 | regime復活/sell分離/特徴量統合 (Appendix F) + D8テスト |
| E. インフラ/コード | 12 | 4→7 | API/post_only/非定常性実測/MC統合✅ |
| F. 未検討提案 | 24 | —→**10** | §5 全 10 件 disposition 完了 (App C) |
| G. v461+ | 13 | 2 | 棚上げ確認済 |

---

## §1 現在地の客観評価

### 1.1 v460 の Phase フロー (000# §2 に基づく)

```
ph0 (計画)  ──✅──→  ph1 (G1-info)  ──✅──→  ph2 (G1.1-exec)  ──⚠️──→  ph3 (G2)
                                          ↑ 現在ここ (fill_test 144.8h)
```

ph2 開始から **6 日間** が経過し、1099 cycle を蓄積。
ph1 での G1-info PASS (XGBoost OOS Spearman IC > 0.02) を根拠に maker 執行検証に進んだ流れは健全。

### 1.2 v459 教訓との照合

| v459 教訓 (000# §1) | v460 対応状況 | 評価 |
|---------------------|-------------|------|
| ①特徴量を先に検証 | G1-info で XGBoost 検証済。70# で 72+ モデルサーチ | ✅ |
| ②単一 seed を信じるな | ph3 で 4-seed 検証予定。fill_test は 1 設定 | ⚠️ 未検証 |
| ③Oracle テスト早期実行 | **未実施**。111# §6.3 #3 で提案あり | ❌ |
| ④手数料は前提条件 | maker 0% (Coincheck) で設計。post_only 実装済 | ✅ |

**教訓③ Oracle テストの未実施は重大な見落とし**。
v459 では「完全予測でも taker 0.1% では費用負け」が判明したのが Phase E (遅すぎた)。
v460 でも maker 0% とはいえ、AS コストが Oracle PnL を超えていないか、
ph3 進入前に Oracle baseline を確認すべき。

### 1.3 プロセス停止の問題

fill_test (PID 31236) は 2026-02-19 19:31 (cycle 1100) で停止。
最終ログ: `[lock] Released lockfile` — プロセスが正常終了した可能性と異常終了の両方あり。

**影響**:
- F7 (calendar_coverage ≥7 日): 実稼働は 2/13 18:39 → 2/19 19:31 = **5.98 日** (ギリギリ不足)
- 「7 暦日」基準では 2/13→2/20 0:00 をカバーすべき → **4.5h 不足**
- 再起動して残り 23.2h を走らせる必要がある

---

## §2 Gate 判定直結の課題 (P0)

### 2.1 F4: PnL30 が有意に負でないこと

**現状**: mean = -0.172bps (n=740)。F4 は片側 t 検定で p ≥ 0.05 を要求。

```
帰無仮説: μ ≤ 0 (PnL30 は 0 以下)
対立仮説: μ > 0 (PnL30 は正)
```

PnL30 の標準偏差が大きい場合 (BTC/JPY の板でσ ≈ 3-5 bps は典型的)、
n=740 でも mean=-0.172 は p < 0.05 になる可能性が十分ある。

**考察**: F4 の解釈は「有意に負でないこと」= 片側検定で「μ < 0 を棄却できない」こと。
つまり **H0: μ ≥ 0 vs H1: μ < 0** の片側検定で **p ≥ 0.05 なら PASS**。
mean = -0.172bps でも分散が大きければ PASS する可能性はある。

**対策案**:
1. PnL30 の分散を下げる → AS 回避強化 (§3 で詳述)
2. fill_test を延長して n を増やし、正のトレンドが拾えるか観察
3. PnL120 (= +0.220bps) が正であることは、hold 期間延長が有効な可能性を示唆

### 2.2 F5: AS_ratio ≤ 30%

**現状**: 30.8% (228/740)。0.8pt 超過。

**内訳**:
- BUY: AS 29.8% (111/373) — ✅ 閾値内
- SELL: AS 31.9% (117/367) — ❌ 超過

SELL 側の AS 率が 2.1pt 高い。これは 098# §4.2 で指摘された
「sell SkipGate 逆選別」問題と一致する (P(AS)≥0.50 の売群が実は好 PnL)。

**対策案**:
1. sell 側 SkipGate skip 強化 (逆効果の可能性あり — 上記逆選別問題)
2. sell 側 offset 引き上げ (sell_offset_floor 0.08→0.12)
3. sell 保持時間延長 (120s でプラス転換するデータあり)
4. fast_fill_defense の sell 側再設計 (098# P0-A)

### 2.3 Holm-Bonferroni 補正の未実装 (A4)

000# §3.7 で明記された統計仕様だが、**自動判定パイプラインへの組込みは未実施**。
G1 (9 ターゲット) とG1.2 (F1-F8 の 8 指標) の多重検定に必須。

**影響**: 補正なしの場合、偶然の PASS を含む可能性がある。
特に F4 (PnL) がボーダーラインの場合、Holm 補正後に FAIL に転じるリスク。

**実装コスト**: 低。`ztb.metrics.metrics.p_mean_method()` が既存。
`scipy.stats` の `multipletests(method='holm')` を使えば数行。

### 2.4 Gate 自動判定パイプライン (A6)

116# で `g1_1_quick_judgment()` / `g1_2_full_judgment()` が実装されたが、
111# §10 の SLO 閾値表 (11 指標) との統合は手動。

**提案**: 168h 到達時の自動判定スクリプト (`scripts/v460/gate_judgment.py`) を整備し、
fill_records JSONL から直接判定を出力する。

---

## §3 収益性直結の未修正問題 (P0–P1)

### 3.1 fast_fill_defense has_negative_edge 検出漏れ 50% [P0]

**出典**: 098# §3.1, 099# P0-3

**問題**: `has_negative_edge` は `fill_price < mid_price` で判定するが、
maker sell の場合 `fill_price ≥ ask ≥ mid` が常態のため、**sell 側は常に `fill_price > mid` → 非検出**。
buy 側でも遅延による mid 変動で検出漏れが発生し、全体で **検出率 50%**。

**未実装の改善案**:
```
[現在] proxy判定: fill_price < mid → 即時ブースト
[提案] 二層化:
  Layer 1: 即時 proxy判定 (現行の改良版: side別閾値)
  Layer 2: 30s 後の事後 PnL 確定判定 → 次サイクル反映
```

**収益インパクト**: 098# 分析で未検出 AS 18 件 × -4.9bps の半数回避 → **約 +0.2bps**

**考察**: fast_fill_defense は v460 の逆選択防御の中核コンポーネントだが、
現在は **実質的に sell 側で機能していない**。
これは SELL PnL30 = -0.632bps (BUY +0.281bps) の非対称性の直接原因の一つ。
二層化の実装コストは中程度 (30s 後の PnL をフィードバックするコールバック追加) だが、
**最もコスト対効果の高い施策の一つ**。

### 3.2 SkipGate warm_start 閾値未復元 [P0]

**出典**: 098# §3.2, 099# P0-1

**問題**: プロセス起動時、SkipGate threshold は warm_start で step=0.02 ずつ目標に近づく。
目標到達まで **6 cycle (= 約 12 分) が無防備**。

**未実装の改善案**:
- warm_start 完了後に 1 回 `calibrate()` を呼び、目標値に即時復元
- または adaptive_step を 0.02 → 0.05 or 0.10 に引き上げ (3 cycle → 1-2 cycle)

**収益インパクト**: 起動初動 6 cycle の AS 率は通常より高い → **約 +0.1bps**

**考察**: 極めて低コストの修正 (数行変更)。リスクも低い。
**即座に実装すべき項目**。

### 3.3 Regime Detector warm-up 問題 [P1]

**出典**: 098# §3.4

**問題**: `RegimeDetector` は `window=20` で 20 サイクル分のデータが必要。
097# の短期実験 (13 cycle) では **全件 "unknown"** で trending/ranging ブーストが一切不発動。

**現在の fill_test (1099 cycle)** では warm-up 済みだが、再起動のたびに 20 cycle が "unknown" に。

**未実装の改善案**:
- warm_start 時に直近 N 件の price feed を injection
- window を 20 → 10 に短縮 (感度は上がるがノイズも増加)
- state persistence で regime 状態を保存・復元

**考察**: 113# で StatePersistence は実装済みだが、regime_detector の状態は
含まれていない可能性が高い。状態保存の対象に追加するのが最善。

### 3.4 param_adapter の全履歴使用問題 [P1]

**出典**: 098# §3.6, 095# §5.2

**問題**: `param_adapter` は **全レコード** を使ってパラメータを学習するため、
古い設定下のデータが最新設定の効果を薄める (dilution effect)。

**提案**: recency window (直近 N 件) を導入。I/O も逆順読みで O(total) → O(window)。

**考察**: 1099 件蓄積された今、この問題は実測可能。
直近 200 件と全件で param_adapter の出力を比較し、乖離が大きければ
window 導入の根拠になる。低コスト検証として優先度を上げるべき。

### 3.5 Sell 側 PnL 非対称性の根本原因 [P0 総合]

**データ**: BUY +0.281bps vs SELL -0.632bps (差: 0.913bps)

この非対称性の原因は複合的:
1. **fast_fill sell 側無効** (§3.1) → sell 側の AS 防御が弱い
2. **sell SkipGate 逆選別** (098# §4.2) → skip すべきでない取引を skip
3. **sell offset が低い** (sell_offset_floor 0.08→引き上げ余地)
4. **BTC/JPY 市場の構造的バイアス**: sell 注文は bid side に置くため、
   上昇トレンドで informed buyer に taken されやすい

**重要**: PnL120 で見ると mean = +0.220bps と反転する。
つまり **30 秒で逆行しても 120 秒で回復するケースが多い**。
これは sell 側の「保持期間延長」が有効な施策であることを示唆。
095# §7 M7 / 099# §4 P2 で提案されたが**未検討**。

---

## §4 SkipGate / ML の構造的限界

### 4.1 事実の整理

| 時期 | データポイント |
|------|-------------|
| 095# | SkipGate は**完全ランダム分類器** (P(AS)=0.50±0.03, skip 0 件, ROC-AUC=0.442) |
| 097# | preorder-only features で再訓練 → skip20% で +0.405bps 改善 |
| 107# | target_skip_rate 引上げ → fill_test 投入 |
| 現在 | skip_gate_ratio 8.8% (97/1099)。閾値 ~0.53 で AS_prob > 0.53 の時のみ skip |

### 4.2 問題の構造

SkipGate の AS 予測精度が低い (AUC ≈ 0.5) ため、
skip は事実上「高い AS_prob を持つサイクルをランダムに除外」する効果しかない。

**売買非対称性**: sell 側では SkipGate が**逆選別**を起こしている疑い:
- skip した sell 群の真の PnL > keep した sell 群の真の PnL
- 理由: sell 側の AS ドライバーが buy 側と異なる (spread_bps, hour_sin/cos が効く)

### 4.3 根本的選択肢

| 選択肢 | 実装コスト | 期待効果 | リスク |
|--------|----------|---------|-------|
| A. 現行維持 + target_skip 微調整 | 低 | 限定的 | 逆選別継続 |
| B. sell 側 SkipGate 無効化 | 極低 | sell逆選別解消 | AS増加 |
| C. buy/sell 別モデル | 高 (データ分割→n不足) | 高 | 過学習 |
| D. ルールベース併用 (threshold + rule) | 中 | 中 | ルール設計依存 |
| E. 特徴量再設計 (sell 専用) | 高 | 高 | 時間コスト |
| F. SkipGate 全廃 + time_filter で代替 | 低 | 中 | 機会損失 |

**推奨**: **B + D の組合せ** — sell 側 SkipGate を一旦無効化し、
ルールベース (spread < 10bps AND vpin > 0.7 → skip) で代替。
データが 500+ 蓄積した段階で C に移行。

### 4.4 500 サンプル再訓練の展望

現在 1099 records (うち filled 740)。097# では 215 サンプルで訓練。
filled 740 件は十分な量だが、**sell 側は 367 件のみ**。
buy/sell 分割訓練には各 200+ が最低ライン → **可能** (buy: 373, sell: 367)。

sell 専用特徴量候補 (098# §4.3):
- `spread_bps` (sell 側は narrow spread で AS 増加する傾向)
- `hour_sin / hour_cos` (時間帯依存性が強い)
- `recent_sell_pnl_mean` (直近 sell PnL の running average)

---

## §5 未検討のまま棚上げされた提案 — Disposition (追記)

以下は各ドキュメントで提案されながら、後続文書で明示的に検討・却下されていない項目。
**追記で全件 disposition を確定:**

### 5.1 AS 判定 horizon 30s → 60s 変更 [098# §6]

**状態**: 📋 **棚上げ妥当 (E3 informational で代替済)**

**提案内容**: 30s 時点の逆行で AS と判定するが、120s では回復するデータがある。
AS 判定を 60s に延長すれば、AS_ratio が大幅に低下する可能性。

**実測データ (追記)**:
- PnL30 = +0.002bps (n=759) → 前回 -0.172bps から改善
- PnL120 は正値を維持 (VG群 +1.735bps)
- B2 で F4b (PnL60s) / F4c (PnL120s) を Gate に追加済

**結論**: Gate 定義の根幹に関わるため変更すべきでない。
**E3 (120s) PnL は B2 で Gate の informational 指標に追加済み** — 実質対応完了。

### 5.2 Event-driven サイクル間隔 [088# P2-1]

**提案内容**: 現在の 120s 固定サイクルを、板変動・約定フローをトリガーにした
イベント駆動に変更。

**考察**:
- 091# で「120s 維持」と結論。092# で「LOW」とタグ付け
- **正当な棚上げ**。120s 固定はシンプルで予測しやすく、
  イベント駆動は WebSocket の安定性に依存してリスクが高い
- ph5 以降の改善候補としては有望だが、ph2 段階では不要

**結論**: **棚上げ妥当**。v461+ の検討事項。

### 5.3 Round-trip を primary KPI に [088# P1-1]

**提案内容**: 現在の 1-leg (buy or sell) ベースの KPI を、
round-trip (buy→sell の往復) ベースに変更。

**考察**:
- 092# で E6 (round-trip KPI) を `informational=True` で追加済
- round-trip KPI は**より実態に近い**が、fill_test は 1-leg 設計
- round-trip 化すると hold 期間管理が必要 → 設計変更が大きい
- **1-leg が黒字でも round-trip で赤字になるケースはあり得る** (逆も然り)

**結論**: E6 informational の蓄積を先行。Gate 昇格は ph3 以降。

### 5.4 Volatility Guard [107# Phase 2]

**状態**: ✅ **実装済み + 効果測定完了 (追記)**

fill_test ログで `volatility_guard` 発動を確認:
```
[volatility_guard] 107# buy offset boosted: 0.1125→0.2250 (vpin=0.95)
```

**§9-A4 効果測定結果** (`vg_and_trend.py`):
- VG 発動 135 サイクル (全体の 14.1%)
- AS rate: VG群 **21.5%** vs 非VG群 **29.2%** → **-7.7pt 改善 (有効)**
- PnL120: VG群 **+1.735bps** vs 非VG群 **-0.148bps** → VG 環境でも長期回復
- fill_rate: VG群 58.5% vs 非VG群 71.7% → offset boost で約定率は低下するが AS 防御に成功
- **結論: VG は有効。offset_boost_factor 2.0x は適切。**

### 5.5 Smart Side 再有効化 [091# §1#7]

**状態**: 📋 **次回再訓練時にデータ分析 → 実装判断**

**提案内容**: order book imbalance に基づく smart side selection を再有効化。
現在は OB 取得無効化のため `_last_imbalance` が更新されず、dead code 化。

**データによる考察 (追記)**:
- buy PnL30 = **+0.222bps** vs sell PnL30 = **-0.678bps** → **BUY が明確に優位**
- 071#/072# のOBトグル化で OB 特徴量不使用に移行済
- price_velocity_60s は preorder 16特徴量に含まれ (097#)、
  side_aligned_velocity として SkipGate に既に使用中
- BUY/SELL どちらが有利かの判断は既に **side_aligned_velocity** で暗黙的に行われている

**結論**: side_selector.py への直接組込みは不要。
再訓練時に `side_aligned_velocity` の重要度を確認し、
重要度が高ければ side selection に反映するルールを検討。
**現状は price 情報が SkipGate 経由で間接的に side 判断に寄与しており、棚上げ妥当**。

### 5.6 time_filter 段階的廃止 [107# Phase 3]

**提案内容**: 3 段階で time_filter を縮小→最終的に全廃。
- Step 1: BUY 7h→3h, SELL 5h→3h (48h 観察)
- Step 2: 最悪時間帯のみ (BUY 1h, SELL 1h)
- Step 3: 全廃

**考察**:
- 107# Phase 1/2 (SkipGate 強化 + Volatility Guard) は実装済み
- **Phase 3 Step 1 は未着手** (48h 観察が必要とされていた)
- time_filter の遮断は BUY 29.2% / SELL 25.0% → **大量の機会損失**
- **Volatility Guard が機能していれば、time_filter の必要性は低下する**
- ただし、Volatility Guard の効果測定なしに time_filter を緩和するのはリスク

**結論**: Volatility Guard の効果を定量評価した上で、Step 1 を実施すべき。
**次回の fill_test 再起動時に time_filter パラメータを変更するのが自然なタイミング**。

### 5.7 Order timeout 短縮 300s → 120-150s [095# M4]

**状態**: ✅ **解決済み** (096# で 300s → 90s に短縮)

**提案内容**: timeout 115 件 (cancel の 31.9%)。300s は長すぎる。

**考察**:
- wait_median = 12.7s。filled の大半は 60s 以内で約定
- 300s 待機中の市場変動リスクが AS の原因になっている可能性
- **timeout を 150s にしても fill_rate への影響は小さい** (60s 以降の約定は少ない)
- ただし、timeout 短縮は fill_rate を僅かに低下させる → F1 への影響

**結論**: fill_records の wait_sec 分布を分析し、
p95 以下のカットオフで timeout を設定すべき。150s が妥当な推定。

### 5.8 Offset 体系的探索 [095# M10, 098# §2.6]

**提案内容**: 現在の offset (0.1125~0.30) を体系的に AB テストで探索。

**考察**:
- 098# §2.6 で「high offset ≒ 悪環境」の因果バイアス指摘
- offset が高いサイクルは Volatility Guard やregime による自動引き上げ
- 真の offset 効果を分離するには、**ランダム化 AB テスト** が必要
- 093# §6#2 で narrow_spread_bps 探索も提案されているが未着手

**結論**: fill_test 168h 完了後、Gate 判定の結果次第で AB テスト設計に進む。
Gate FAIL なら AB テストの前に §3 の P0 施策を優先。

### 5.9 Sell 保持期間延長 [095# M7, 099# P2]

**状態**: ⏳ **ph3 以降で検討 (B2 で PnL120 Gate 追加済)**

**提案内容**: sell 30s PnL = -0.632bps だが、120s PnL はプラス転換するデータあり。
sell 約定後の保持期間を延長する。

**データ実績 (追記)**:
- sell PnL30 = **-0.678bps** (n=377) — 依然として負
- VG群の PnL120 = **+1.735bps** → 120s で回復傾向は健在
- BTC/JPY は平均回帰傾向あり (30s 逆行 → 120s 回復)
- fill_test は 1-leg 設計で「保持」概念がない

**結論**: PnL120 は **B2 で F4b/F4c として Gate に追加済** — データ蓄積は完了。
ph3 (SAC) で保持期間を action に組込む設計が自然。
fill_test 段階での設計変更は不要。**ph3 以降に defer**。

### 5.10 fast_fill_defense 持続時間制御 [093# §6#4]

**提案内容**: 現在は 1 cycle で boost が消滅。N cycle 持続させる。

**考察**: fast_fill_defense が売側で実質無効 (§3.1) な状態で持続時間を延ばしても効果は限定的。
先に §3.1 の二層化を実装すべき。

**結論**: §3.1 解消後に再検討。

---

## §6 本番前 (ph5) に必須のインフラ課題

### 6.1 OrderManager.execute_trade() TODO [013# D-1]

**問題**: 実取引パスが TODO のまま。fill_test は maker limit order の place/cancel のみで、
market order や成行変換のロジックが未実装。

**ph3/ph5 影響**: SAC エージェントが出す action は {buy, sell, hold} で、
execute_trade() がそれを注文に変換する。TODO のままでは ph4 以降に進めない。

**考察**: fill_test 完了後の ph3 移行時にブロッカーになる。
ただし、fill_test の maker-only 戦略をそのまま ph3 以降も使うなら、
execute_trade() は fill_test のロジックをラップする形で実装可能。

### 6.2 v458 Walk-Forward バグ 6 件 [111# §5.1-5.2]

| # | バグ | 影響 |
|---|------|------|
| P0-1 | Entry Gate crash (np.nan 除算) | WF 全体停止 |
| P0-2 | Fee 二重カウント (0.1% + 0.1%) | 訓練の reward信号が汚染 |
| P0-3 | Val/Test 汚染 (1-fold overlap) | 過学習を検出できない |
| P1-1 | Trade 誤分類 (partial fill = cancel) | Profit Factor 過大評価 |
| P1-2 | Reporter 3 重定義 | メンテナンス困難 |
| P1-3 | CalibrationMap 未ロード | 実行時 crash |

**考察**: ph3 (SAC 4-seed 訓練) で Walk-Forward を使う場合、
P0-1 ~ P0-3 は **確実に訓練結果を汚染** する。
ph3 進入前の修正が必須。

### 6.3 運用失敗モードテスト [112# §3.3, 113# §7 A2]

**テスト対象**:
- API 429/5xx burst (rate limit 超過)
- 再起動復元 (state persistence の整合性)
- OOM (memory leak regression)
- 板薄化/急変 (spread 急拡大)

**考察**: CircuitBreaker と HealthMonitor は実装済み (113#) だが、
**実際に障害を注入してテストしていない**。
「テストされていない回復機構は回復機構ではない」。

### 6.4 ph3 Stop 条件の明文化 [112# §3.4]

**提案内容**:
- 4-seed worst ROI 閾値: 具体的数値の確定
- Oracle fee 確認: 完全予測で maker 0% でも収益が出るか
- WF P0/P1 未解消 → 凍結

**考察**: ph2 Gate PASS 後に ph3 に突入する際、
「どうなったら ph3 を中止するか」が明文化されていない。
000# §3.4 に G2-train の条件はあるが、**中止条件が弱い** (worst-seed ROI > -2%)。

---

## §7 コード品質・技術的負債 (低優先)

| # | 項目 | 行数影響 | 優先度 | 備考 |
|---|------|---------|--------|------|
| R3 | SkipGate warm_start 単体テスト | — | LOW | テストカバレッジ |
| R4 | ドキュメント命名違反 28 件 | — | LOW | phX/type 欠落 |
| R5 | lib → ztb 移動 (4 モジュール) | — | v461 | fast_fill, param_adapter 等 |
| R6 | utils 70+ ファイル分割 | — | v461 | God package |
| R7 | config/ vs configs/ 整理 | — | LOW | 重複ディレクトリ |
| R8 | `# type: ignore` 残 2 箇所 | — | N/A | 正当判定→維持 |
| Tier-3 | RiskRuleEngine, Reconciliation | ~2000 行 | ph5 | 未着手 |
| DUP3 | UnifiedTrainer 2835 行 | — | v461+ | God Object |

**考察**: これらは ph2 Gate 判定に影響しないため、急ぐ必要はない。
ただし、**R3 (SkipGate テスト) は SkipGate 改修作業の前提**として
先に実施した方がリグレッション検知に役立つ。

---

## §8 新規考察: ドキュメントから見えない盲点

以下は既存文書に明示されていないが、プロジェクト全体を俯瞰して浮上する課題。

### 8.1 Coincheck API の rate limit 実態

**状態**: ✅ **実測完了 (追記) — 問題なし**

013# で Coincheck rate limit = 4 req/s と記載。fill_test ログ分析結果:

| cancel reason | 件数 | unfilled内% |
|---|---:|---:|
| timeout | 115 | 31.6% |
| skip_gate | 98 | 26.9% |
| postonly_reject | 47 | 12.9% |
| **api_error** | **34** | **9.3%** |
| unknown/status_unknown | 48 | 13.2% |
| orderbook_error | 16 | 4.4% |
| その他 | 6 | 1.6% |

api_error 34 件の内訳:
- "Bad Request" (400): **22 件** — 注文量不足 ("Amount 量が最低量を下回っています")
- その他 400: 12 件
- **429 (rate limit): 0 件** → RateLimiter 4 req/s (013# D-5) が有効に機能中
- side 分布: buy 15 / sell 19 → 差なし

**結論**: rate limit は問題なし。api_error の主因は注文最低量不足 (lot 設定起因) であり、
ph3 で lot を引き上げる際に自然解消。**対応不要**。

### 8.2 post_only reject 率の高さ

**状態**: ✅ **実測完了 (追記) — 原因特定済**

cancel 理由で `postonly_reject` = **47 件 (12.9%)** — 依然として高い。

**spread 分析結果 (追記)**:
| 区分 | spread_at_order 平均 | 中央値 | n |
|---|---:|---:|---:|
| postonly_reject | **1,915 JPY** | 2,035 JPY | 47 |
| 全体平均 | 2,409 JPY | 2,447 JPY | 832 |
| filledのみ | 2,413 JPY | 2,425 JPY | 552 |

- postonly_reject の spread は全体より **20% 狭い** (1,915 vs 2,409)
- side 分布: sell 29 / buy 18 → sell に偏り
  - sell は bid 側に置くため、narrow spread 時に taker 側になりやすい

**対策案**: `narrow_spread_bps` の引き上げ (093# §6#2 提案と一致)
現在 narrow_spread_boost_sell=2.0 が発動しているが、それでも reject が発生。
**minimum_spread_guard** (e.g. spread < 1500 JPY なら skip) を検討余地あり。

**結論**: 主因は narrow spread + sell 側の構造的問題。
47/1123 (4.2%) は許容範囲だが、**Gate FAIL 時の D2 sell offset 引き上げで自然に改善される**。
単独対応は LOW 。

### 8.3 fill_test の再起動耐性

**状態**: ⚠️ **部分解決。自動再起動は ph5 で実装**

今回 PID 31236 が 144.8h で停止。§9-A1 で手動再起動済。
113# の StatePersistence は実装済みだが、**自動再起動の仕組みがない**。

**追記実績**: 再起動後の fill_test は PID 51480 で安定稼働中 (Cycle 1123+)。
StatePersistence が機能し、param_adapter/SkipGate の warm_start が正常に完了。
ただし、regime_detector の warm-up は 20 cycle の "unknown" が発生。

**ph2 での対策**: 手動監視 + 再起動で十分。
**ph5 での対策**: systemd / nssm / Task Scheduler による自動再起動。

**結論**: ph2 段階では対応不要。**ph5 必須項目として C7 に統合**。

### 8.4 累積 PnL の JPY 実額

fill_test ログの最終行: `cumPnL=-32.5JPY`。
000# §3.9 の実損キャップは **10,000 JPY**。余裕はあるが、
100 万円規模での運用を想定した場合、lot が 0.001 BTC (≈10,400 JPY) のため
レバレッジ後の実損は桁が変わる。

**考察**: 0.001 BTC × -0.172bps = 約 -0.002 JPY/cycle。
月 21,600 cycle (120s 間隔) で **約 -37 JPY/月**。
lot を 0.005 BTC に上げると **-185 JPY/月**。
lot 0.01 BTC で **-370 JPY/月**。

一方、PnL30 がゼロ bps に改善されれば損失はゼロ。
改善施策 (§3) が **+0.2~0.5bps** のインパクトを持つなら、
0.01 BTC lot で **月 +430~1,080 JPY** のプラスに転じる。
0.1 BTC lot なら **月 +4,300~10,800 JPY**。

**大義 (短期高収益) への到達**: 月間 10 万円級の利益を目指すなら、
lot 1.0 BTC × PnL +1.0bps が必要 → **約 216,000 JPY/月**。
これは spread_bps ≈ 1.2bps の BTC/JPY 市場において楽観的だが不可能ではない。

### 8.5 Oracle テストの欠如 (再掲)

v459 教訓③の再掲。完全予測 (Oracle) エージェントが maker 0% で
どの程度の PnL を達成するか測定していない。

Oracle PnL が AS コスト以下なら、**いかなる ML 改善も理論上限に届かない**。
ph3 進入前の必須チェック。

### 8.6 時系列非定常性への対処

**状態**: ✅ **§9-B4 で実測完了 — 改善トレンド確認**

**vg_and_trend.py による日別 AS トレンド**:
- 2/13: 40.5% → 2/14: 31.1% → 2/15: 34.7% → 2/17: 28.5% → 2/18: 18.8% → 2/19: 23.8%
- **前半平均 35.4% → 後半 26.7% → 改善トレンド**
- sell AS は A3 (sell SkipGate 無効化) 後に 41.7% → 23.0% → 22.2% と改善

**regime 分布実測 (追記, (n=1123)**:
| regime | 件数 | 全体% | AS% | PnL30 |
|---|---:|---:|---:|---:|
| ranging | 267 | 32.2% | 50.2% | +0.097bps |
| none | 178 | 31.6% | 48.3% | -0.127bps |
| trending | 132 | 15.7% | 51.5% | +0.076bps |
| **unknown** | **93** | **10.3%** | **60.2%** | **-0.891bps** |

**重要発見**: `unknown` regime は AS=60.2%, PnL=-0.891bps と最悪。
これは再起動時の warm-up 20サイクルが主因 (§3.3 参照)。
regime が確定して〜いるレコード (ranging/trending) の方が明確に良好。
**regime_detector は実効的だが、warm-up 問題の解決が最優先** (§3.4 warm-up state persistence 参照)。

### 8.7 Coincheck 以外の取引所への展開可能性

000# §4 で「Coincheck (主) / Bitflyer / Zaif」と多取引所設計が明記されているが、
実装は Coincheck adapter のみ。

**考察**: Coincheck の板が薄い場合 (spread 拡大 → AS 増加)、
bitFlyer への切替が必要になる可能性。
BaseExchangeAdapter の抽象化は 013# C-4 で実装済みだが、
bitFlyer adapter は不完全 (013# C-5: product_code 正規化未済)。

ph2 段階で bitFlyer への fail-over は不要だが、
**ph5 本番では API 障害時の自動切替が必要**。

### 8.8 SkipGate と time_filter の役割重複

**状態**: 📋 **§9-A4 で VG 有効性確認済。段階的統合計画は健全**

SkipGate (ML) と time_filter (ルールベース) が同じ「高リスクサイクル除外」役割。

**特徴量分析による整理 (追記)**:
- SkipGate preorder 16 特徴量に `hour_sin/cos` は既に含まれる (097#)
- LR coeff 重要度: `hour_cos`=0.0406 (5位), `hour_sin`=0.0053 (10位)
  → 時間帯情報は既にモデルが学習中だが、AUC=0.442 では不十分
- time_filter の神通力: **sell AS 41.7% → 22.2%** (A3 と併せて効果大)
- VG: AS -7.7pt 改善 (A4 実証済)

**今後の統合ロードマップ** (107# Phase 3 踏襲):
1. ✅ Phase 1: SkipGate target_skip_rate 引上げ (107# 実装済)
2. ✅ Phase 2: Volatility Guard (107# 実装, §9-A4 有効性確認)
3. ⏳ Phase 3 Step 1: BUY 7h→3h, SELL 5h→3h (VG 有効確認済のため実行可)
4. ⏳ Phase 3 Step 2: 最悪時間帯のみ
5. ⏳ Phase 3 Step 3: 全廃 (SkipGate 再訓練後)

**結論**: 現行の Phase 3 計画は健全。VG 有効性が §9-A4 で確認されたため、
**次回 fill_test で Step 1 を実行する根拠が揃った**。
ただし SkipGate AUC が低いまま time_filter を縮小するので、
**Step 1 実施前の SkipGate 再訓練が望ましい** (Appendix F 参照)。

---

## §9 提案: 次の行動計画

### Phase A: 即時 (fill_test 再起動前)

| # | 施策 | 出典 | コスト | インパクト | 状態 |
|---|------|------|-------|----------|------|
| A1 | fill_test 再起動 (残 23h) | — | 5 min | F7 PASS に必須 | ✅ 完了 |
| A2 | warm_start threshold 即復元 | 098# §3.2 | 数行 | +0.1bps | ✅ `db41b7c57` |
| A3 | sell SkipGate 無効化 (実験) | 098# §4.2 | YAML 1 行 | 逆選別解消 | ✅ `db41b7c57` |
| A4 | Volatility Guard 効果測定 | 107# | 分析スクリプト | 意思決定の基盤 | ✅ vg_and_trend.py (AS -7.7pt, VG有効) |

### Phase B: fill_test 168h完了 — Gate 判定

| # | 施策 | 出典 | コスト | 状態 |
|---|------|------|-------|------|
| B1 | G1.1-quick / G1.2-full 自動判定実行 | 116# | 中 | ✅ `8a27ce2af` gate_judgment.py |
| B2 | Holm-Bonferroni 補正実装 | 000# §3.7 | 低 | ✅ `8a27ce2af` F4/F4b/F4c 3TF |
| B3 | PnL30 / PnL120 の t 検定結果報告 | — | 低 | ✅ `8a27ce2af` FillMetrics 統合 |
| B4 | AS rate の日別トレンド分析 | 新規 | 低 | ✅ vg_and_trend.py (前半35.4%→後半26.7%) |

### Phase C: Gate PASS 後 — ph3 準備

| # | 施策 | 出典 | コスト |
|---|------|------|-------|
| C1 | v458 Walk-Forward バグ 6 件修正 | 111# §5 | 高 |
| C2 | Oracle テスト実施 | 000# §1, 111# §6.3 | 中 |
| C3 | ph3 Stop 条件明文化 | 112# §3.4 | 低 |
| C4 | execute_trade() 実装 | 013# D-1 | 中 |

### Phase D: Gate FAIL 時 — 改善施策

| # | 施策 | 出典 | コスト | 期待効果 | 状態 |
|---|------|------|-------|---------|------|
| D1 | fast_fill_defense 二層化 | 098# P0-A | 中 | +0.2bps | ✅ 解決済 (100# L2 実装済) |
| D2 | sell offset 引き上げ | 098# P0-C | 低 | sell AS 削減 | Gate判定結果待ち |
| D3 | param_adapter recency window | 098# §3.6 | 低 | 適応精度向上 | ✅ 解決済 (096# window=120) |
| D4 | time_filter Phase 3 Step 1 | 107# | 低 | 機会損失解消 | A4でVG有効確認→次回FT候補 |
| D5 | order timeout 150s 短縮 | 095# M4 | 低 | timeout cancel 削減 | ✅ 解決済 (096# 300→90s) |
| D6 | SkipGate buy/sell 分離訓練 | 099# P2-1 | 高 | AS 削減 | Gate判定結果待ち |

### 行動の意思決定フロー

```
fill_test 再起動 (A1-A4)
    │
    ▼
168h 完了 → Gate 判定 (B1-B4)
    │
    ├─ PASS → Phase C (ph3 準備)
    │
    ├─ WATCH → パラメータ凍結 + D1-D3 の低コスト施策 → 再測定
    │
    └─ FAIL
        ├─ F4 (PnL) FAIL → D1, D2, D3 → 再 fill_test (168h)
        ├─ F5 (AS) FAIL → D1, D2, D6 → 再 fill_test
        └─ F1 (fill_rate) FAIL → D4, D5 → 再 fill_test
```

---

## Appendix A: 53 OPEN items 全件一覧

### A. Gate 判定関連 (7 件)

| ID | 文書 | 内容 | 優先度 | 追記 状態 |
|----|------|------|--------|-----------|
| A1 | 014# | G1.2-full 判定: 各指標の実測値 vs 閾値 | P0 | ✅ gate_judgment.py |
| A2 | 092# | E1 閾値 (90%→85%) と G1.2 F1 (70%) の整合性確認 | P0 | Gate判定結果待ち |
| A3 | 014# | 継続中止ルール (n≥200 fill_rate<70% → 中止) | P0 | 運用中 |
| A4 | 111# | Holm-Bonferroni 補正の実装 | P1 | ✅ §9-B2 |
| A5 | 107# | time_filter Phase 3 Step 1 (fill 数に影響) | P1 | VG有効確認→次回FT |
| A6 | 112# | SLO/Gate 自動判定パイプライン | P1 | ✅ §9-B1 |
| A7 | 092# | Round-trip/Net Inventory の Gate 昇格検討 | P2 | 棚上げ妥当 |

### B. 収益性直結 (10 件)

| ID | 文書 | 内容 | 優先度 | 追記 状態 |
|----|------|------|--------|-----------|
| B1 | 095# | SkipGate ランダム分類器問題 | P0 | sell無効化+buy継続 (A3) |
| B2 | 095# | fast_fill / param_adapter 状態競合 | P0 | ✅ 100# side分離で解消 |
| B3 | 098# | fast_fill has_negative_edge 検出漏れ 50% | P0 | ✅ 100# L2実装済 |
| B4 | 098# | SkipGate warm_start 閾値未復元 | P0 | ✅ §9-A2 直接quantile |
| B5 | 098# | Regime Detector warm-up (全件 unknown) | P1 | ✅ 101# P1-5 warm-up (window×3=60件) で update() フル実行。hysteresis含め状態復元済。unknown 93件は初回起動20cycle |
| B6 | 095# | param_adapter 全レコード使用 (dilution) | P1 | ✅ 096# window=120 |
| B7 | 098# | Offset vs PnL 因果分析未着手 | P1 | Gate PASS後 AB テスト |
| B8 | 098# | AS 判定 horizon 30s→60s 検討 | P2 | 棚上げ妥当 (B3 PnL120で代替) |
| B9 | 095# | 10-30s wait band 全損失の 65% | P1 | 096# timeout 90s で緩和 |
| B10 | 098# | sell offset 0.12→0.15 / floor 引き上げ | P1 | 105# floor 0.10済、追加はGate判定後 |

### C. 本番前必須 (10 件)

| ID | 文書 | 内容 | 優先度 |
|----|------|------|--------|
| C1 | 013# | エラーハンドリング全 API 横展開 | P1 |
| C2 | 013# | async/sync 整合 (asyncio.to_thread) | 済 |
| C3 | 013# | post_only 実装 | 済 |
| C4 | 013# | BaseExchangeAdapter 継承化 (150行重複) | P2 |
| C5 | 111# | v458 WF バグ 6 件 | P0 (ph3) |
| C6 | 111# | BacktestReporter 統一 + CheckVenueHealth | P1 (ph3) |
| C7 | 112# | 運用失敗モード試験 | P1 |
| C8 | 112# | ph3 Stop 条件明文化 | P1 (ph3) |
| C9 | 014# | Seed 非決定性 + チェックポイント共有 | P2 (ph3) |
| C10 | 013# | RateLimiter 4 req/s | 済 |

### D. SkipGate/ML (8 件)

| ID | 文書 | 内容 | 優先度 | 追記 状態 |
|----|------|------|--------|-----------|
| D1 | 097# | データ 500 蓄積で再訓練 | P1 | ✅ filled 759件→可能。Appendix F に再訓練計画 |
| D2 | 097# | regime 特徴量全 fold 定数 0 | P1 | ✅ 実測: regime分布 ranging=32%/trending=16%/unknown=10% → 定数0問題は097#の13サイクル限定。再訓練で自然解消 |
| D3 | 097# | Skip10% 効果なし → 15-25% が妥当 | INFO | YAML target 20-25% 設定済 |
| D4 | 098# | sell SkipGate 逆選別 | P0 | ✅ §9-A3 sell無効化 |
| D5 | 098# | sell 専用追加特徴量候補 | P2 | ✅ 過去資料統合完了: spread_bps(058# §3, 097# coeff=0.0494), hour_cos(097# coeff=0.0406), recent_sell_pnl_mean(098# §4.3)。Appendix F で整理 |
| D6 | 095# | SkipGate 抜本見直し | P0 | A3暫定+再訓練(Appendix F)で本格対応 |
| D7 | 099# | sell 専用 SkipGate モデル | P2 | ✅ 098# §4.2 で sell P(AS) 逆効果実証。再訓練で buy/sell 分離推奨。buy 382件, sell 377件で分割可能 |
| D8 | 106# | SkipGate evaluate/warm_start テスト不足 | P2 | ✅ test_skip_gate_d8.py 30テスト: A3 side無効化7件 + A2 warm_start即収束6件 + build_features 10件 + save/load 2件 + evaluate edge 5件 |

### E. インフラ/コード品質 (12 件)

| ID | 文書 | 内容 | 優先度 | 追記 状態 |
|----|------|------|--------|-----------|
| E1 | 106# | run_fill_test.py God Object | ✅ 完了 (121#) | — |
| E2 | 106# | ドキュメント命名違反 28 件 | LOW | 棚上げ |
| E3 | 106# | lib → ztb 移動 | v461 | 棚上げ |
| E4 | 106# | utils 70+ ファイル分割 | v461 | 棚上げ |
| E5 | 106# | config/ vs configs/ 整理 | LOW | 棚上げ |
| E6 | 013# | SimBroker リネーム | LOW | 棚上げ |
| E7 | 013# | StreamBuffer v460 組込み | 将来 | 棚上げ |
| E8 | 111# | Dead code 整理 (adaptation/) | LOW | 棚上げ |
| E9 | 113# | 運用失敗モードテスト | MEDIUM | ph3 前に実施。§8.3 参照 |
| E10 | 113# | PnL Monte Carlo 定期実行 | MEDIUM | ✅ gate_judgment.py に --monte-carlo フラグで統合。19テスト (test_gate_judgment.py) |
| E11 | 106# | type: ignore 残 2 箇所 | N/A | 正当→維持 |
| E12 | 112# | God Object 再発リスク監視 | INFO | 1568行で安定 |

---

## Appendix B: 22 → 42 RESOLVED items

| 元 ID | 文書 | 内容 | 解決時期 |
|--------|------|------|---------|
| — | 014# T3 | .env 自動読込 | 014# |
| — | 014# T4 | WebSocket API | 014# |
| — | 014# T5 | PnL Monte Carlo | 014# |
| — | 013# C-4 | asyncio.to_thread | 013# App-E |
| — | 013# D-3 | post_only | 013# App-E |
| — | 013# D-5 | RateLimiter 4 req/s | 013# App-E |
| — | 107# Ph1 | SkipGate target_skip_rate 引上げ | 107# |
| — | 107# Ph2 | Volatility Guard 新設 | 107# |
| — | 113# | CircuitBreaker / HealthMonitor 実装 | 113# |
| — | 113# | R1 God Method 分割 (755→307 行) | 113# |
| — | 113# | StatePersistence | 113# |
| — | 116# | 二段階ゲート実装 (K1-K6 / F1-F8) | 116# |
| — | 117# | Import chain fix + 二重キャンセル防止 | 117# |
| — | 106# R1 | run_fill_test.py 分割 3411→1568 行 | 119#-121# |
| — | 103# | YAML 設定外部化 | 103# |
| — | 109# | Any 型完全撤去 | 109# |
| — | 109# | 耐障害性強化 | 109# |
| — | 110# | time_filter デッドロック修正 | 110# |
| — | 080# | 重複排除 ~3,000 行削減 | 080# |
| — | 063# | SAC 重複実装整理 246 行削除 | 063# |
| — | 092# E6 | Round-trip KPI (informational) 追加 | 092# |
| — | 092# E7 | Net Inventory (informational) 追加 | 092# |
| B3 | 098# | fast_fill has_negative_edge 50% → L2 対応 | 100# |
| B4 | 098# | warm_start 閾値即復元 | `db41b7c57` (§9-A2) |
| B6 | 095# | param_adapter recency window | 096# window=120 |
| D4 | 098# | sell SkipGate 逆選別 | `db41b7c57` (§9-A3) |
| A4(§9) | 111# | Holm-Bonferroni 3TF 補正 | `8a27ce2af` (§9-B2) |
| A6(§9) | 112# | Gate 自動判定パイプライン | `8a27ce2af` (§9-B1) |
| §9 A4 | 107# | Volatility Guard 効果測定 | §9-A4 vg_and_trend.py |
| §9 B4 | 新規 | AS 日別トレンド分析 | §9-A4 vg_and_trend.py |
| D5(§9) | 095# | order timeout 300→90s | 096# |
| B2 | 095# | fast_fill/param_adapter 状態競合 | 100# side分離 |
| §5.1 | 098# | AS判定 horizon → E3 Gate追加で代替 | §9-B2 PnL120 |
| §5.5 | 091# | Smart Side → side_aligned_velocity で代替 | 追記分析 |
| §8.1 | 013# | API rate limit 実測: 429=0件 | 追記ログ分析 |
| §8.2 | 013# | post_only reject: narrow spread起因特定 | 追記spread分析 |
| §8.6 | 新規 | 時系列非定常性 → B4改善トレンド確認 | §9-B4 vg_and_trend |
| D2 | 097# | regime特徴量定数0 → 759件で自然解消 | 追記regime分布実測 |
| B5 | 098# | Regime warm-up → 101# P1-5 warm-up で解決 | 追記update()フル実行確認 |
| D8 | 106# | SkipGate テスト不足 → 30テスト追加 | test_skip_gate_d8.py |
| E10 | 113# | MC定期実行 → gate_judgment.py統合+19テスト | test_gate_judgment.py |

---

## Appendix C: 未検討提案 §5 全件 Disposition サマリ (追記)

| §5 項 | 提案内容 | 出典 | Disposition | 根拠 |
|-------|---------|------|------------|------|
| 5.1 | AS判定 horizon 30s→60s | 098# §6 | 📋 棚上げ妥当 | B2 で PnL60/120 を Gate に追加済 |
| 5.2 | Event-driven サイクル間隔 | 088# P2-1 | 📋 棚上げ妥当 | v461+。120s 固定はシンプルで安全 |
| 5.3 | Round-trip を primary KPI | 088# P1-1 | 📋 棚上げ妥当 | E6 informational 蓄積中。ph3 以降 |
| 5.4 | Volatility Guard | 107# Ph2 | ✅ 実装+測定完了 | AS -7.7pt, boost 2.0x 適切 |
| 5.5 | Smart Side 再有効化 | 091# §1#7 | 📋 棚上げ妥当 | side_aligned_velocity で間接的に対処済 |
| 5.6 | time_filter 段階的廃止 | 107# Ph3 | ⏳ VG有効確認→Step 1 可 | §8.8 ロードマップ参照 |
| 5.7 | Order timeout 短縮 | 095# M4 | ✅ 解決済 | 096# で 300→90s |
| 5.8 | Offset 体系的探索 | 095# M10 | ⏳ Gate 判定後 | AB テスト設計が必要 |
| 5.9 | Sell 保持期間延長 | 095# M7 | ⏳ ph3 以降 | B2 PnL120 Gate追加で実質対応 |
| 5.10 | fast_fill 持続時間制御 | 093# §6#4 | 📋 棚上げ妥当 | 100# L2 実装済で効果限定的 |

---

## Appendix D2: §8 全件 Disposition サマリ (追記)

| §8 項 | 内容 | Disposition | 根拠 |
|-------|------|------------|------|
| 8.1 | API rate limit | ✅ 実測完了 | 429=0件。主因は最低注文量不足、lot引上げ時に自然解消 |
| 8.2 | post_only reject | ✅ 原因特定済 | narrow spread + sell偏り。D2 sell offset引上げで改善見込み |
| 8.3 | 再起動耐性 | ⚠️ 部分解決 | StatePersistence機能中。自動再起動はph5へ |
| 8.4 | 累積PnL JPY実額 | 📋 INFO | -32.5JPY。損失キャップ10,000JPY内。改善施策で転換可能 |
| 8.5 | Oracle テスト | ⏳ ph3前に必須 | C2 として Phase C に計上済 |
| 8.6 | 時系列非定常性 | ✅ B4で実測完了 | AS改善トレンド確認。regime実効性も確認 |
| 8.7 | 多取引所展開 | 📋 棚上げ | BaseExchangeAdapter抽象化済。ph5以降 |
| 8.8 | SkipGate/time_filter重複 | ✅ 整理完了 | 107# Phase 3 ロードマップ健全。VG有効でStep 1可 |

---

## Appendix F: SkipGate 再訓練計画 (特徴量統合)

### F1. データ準備状況

| 指標 | 097# 訓練時 | 現在 | 改善 |
|------|-----------|----------|------|
| filled records | 215 | **759** | 3.5x |
| buy / sell | ~110/~105 | **382 / 377** | 各3.5x |
| AS label 付き | 215 | ~670 | 3.1x |
| spread_at_order 保有 | 166 | **552** | 3.3x |
| regime 有効 | 0 (全 unknown) | **399** (ranging+trending) | ∞ |
| Walk-forward folds | 8 (min_train=50) | **~20** (min_train=50, step=30) | 2.5x |

**結論**: 759 filled records は十分。buy/sell 分割訓練 (各 ~380) も実行可能。

### F2. 過去資料に基づく特徴量整理

#### 現行 preorder 16 特徴量 (097#)

| # | Feature | 097# SelectKBest | 097# LR |coeff| | 065# LR |coeff| | 安定性 |
|---|---------|---:|---:|---:|---|
| 1 | `side_buy` | ❌除外 | — | 0.0217 | 不安定 |
| 2 | `hour_sin` | ✅ | 0.0053 | — | 不安定 |
| 3 | `hour_cos` | ✅ | 0.0406 | 0.0451 | **安定** |
| 4 | `spread_jpy` | ✅ | **0.0494** | 0.0368 | **安定** |
| 5 | `offset_ratio` | ✅ | 0.0455 | — | 不安定 |
| 6 | `regime_trending` | ❌除外 | — | — | 097#全件定数0→除外 |
| 7 | `regime_ranging` | ❌除外 | — | — | 097#全件定数0→除外 |
| 8 | `regime_high_vol` | ❌除外 | — | — | 097#全件定数0→除外 |
| 9 | `trade_count_60s` | ✅ | 0.0440 | 0.0126 | **安定** |
| 10 | `buy_ratio` | ✅ | 0.0327 | **0.0630** | **安定** (Always selected) |
| 11 | `trade_flow_imbalance_60s` | ✅ | 0.0327 | **0.0630** | **安定** (Always selected) |
| 12 | `avg_trade_size` | ✅ | 0.0484 | 0.0278 | **安定** |
| 13 | `price_velocity_60s` | ✅ | 0.0298 | — | 不安定 |
| 14 | `vpin_60s` | ❌除外 | — | 0.0123 | 不安定 |
| 15 | `side_aligned_tfi` | ❌除外 | — | 0.0346 | **安定** (Always selected) |
| 16 | `side_aligned_velocity` | ✅ | 0.0354 | 0.0284 | **安定** (Always selected) |

**Always Selected (4)** (065# Jaccard analysis): `buy_ratio`, `side_aligned_velocity`, `trade_count_60s`, `trade_flow_imbalance_60s`

#### 再訓練時の特徴量変更提案

**期待改善点** (097#→現在の差分):

1. **regime 特徴量の復活** (D2):
   - 097# では 13 cycle / 全件 unknown → regime 特徴量が定数 0 → SelectKBest で除外
   - 現在: ranging=267件(32%), trending=132件(16%), unknown=93件(10%)
   - **unknown regime の AS=60.2%, PnL=-0.891bps は最悪** → regime 特徴量に情報量あり
   - 推奨: `regime_trending`, `regime_ranging` を**強制 include** (SelectKBest で除外させない)

2. **sell 専用特徴量** (D5, 098# §4.3):
   - 098# §4.2 で sell P(AS)≥0.50 群は PnL +0.027bps → **逆効果** (skip群の方が良い)
   - 原因候補: sell の AS ドライバーが buy と異なる
   - `spread_bps`: sell は narrow spread 時に AS 率増加 (098# 文書)
   - `hour_cos`: sell は時間帯感度が高い (097# 分析)
   - `recent_sell_pnl_mean`: 直近 sell PnL の running average (098# §4.3 提案)
   - 推奨: buy/sell 分割訓練 (D7) で各 ~380 サンプル使用

3. **V2 マルチタイムフレーム特徴量** (060#):
   - 060# で追加された `vpin_30s`, `tfi_30s`, `velocity_30s`, `vpin_acceleration` 等
   - 060# 結果: ROC-AUC 0.5007 (+0.01 微改善)、`return_60s` が最重要 (coeff=1.03)
   - **ただし preorder では `return_*` は情報リーク** (096# で除外済)
   - 推奨: `vpin_acceleration` (vpin_30s - vpin_60s) のみ追加候補。効果限定的で低優先

4. **SelectKBest k 値** (097# §2):
   - 097#: k=10 が最良 (Skip20%=+0.405bps)
   - 759 サンプルではサンプル/特徴量比が十分 → k=12-14 への拡大を検討
   - regime 特徴量復活で実質 16→16 (0 ではなくなる) → k=10-12 が推奨

### F3. 再訓練実行計画

```
前提条件: Gate 168h 判定完了後

Step 1: データ準備
  - fill_records 759件 + spread_at_order 552件
  - enrich_fill_records() で enriched_df 作成
  - build_preorder_as_features() で X, y 構築

Step 2: 全データ統合訓練 (ベースライン)
  - k=10, C=0.01, LR pipeline (現行と同じ)
  - Walk-forward 20-fold (min_train=50, step=30)
  - ROC-AUC, Skip20% 改善を 097# 結果と比較

Step 3: buy/sell 分割訓練 (D7)
  - buy: n=382, k=8-10, C=0.01
  - sell: n=377, k=8-10, C=0.01
  - sell モデルの ROC-AUC が 0.5 を超えるか検証
  - sell で skip 群 PnL > kept 群 PnL (逆効果) が解消するか確認

Step 4: regime 特徴量強制 include
  - SelectKBest のスコアに関わらず regime_trending/ranging を含める
  - k=12 で regime + base 10 の 12 特徴量

Step 5: 評価・デプロイ
  - Skip20% 改善 > +0.3bps なら更新
  - sell 分割モデルの方向性が正しければ sell_enabled 再有効化検討
  - fill_test YAML 更新 + warm_start で新閾値に即収束 (A2 修正済)
```

### F4. 特徴量進化の時系列 (ドキュメントトレース)

| 文書 | 特徴量数 | Key Finding |
|------|---------|------------|
| 057# | 10 (base) | ROC-AUC 0.528。ランダム水準 |
| 058# | 21 (base+micro+interact) | AS分類 +0.01, PnL回帰が有効 |
| 060# | 39 (v2 multi-TF追加) | return_60s が最重要 (情報リーク) |
| 065# | 16 (preorder-only) | Always selected 4特徴量特定 |
| 070# | 72+構成×3セット | **全 ROC-AUC ≤ 0.54。SNR=0.11** |
| 097# | 16→10 (SelectKBest) | preorder 統一。Skip20%=+0.405bps |
| **現在** | **16 (regime 復活見込)** | **759 samples。regime/sell分離が鍵** |
