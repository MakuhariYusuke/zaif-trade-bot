# 118# 残課題・未検討提案の深掘り考察

**Date**: 2026-02-19  
**Phase**: phg (フェーズ横断)  
**前提文書**: 000#–117# 全 96 文書 + fill_test 144.8h 実測データ

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
- [Appendix B: 22 RESOLVED items](#appendix-b-22-resolved-items)

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

### 全体像: 53 OPEN / 22 RESOLVED / 24 提案未検討

| カテゴリ | OPEN | RESOLVED | 要点 |
|---------|------|----------|------|
| A. Gate 判定関連 | 7 | 3 | F4/F5 がボーダー。Holm-Bonferroni 未実装 |
| B. 収益性直結 | 10 | 4 | fast_fill/warm_start/sell_offset が最大レバレッジ |
| C. 本番前必須 | 10 | 6 | execute_trade() TODO, WF バグ 6 件 |
| D. SkipGate/ML | 8 | 3 | 分離力不足 (AUC=0.442)。sell 逆選別 |
| E. インフラ/コード | 12 | 4 | Tier-3 全未着手。R3-R7 大半未着手 |
| F. 未検討提案 | 24 | — | 下記 §5 で個別考察 |
| G. v461+ | 13 | 2 | lib→ztb, utils 分割等 |

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

## §5 未検討のまま棚上げされた提案

以下は各ドキュメントで提案されながら、後続文書で明示的に検討・却下されていない項目。

### 5.1 AS 判定 horizon 30s → 60s 変更 [098# §6]

**提案内容**: 30s 時点の逆行で AS と判定するが、120s では回復するデータがある。
AS 判定を 60s に延長すれば、AS_ratio が大幅に低下する可能性。

**考察**:
- 058# で AS は 30s mid との比較で定義。これは v460 全体の KPI 基盤。
- 判定 horizon を変えると、SkipGate のラベルも変わるため再訓練が必須
- Gate 閾値 (F5: AS_ratio ≤ 30%) も再定義が必要
- **しかし、PnL120 が +0.220bps であることは、30s 判定が厳しすぎる可能性を示唆**

**結論**: AS 判定 horizon の再検討は **データに基づく正当な提案**だが、
Gate 定義の根幹に関わるため、安易に変更すべきではない。
代わりに **E3 (120s) PnL を Gate に追加する** (informational として) が現実的。

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

**状態**: ✅ **実装済み** (price_velocity_60s + VPIN → offset boost)

fill_test ログで `volatility_guard` 発動を確認:
```
[volatility_guard] 107# buy offset boosted: 0.1125→0.2250 (vpin=0.95)
```

**考察**: 実装はされたが、その **効果測定が未実施**。
Volatility Guard 発動時の PnL と非発動時の PnL を比較し、
offset boost 倍率 (2.0x) が適切か検証すべき。

### 5.5 Smart Side 再有効化 [091# §1#7]

**提案内容**: order book imbalance に基づく smart side selection を再有効化。
現在は OB 取得無効化のため `_last_imbalance` が更新されず、dead code 化。

**考察**:
- 071# / 072# で OB 特徴量トグル化。OB 依存なしモデルに移行
- smart side の価値は「逆行しにくい方向に注文する」こと
- **price-only 代替**: `price_velocity` / `mid_trend_5s` で代替可能
  - 上昇トレンドなら buy を優先 (ask 側に置く方が安全)
  - 下降トレンドなら sell を優先

**結論**: price-only Smart Side は **低コストで実装可能** (side_selector.py に組込み)。
fill_test のデータで「BUY トレンド/ranging 別 PnL」を分析し、
効果が見込めれば実装。

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

**提案内容**: sell 30s PnL = -0.632bps だが、120s PnL はプラス転換するデータあり。
sell 約定後の保持期間を延長する。

**考察**:
- **PnL120 = +0.220bps** (全体)。sell のみの 120s PnL は更に良い可能性
- BTC/JPY は平均回帰傾向があり、30s の逆行が 120s で戻る
- ただし、fill_test は 1-leg 設計で「保持」概念がない
- 実装するなら、sell 約定後に **exit trigger を 30s → 120s に延長** する
  (= PnL 計測タイミングを遅らせる)

**結論**: 実装は中規模。fill_test の設計変更が必要。
ただし、**PnL120 を Gate 判定の補助指標に加える** のは低コストで有効。

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

013# で Coincheck rate limit = 4 req/s と記載されたが、
fill_test ログで `api_error` が 34 件 (cancel 理由の 9.5%)。
これが rate limit 由来か、他のエラーかの切り分けが未実施。

**提案**: api_error の詳細ログを集計し、
429 (rate limit) / 5xx (server error) / その他の内訳を可視化。
rate limit 超過が多ければ、request 間隔の調整が必要。

### 8.2 post_only reject 率の高さ

cancel 理由で `postonly_reject` = 43 件 (11.9%)。
これは SkipGate skip (97 件) に次いで 3 番目に多い。

**意味**: maker 注文が即約定 (taker) として拒否されるケースが頻発。
offset が小さすぎるか、bid-ask が急接近した瞬間に発注している可能性。

**提案**: postonly_reject 発生時の spread_at_order を分析。
spread が狭い時に集中しているなら、最小 spread ガードの導入が有効。

### 8.3 fill_test の再起動耐性

今回 PID 31236 が 144.8h で停止。原因は不明 (OOM, crash, 自然終了?)。
113# の StatePersistence は実装済みだが、**自動再起動の仕組みがない**。

168h テストで途中停止は致命的 (F7: calendar ≥7 日)。

**提案**:
1. systemd / nssm / Task Scheduler による自動再起動
2. watchdog プロセス (HealthMonitor ベース)
3. state persistence からの exact resume (現在は新 run_id で再開)

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

fill_test データは 6 日間。BTC/JPY の市場構造は
数日単位で変化する (レジーム遷移)。

現在の AS_ratio 30.8% は 6 日間の平均であり、
日別に見るとバラつきが大きい可能性。

**提案**: 日別・8h 別の KPI トレンドを可視化し、
悪化トレンドか改善トレンドかを確認。
regime_detector の実効性評価にもなる。

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

SkipGate (ML) と time_filter (ルールベース) が同じ役割 (高リスクサイクルの除外) を
異なるメカニズムで行っている。

- time_filter: 時間帯ベース (UTC の特定時間帯を遮断)
- SkipGate: 特徴量ベース (AS 確率予測)

107# Phase 3 で time_filter 全廃 → SkipGate に統合する計画は妥当だが、
SkipGate の AUC が 0.442 の状態で time_filter を廃止すると AS 率が悪化するリスク。

**提案**:
1. SkipGate に時間帯特徴量 (hour_sin/cos) を追加して再訓練
2. 再訓練後に time_filter Phase 3 Step 1 を実施
3. SkipGate が時間帯情報を内在化した状態で time_filter 廃止を評価

---

## §9 提案: 次の行動計画

### Phase A: 即時 (fill_test 再起動前)

| # | 施策 | 出典 | コスト | インパクト |
|---|------|------|-------|----------|
| A1 | fill_test 再起動 (残 23h) | — | 5 min | F7 PASS に必須 |
| A2 | warm_start threshold 即復元 | 098# §3.2 | 数行 | +0.1bps |
| A3 | sell SkipGate 無効化 (実験) | 098# §4.2 | YAML 1 行 | 逆選別解消 |
| A4 | Volatility Guard 効果測定 | 107# | 分析スクリプト | 意思決定の基盤 |

### Phase B: fill_test 168h完了 — Gate 判定

| # | 施策 | 出典 | コスト |
|---|------|------|-------|
| B1 | G1.1-quick / G1.2-full 自動判定実行 | 116# | 中 |
| B2 | Holm-Bonferroni 補正実装 | 000# §3.7 | 低 |
| B3 | PnL30 / PnL120 の t 検定結果報告 | — | 低 |
| B4 | AS rate の日別トレンド分析 | 新規 | 低 |

### Phase C: Gate PASS 後 — ph3 準備

| # | 施策 | 出典 | コスト |
|---|------|------|-------|
| C1 | v458 Walk-Forward バグ 6 件修正 | 111# §5 | 高 |
| C2 | Oracle テスト実施 | 000# §1, 111# §6.3 | 中 |
| C3 | ph3 Stop 条件明文化 | 112# §3.4 | 低 |
| C4 | execute_trade() 実装 | 013# D-1 | 中 |

### Phase D: Gate FAIL 時 — 改善施策

| # | 施策 | 出典 | コスト | 期待効果 |
|---|------|------|-------|---------|
| D1 | fast_fill_defense 二層化 | 098# P0-A | 中 | +0.2bps |
| D2 | sell offset 引き上げ | 098# P0-C | 低 | sell AS 削減 |
| D3 | param_adapter recency window | 098# §3.6 | 低 | 適応精度向上 |
| D4 | time_filter Phase 3 Step 1 | 107# | 低 | 機会損失解消 |
| D5 | order timeout 150s 短縮 | 095# M4 | 低 | timeout cancel 削減 |
| D6 | SkipGate buy/sell 分離訓練 | 099# P2-1 | 高 | AS 削減 |

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

| ID | 文書 | 内容 | 優先度 |
|----|------|------|--------|
| A1 | 014# | G1.2-full 判定: 各指標の実測値 vs 閾値 | P0 |
| A2 | 092# | E1 閾値 (90%→85%) と G1.2 F1 (70%) の整合性確認 | P0 |
| A3 | 014# | 継続中止ルール (n≥200 fill_rate<70% → 中止) | P0 |
| A4 | 111# | Holm-Bonferroni 補正の実装 | P1 |
| A5 | 107# | time_filter Phase 3 Step 1 (fill 数に影響) | P1 |
| A6 | 112# | SLO/Gate 自動判定パイプライン | P1 |
| A7 | 092# | Round-trip/Net Inventory の Gate 昇格検討 | P2 |

### B. 収益性直結 (10 件)

| ID | 文書 | 内容 | 優先度 |
|----|------|------|--------|
| B1 | 095# | SkipGate ランダム分類器問題 | P0 |
| B2 | 095# | fast_fill / param_adapter 状態競合 | P0 |
| B3 | 098# | fast_fill has_negative_edge 検出漏れ 50% | P0 |
| B4 | 098# | SkipGate warm_start 閾値未復元 | P0 |
| B5 | 098# | Regime Detector warm-up (全件 unknown) | P1 |
| B6 | 095# | param_adapter 全レコード使用 (dilution) | P1 |
| B7 | 098# | Offset vs PnL 因果分析未着手 | P1 |
| B8 | 098# | AS 判定 horizon 30s→60s 検討 | P2 |
| B9 | 095# | 10-30s wait band 全損失の 65% | P1 |
| B10 | 098# | sell offset 0.12→0.15 / floor 引き上げ | P1 |

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

| ID | 文書 | 内容 | 優先度 |
|----|------|------|--------|
| D1 | 097# | データ 500 蓄積で再訓練 | P1 |
| D2 | 097# | regime 特徴量全 fold 定数 0 | P1 |
| D3 | 097# | Skip10% 効果なし → 15-25% が妥当 | INFO |
| D4 | 098# | sell SkipGate 逆選別 | P0 |
| D5 | 098# | sell 専用追加特徴量候補 | P2 |
| D6 | 095# | SkipGate 抜本見直し | P0 |
| D7 | 099# | sell 専用 SkipGate モデル | P2 |
| D8 | 106# | SkipGate evaluate/warm_start テスト不足 | P2 |

### E. インフラ/コード品質 (12 件)

| ID | 文書 | 内容 | 優先度 |
|----|------|------|--------|
| E1 | 106# | run_fill_test.py God Object | ✅ 完了 (121#) |
| E2 | 106# | ドキュメント命名違反 28 件 | LOW |
| E3 | 106# | lib → ztb 移動 | v461 |
| E4 | 106# | utils 70+ ファイル分割 | v461 |
| E5 | 106# | config/ vs configs/ 整理 | LOW |
| E6 | 013# | SimBroker リネーム | LOW |
| E7 | 013# | StreamBuffer v460 組込み | 将来 |
| E8 | 111# | Dead code 整理 (adaptation/) | LOW |
| E9 | 113# | 運用失敗モードテスト | MEDIUM |
| E10 | 113# | PnL Monte Carlo 定期実行 | MEDIUM |
| E11 | 106# | type: ignore 残 2 箇所 | N/A |
| E12 | 112# | God Object 再発リスク監視 | INFO |

---

## Appendix B: 22 RESOLVED items

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
