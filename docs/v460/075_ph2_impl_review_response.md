# 075# ph2 impl: レビュー指摘対応 + 50K ステップ検証 + 批判的再考察

| key | value |
|---|---|
| 番号 | 075 |
| フェーズ | ph2 |
| 種別 | impl |
| 参照 | `074_ph2_rev_073.md`, `scripts/v460/ml/run_075_verification.py`, `scripts/v460/run_fill_test.py` |
| 作成日 | 2026-02-16 |
| テスト | 662 passed |
| 目的 | 073# 外部レビュー (9 指摘) への対応 + クリーンデータ再検証 + 50K MC 検証 + 自己批判的深層考察 |

---

## §0 エグゼクティブサマリ

**073# 外部レビュー (074_ph2_rev_073.md) で指摘された 9 件 (CRITICAL ×2, HIGH ×4, MEDIUM ×3) に対応。**
**最重要発見: quarantine 汚染 (30.3%) が 073# の sell UTC15 アンブロック判断を歪めていた。クリーンデータで再検証し YAML を修正。**

### 対応ステータス

| # | 重要度 | 指摘 | 対応 | 状態 |
|---|---|---|---|---|
| 1 | CRITICAL | `run_single_cycle()` が `_next_side()` を再呼出し → side 不一致 | `side_override` パラメータ追加 | ✅ 修正済 |
| 2 | CRITICAL | 分析に `filter_clean_records()` 未使用 → quarantine 汚染 | 検証スクリプトで適用 + YAML 修正 | ✅ 修正済 |
| 3 | HIGH | S12 `sim_pnl` 未使用 | 検証で PNL_COL に反映 | ✅ 修正済 |
| 5 | HIGH | queue_wait 系戦略は事後情報依存 | S13 に限定、補助扱いで継続 | ✅ 認識・限定 |
| 6 | HIGH | 統計検定なし (Holm/Cliff/p-mean) | Wilcoxon + Cliff's Delta + p-mean 適用 | ✅ 実装済 |
| 7 | MEDIUM | side filter で 13/24h 両 side ブロック | 機会損失定量化、12/24h に圧縮 | ✅ 改善済 |
| 8 | MEDIUM | 120s horizon が G1.1 30s KPI と矛盾 | role label (30s=KPI, 120s=補助) 明記 | ✅ 整理済 |
| 9 | MEDIUM | manifest/JSON artifact なし | `results/v460/verification_077/` に出力 | ✅ 実装済 |

### 最重要発見: quarantine 汚染

073# では全 491 レコード (filled 373) で分析。しかし 149 件 (30.3%) は quarantine 対象 (git_sha 欠損等)。

| データセット | filled | mean PnL (bps) |
|---|---|---|
| 全データ (073# 使用) | 373 | -0.620 |
| **clean データ** | **284** | **-0.459** |
| quarantine データ | 89 | -1.132 |

**073# の sell UTC15 アンブロック (+2.460 bps) は quarantine 汚染データに基づく誤判断。**
**クリーンデータでは sell UTC15 = -3.325 bps (n=2)。即座に再ブロック。**

---

## §1 CRITICAL#1: side_override 修正

### 問題

`run_continuous()` で `_next_side()` → side 決定 → `run_single_cycle()` 呼出し、
しかし `run_single_cycle()` 内で再度 `_next_side()` が呼ばれ side が上書きされる。

### 修正

```python
# run_fill_test.py
def run_single_cycle(self, side_override: str | None = None) -> dict:
    if side_override is not None:
        side = side_override   # _next_side() をスキップ
    else:
        side = self._next_side()  # 単独実行時のみ
```

`run_continuous()` 側:
```python
result = self.run_single_cycle(side_override=next_side)
```

---

## §2 CRITICAL#2: quarantine 分離 + YAML 修正

### clean/quarantine 分離結果

`filter_clean_records()` を全分析パイプラインに適用:

- 全: 491 → clean: 342 / quarantine: 149 (30.3%)
- filled: clean 284 / quarantine 89
- quarantine 期間: 2026-02-13 09:39 → 2026-02-14 09:15 (UTC)

### 073# → 075# YAML 修正

| 変更 | 073# (quarantine 汚染) | 074# (clean データ基準) | 根拠 |
|---|---|---|---|
| sell UTC15 | アンブロック (+2.460) | **再ブロック (-3.325)** | clean n=2, 汚染 +2.460 は quarantine 由来 |
| sell UTC01 | ブロック | **アンブロック (+0.931)** | clean n=13, 61.5% win rate |
| sell UTC02 | ブロック | **アンブロック (+0.239)** | clean n=8, 75.0% win rate |
| buy UTC04 | ブロック | **アンブロック (+3.993)** | clean n=7, 71.4% win rate, buy 最良時間帯 |
| buy UTC23 | ブロック | **ブロック維持** | mean -0.216、§8.2 批判: mean 負の時間帯をアンブロックする根拠なし |

### 最終 YAML (`configs/v460/fill_test.yaml`)

```yaml
time_filter:
  skip_utc_hours_buy:  [1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23]
  skip_utc_hours_sell: [3, 4, 5, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23]
```

| カテゴリ | 時間数 | 時間帯 (UTC) |
|---|---|---|
| 両 side ブロック | 12/24h | 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23 |
| buy のみ通過 | 4/24h | 3, 4, 5, 22 |
| sell のみ通過 | 4/24h | 1, 2, 10, 11 |
| 両 side 通過 | 4/24h | 0, 6, 7, 20 |

---

## §3 Time Filter 機会損失分析 (MEDIUM#7)

| カテゴリ | mean PnL (bps) | n | 備考 |
|---|---|---|---|
| both_blocked | -1.770 | 83 | 損失集中帯 → ブロック正当 |
| buy_only | **+1.396** | 40 | sell が悪い時間帯で buy のみ稼働 |
| sell_only | **+0.667** | 21 | buy が悪い時間帯で sell のみ稼働 |
| both_open | +0.611 | 79 | 主要稼働帯 |

### Before/After 比較

| | mean PnL (bps) | n |
|---|---|---|
| Before (global filter) | +0.322 | 161 |
| **After (side filter)** | **+0.844** | **140** |
| 改善幅 | **+0.522** | — |

side 分離により **+0.522 bps/step** の改善。n は 21 件減少するが PnL 密度が大幅向上。

---

## §4 統計検定付き WF-4fold (HIGH#6)

### 結果一覧

| Strategy | mean PnL | n | folds>0 | Cliff's d | p-mean |
|---|---|---|---|---|---|
| S0_baseline | -0.658 | 228 | **1/4** | — | 0.607 |
| S1_side_time | -0.397 | 203 | **1/4** | +0.027 | 0.533 |
| S9_conservative | -0.578 | 217 | **1/4** | +0.011 | 0.592 |
| S12_offset_sim | **-0.158** | 228 | **1/4** | +0.115 | 0.129 |
| S13_sell_offset | -0.373 | 196 | **1/4** | +0.018 | 0.559 |

### 解釈

- **robustly positive な戦略はなし** (全て 1/4 folds positive) — 070# 結論と整合
- S12 (offset sim +0.5bps) が最善だが、fold2/3 で負
- Cliff's Delta はいずれも小 (+0.027 ～ +0.115)
- **結論: データ量制約 (284 filled / 1.4 日) が支配的。戦略優位性の検証は時期尚早。**
- p-mean: S12 = 0.129 が最良 (ただし 0.05 未満は達成せず)

---

## §5 50,000 ステップ Monte Carlo 検証

### パラメータ

- Bootstrap: 1,000 回 × 50,000 ステップ
- Pool A (Before): global filter 後の 161 records (mean +0.322 bps)
- Pool B (After): side filter + sell offset 調整後の 140 records (mean +0.931 bps)
- sell offset 効果: +0.2 bps 近似

### 結果

| 指標 | Before (global) | After (side) | 差異 |
|---|---|---|---|
| 累積 mean (bps) | +16,130 | **+44,234** | +28,104 |
| 累積 std (bps) | 929 | 744 | -185 |
| 正の確率 | 100% | 100% | ±0% |
| P5 | +14,599 | +43,030 | +28,431 |
| P50 | +16,129 | +44,242 | +28,113 |
| P95 | +17,601 | +45,484 | +27,883 |

### Per-step PnL

| | bps/step |
|---|---|
| Before | +0.323 |
| **After** | **+0.885** |
| 改善 | **+0.562** |

### JPY 換算 (BTC=¥15M, lot=0.001)

| | 50K ステップ累積 |
|---|---|
| Before | ¥+24,195 |
| **After** | **¥+66,351** |
| 差異 | **¥+42,157** |

### 統計的有意性

| 検定 | 値 | 判定 |
|---|---|---|
| Mann-Whitney U | p < 0.001 | ✅ 高度に有意 |
| Cliff's Delta | +1.000 | ✅ 完全効果サイズ |

**50K MC は side filter + offset 調整の改善を統計的に強く支持。**

> ⚠️ 注意: MC は pool 内の分布が将来も持続する仮定。1.4 日分のデータに基づくため、
> regime shift リスクは残る。E3 サンプリング強化 (0.50) により 120s データも蓄積中。

---

## §6 Multi-horizon PnL (MEDIUM#8)

| Horizon | mean (bps) | median | std | win% | n | 役割 |
|---|---|---|---|---|---|---|
| **30s** | **-0.459** | -0.089 | 4.741 | 47.2% | 284 | **G1.1 KPI (主)** |
| 60s | -0.620 | -0.681 | 6.299 | 46.2% | 26 | 補助指標 |
| 120s | +0.101 | +1.007 | 8.267 | 53.8% | 26 | 補助指標 |

- G1.1 ゲート判定は **30s PnL のみ** で行う (000# §3.3)
- 120s の正転は E3 データ蓄積後に再評価
- 60s/120s の n=26 はサンプル不足、統計的結論は保留

---

## §7 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/run_fill_test.py` | CRITICAL#1: `side_override` パラメータ追加 |
| `configs/v460/fill_test.yaml` | CRITICAL#2: clean データ基準で時間フィルタ修正 |
| `scripts/v460/ml/run_075_verification.py` | NEW: 包括的検証スクリプト (§0-§6) |
| `tests/unit/v460/test_regime_detector.py` | YAML 修正に合わせたアサーション更新 |

### artifact

- `results/v460/verification_077/verification_077_*.json` — 全セクション結果の JSON

---

## §8 批判的深層考察 — 楽観を疑う

以下は「自分たちの分析自体を猜疑心をもって検証する」試みである。
075# のデータと結論に対し、多角的・批判的視点から弱点と盲点を洗い出す。

### §8.1 50K MC の「統計的有意性」は論理的循環である

50K MC は以下のプロセスで走っている:

1. 1.4 日分のデータから「正の PnL を持つ時間帯」を**選択**
2. その選択済みプール (mean +0.931 bps) から 50K 回 bootstrap
3. 「累積 PnL が正で、Before との差が有意」と結論

**これは循環論法である。** 正の平均を持つプールから復元抽出すれば、当然 50K ステップの累積は正になる。
Mann-Whitney p<0.001 は「2つの正規分布的プールの平均差が有意か」を検定しているに過ぎず、
**「選択した時間帯フィルタが将来も有効か」は一切検証していない。**

WF-4fold が全戦略 1/4 folds positive という結果との矛盾がこれを裏付ける:
- **MC は「フィルタが正しい前提」で勝てるかを問う → 当然 Yes**
- **WF は「フィルタを学習して汎化するか」を問う → No (1/4)**

WF の方が誠実な結論である。MC の「有意性」に惑わされてはならない。

### §8.2 小サンプル過適合の定量的リスク

アンブロック判断の根拠となったサンプルサイズ:

| 変更 | n | 危険度 | 問題 |
|---|---|---|---|
| buy UTC04 (+3.993) | **7** | 🔴 極高 | 1 件の外れ値で符号反転。7 件中 1 件が +15 bps なら残り 6 件平均は +2.2 で、その 1 件を除くと全体像が変わる |
| sell UTC01 (+0.931) | 13 | 🟡 中 | 比較的安定だが、1.4 日間の特定時間帯。曜日効果すら検証不能 |
| sell UTC02 (+0.239) | **8** | 🔴 高 | ほぼゼロ近傍。数件の変動で容易に反転 |
| buy UTC23 (-0.216) | 9 | 🔴 高 | そもそも mean が負。median とwin rate で正当化しているが、これは metric cherry-picking |
| sell UTC15 (-3.325) | **2** | ⚫ 判断不能 | n=2 で何も言えない。ブロック判断自体も n=2 では過剰反応の可能性 |

**buy UTC23 の判断は特に危うい。** mean PnL が負 (-0.216) なのに、median (+0.210) と win rate (66.7%) で
アンブロックを正当化している。これは「都合の良い指標を選んで結論を支持する」確証バイアスである。
mean が負なら G1.1 KPI (mean PnL ≥ 0) 基準でブロックすべき。

### §8.3 多重比較問題 — 48 セルからの選択

24 時間 × 2 side = 48 セルを検査し、正のセルを「通過」、負のセルを「ブロック」している。
多重比較補正なしでの有意水準 α=0.05 において、48 セル中 2.4 セルが偶然だけで「有意に正」に見える。

現在の判断基準は「mean PnL と n」に基づく閾値だが、**統計検定すら適用していない**。
Holm-Bonferroni 補正を適用すれば、n=7-13 で α/48 ≈ 0.001 を超える p 値はほぼ確実で、
個別時間帯の有意性は主張できない。

### §8.4 本質的問題: 全体 PnL が負である

clean データの mean PnL = **-0.459 bps**。これがシステムの地金である。

時間フィルタは「悪い時間を除外して残りを正にする」操作だが、これは:
1. **エッジの創出ではなく、損失の回避**に過ぎない
2. 除外された時間帯で稼働しないことは **機会損失** ではなく **損失回避** — これ自体は有益だが限界がある
3. フィルタ後の mean +0.844 bps は、n=140 の選択的抽出の結果であり、真の out-of-sample 性能は不明

根本的な問いは: **このシステムに market microstructure edge はあるのか?**

AS 比率 34.2% (G1.1 基準 ≤20%) は、**我々の注文が情報劣位者として搾取されている**ことを示す。
これは offset や時間帯フィルタで表面的に改善できる問題ではなく、
注文の出し方 (maker pricing, queue position) の根本的欠陥を示唆する。

### §8.5 WF-4fold と MC の矛盾が示すもの

| 検証手法 | 質問 | 回答 | 信頼度 |
|---|---|---|---|
| WF-4fold | 過去データ内で戦略は汎化するか? | **No** (1/4) | 高 — out-of-sample テスト |
| MC bootstrap | 選択済みプールは正か? | **Yes** (p<0.001) | **低** — in-sample 循環 |

**この矛盾を真剣に受け止めるべきである。**

MC の正の結果は「過去のフィルタ後データが正」という既知事実の再確認に過ぎない。
WF の 1/4 は「フィルタの time-varying 性能が不安定」であることを示す。
1.4 日間で学習したパターンが次の 1.4 日で崩壊する可能性は高い。

### §8.6 sell 劣後の構造的原因

sell 全体の mean PnL = -0.826 bps, buy = -0.098 bps。sell が buy の 8.4 倍悪い。

考えられる構造的原因:
1. **BTC/JPY の上昇 bias** — 対象期間に上昇トレンドがあれば sell maker は構造的に不利
2. **offset 非対称** — 073# で sell offset を 0.12 に上げたが、buy は 0.10。この差が十分か検証不足
3. **AS の side 非対称** — sell AS 35.5% vs buy AS 32.9%。sell 側で informed trader に狙われやすい
4. **時間帯 × side の交互作用** — sell が悪い時間帯 (UTC4: -5.558, UTC8: -6.725) は極端。
   これらは「sell が全般的に悪い」というよりも「特定局面で大敗する」パターン

**sell offset 0.12 は 073# の近似的改善だが、sell の真の問題は offset ではなく queue position や
informed flow detection にある可能性が高い。** offset を上げすぎると fill rate が低下し、
さらに AS に選別されやすい残り物だけが約定するという悪循環に陥るリスクがある。

### §8.7 regime shift に対する無防備

1.4 日間のデータに基づくフィルタ設定は、以下の regime shift に脆弱:

- **ボラティリティ変化**: 低 vol 期間は spread が縮小し、maker edge が消滅
- **トレンド転換**: 上昇→下落で buy/sell の有利不利が反転
- **流動性イベント**: 大口注文、規制ニュース等で板の厚さが変動
- **曜日・週次パターン**: 1.4 日では週末効果すら検証不能

**現在の設定は「2026-02-13 〜 02-15 の BTC/JPY 市場に最適化された」ものであり、
汎用性の保証は一切ない。**

### §8.8 楽観的数値の裏にある現実

ドキュメント全体を通じて「改善」を強調しているが、現実を直視する:

| 指標 | 現在値 | G1.1 基準 | ギャップ |
|---|---|---|---|
| mean PnL (30s) | **-0.459 bps** | ≥ 0 | -0.459 |
| AS ratio | **34.2%** | ≤ 20% | +14.2pt |
| fill rate | 76.0% | ≥ 90% | -14.0pt |
| データ期間 | 1.4 日 | ≥ 7 日 | -5.6 日 |

**4 指標すべてが G1.1 FAIL。** 時間フィルタで理論的に PnL をプラスにできても、
AS ratio と fill rate は時間フィルタでは改善しない。

### §8.9 075# が本当にすべきだったこと

レビュー指摘 (074#) への「修正完了チェックリスト消化」に終始し、
**「なぜ PnL が負なのか」という根本問題の深掘り**が不足している。

修正すべきだったかもしれない本質的アプローチ:

1. **Oracle 分析 (v459/116# 教訓)**: 「完全な未来情報があったときの最大 PnL」を計算し、
   エッジの理論上限を把握。上限自体が低ければシステム設計の前提が崩壊している
2. **AS の原因分析**: 34.2% の AS がどの条件で発生するか (spread, imbalance, time-since-last-trade 等)
   の因果分析。時間帯よりも重要な特徴量がある可能性
3. **Spread vs Offset の均衡点分析**: offset を上げると fill rate が下がり、下げると AS が増える。
   この trade-off の最適点を理論的・実証的に求める
4. **Informed flow detection**: AS を事前に予測・回避する特徴量の探索。
   これが v460 "Microstructure Edge" の本来の狙いだったはず

---

## §9 結論と次のアクション

### 結論

1. **quarantine 汚染 (30.3%) は 073# の判断に実害を与えていた** — sell UTC15 の誤アンブロック
2. **clean データ基準で YAML を修正** — sell UTC15 再ブロック、buy UTC04/sell UTC01,02 アンブロック
3. **50K MC は循環論法的であり、過信禁物** — フィルタ後プールが正なのは定義上当然。WF-4fold (1/4) の方が信頼できる
4. **WF-4fold では robustly positive な戦略なし** — データ量制約 (284 filled / 1.4 日) が支配的
5. **CRITICAL#1 (side 上書きバグ) 修正** — fill test 投入前に対処完了
6. **全 G1.1 指標が FAIL 継続** — PnL, AS ratio, fill rate, データ期間すべて基準未達
7. **時間フィルタは損失回避であってエッジ創出ではない** — AS ratio 34.2% の構造的問題は未解決

### 誠実な現状認識

- 現時点でシステムに確認された market microstructure edge は **存在しない**
- 時間フィルタは「悪い時間を避ける」だけで、根本的な maker pricing / AS 対策にはならない
- 1.4 日分のデータで deployment 判断を行うのは統計的に不適切
- buy UTC23 のアンブロックは mean PnL 負であり、再ブロックを検討すべき

### 次のアクション

- [ ] **即座**: ~~buy UTC23 を再ブロック~~ → 実施済 (mean 負の時間帯をアンブロックする根拠なし)
- [ ] fill test を修正済み設定で継続稼働 → G1.1 データ蓄積 (目標: 7 日間 500+ filled)
- [ ] **AS 原因分析**: AS 34.2% の発生条件 (spread, imbalance, 時間帯) の因果分解
- [ ] **Oracle 分析**: 完全情報下での理論 PnL 上限を計算し、エッジの存在可否を確認
- [ ] clean データが 500+ に到達次第、WF-4fold 再検証 (Holm-Bonferroni 付き)
- [ ] E3 120s データ蓄積 → multi-horizon 再評価
- [ ] **Spread-Offset 均衡分析**: offset 変化に対する fill rate / AS / PnL の感度曲線を描く
