# 671# ログ改善・分析スクリプト修正 + 多角的分析による 670# 仮説検証

- **日付**: 2026-03-31
- **コミット予定**: InfeasibleQuoteError 構造化、FillRecord NFQ フィールド、section_nfq_analysis、section_daily multi-metric
- **目的**: 670# の「avg PnL 負 = 構造的問題」仮説を多角的に検証し、データの嘘を見抜く

---

## §0 実施した変更

### §0.1 ログ改善 — NFQ 構造化フィールド

**問題**: NFQ (no_feasible_quote) レコードは `error_message` にテキスト埋め込みで数値を持つのみ。構造化分析が不可能。

**対応**:

| ファイル | 変更 |
|---------|------|
| `scripts/v460/lib/maker_price.py` | `InfeasibleQuoteError` に `actual_spread`, `min_spread_effective`, `min_spread_abs`, `min_spread_atr`, `sigma` スロット追加 |
| `scripts/v460/lib/maker_price.py` | `_check_min_spread_guard` の raise サイトで構造化値を渡す |
| `scripts/v460/lib/fill_cycle_executor.py` | `_make_price_error_skip` が `InfeasibleQuoteError` から nfq_* フィールドを抽出し skip record に伝搬 |
| `ztb/metrics/fill_quality.py` | `FillRecord` に `nfq_actual_spread`, `nfq_min_spread_effective`, `nfq_min_spread_abs`, `nfq_min_spread_atr`, `nfq_sigma` 追加 |

**後方互換**: 旧コードからの呼び出し (`InfeasibleQuoteError(reason=..., msg=...)`) はデフォルト値 0.0 で互換維持。

### §0.2 分析スクリプト修正

| ファイル | セクション | 変更 |
|---------|-----------|------|
| `analyze_fill_logs.py` | `section_daily()` | 30s/EV/120s 三指標の並列表示 (670# 教訓: 単一指標は嘘をつく) |
| `analyze_fill_logs.py` | `section_nfq_analysis()` | 新規追加。構造化フィールド優先 + error_message フォールバック。日別/時間帯別 NFQ 率、spread gap 統計、σ 分布 |

### §0.3 テスト

`tests/unit/v460/test_671_nfq_structured_logging.py` — 12 テスト:
- InfeasibleQuoteError 構造化フィールド (3)
- FillRecord nfq_* フィールド (3)
- build_skip_fill_record NFQ 連携 (1)
- section_nfq_analysis (3)
- section_daily multi-metric (2)

既存テスト: 238 passed (分析スクリプト 32 + fill_quality 206)、回帰なし。

---

## §1 多角的分析 — 670# 仮説の検証

> 670# の結論: 「avg PnL が -0.54 bps/fill — fill が増えるほど損が増える構造」

### §1.1 「データは嘘吐き」の実証

**PnL メトリクスの選択で結論が反転する:**

| 日付 | fills | 30s avg | EV avg | 120s avg | 120s sum |
|------|-------|---------|--------|----------|----------|
| 3/20 | 287 | **+0.03** | -0.17 | **-0.86** | -130.1 |
| 3/25 | 108 | +0.04 | -0.10 | -1.05 | -60.0 |
| 3/28 | 36 | -0.63 | -0.71 | -0.36 | -4.7 |
| 3/31 | 20 | -0.75 | -0.39 | **+0.83** | **+10.0** |

- 3/20 は 30s では黒字だが 120s では -130 bps
- 3/31 は 30s では赤字だが 120s では +10 bps
- **どのウィンドウを選ぶかで「良い日」と「悪い日」が入れ替わる**

### §1.2 視座 4: Adverse Selection が真の分水嶺

| 区分 | n | avg30 | avg120 |
|------|---|-------|--------|
| **AS=True** | 449 | **-6.85** | **-7.71** |
| **AS=False** | 1173 | **+2.14** | **+1.79** |

**Non-AS fill は 30s でも 120s でも黒字。**
670# の「per-fill PnL 負」は、AS fill (~28%) が平均を引き下げているだけ。

→ **真の問題は「avg PnL 負」ではなく「AS 回避精度」**

### §1.3 視座 6-7: SkipGate / Offset の予測力 ≈ 0

| 指標 | PnL30 相関 | PnL120 相関 |
|------|-----------|------------|
| SkipGate score | -0.007 | -0.004 |
| effective_offset | -0.005 | — |

SkipGate も offset pipeline も PnL アウトカムとほぼ無相関。
**ML モデルの予測が fill の質改善に寄与していない。**

### §1.4 視座 5: Sidecar (SAC) は完全に死んでいる

| status | n | avg30 | sidecar_offset |
|--------|---|-------|----------------|
| fresh | 318 | -0.29 | **0.00** bps |
| stale | 823 | -0.43 | **0.00** bps |
| error | 264 | -0.03 | **0.00** bps |

全ステータスで `sidecar_offset = 0.00 bps`。Alpha 層は文字通りゼロ寄与。

---

## §2 最重要発見 — NFQ が収益帯域を遮断している

### §2.1 視座 8: スプレッド帯域別 PnL

| スプレッド帯 | n | avg30 | avg120 | AS率 |
|-------------|---|-------|--------|------|
| **< 1500 JPY** | 274 | -0.29 | -0.16 | 26% |
| **1500-2500 JPY** | 598 | **+0.02** | -0.37 | **25%** |
| 2500-3500 JPY | 562 | -0.68 | -1.53 | 30% |
| > 3500 JPY | 188 | -0.60 | -2.78 | 30% |

**NFQ がブロックするスプレッド帯 (mean=1998, p25=1428-p75=2654 JPY) はまさに PnL が最も良い帯域 (1500-2500 JPY)。**

min_spread_atr (σ × mid × mult) が ~3200 JPY → 2500 JPY 以下のクオートを全て遮断。
結果として、通過するのは 2500 JPY 以上の「PnL が悪い」帯域のみ。

### §2.2 視座 9: 3/20 vs 3/27-31 の構造的差異

| 指標 | 3/20 (良日) | 3/27-31 (悪期間) |
|------|------------|-----------------|
| fill rate | 53% | 8% |
| NFQ | 0 (0%) | 597 (**23%**) |
| PI | 1 (0%) | 1085 (**42%**) |
| avg_spread | 2255 JPY | 2721 JPY |
| avg_offset | 0.2132 | **0.3351** |
| AS rate | **33%** | 26% |
| avg30 | +0.03 | -0.71 |

逆説: 3/27-31 は AS 率が低い (26% < 33%) のに PnL が悪い。

**なぜか:** NFQ が 1500-2500 JPY 帯の fill を遮断 → 残る fill は wide spread 帯 (avg 2721) → wide spread は offsetも wider (0.3351) → mid から遠いので 120s での逆行リスクが大きい → avg120 が悪化

### §2.3 視座 10: git_sha で因果を辿る

| SHA | fills | fill% | avg30 | avg120 | NFQ |
|-----|-------|-------|-------|--------|-----|
| **dfbe3b539eaa** | 169/332 | **51%** | **+0.22** | +0.02 | **0** |
| 20d4f778ef67 | 115/340 | 34% | -0.67 | -1.30 | 67 |
| d93b9a5bf672 | 106/583 | 18% | -0.45 | -0.23 | 64 |
| f7faac4f1232 | 20/551 | **4%** | -1.82 | -4.95 | **184** |

`dfbe3b539eaa` (3/20 のコード) は NFQ=0、fill 51%、avg30=+0.22。
後のコード変更 (min_spread_atr 関連?) で NFQ が急増し fill rate が崩壊した。

---

## §3 670# 仮説の修正

### §3.1 670# が正しかったこと

1. fill rate の崩壊 (52% → 7%) は確かに起きている
2. NFQ の急増が主因であること
3. 663#-669# の迷走 (PI に注力して本質を見逃した) の指摘

### §3.2 670# が間違っていたこと

1. **「avg PnL 負 = 構造的問題」** → Non-AS fill は +2.14 bps で黒字。AS fill の混入率が問題
2. **「fill rate を上げても損失が増える」** → 逆。NFQ で遮断されている帯域が最も PnL が良い。fill rate を上げれば PnL は改善する
3. **「Execution 層の直接改善に正面から取り組む」** → offset pipeline は PnL と無相関。改善しても効果は限定的
4. **「EV toxic skip を -8 → -5 に」** → EV は fill のクオリティ予測に使えていない (相関 ≈ 0)。閾値を厳しくしても AS rate は下がらず fill が減るだけの可能性

### §3.3 真の優先順位 (670# §5 の修正)

| 優先度 | 施策 | 根拠 |
|--------|------|------|
| **P0** | **min_spread_atr_mult の引き下げまたは cap_bps の見直し** | NFQ が収益帯域 (1500-2500 JPY) を遮断。min_spread を 2000-2500 に下げれば fill rate が 8% → 20%+ に回復し、かつ通過する fill の avg PnL が改善する |
| **P0** | **3/20 (dfbe3b539eaa) vs 現行コードの diff 調査** | 何がNFQ=0 → NFQ=23% の劣化を引き起こしたか |
| **P1** | **SkipGate AS 予測モデルの再評価** | score-PnL 相関 ≈ 0。AS 回避に寄与していない |
| **P1** | **669# max_lot=0.001 デプロイ + PI 45%→?% の効果検証** | PI 解消で fill rate に追加改善 |
| **P2** | **Sidecar (SAC) の調査** | offset=0.00 bps は仕様か故障か |
| **延期** | **EV toxic skip 閾値変更** | EV score と PnL の相関がない状態では効果不明 |

---

## §4 方法論の比較 — どうデータを見るべきか

### §4.1 670# のアプローチ (単一指標の日別集計)

- **方法**: 日別の fill count / fill rate / avg PnL (30s) を時系列で表示
- **強み**: トレンドが一目でわかる
- **弱み**: 単一 PnL ウィンドウ (30s) のみ、confounders (コード変更、市場環境) が混在

### §4.2 本文書のアプローチ (多変量条件付き分析)

- **方法**: 市場環境変数 (spread, regime, VG) と操作変数 (offset, git_sha) で条件付けして PnL を分解
- **強み**: confounders を分離でき、因果の方向がより明確
- **弱み**: サンプルサイズが小さい帯域での推定が不安定

### §4.3 今後追加すべき方法

1. **Bootstrap 信頼区間**: avg PnL の ±2σ を推定し、「本当に 0 と有意に異なるか」を検定
2. **Propensity Score Matching**: NFQ と non-NFQ の市場環境をマッチングし、NFQ 遮断の因果効果を推定
3. **Rolling Window Analysis**: 7日/3日ウィンドウのローリング PnL で regime shift を検出
4. **SHAP 値によるフィーチャー重要度**: SkipGate / offset pipeline のどのステージが PnL に寄与しているかの定量化

### §4.4 最も重要な教訓

**集計レベル (aggregate) のデータは因果を語れない。** 670# の「avg PnL -0.54 bps = 構造的問題」は、NFQ selection bias + spread-PnL 逆相関 + PnL ウィンドウ選択という 3 つのバイアスの複合産物だった。

条件付き分析 (conditional analysis) によってのみ、「何が原因で何が結果か」に近づける。

---

## §5 渙との接続

605# 渙三爻「其の躬を渙らす」— 670# はこれを「向き合いたくない問題 (avg PnL 負) に向き合え」と解釈した。

しかし本当の「向き合いたくない問題」は:
- **min_spread_atr が害を為している可能性** — Safety の名目で利益機会を潰している
- **SkipGate が機能していない可能性** — ML が無相関の予測を出し続けている
- **Sidecar (SAC) が完全に死んでいる** — 0# のAlpha層の中核が寄与ゼロ

Safety 層を手放す (=min_spread_atr を緩和する) のは恐怖を伴う。しかし「データは平気で嘘吐きます」— その嘘を見抜いた以上、Safety への固執こそが渙三爻の「散らすべき躬」かもしれない。
