# 204# 包括的トレード分析レポート — 市場微細構造理論に基づく診断

| key | value |
|-----|-------|
| type | ph2_rpt (分析レポート) |
| scope | fill_test 全期間実績データ |
| date | 2026-03-01 |
| depends | 198#, 199#, 200#, 201#, 202#, 203# |
| purpose | 2,172 fills の包括的分析 — MM理論・一目均衡表・市場微細構造理論に基づく診断と改善候補 |

---

## §1 サマリ統計

| 指標 | 値 |
|------|-----|
| 分析期間 | 2026-02-13 ～ 2026-03-01 (17日間) |
| 総 fill 数 | 2,172 |
| 勝率 | 45.9% (998W / 1,167L / 7BE) |
| 累計 PnL | **-768.1 bps** |
| 勝ちトレード平均 | +3.52 bps |
| 負けトレード平均 | -3.67 bps |
| Risk/Reward 比 | 1.04:1 |
| 勝ち日 / 負け日 | 4 / 13 |
| 大損 (>10bps) | 82件, **-1,290.7 bps** (全損失の 168%) |
| 連敗 3+ストリーク | 162回, 累計 -2,498.9 bps |

**根本問題**: 累計 -768.1bps のうち、82件の大損 (-1,290.7bps) が全損失を超過。
つまり **大損トレードを排除すれば +522.6bps の黒字** になる。

---

## §2 勝ちトレード分析 — 「何故勝つのか」

### §2.1 Big Winner (>5bps) の特徴
- 211件中 **77% は VG 非発火** 状態で勝利
- 120s後も **90% が利益維持** (avg 120s = +9.3bps)
- buy 99件 (avg +10.0), sell 112件 (avg +10.2) → **side 偏りなし**
- スプレッド: avg 2.59bps, med 2.58bps → 広すぎず狭すぎない帯域

### §2.2 勝つ時間帯 (JST)

| 時間帯 | Net PnL | n | Avg | WR |
|--------|---------|---|-----|-----|
| 05:00 | +78.9 | 147 | +0.54 | 54% |
| 10:00 | +72.3 | 89 | +0.81 | 53% |
| 20:00 | +42.1 | 93 | +0.45 | 54% |
| 12:00 | +30.8 | 113 | +0.27 | 50% |

### §2.3 勝つ日の特徴

| 日 | PnL | n | 特徴 |
|----|-----|---|------|
| 2/17 | +61.5 | 137 | buy +67.7, sell -6.2 → **buy 一方向勝ち** |
| 2/18 | +52.6 | 149 | buy +100.4, sell -47.8 → **buy が大勝** |
| 2/24 | +104.4 | 157 | buy +112.6, sell -8.2 → **buy が大勝** |
| 2/27 | +44.8 | 204 | buy -41.4, sell +86.1 → **sell が大勝** |

**考察**: 勝つ日は必ず「片方の side が圧倒的に勝つ」 → トレンドに乗った side が利益を生む。

### §2.4 勝ちの本質 — 学術的解釈

#### Avellaneda-Stoikov モデル (2008) の観点
- 最適スプレッド: $\delta^* = \gamma\sigma^2 T + \frac{2}{\gamma}\ln(1 + \gamma/\kappa)$
- 勝ちトレードは $\sigma$ (ボラティリティ) が低〜中程度で、$\kappa$ (注文到着率) が安定している局面
- スプレッド 2.5〜3.5bps の帯域が最も効率的

#### Kyle モデル (1985) の観点
- $\lambda$ (price impact coefficient) が低い = 情報トレーダーの影響が小さい局面
- VG非発火で勝っている (77%) → velocity が穏やかで逆選択リスクが低い

#### Ho-Stoll モデル (1981) の観点
- ディーラーの最適スプレッドは在庫リスク + 逆選択コストの関数
- 勝ち日 = 片 side 大勝 → 在庫が**一方向に増減し、トレンドと整合**していた
- 在庫偏向がリスクプレミアムを正方向に生む稀な局面

### §2.5 勝ちトレードの要約

1. **スプレッド 2.5-3.5bps** の安定した板状態
2. **velocity が穏やか** (VG 非発火 77%)
3. **トレンド方向と一致した side** の約定
4. **120s後も 90% が利益維持** → genuine edge (noise ではない)

**本質**: 勝ちトレードは「静かな市場で、トレンドに逆らわない方向に、適切なスプレッドで約定」
= **受動的スプレッド収穫** + **弱トレンドフォロー** の組合せ。

---

## §3 負けトレード分析 — 「何故負けるのか」

### §3.1 構造的問題

#### A. Edge Capture Ratio が負値

```
avg_spread = 2.51 bps
half_spread (理論的エッジ) = 1.26 bps
actual avg PnL = -0.30 bps
Edge Capture Ratio = -24%
```

**Market Microstructure における致命的指標**: スプレッドの半値 (1.26bps) を稼ぐどころか、
平均で -0.30bps 失っている。**逆選択コストがスプレッド獲得を完全に上回る**。

#### B. Adverse Selection (逆選択) の深刻さ

- 922件中 **340件 (37%) が "persistent loss"** (30s/120s 共に負け)
- Persistent loss: avg30 = -4.29bps, avg120 = -7.08bps → **時間が経つほど悪化**
- **情報トレーダー (informed traders) に狩られている** 証拠

#### C. Sell Side の脆弱性

| Side | Fills | WR | Sum PnL | Avg PnL |
|------|-------|-----|---------|---------|
| Buy | 1,097 | 47.1% | -305.7 | -0.28 |
| Sell | 1,075 | 44.7% | -462.4 | -0.43 |
| **差** | | | **-156.7** | **-0.15** |

Sell が **50% 多く負けている**。`base_offset_ratio_sell = 0.18` (buy の 3.6倍) にもかかわらず。

### §3.2 大損パターン (>10bps, 82件)

- **Sell が 60%** (49/82)
- VG発火中でも大損: 20/82 (24%) → VG boost が不十分
- **JST 22-23時に集中** (24/82 = 29%) → 深夜の流動性低下局面
- 120s で回復するのは 45% のみ → 55% は permanent adverse selection
- **曜日集中**: 金曜日 25件 (30%) → 週末前のポジション解消フローに巻き込まれ

### §3.3 最悪時間帯 (JST)

| 時間帯 | Net PnL | n | Avg | WR |
|--------|---------|---|-----|-----|
| 23:00 | -148.2 | 63 | -2.35 | 40% |
| 06:00 | -126.4 | 85 | -1.49 | 41% |
| 22:00 | -97.7 | 97 | -1.01 | 49% |
| 17:00 | -90.8 | 37 | -2.45 | 32% |
| 01:00 | -82.9 | 36 | -2.30 | 39% |

### §3.4 スプレッド帯別分析

| Spread | Count | Avg PnL | WR | Sum PnL |
|--------|-------|---------|-----|---------|
| < 3bps | 1,255 | -0.31 | 45.4% | -387.6 |
| 3-5bps | 538 | -0.30 | 46.7% | -159.6 |
| ≤ 2bps | 523 | **-0.66** | **43%** | **-345.5** |

**スプレッド ≤ 2bps は特に危険**: WR 43%, avg -0.66bps → narrow スプレッド時に
逆選択が深刻化。

---

## §4 市場理論に基づく診断

### §4.1 Market Microstructure Theory (市場微細構造論)

#### (1) Glosten-Milgrom モデル (1985): 情報の非対称性と Bid-Ask スプレッド

- Bid-Ask スプレッドは逆選択コストを内包する
- 狭スプレッド時は情報の非対称性が低いか、MM がフィルタリングできている場合のみ有効
- **現状**: SkipGate のフィルタリングが不十分。スプレッド ≤ 2bps 時に WR 43% は、
  情報トレーダーの注文を選別できていない
- **処方**: Glosten-Milgrom の逆問題として、fill データからの事後的 $\mu$ (情報トレーダー比率) 推定がゲート精度向上に直結

#### (2) Kyle モデル (1985): 流動性と Price Impact

- 情報トレーダーの影響度 $\lambda$ は流動性に反比例
- 深夜 (JST 22-01時) は流動性低下 → $\lambda$ 上昇 → 逆選択リスク増大
- **現状**: 時間帯フィルターは存在するが、深夜帯でのオフセット拡大が不十分

#### (3) Avellaneda-Stoikov モデル (2008): 最適 Market Making

- 最適な quote 位置は $\sigma$ (ボラティリティ) に二次で依存
- **現状**: `base_offset_ratio` は固定値 → σ 変動への追従が不足
- σ急上昇時に offset が追いつかず、逆選択される

#### (4) Easley-O'Hara PIN モデル (1996): 情報取引の確率

- PIN (Probability of Informed Trading) = $\frac{\alpha\mu}{\alpha\mu + 2\varepsilon}$
- α = 情報イベント発生率、μ = 情報トレーダー到着率、ε = uninformed 到着率
- **現状**: Persistent loss 37% ≈ 高 PIN 環境で取引してしまっている
- **処方**: 約定フロー (OFI: Order Flow Imbalance) から PIN を近似推定し、
  高 PIN 時は offset を拡大 or スキップ

#### (5) Ho-Stoll モデル (1981): 在庫リスクとディーラーのスプレッド

- 最適スプレッド = 逆選択コスト + 在庫リスクプレミアム + 注文処理コスト
- 在庫 (ポジション) が一方に偏ると、解消方向の quote を aggressive にする必要
- **現状**: `inv_skew` で部分的に対応しているが、balance_forced_skip (10.4%) は
  在庫管理不足を示唆

#### (6) Almgren-Chriss モデル (2001): 最適執行と Market Impact

- 大口注文の最適分割執行: temporary impact + permanent impact を最小化
- **直接的には適用範囲外** (本 bot は小ロット maker) だが、
  他のトレーダーの大口注文の impact に巻き込まれるリスクの分析に応用可能
- 大損パターンの 55% が permanent adverse selection → 他者の情報取引の permanent impact と整合

#### (7) Roll モデル (1984): 有効スプレッドの推定

- 連続リターンの系列共分散から有効スプレッドを推定: $S_{eff} = 2\sqrt{-\text{Cov}(r_t, r_{t-1})}$
- **処方**: fill 前後の mid-price リターン共分散から「実効スプレッド」を計測し、
  理論スプレッドとの乖離を監視 → 乖離大 = 逆選択環境

### §4.2 一目均衡表の視点

| 一目均衡表の概念 | 対応する市場状態 | bot の現状 |
|-----------------|----------------|-----------|
| **三役好転** | 強いトレンド発生、方向性明確 | 勝ち日は片 side 大勝 — 検知はしているが活用不足 |
| **三役逆転** | 強い下落トレンド | sell 側の大損が集中 — 逆転検知が不十分 |
| **雲の中** (持ち合い) | レンジ相場、spread 安定 | WR 最高帯域 — **ここに注力すべき** |
| **雲の外** (離脱) | ブレイクアウト、急変 | 大損のほとんどがここで発生 |
| **遅行線の乖離** | 過去対比の極端な変動 | velocity_60s で部分的に代替、しかし不完全 |
| **転換線と基準線の交差** | トレンド転換点 | **最も危険な局面 — 現在未検知** |

### §4.3 Garman-Klass ボラティリティ

High-Low-Close ベースの Garman-Klass 推定量は OHLC から効率的に σ を計算:

$$\sigma_{GK}^2 = 0.5 \ln(H/L)^2 - (2\ln 2 - 1)\ln(C/O)^2$$

周期的に算出し、offset の動的調整に使えば A-S モデルに近づく。

### §4.4 追加理論フレームワーク

#### Cont-Kukanov-Stoikov OFI (2014): Order Flow Imbalance

- OFI = Σ (bid_size 変化 × I(bid上昇) - ask_size 変化 × I(ask下降))
- 短期的な価格変動の 65% 以上を説明するとされる
- **処方**: OFI を算出し、OFI が大幅負 (= 売り圧力) 時の buy 注文をスキップ or offset 拡大

#### Amihud 非流動性尺度 (2002)

- $ILLIQ = \frac{1}{D}\sum_{d=1}^{D}\frac{|r_d|}{V_d}$
- 出来高あたりの価格変動 → 流動性枯渇の指標
- **処方**: 直近 N 分の ILLIQ が閾値超過で offset 拡大 (Kyle λ の代理変数)

#### ボラティリティ・クラスタリング (Mandelbrot 1963, Engle 1982)

- 暗号資産は特にボラティリティのクラスタリングが顕著
- 大損 82 件の時系列分布を検証 → **連続的に発生する傾向がある** (連敗ストリーク 162 回)
- GARCH 的なボラティリティ推定を offset に反映すべき

---

## §5 What-If シミュレーション

| シナリオ | 結果 | 備考 |
|---------|------|------|
| 実際 | -768.1 bps | |
| 損失 -10bps キャップ | **-297.3 bps** | +470.7 改善 |
| 最悪 3時間除外 (23,6,22時 JST) | **-395.8 bps** | 245 fills 除外 |
| spread > 2.5bps のみ | -379.6 bps | 921 fills に半減 |
| VG + 勝ちのみ | +844.1 bps | 203 fills のみ |

**最もインパクトのある改善**: 損失キャップ (-10bps) → **+470.7bps 改善**

---

## §6 改善候補 (Priority 順)

### P0: 即効性のある施策

#### 204# H: 時間帯別リスク係数 (Time-Weighted Risk)
- JST 22-01, 06, 17時台は WR < 41%, avg < -1.5bps
- **施策**: これらの時間帯で offset を 2.0x に拡大、または取引停止
- **期待効果**: -148.2 - 126.4 - 82.9 - 90.8 = **-448.3bps 削減可能**

#### 204# I: Per-Fill Loss Cap (単発損失上限)
- 82件の >10bps 損失が -1,290.7bps → 全体損失の 168%
- **施策**: fill 後30秒で -10bps 超過の場合、次の同 side 取引で offset を 3x に拡大
- **期待効果**: -470.7bps 改善 (理論値)

#### 204# J: Narrow Spread Guard (狭スプレッド回避)
- spread ≤ 2bps で WR=43%, avg=-0.66bps → 損失密度最高
- **施策**: `min_spread_bps` を 2.0bps に引き上げ、または narrow spread 時に offset 1.5x
- **期待効果**: -345.5bps → 推定 -100bps 程度に改善

### P1: 構造的改善

#### 204# K: σ連動オフセット (Avellaneda-Stoikov 近似)
- 直近 N 秒の価格変動率 (Garman-Klass or realized vol) を算出
- offset = base_offset × (1 + α × σ/σ_avg) で動的調整
- σ 高騰時にオフセット拡大 → 逆選択防御

#### 204# L: トレンド転換検知 (一目均衡表インスパイア)
- 短期 EMA (転換線) と長期 EMA (基準線) の交差でトレンド転換を検知
- 交差直後は「三役逆転」リスク → offset 拡大 or 取引停止
- 現在の regime_detector は粗すぎる (ranging/trending の 2 分類)

#### 204# M: 深夜流動性ペナルティ
- 板の depth が薄い時間帯を自動検知 (OB depth ベース)
- 流動性メトリクスが閾値以下で offset 拡大

#### 204# P: PIN 近似フィルター (Easley-O'Hara インスパイア)
- 約定フロー (buy/sell 到着率) から PIN を近似推定
- 高 PIN (情報取引が活発) 環境では offset 拡大 or スキップ
- persistent loss 37% の構造的原因に対処

#### 204# Q: OFI (Order Flow Imbalance) ベースフィルター
- Cont-Kukanov-Stoikov OFI をリアルタイム算出
- OFI 極端値で逆方向取引を回避

### P2: 中期的改善

#### 204# N: 逆選択フィルター強化
- Persistent loss (30s+120s 共に負け) パターンの特徴を学習
- SkipGate に「逆選択確率」予測を追加

#### 204# O: Friday Effect ガード
- 金曜日の大損が突出 (25/82 = 30%)
- 金曜 JST 18時以降は offset 2x or lot 半減

---

## §7 vXXX シリーズ資産 — 204# 改善に活用可能なもの

> 出典: 111# (v456–v459 レガシー資産調査), 168# (既存資産活用計画)

### §7.1 即時活用推奨

| 資産 | 出典 | パス | 活用方法 | 204# 対応 |
|------|------|------|---------|----------|
| **DrawdownController** | v459 | `ztb/risk/drawdown_controller.py` (255L) | 段階的 DD 制御: warning→reduction→emergency_stop。203# の DailyDrawdownGuard を補完 | H, I |
| **CircuitBreaker** | v456+ | `ztb/utils/circuit_breaker.py` (229L) | API 障害時の自動遮断 → 無駄サイクル削減 | ― (インフラ) |
| **PnL Monte Carlo** | v460 | `ztb/risk/pnl_monte_carlo.py` (412L) | 月次 PnL 期待値の信頼区間推定 → 施策効果の統計的評価 | H, I, J の事前評価 |
| **Frequency Control Wrapper** | v457 | `backtest_v456.py` 内 cooldown/threshold | 連敗後のクーリングオフ制御。大損後の連鎖防止 | I |
| **Cyclical Time Features** | v457 | `fast_intraday_env_v456.py` 内 sin/cos | 時間帯エンコーディング → SkipGate の時間帯認識強化 | H |

### §7.2 中期的活用候補

| 資産 | 出典 | パス | 活用方法 | 204# 対応 |
|------|------|------|---------|----------|
| **DynamicPositionSizer** | v459 | `ztb/risk/dynamic_position_sizer.py` (260L) | ボラティリティ・DD・レジーム連動の動的ロットサイジング | K, O |
| **Walk-Forward Evaluation** | v458 | `ztb/evaluation/walk_forward/` | SkipGate 再訓練のアウトオブサンプル評価パイプライン | N |
| **Oracle Test Framework** | v459 | `scripts/v459/run_phase_e1_counterfactual.py` | 理論上限との乖離追跡 (maker 0% 環境での ceiling) | 全体効果検証 |
| **Multi-Seed Training** | v458 | config-driven seed 管理 | SkipGate 再訓練での再現性検証 | N |
| **HealthMonitor** | v460 | `ztb/trading/live/core/health_monitor.py` (119L) | 長時間稼働 (72h+) でのリソース監視 | ― (運用) |

### §7.3 既存分析ツール — 活用不足

| ツール | パス | 推奨活用 |
|-------|------|---------|
| **hindsight_filter** | `scripts/v460/analysis/hindsight_filter.py` (996L) | skip/cancel 機会損失の定量化 → SkipGate 精度評価 |
| **oracle_baseline** | `scripts/v460/analysis/oracle_baseline.py` (411L) | 理論上限との乖離を 204# 施策前後で比較 |
| **PnL Monte Carlo runner** | `scripts/v460/run_pnl_monte_carlo.py` (122L) | 施策導入前後の期待値変化を Monte Carlo で評価 |
| **stopgap_daily_report** | `scripts/v460/analysis/stopgap_daily_report.py` | 日次ヘルスチェック自動化 → 施策効果の継続監視 |
| **vg_and_trend** | `scripts/v460/analysis/vg_and_trend.py` (549L) | VG 効果の経時変化追跡 → K (σ連動) の効果測定 |

---

## §8 Top Winner パターンからの示唆

```
2026-02-24 01:17 buy  +50.1bps  VG  → BTC急騰に乗った
2026-02-25 23:41 sell +32.4bps  VG  → 急落局面でVGがoffsetを拡大 → 底で約定
2026-02-28 06:18 buy  +32.1bps  VG  → 同上パターン
```

VG 発火 + トレンド一致 = 大勝。**VG の精度向上が最大のレバレッジポイント**。

---

## §9 Codex / Gemini への分析指示案

### 分析依頼事項:

1. **逆選択パターンの特徴量分析**: persistent loss (30s+120s 負け) と winning trade の
   feature 分布差 (spread, 時間帯, side, 連続 sign, regime) を定量化

2. **最適 offset 関数の推定**: Avellaneda-Stoikov の $\sigma^2 T$ 項に相当する、
   realized volatility と offset の最適関係を historical data から回帰推定

3. **時間帯別 regime 切替**: 深夜・早朝の regime をマイクロ分類
   (not just ranging/trending, but liquidity-thin / weekend-pre / tokyo-open etc.)

4. **SkipGate feature importance の再評価**: 現在の feature set で
   「逆選択される確率」の予測精度を検証 + PIN/OFI 特徴量の追加検討

5. **損失分布のテールリスク分析**: 82件の大損トレードの条件付き分布を分析し、
   VaR/CVaR ベースのリスク制限を設計

6. **vXXX 資産の統合優先度**: §7 の資産リストについて、
   実装コスト/期待効果のトレードオフ評価

---

## §10 結論

### 核心的問題

**「大損トレードの排除」が最大のレバレッジポイント。**
82件の >10bps 損失を排除するだけで、累計が -768 → +523bps にスイングする。

### 適用した理論フレームワーク (計 11)

| # | 理論 | 貢献 |
|---|------|------|
| 1 | **Glosten-Milgrom (1985)** | 狭スプレッド時の逆選択コスト過大を説明 |
| 2 | **Kyle (1985)** | 深夜流動性枯渇が price impact を増幅 |
| 3 | **Avellaneda-Stoikov (2008)** | σ連動 offset の不在 = 固定 offset の限界 |
| 4 | **一目均衡表** | 「雲の外」での損失集中、転換点未検知 |
| 5 | **Garman-Klass** | 効率的σ推定 → offset 動的調整の基盤 |
| 6 | **Easley-O'Hara PIN (1996)** | 情報取引確率の推定 → 逆選択フィルター |
| 7 | **Ho-Stoll (1981)** | 在庫リスクと最適スプレッドの関係 |
| 8 | **Almgren-Chriss (2001)** | 大損の permanent impact との整合性 |
| 9 | **Roll (1984)** | 有効スプレッド推定 → 逆選択環境の検出 |
| 10 | **Cont-Kukanov-Stoikov OFI (2014)** | 短期価格変動の説明力 → フィルター |
| 11 | **Amihud (2002) + Mandelbrot/Engle** | 非流動性尺度 + ボラティリティクラスタリング |

### 次の一手

1. **時間帯フィルター強化** (204# H) — 即効性最高
2. **損失キャップ** (204# I) — 実装コスト低、効果大
3. **σ連動 offset** (204# K) — 本質的改善
4. **PIN 近似フィルター** (204# P) — 逆選択の根本対処

---

## Appendix A: 分析スクリプト

| ファイル | 内容 |
|---------|------|
| `analysis/204_trade_analysis.py` | 12+ 分析関数: daily_summary, win_loss, spread_vs_pnl, velocity_regime, adverse_selection, MM理論診断 |
| `analysis/204_trade_analysis_p2.py` | 勝ちトレード深堀り, what-if シミュレーション, スプレッド-PnL 相関 |
