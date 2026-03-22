# 537# 536番「風水渙に基づく断捨離」レビューと拡張提案

- **日付**: 2026-03-22
- **目的**: 536# の3シナリオを現行コードベースと金融工学理論で検証し、具体的な実装方針と追加の収益向上提案を行う
- **入力**: 536#, 531#, 532#, 533#, 535#, 現行コード・設定値

---

## §0 総合評価

536# の診断は**概ね正確**であり、特に以下3点は実データで裏付けられている：

1. **Pipeline 100% ceiling 飽和** — 531# 実証: buy 35/35, sell 19/19 が 0.25 でクランプ → Pipeline の 8 段の演算は取引価格に一切反映されていない
2. **事後反応型 Kill の遅効性** — 532# 実証: C14902 で -20.79bps 損失 → *1分後*にようやく kill 発動 → 損失防止ではなくサンドバッグ
3. **時間帯ベース固定値の過学習性** — 533# が正当に批判: 市場構造が変われば 12h/15h の特異性は消滅し得る

ただし536# には**過小評価・欠落している現実的制約**がある。以下で精密に検証する。

---

## §1 536# の §2「現在の構造」に対するファクトチェック

### 1.1 レイヤー1「実行前ゲート」— 部分的に不正確

536# は `cross_venue_lead_lag_veto`, `min_spread_jpy`, `balance_insufficient` の3つのみ列挙しているが、実際の Gate 構造は**9ゲート（Hard 4 + Soft 5）**の複合体である。

| 区分 | Gate | 現状 | 536# の言及 |
|------|------|------|-------------|
| Hard | Gate 4: `buy_dynamic_kill` | Rolling PnL → binary kill | ❌ 記載なし |
| Hard | Gate 5: `sell_dynamic_kill` | Rolling PnL → binary kill | §2 で「事後対応ガード」分類 |
| Hard | Gate 8: `narrow_spread_pause` | spread < min → 停止 | ❌ 記載なし |
| Hard | Gate 9: `maker_price_precheck` | min_spread_jpy + sell_guard | §2 で一部言及 |
| Soft | Gate 1,7: `unknown_regime_skip` | regime=UNKNOWN → skip | ❌ 記載なし |
| Soft | Gate 2,2b: `ranging_low_vol` | 低ボラ ranging → skip | ❌ 記載なし |
| Soft | Gate 3: `trending_sell_skip` | trending_up → sell skip | §2 で「事後対応ガード」分類 |
| Soft | Gate 6: `velocity_skip` | 急変動 → skip | ❌ 記載なし |

**影響**: 536# が「デッドロック（死場）」と述べる実行前ゲートは3つではなく9つあり、その相互干渉が根本問題。特に **491# composite_risk（現在 disabled）** を有効化すれば、Soft Gate の hard-block 問題の一部は解消可能な既存実装がある。

### 1.2 レイヤー2「オフセット演算」— 正確だが深さ不足

536# は `base → as_shift → regime → spread_adapt → vol_guard → final` と記述するが、実際のパイプラインは**9段の乗算チェーン**：

```
193# EV → 195# Velocity → 196# Trending → 240# Toxicity →
202# VG → 458# Macro → 215# Alert → 372# Sidecar → 421# Clamp
```

各段が `effective_offset_ratio *= mult` で累積するため、典型的な経路：

```
base=0.05 × EV(1.2) × Vel(1.3) × Trend(1.5) × Tox(2.0) × VG(1.2) × Macro(1.6) × Alert(1.0)
= 0.05 × 1.2 × 1.3 × 1.5 × 2.0 × 1.2 × 1.6 × 1.0
= 0.05 × 8.9856
= 0.449

→ ceiling 0.25 でクランプ → 出力の 44% が切り捨て
```

**定量的帰結**: P(output < 0.25) < 5%（531# 実測で 0%）。パイプラインは「全力で保守的になれ」と叫び、天井が「いや静かにしろ」と押さえ込んでいる構図。

### 1.3 レイヤー3「事後対応ガード」— 正確

`sell_dynamic_kill` が rolling PnL の**遅行指標**であることは 532# C14902 事例で実証済み。536# の「サンドバッグ」という表現は妥当。

### 1.4 追加: 536# が完全に見落としている要素

| 要素 | 実態 | 536# の扱い |
|------|------|-------------|
| **491# composite_risk** | Soft Gate を重み付き累積 → threshold で判定（disabled） | 未記載 |
| **219# unknown_bypass** | 5回連続 UNKNOWN → 強制通過 | 未記載 |
| **Sidecar (372#)** | SAC 出力 → ±0.15bps（但し TTL 切れで 92% stale） | 未記載 |
| **533# deadlock prevention** | BTC=0 時 veto 緩和 + 連続上限 | 未記載 |
| **min_spread_jpy: 500** | 535# で 700→500 に修正済み | 「700」と記載（古い値） |
| **OB 5-level depth** | SkipGate ML 用に取得可能 | 536# は OFI 提案するが OB 利用可能性に言及なし |

---

## §2 シナリオ A「抜本的ハブ化」の金融工学的検証

### 2.1 理論的妥当性: ★★★★☆

536# が提案する OFI ベースの「逆選択コスト内部化」は、マーケットメイキング理論の正道である。

- **Glosten-Milgrom (1985)**: スプレッドは情報の非対称性コストを内包すべき → 固定 min_spread ではなく動的 AS コスト反映が正
- **Avellaneda-Stoikov (2008)**: 最適スプレッド ≈ γσ²τ + (2/γ)ln(1 + γ/κ) → σ（ボラティリティ）連動が自然
- **Kyle (1985)**: 流動性 λ は情報トレーダーの存在強度に比例 → lambda_proxy は既に SAC features に実装済み

### 2.2 実装上の現実的障壁

**問題 A: OFI のリアルタイム計算の課題**

536# は `ofi = (bid_vol_diff - ask_vol_diff) / total_vol` を提案するが：

| 要件 | 現状 | ギャップ |
|------|------|---------|
| L2 板情報の取得 | 5-level snapshot 取得可能（SkipGate 経由） | ✅ 利用可能 |
| Tick-by-tick 更新 | OHLCV バー単位での更新のみ | ⚠️ `bid_vol_diff` は連続 OB snapshot の差分が必要。現行は cycle 間隔（数秒）での snapshot 取得のみ |
| OFI Window 長 | 536# は言及なし | ⚠️ 短すぎるとノイズ、長すぎると遅行。20~50 tick が学術的標準だが、1回/cycle の OB 取得では window 構築に数分必要 |

**結論**: OFI は cycle 毎の OB snapshot 差分で**近似的に計算可能**。ただし精度は tick-by-tick に劣る。Coincheck の WebSocket が tick-level OB diff を提供するなら理想的。

**問題 B: ATR 動的スプレッドの制御可能性**

536# の `base_spread = ATR * 0.5` は理想的だが：

- ATR が急落すると min_spread が極小化 → fill rate 向上だが逆選択リスクが爆発
- ATR が急騰すると min_spread が巨大化 → 事実上の取引停止（≒現行の deadlock と同じ）
- **必要**: ATR 連動スプレッドの上下限クランプ（`max(200, ATR * k, min(1500, ...))`）

**問題 C: Ceiling 廃止は漸進的に**

536# は ceiling を「撤廃」と述べるが、531# が示す 100% 飽和は**ceiling の値が小さすぎる**問題であり、ceiling 自体の廃止とは別問題。

- **推奨**: まず ceiling を 0.25 → 0.35 → 0.50 と段階引き上げ、各段で fill_rate と AS-PnL を計測
- ceiling なしの「青天井」は、パイプラインのバグ1つで発注価格が mid を超える catastrophic error を招く

### 2.3 シナリオ A の修正版評価

| 項目 | 536# 原案 | 修正後推奨 |
|------|----------|-----------|
| OFI 導入 | 「固定値を全て OFI で置換」 | まず OFI を **240# Toxicity 段のシグナルとして追加**（VPIN と並列入力）。全置換は OFI の実績検証後 |
| ATR 動的スプレッド | `base_spread = ATR * 0.5` | `min_spread_jpy = clamp(ATR * k, floor=200, cap=1500)` |
| Ceiling 廃止 | 「撤廃」 | 段階的引き上げ（0.35 → 0.50）+ catastrophic guard 残存 |
| sell_dynamic_kill 廃止 | 「散らす」 | **Toxicity Budget (240#) に統合**（kill binary → graduated offset）。kill 機構自体は max_duration=600s に短縮して残す |

---

## §3 シナリオ B「パイプライン正規化」の検証

### 3.1 理論的妥当性: ★★★☆☆

Sigmoid 正規化による加重平均は統計学的に合理的だが、以下の固有問題がある：

- **乗算→加算の変換**: 現行パイプラインは `ratio *= mult` の乗算累積。これを `score = Σ(w_i × σ(x_i))` に変えると、各段の「防衛的意図」が薄まる。極端な1指標（例: VPIN=0.95）の警報が加重平均で希釈される
- **重み w_i の決定問題**: 9段の重みをどう決めるか。手動チューニングは現行の固定閾値と同種の問題、ML学習は過去の「パラメータ発散」の二の舞
- **Sigmoid の値域圧縮**: σ(x) ∈ [0,1] に潰すと、Toxicity: 0.9 と EV: 0.95 が「ほぼ同等のスコア」として合算される。意味論的には全く異なるリスク軸

### 3.2 実はもっと簡単な解がある

パイプラインの根本問題は**各段が independent に乗算し合い、結果が指数的に膨張する**こと。

正規化するなら、全段を Sigmoid にする必要はない。**乗算の上限を各段に設ける**だけでよい：

```
stage_capped_mult = min(raw_mult, stage_max_mult)
```

例: 各段の max_mult を 1.5 に制限 → 9段全て max でも `0.05 × 1.5^8 ≈ 1.28`（Sidecar は加算なので除外）。ceiling 0.50 で十分収まる。

**これは現行コードの `_apply_offset_multiplier()` に 1行追加するだけで実装可能**。

---

## §4 シナリオ C「枯死機能の剪定」の検証

### 4.1 理論的妥当性: ★★★★★（リスク最小・効果確実）

536# の「まず明らかな不要物だけを風に飛ばす」は最も合理的な出発点。

#### 4.1.1 即座に剪定すべきもの

| 対象 | 根拠 | 実装コスト |
|------|------|-----------|
| **sell_dynamic_kill → Toxicity Budget 統合** | 532# 実証: 反応型 kill は損失後。240# Toxicity Budget が graduated response を既に提供 | 低: kill trigger を toxicity KILL level に統合 |
| **Ceiling 0.25 → 0.35 引き上げ** | 531# 実証: 100% 飽和。0.35 で上位 ~30% が解放、パイプライン機能部分回復 | 極低: YAML 1行変更 |
| **sell_hour_offset_boost の一部削除** | 533# 批判: 過学習。但し 12h/15h の AS clustering は 532# で統計的に有意 → 完全削除ではなく **OFI シグナルが稼動するまでの橋渡し** として維持 |
| **composite_risk: true 有効化** | 491# 実装済み・disabled。Soft Gate が binary block → weighted accumulation に変わり、deadlock 緩和 | 極低: YAML 1行変更 |

#### 4.1.2 剪定してはいけないもの

| 対象 | 理由 |
|------|------|
| **min_spread_jpy 自体** | floor がないと逆選択コスト＞利益の注文が約定。535# で 700→500 に下げたがゼロにはできない |
| **cross_venue_lead_lag_veto** | 他取引所の先行情報は**唯一の予測的防衛指標**。533# も veto 自体は肯定、閾値調整を推奨 |
| **velocity_skip (Gate 6)** | C4/C5 急変動検知は reactive ではなく **concurrent**（発生中の検知）|
| **unknown_regime_skip** | UNKNOWN 状態はモデル不確実性の表明。219# bypass が安全弁 |

---

## §5 536# が触れていない収益向上の提案

以下は 536# の3シナリオとは独立に、収益性向上に直結する施策。

### 5.1 提案 P1: Maker-Taker 非対称性の活用（即効性 ★★★★★）

**現状**: buy/sell を対称的に扱っている（fill_test.yaml の多くのパラメータが buy/sell 共通）。

**金融工学的根拠**: BTC/JPY マーケットメイキングにおいて、buy と sell は本質的に非対称である：
- **Inventory risk**: JPY→BTC（buy）は「ポジション取得」、BTC→JPY（sell）は「ポジション解消」
- **Adverse selection**: uptick での sell は informed trader の可能性が高い（Kyle 1985）
- **データ実証**: sell AS avg = -7.31bps vs buy 側は比較的安定

**提案**:
- buy の ceiling を sell より低く維持（buy: 0.30, sell: 0.40）— buy は aggressive に fill を取りに行く
- sell の min_spread_jpy を buy より高く設定（sell: 600, buy: 400）— sell は AS コスト内部化
- これにより「安く買って高く売る」基本戦略を構造的に enforcement

### 5.2 提案 P2: Composite Risk の即時有効化（即効性 ★★★★★）

**現状**: 491# で実装済みだが `composite_risk_enabled: false`。

**効果**: Soft Gate が binary skip → weighted accumulation に変わる。
- 例: unknown_regime(0.6) + ranging_low_vol(0.5) = 1.1 > threshold(1.0) → skip
- 例: unknown_regime(0.6) のみ = 0.6 < threshold → **通過**（現行は hard block）
- **Deadlock 緩和**: 単一 soft gate による不必要な全面停止を解消

**リスク**: threshold の calibration が必要だが、1.0 をデフォルトとして開始し、fill_rate を見ながら調整可能。

### 5.3 提案 P3: OFI-Lite — Cycle 差分 OB Imbalance（中期 ★★★★☆）

536# の OFI 完全版は実装コストが高い。代わりに：

```python
# 前 cycle の OB snapshot との差分で近似 OFI
bid_delta = sum(current_bids[:5]) - sum(prev_bids[:5])  # 5-level 合計
ask_delta = sum(current_asks[:5]) - sum(prev_asks[:5])
ofi_lite = (bid_delta - ask_delta) / (abs(bid_delta) + abs(ask_delta) + eps)
```

**利点**:
- OB snapshot は SkipGate が既に cycle 毎に取得（355# L-1 prefetch）
- ofi_lite ∈ [-1, +1]。-1 = 売圧、+1 = 買圧
- **240# Toxicity 段への入力として追加**するだけ。pipelineの構造変更不要

**金融工学的裏付け**: Cont, Kukanov & Stoikov (2014) "The Price Impact of Order Book Events" — 板の差分変化は短期価格変動の最良予測子。

### 5.4 提案 P4: 各段 Stage Max Mult の導入（即効性 ★★★★★）

**現状**: 各段の乗数に上限なし → 乗算的膨張 → 100% ceiling hit。

**提案**: `_apply_offset_multiplier()` に各段の max_mult を追加：

```python
capped_mult = min(offset_mult, stage_max_mult)  # e.g., 1.5
```

**効果**: 9段全 max でも `0.05 × 1.5^8 = 1.28`。ceiling 0.50 で自然に収まる。
- パイプラインの各段が*意味のある差異*を出力に反映できるようになる
- 「全段フルスロットル → ceiling で全潰し」から脱却

### 5.5 提案 P5: Sidecar TTL 修正と SAC アクション幅拡大（中期 ★★★☆☆）

**現状**: Sidecar TTL=600s vs retrain_interval=7200s → 92% stale。±0.15bps の影響力も極小。

**提案**:
- TTL を retrain_interval に合わせる（7200s or 無制限で最新推論値を常時使用）
- SAC のアクション幅を ±0.15bps → ±5.0bps に拡大（A-S 最適スプレッドモデルの 1σ 程度）
- SAC が*意味のある*価格調整を行えるようにする

**リスク**: SAC の学習が不安定な場合に大きな損失を生む → **SAC offset にも per-stage max_mult（例: 3.0bps）を設定**。

### 5.6 提案 P6: Avellaneda-Stoikov 最適スプレッドの導入（長期 ★★★★★）

**理論**: Avellaneda & Stoikov (2008) のマーケットメイカー最適化問題：

$$
\delta^{bid/ask} = \frac{1}{\gamma} \ln\left(1 + \frac{\gamma}{\kappa}\right) + \frac{\gamma \sigma^2 (T-t)}{2} \mp q \gamma \sigma^2 (T-t)
$$

- $\gamma$: リスク回避パラメータ
- $\sigma$: ボラティリティ（Parkinson σ が既に実装済）
- $\kappa$: 注文到着強度（fill_rate から推定可能）
- $q$: 在庫ポジション（inventory_imbalance が既に実装済）
- $T-t$: 残存ホライズン

**現行システムとの対応**:

| A-S パラメータ | 現行の対応物 | 精度 |
|---------------|-------------|------|
| σ | Parkinson σ (305#) | ✅ 実装済 |
| q | inventory_imbalance (226# P5) | ✅ 実装済 |
| κ | fill_rate 計測 | ⚠️ 要追加 |
| γ | 手動パラメータ | ⚠️ 要設定 |

**提案**: A-S 最適スプレッドを**参照スプレッド（reference spread）**として計算し、現行の固定 `min_spread_jpy` + `spread_offset_ratio` をこれに置換する。パイプラインの base を理論的最適値から開始する。

```python
# 参照実装（概念）
sigma = parkinson_sigma(recent_bars, window=20)
kappa = estimate_arrival_intensity(fill_history, window=50)
gamma = config.risk_aversion  # 例: 0.01
q = inventory_imbalance
T_minus_t = config.horizon_sec  # 例: 300s

optimal_half_spread = (1/gamma) * math.log(1 + gamma/kappa)
inventory_skew = q * gamma * sigma**2 * T_minus_t
bid_offset = optimal_half_spread + inventory_skew
ask_offset = optimal_half_spread - inventory_skew
```

### 5.7 提案 P7: Fill Quality フィードバックループ（長期 ★★★★☆）

**現状**: Fill 後の PnL は DynamicKill の rolling mean に反映されるのみ。

**提案**: Fill Quality を**即時フィードバック**としてパイプラインに還元: 
- 直近 N fills の AS-PnL を exponential moving average で追跡
- AS-PnL が悪化傾向 → 自動的に offset を引き上げ（予防的）
- AS-PnL が好転傾向 → offset を引き下げ（aggressive に fill を狙う）
- **実装**: 240# Toxicity Budget が既にこの構造を持つ。入力を rolling_mean ではなく EWMA にするだけ（344# で部分的に実装済み）

### 5.8 提案 P8: 逆張り vs 順張りの動的切替（革新的 ★★★★☆）

**現状**: 常に「spread を取る」マーケットメイキング戦略。

**金融工学的洞察**: マーケットメイカーの最大の敵は「情報トレーダーとの取引」（逆選択）。しかし、**情報の非対称性が低い時間帯**（ranging, 低ボラ）では spread が薄くても安全。

**提案**: regime に応じた戦略切替：
- **RANGING + 低 VPIN**: aggressive fill（offset 縮小, min_spread 縮小）→ 薄利多売
- **TRENDING + 高 VPIN**: defensive（offset 拡大, 場合により skip）→ 損失回避
- **HIGH_VOL + 高 OFI**: 様子見 or 逆張り的に深い offset で待機

これは実質的に 193# EV Offset が意図していた機能だが、現行では ceiling で潰されている。

---

## §6 実装ロードマップ — 段階的「渙（と）かし」

536# の「シナリオ C から入り、A を目標に」という方針に同意する。ただし具体的な順序を再構成する。

### Phase 0: 即日実行可能（YAML 変更のみ）

| # | 施策 | 変更箇所 | 期待効果 |
|---|------|---------|---------|
| 0-1 | Ceiling 引き上げ 0.25 → 0.35 | fill_test.yaml: `offset_ceiling_ratio_buy/sell` | Pipeline 上位 30% の情報が取引価格に反映 |
| 0-2 | `composite_risk_enabled: true` | fill_test.yaml | Soft Gate deadlock 緩和 |
| 0-3 | sell_dynamic_kill の max_kill_duration: 1800 → 600 | fill_test.yaml | kill 状態の長期化防止 |

**検証**: 24h 稼動後に fill_rate, AS-PnL, clamp_rate を計測。

### Phase 1: 低コスト構造変更（コード数行）

| # | 施策 | 変更箇所 | 期待効果 |
|---|------|---------|---------|
| 1-1 | Stage Max Mult 導入 (P4) | `pre_order_adjustments.py` に 1行追加 | 乗算膨張の構造的解消 |
| 1-2 | OFI-Lite 導入 (P3) | `offset_pipeline.py` 240# Toxicity 段に OB diff 入力追加 | 予測的 AS 検知の第一歩 |
| 1-3 | buy/sell 非対称 ceiling (P1) | fill_test.yaml: buy: 0.30, sell: 0.40 | 構造的な「安く買って高く売る」|

### Phase 2: 中規模リファクタリング

| # | 施策 | 変更箇所 | 期待効果 |
|---|------|---------|---------|
| 2-1 | min_spread_jpy の ATR 連動化 | maker_price.py | 市場適応的な最小スプレッド |
| 2-2 | Sidecar TTL 修正 + アクション幅拡大 (P5) | sidecar_types.py, fill_test.yaml | SAC の実効的影響力回復 |
| 2-3 | sell_dynamic_kill → Toxicity Budget 完全統合 | sell_dynamic_kill.py, offset_pipeline.py | binary kill の廃止、graduated response への移行 |

### Phase 3: A-S ハブ化（536# シナリオ A）

| # | 施策 | 変更箇所 | 期待効果 |
|---|------|---------|---------|
| 3-1 | A-S 最適スプレッド参照値導入 (P6) | 新モジュール + maker_price.py | 理論的最適値からの出発 |
| 3-2 | OFI 完全版（tick-level or WebSocket） | ztb/features/ + pipeline | 高精度 AS 予測 |
| 3-3 | 動的戦略切替 (P8) | regime × VPIN → strategy selector | 環境適応型 MM |

---

## §7 536# の「風水渙」解釈についての批評

### 7.1 「廟（みたまや）に祭る」の再解釈

536# は「廟 = 最終防波堤・コアコンセプト」と解釈しているが、金融工学的に「廟に祭るべき不動の原則」は以下であるべき：

1. **Spread > Adverse Selection Cost** — スプレッドは常に逆選択コストを上回る（Glosten-Milgrom の存在条件）
2. **Inventory Mean-Reversion** — 在庫は必ず中立に回帰させる（Ho-Stoll の生存条件）
3. **Catastrophic Loss Prevention** — 単一 fill で資本の X% 以上を失わない（Kelly 基準の下限）

これらが「廟」であり、その他の一切（ceiling, kill, veto, hour boost, regime skip）は全て手段であって目的ではない。手段の氷を溶かし、目的の廟だけを残す — これが「渙」の正しい実装。

### 7.2 「ML への全委譲」への警戒は正しい

536# の「過去の知見の通りMLに逃げると発散する」は本システムの歴史と整合する。SAC Sidecar が ±0.15bps という極小影響力しか持たない現状は、逆に言えば「ML を信頼していない」ことの表明。

**正しいアプローチ**: ML に「最終判断」を委ねるのではなく、ML に「情報の要約」を委ねる。
- **NG**: SAC が直接 `buy/sell/hold` を決定
- **OK**: SAC が `regime_confidence`, `as_risk_score`, `inventory_pressure` を出力 → ルールベースが最終判断

これは現行の Sidecar 設計の延長線上にある。Sidecar の影響力を拡大しつつ、ルールベースの「廟」は堅持する。

---

## §8 結論と推奨

### 最優先（今日やるべきこと）

1. **Ceiling 0.25 → 0.35** — YAML 1行。100% 飽和を≒30% 飽和に軽減
2. **composite_risk_enabled: true** — YAML 1行。Soft Gate deadlock 解消
3. **sell max_kill_duration: 600s** — YAML 1行。kill 長期化防止

### 今週やるべきこと

4. **Stage Max Mult 導入** — コード 1行。乗算膨張の構造的解消
5. **OFI-Lite 導入** — 既存 OB snapshot を活用した予測的 AS 検知

### 来週以降

6. A-S 最適スプレッドの参照実装
7. sell_dynamic_kill → Toxicity Budget 統合
8. Sidecar 影響力拡大

536# のシナリオ A は最終目標として正しいが、一足飛びに到達するのではなく、Phase 0 → 1 → 2 → 3 の段階的「渙き」で、各段階の fill_rate と AS-PnL を計測しながら進むことを強く推奨する。

---

*以上*
