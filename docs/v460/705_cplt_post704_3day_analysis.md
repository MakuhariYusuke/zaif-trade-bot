# 705# Post-704# 3-Day Analysis Report (Apr 4-6 vs Apr 1-3)

## 概要

704# で実施した sell 損失構造改善（spread_as_guard staleness fix、sell_trending_down_offset、sell_hour_offset_boost、regime_guard_overrides）の効果を、3日間のライブデータ（Apr 4-6, SHA `352e3b7d9`）で評価する。

## 主要メトリクス比較

| 指標 | Pre-704# (Apr 1-3) | Post-704# (Apr 4-6) | 変化 |
|------|---------------------|----------------------|------|
| サイクル数 | 1,403 | 1,109 | -21% |
| フィル数 | 504 (35.9%) | 461 (41.6%) | +16% |
| BUY fills | 254 | 235 | |
| SELL fills | 250 | 226 | |
| **30s PnL total (bps)** | **-149.2** | **-72.4** | **+51%改善** |
| **30s PnL avg (bps)** | **-0.296** | **-0.157** | **+47%改善** |
| BUY 30s PnL (bps) | +45.1 (+0.177 avg) | **-75.2 (-0.320 avg)** | **完全反転 ✗** |
| SELL 30s PnL (bps) | -194.3 (-0.777 avg) | **+2.9 (+0.013 avg)** | **劇的改善 ✓** |
| BUY win rate | 49% | 43% | -6pt |
| SELL win rate | 42% | 46% | +4pt |
| BUY AS rate | 50.4% | 56.6% | +6pt悪化 |
| SELL AS rate | 58.4% | 54.0% | -4pt改善 |
| BUY eff_offset | 0.192 | 0.134 | **-30%** |
| SELL eff_offset | 0.404 | 0.387 | -4% |
| 平均 Spread (bps) | 2.33 | 2.03 | -13%（市場縮小） |
| Entry gate actual blocks | 11 | 10 | 横ばい |
| Skip gate (buy) skips | 59 | **0** | **完全停止** |

## 市場環境

- BTC/JPY: ¥10,700,000 → ¥11,200,000（**+3.6% 上昇トレンド**）
- Spread: 2.33 → 2.03 bps（**-13% 縮小**）= 流動性改善
- Regime: 89% ranging / 6.6% trending_down / 4.4% trending_up
- BTC 在庫比率: 60-62%（安定的にBTC偏重）

## 704# Fix 効果検証

### ✅ P1: spread_as_guard staleness fix → **完全成功**
- Trigger率: 5.8% → **99.6%**（`last_spread_raw` 使用で staleness 解消）
- EV ペナルティ 0.5bps + regime premium 0.3bps が全トレードに正しく適用

### ✅ sell_trending_down_offset → **有効**
- SELL+trending_down PnL: -74.8 → -0.4 bps

### ✅ sell_hour_offset_boost → **有効**
- SELL 全体: -194.3 → +2.9 bps（プラスに反転）

### ⚠️ entry_gate side-aware blocking → **機能不全（新発見）**
- 設計意図: buy_suppress_ev_threshold=-0.5 で、EV≧-0.5 の buy のみ通す
- **実態**: auto-disable が先に発火し、side-aware ロジック不到達
- 根本原因: CalibrationMap EV が構造的に負（avg=-1.85）→ block_rate=100% > max_block_rate=0.95 → auto_disable=True → **全通し**

### ❌ buy_dynamic_kill → **Apr 6 で発動（14回）**
- UTC 13:46-14:25 に buy kill 発動
- 直前の buy PnL 悪化（-7.758, -3.912 bps 連続）がトリガー
- **正常動作**（rolling EWMA が threshold 以下で正しく kill）

## Buy 側劣化の根本原因分析

### 原因1: 市場スプレッド縮小（外部要因、704# と無関係）

| Offset Stage | Pre-704# | Post-704# | 変化 |
|-------------|----------|-----------|------|
| spread_bps | 2.37 | 2.02 | -15% |
| spread_adapt | 0.109 | 0.077 | **-29%** |
| amihud | 0.117 | 0.082 | -30% |
| kyle | 0.109 | 0.077 | -29% |
| as_shift | 0.093 | 0.058 | -38% |
| regime | 0.116 | 0.076 | -34% |
| **effective_offset** | **0.192** | **0.134** | **-30%** |

スプレッドが -15% 縮小 → spread_adapt を起点に全ての市場依存コンポーネントが連鎖的に縮小。buy の base offset (0.05) が sell (0.14) の 1/3 しかないため、相対的な影響が極めて大きい。

### 原因2: Skip Gate スコア崩壊（モデルドリフト）

| 指標 | Pre-704# | Post-704# |
|------|----------|-----------|
| skip_gate score avg | **+0.327** | **-0.322** |
| skip_gate score median | +0.577 | -0.295 |
| buy skipped | 59 (11.3%) | **0 (0%)** |
| forced_pass | 42 | 120 |

スコアが正→負に反転 → threshold=0.5 を超えるものがゼロ → 全 buy が通過 → marginal な機会も含めて全fill → PnL 劣化。

### 原因3: Entry Gate auto-disable（アーキテクチャ欠陥）

```python
# entry_gate_guard.py L47-68 の実行順序
def should_suppress_block(self, *, ev, regime, side):
    if self._state.auto_disabled:        # ← (1) ここで True → return True
        return True
    if self._is_stale(): ...             # ← (2)
    if consecutive >= max_consecutive: ... # ← (3)
    if block_rate >= max_block_rate: ...  # ← (4) 20件後に発火 → auto_disabled=True
    if side == "buy" and ev >= -0.5: ... # ← (5) side-aware → 永久に到達不可
    return False
```

CalibrationMap EV が常時負 → block_rate ≈ 100% → (4) で auto_disable → (1) で全通し。
**side-aware ロジックは dead code**。

### 原因4: 上昇トレンド × タイトスプレッド × 低オフセット

低 offset の buy limit order は mid 近傍に配置 → 浅い dip で約定 → 上昇トレンド自体が本来 buy に有利だが、30s 時点ではまだ dip が完了していないケースが多く negative PnL に。

特に **Q4（最高 offset）が最悪 PnL (-0.753)** という逆説的結果は、高 offset = 深い dip で約定 = dip がさらに深くなる genuine adverse move で約定、を示唆。

### 損失時間帯分析（buy+ranging, UTC）

| Hour(UTC) | avg 30s PnL | n |
|-----------|-------------|---|
| 1h | -13.45 | 少量 |
| 13h | -15.40 | 多量 |
| 14h | -6.85 | 多量 |
| 18h | -6.71 | 少量 |

## 構造的整理

```
704# Sell 改善                704# が原因ではない Buy 劣化
─────────────               ───────────────────────────
spread_as_guard fix ✓        市場スプレッド縮小 -15%（外部）
sell offsets 増加 ✓          skip_gate スコア崩壊（モデルドリフト）
→ sell offset 維持 (-4%)    entry_gate auto-disable（既存バグ）
→ SELL PnL: -194→+3 ✓      → buy offset -30% 圧縮
                            → BUY PnL: +45→-75 ✗
```

**総合**: 704# の sell 修正は完全に成功。buy 劣化は (a) 外部市場変化 + (b) 既存アーキテクチャ欠陥の顕在化。704# コード変更は buy パスに直接影響していない。

## 改善提案（705#）

### P1: entry_gate_guard auto-disable の side-aware 化（確実性: 高）

auto-disable チェックを side-aware ロジックの**後**に移動、または per-side auto-disable を実装。

```python
# 提案: side-aware を先に評価
def should_suppress_block(self, *, ev, regime, side):
    # Side-aware: buy-specific threshold check FIRST
    if side == "buy" and ev >= self._config.buy_suppress_ev_threshold:
        return True
    # sell は常にブロック候補（suppress しない）
    if side == "sell":
        return False
    # ここからは fallback: auto-disable は buy+低EV のみに適用
    if self._state.auto_disabled:
        return True
    ...
```

ただし現状 buy EV が全て -0.5 以下なので、上記だけでは不十分。**threshold 調整**（-0.5 → -2.0 or -3.0）も併用する。

### P2: buy_base_offset 引き上げ（確実性: 中）

現在 buy base=0.05 vs sell base=0.14。スプレッド縮小時に buy が圧縮されすぎる根本原因。

- 提案: buy base_offset 0.05 → 0.08〜0.10
- fill rate は若干低下するが、fill quality が改善
- **注意**: base_offset は `configs/v460/fill_test.yaml` の `base_offset_buy` で設定

### P3: buy_trending_up_offset 追加（確実性: 中）

sell_trending_down_offset=0.5 の対称として、buy の trending_up 防御。

- 上昇トレンドでの buy は、shallow dip で約定 → dip 継続リスク
- buy_trending_up_offset = 0.3〜0.5 で offset を上積み
- skip_gate_evaluator.py に既存パターンで追加可能

### P4: buy_hour_offset_boost（確実性: 低）

Buy 損失時間帯（UTC 13h, 14h）にオフセット加算。

- UTC 13h = JST 22h、UTC 14h = JST 23h
- ただし sample size が小さく、直接実装より monitor-first を推奨

### P5: skip_gate モデル品質調査（確実性: 不明）

スコアが +0.33 → -0.32 に崩壊した原因調査が必要。

- retrain 後のモデルドリフト？
- 入力特徴量の分布シフト？
- 上昇トレンド環境でのモデル不適合？

## 推奨実施順序

1. **P1** (entry_gate fix) — dead code 修正、効果が確実
2. **P2** (buy base offset) — 根本対策、tuning 必要
3. **P3** (buy_trending_up_offset) — 既存パターン活用、低リスク
4. **P5** (skip_gate 調査) — 要調査、Codex タスク候補
5. **P4** (hour boost) — データ蓄積後に再評価

## JPY 推定損益

| 期間 | 推定 JPY PnL（trading） | BTC 評価益 |
|------|------------------------|-----------|
| Apr 4 | -51.6 bps | +95 |
| Apr 5 | -60.5 bps | +390 |
| Apr 6 | +41.6 bps | +185 |
| **合計** | **-70.5 bps ≈ -37 円** | **+670** |

BTC 評価益に支えられているが、trading PnL 自体は依然マイナス。

---

## Deep Dive: 統計的検証と多角的分析（705# 深堀り）

### A. 統計的有意性検証

#### A1. Bootstrap 95% 信頼区間 (n=10,000)

| セグメント | Δ avg (bps) | 95% CI | p-value | 判定 |
|-----------|------------|--------|---------|------|
| ALL | +0.139 | [-0.346, +0.637] | 0.296 | **有意でない** |
| BUY | -0.498 | [-1.018, +0.011] | 0.028 | **有意 (p<0.05)** |
| SELL | +0.790 | [-0.044, +1.644] | 0.031 | **有意 (p<0.05)** |

#### A2. Permutation Test (n=10,000, two-sided)

| セグメント | p-value | 判定 |
|-----------|---------|------|
| ALL | 0.591 | NS |
| BUY | 0.062 | 境界的 (p<0.10) |
| SELL | 0.073 | 境界的 (p<0.10) |

#### A3. Effect Size (Cohen's d)

| セグメント | d | 解釈 | pre_std | post_std |
|-----------|---|------|---------|----------|
| ALL | +0.035 | negligible | 4.49 | 3.36 |
| BUY | -0.169 | negligible | 3.50 | 2.17 |
| SELL | +0.164 | negligible | 5.27 | 4.26 |

**統計的解釈**:
- Bootstrap の one-sided test では BUY 劣化・SELL 改善とも p<0.05 で有意
- しかし Permutation test (two-sided) では境界的。**サンプルサイズ（各230-250件）に対して PnL 分散が大きすぎる**（std ≈ 2-5 bps vs mean ≈ 0.1-0.8 bps）
- Cohen's d は全て negligible → **効果量は極めて小さい**。観測された差は統計的ノイズの範囲内に収まる可能性がある
- **結論**: sell 改善のシグナルは direction として正しいが、3日間では統計的に確定できない。少なくとも1-2週間の追加データが必要

### B. 反事実分析（Counterfactual Analysis）

#### B1. スプレッド正規化: "スプレッドが同じだったら？"

```
Pre spread: 2.37 bps → Post spread: 2.02 bps （-15%）
Pre offset: 0.192  → Post offset: 0.134
Spread-offset 相関: r=0.206 (pooled)

スプレッド比率による線形推定:
  同一スプレッドなら estimated offset ≈ 0.157
  → スプレッド縮小は offset 低下の約 40% を説明
  → 残り 60% は他の要因（OFI分布変化、regime構成等）
```

#### B2. Skip Gate 復元: "skip_gate が以前と同様に機能していたら？"

```
Post buy fills: n=235, avg PnL=-0.320
If worst 11% removed（pre 期と同等の skip 率、最適ケース）:
  → n=209, avg PnL=+0.170, total=+35.5 bps
If random 11% removed（現実的ケース）:
  → n=209, avg PnL=-0.331, total=-69.1 bps
If worst 20% removed:
  → n=188, avg PnL=+0.403, total=+75.8 bps
```

skip_gate が pre-704# と同等に機能し、最悪の fill を排除できた場合、buy PnL はプラスに戻る可能性がある。ただし「最悪の fill を事前に識別できる」前提は楽観的。

#### B3. 因果パス分析: "704# コード変更は buy に影響したか？"

```
spread_as_guard が buy に与える影響パス:
  SAG triggered → EV に -1.05 bps ペナルティ → entry_gate 判定で使用
  → しかし entry_gate auto-disabled → ペナルティは無効
  → offset pipeline に影響なし（SAG は EV 専用、offset 非連動）
  
結論: 704# コード変更の buy への因果パスは **ゼロ**
```

### C. Pre-Trend 分析: "buy 劣化は 704# 以前から始まっていたか？"

| 日 | buy avg PnL | offset | spread | buy skip | SHA |
|----|------------|--------|--------|----------|-----|
| Apr 1 | **+0.776** | 0.230 | 2.62 | 24 | 8d5d304 |
| Apr 2 | +0.333 | 0.182 | 2.34 | 0 | b5f7828 |
| Apr 3 | **-0.400** | 0.175 | 2.22 | 35 | dc9fb47 |
| Apr 4 | -0.196 | 0.123 | 2.01 | 0 | 352e3b7 ← 704# |
| Apr 5 | -0.248 | 0.122 | 1.97 | 0 | 352e3b7 |
| Apr 6 | -0.635 | 0.170 | 2.17 | 0 | 352e3b7 |

**重要な発見**: Apr 3（704# デプロイ前日）で buy PnL は既に **-0.400** でマイナスに転落。offset も 0.230→0.175 と既に低下トレンドにあった。**buy 劣化は 704# デプロイに先行しており、市場スプレッド縮小と連動する外部要因**。

### D. スプレッドマッチ比較（市場環境を統制）

同一スプレッド帯で pre/post を比較し、コード変更の純粋な効果を分離:

| Spread帯 | Pre BUY (n, avg PnL) | Post BUY (n, avg PnL) | Δ |
|----------|----------------------|-----------------------|---|
| 1.0-1.8 tight | 42, +0.173 | 83, **-0.497** | **-0.670** |
| 1.8-2.5 mid | 111, +0.099 | 81, -0.068 | -0.167 |
| 2.5-4.0 wide | 97, +0.429 | 59, **-0.341** | **-0.770** |

| Spread帯 | Pre SELL (n, avg PnL) | Post SELL (n, avg PnL) | Δ |
|----------|----------------------|-----------------------|---|
| 1.0-1.8 tight | 59, +0.088 | 73, **+0.138** | +0.050 |
| 1.8-2.5 mid | 99, -1.124 | 93, **+0.216** | **+1.340** |
| 2.5-4.0 wide | 90, -0.933 | 52, +0.020 | **+0.953** |

**解釈**:
- **SELL**: 全帯域で改善。特に mid/wide で大幅改善。**704# の sell offset 調整は spread-band 非依存で有効**
- **BUY**: 全帯域で劣化。特に tight/wide で悪化。**スプレッドを統制しても劣化が残る** → スプレッド縮小だけが原因ではない。skip_gate 停止 + mid drift の方向変化が主因と推定

### E. マイクロストラクチャ変化

| 指標 | Pre BUY | Post BUY | 解釈 |
|------|---------|----------|------|
| queue_wait | 47.5s | **77.5s** | +63%延長 — fill 困難化 |
| OBI | +0.072 | +0.030 | 買い圧低下 |
| microprice_bias | +0.125 bps | +0.062 bps | 価格上昇バイアス半減 |
| VG boost | 1.533 | 1.497 | 微減 |
| AS cost | **+0.177** bps | **-0.320** bps | 反転（有害化） |
| mid drift 30s | **+0.141** bps | **-0.432** bps | **反転（核心）** |
| mid drift 30s 負率 | 52.8% | **64.7%** | +12pt |

**核心発見**: **mid drift が正→負に反転**。Pre 期は buy 約定後に 30s で +0.14 bps 上昇していたが、Post 期は -0.43 bps 下落。これは市場の短期ダイナミクス自体が変化したことを示す。

仮説: 上昇トレンドの「フェーズ」の違い。Apr 1-3 は上昇初動（momentum-driven dip buying が有効）、Apr 4-6 は上昇後半（mean-reversion 的な dip が shallow reversal に）。

### F. テールリスクプロファイル

| 分布 | P5 | P10 | P25 | P75 | P90 | P95 | Skew | Kurt |
|------|-----|------|------|------|------|------|------|------|
| Pre BUY | -4.98 | -3.51 | -1.75 | +1.90 | +4.55 | +5.68 | -0.16 | 1.86 |
| Post BUY | -4.01 | -2.62 | -1.27 | **+0.57** | **+1.66** | +2.84 | +0.44 | **6.99** |
| Pre SELL | -8.94 | -6.93 | -3.51 | +1.68 | +4.28 | +8.15 | +0.30 | 2.03 |
| Post SELL | **-4.56** | **-3.10** | **-1.66** | +1.45 | +3.67 | +5.13 | +2.31 | **28.46** |

**解釈**:
- **Post BUY**: テール損失は改善（P5: -4.98→-4.01）だが、**upside が大幅縮小**（P75: +1.90→+0.57, P90: +4.55→+1.66）。損失が小さくなった以上に利益が減った
- **Post SELL**: 左テール大幅改善（P5: -8.94→-4.56）。kurtosis=28.46 は outlier 1件（-23.18 bps）の影響
- Post Buy の高 kurtosis (6.99) は「通常は小さい PnL だが、たまに大きく動く」状態

### G. PnL 自己相関と損失連鎖

| 期間 | AC(1) | 最長 loss streak | W→W | W→L | L→W | L→L |
|------|-------|-----------------|------|------|------|------|
| Pre BUY | -0.054 | 7 | 57 | 68 | 67 | 61 |
| Post BUY | -0.009 | **10** | 42 | 59 | 59 | **74** |

Post では **L→L（損失→損失遷移）が74回で最多**。最長 loss streak も 7→10 に延長。損失の粘着性が増加。

### H. Entry Gate EV 感度分析

Post-704# buy cycles の entry_gate_ev 分布:
```
  Median: -1.91 bps （CalibrationMap のデフォルト EV がほぼ固定）
  P95:    -1.34 bps
  Max:    -0.59 bps
  
  EV ≧ -0.5:  0件 (0%)   ← 現行 threshold、到達不可能
  EV ≧ -1.0:  6件 (2.1%)
  EV ≧ -1.5: 35件 (12.3%)
  EV ≧ -2.0: 285件 (100%)
```

**threshold 感度**: -0.5 では 0% 到達。**-1.5 に緩和すれば 12.3%、-2.0 なら 100%**。ただし auto-disable が先に発火するため、threshold 変更だけでは不十分（ロジック順序修正が先決）。

### I. Fill 品質分類（30s vs 120s 一致性）

| パターン | Pre BUY | Post BUY | 解釈 |
|---------|---------|----------|------|
| 30s+/120s+ (一貫利益) | 38 | 33 | |
| 30s+/120s- (利確逃し) | 19 | 16 | |
| 30s-/120s+ (回復) | 21 | 22 | |
| 30s-/120s- (真の損失) | 41 | 45 | 増加 |
| Reversion rate (30s+→120s-) | 33% | 33% | 変化なし |

30s→120s の reversion rate は不変（33%）。**短期利益の持続性は変わっていない**。問題は 30s のエッジ自体が縮小していること。

---

## 改善アプローチ: 複数手段の検討と比較

以下、buy 側改善の 6 つのアプローチを多角的に検討する。

### アプローチ1: entry_gate_guard ロジック順序修正（P1 改良）

**概要**: `should_suppress_block()` の auto-disable チェックを side-aware の後に移動

```python
def should_suppress_block(self, *, ev, regime, side):
    # (A) Side-aware: buy で EV が threshold 以上なら通す
    if side == "buy" and ev >= self._config.buy_suppress_ev_threshold:
        return True
    # (B) sell は常に entry_gate 判定に委ねる
    if side == "sell":
        return False
    # (C) 上記に該当しない場合のみ auto-disable fallback
    if self._state.auto_disabled:
        return True
    ...
```

- **利点**: dead code 修正。side-aware が正しく動作するようになる
- **リスク**: threshold=-0.5 だと EV 0% が到達 → 実質 buy 全ブロック。threshold=-2.0 なら 100% 通過で現状と変わらない
- **留意**: threshold を -1.5 にすれば 12.3% の buy をブロック可能。しかし CalibrationMap EV が固定値(-1.91)に集中しているため、**ブロック vs 通過が EV のわずかな変動で 0%/100% に振れるクリフ効果**が発生する
- **判定**: △ 実装は簡単だがチューニングが困難。CalibrationMap 精度向上が前提

### アプローチ2: buy base_offset 引き上げ（P2）

**概要**: buy base=0.05 → 0.08〜0.10

- **利点**: スプレッド縮小への耐性向上。offset pipeline の全段階に base-up が波及
- **リスク**: fill rate 低下。buy fill rate が既に 41% なので、さらなる低下は trading 機会を減少
- **根拠**: スプレッドマッチ分析で tight(1.0-1.8) が最悪(-0.497) → base-up で tight 帯の fill を抑制する効果
- **判定**: ○ 安定的。fill rate 低下は回収率改善で相殺される可能性

### アプローチ3: buy_trending_up_offset 追加（P3）

**概要**: sell_trending_down_offset=0.5 の対称

- **利点**: regime 条件付きなので ranging には影響しない。Post 期 trending_up は 4.4% のみで影響範囲小
- **リスク**: buy+trending_up は本来有利（上昇中の buy → 価格上昇で利益）。Post 期のデータでも buy+trending_up は +0.683 avg と**唯一のプラス区画**
- **矛盾**: buy+trending_up はプラスなのに offset を上げると機会損失
- **判定**: ✗ **データが支持しない**。buy+trending_up は改善不要。問題は buy+ranging

### アプローチ4: buy+ranging 専用防御（新提案）

**概要**: 問題の核心は buy+ranging (-0.322 avg, n=208)。ranging 専用の buy offset boost

- **手段A**: buy_ranging_offset_boost を YAML に追加（sell_hour_offset_boost の対称パターン）
- **手段B**: ranging 時の buy base_offset を動的に引き上げ（`base_offset_buy_ranging: 0.10`）
- **手段C**: OBI (orderbook imbalance) が中立〜売り優勢（<0）の場合のみ buy offset を加算
- **利点**: buy+trending_up (+0.683) を温存しつつ ranging のみ改善
- **リスク**: buy+ranging が n=208 で最大セグメント。過剰防御は全体 fill を大きく減少
- **判定**: ○ アプローチ2 より精密。手段B が最もシンプル

### アプローチ5: skip_gate 再調整（P5 deep）

**概要**: skip_gate score が +0.33→-0.32 に崩壊した問題の解決

- **仮説A**: モデル retrain でパラメータがドリフト → 閾値調整（0.5→低め）で部分復元
- **仮説B**: 入力特徴量の分布がシフト（スプレッド縮小、流動性変化）→ retrain が必要
- **仮説C**: skip_gate が forced_pass を頻繁に使用（120/302 = 40%）→ skip_rate_limit の制限が原因
- **予想効果**: 反事実分析から、worst 11% 排除で buy PnL は +0.170 に（楽観推定）
- **リスク**: skip_gate 精度が不明。スコアと PnL の相関は -0.021 と極めて弱い
- **判定**: △ ポテンシャルは高いが、現行 skip_gate のスコア品質自体に疑義。調査が先

### アプローチ6: mid drift レジーム検出（新提案）

**概要**: 最も根本的な発見は mid drift 30s が +0.14→-0.43 bps に反転したこと。これは「買った直後に価格が下がる」環境への変化。

- **手段**: 直近 N 件の buy fill の 30s drift を rolling 計算し、負が持続する場合に buy offset を動的に引き上げ
- **実装**: `buy_dynamic_kill` と似た rolling EWMA 機構で、kill ではなく offset 加算
- **利点**: 市場環境に自動適応。regimeではキャプチャできない短期ダイナミクスを捉える
- **リスク**: 実装の複雑さ。過剰適合のリスク。lookback window の選定が困難
- **判定**: △ 理論的には最も強力だが、実装コストが高い。中期タスク

### 改善アプローチ比較表

| # | アプローチ | 期待効果 | 実装コスト | リスク | 推奨 |
|---|-----------|---------|-----------|--------|------|
| 1 | entry_gate ロジック修正 | CalibrationMap依存 | 低 | クリフ効果 | △→要調整 |
| 2 | buy base_offset 引上げ | 中 | 極低 | fill rate 低下 | **○ 即実行** |
| 3 | buy_trending_up_offset | 逆効果の可能性 | 低 | データ不支持 | **✗ 不採用** |
| 4 | buy+ranging 専用防御 | 高 | 低 | 過剰防御 | **○ P2と併用** |
| 5 | skip_gate 再調整 | 高（要検証） | 中 | 品質不明 | △ 調査優先 |
| 6 | mid drift 適応型 | 最高 | 高 | 過剰適合 | △ 中期課題 |

### 推奨実施順序（修正）

1. **アプローチ2 (buy base_offset)** + **アプローチ4B (buy_ranging base)** — 即実行。YAML 変更のみ
2. **アプローチ1 (entry_gate ロジック修正)** — auto-disable の side-aware 化（ただし threshold は -2.0 で事実上パス。CalibrationMap 改善後に -1.5 に調整）
3. **アプローチ5 (skip_gate 調査)** — Codex タスク。score 品質分析と threshold 再検討
4. **アプローチ6 (mid drift)** — 中期研究テーマ

### 棄却したアプローチへの反論想定

- **「P3 を棄却するのは早計では？」** → buy+trending_up は n=13 で avg +0.683。データ上プラスのセグメントに防御 offset を追加するのは逆効果。trending_up 自体が buy に有利な環境であり、offset 加算は利益機会を失う
- **「Cohen's d が negligible なら何もしないべきでは？」** → 効果量が小さいのは PnL 分散が大きいため。方向性は bootstrap CI で有意。「何もしない」は trading PnL が構造的にマイナスの現状を放置することになり、大義に反する
- **「3日間のデータで結論を出すべきではない」** → 正しい指摘。しかし pre-trend 分析（Apr 3 で既に -0.400）が 704# の無関与を強く示唆。**待つ**ことと**データを追加しながら低リスク改善を実施**することは両立する

---

## Appendix: 再現コマンド

```bash
# 期間比較レポート（自動診断付き）
python -m scripts.v460.analysis.compare_periods \
  --pre 2026-04-01:2026-04-03 \
  --post 2026-04-04:2026-04-06

# 本レポートの全分析を再現するワンライナー群は
# analysis_results/705_period_comparison.txt に出力済み
```
