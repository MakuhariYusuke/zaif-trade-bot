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
