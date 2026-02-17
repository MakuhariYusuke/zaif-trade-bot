# 090# fill_test 深掘り分析 v2 — Codex レビュー用資料

> **作成日**: 2026-02-17  
> **対象**: v460 "Microstructure Edge" BTC/JPY 自動取引 fill_test  
> **目的**: 外部 AI コーディングエージェント (Codex) に本分析を提示し、構造損失の根本原因特定と改善提案を求める  
> **Git HEAD**: `93ea3fd4c` (089# time_filter 大幅削減)  
> **前回 Codex レビュー**: 082# (`8a2c589d3`) → 087# で回答受領 → 088# で実装済

---

## 目次

1. [前回 Codex レビュー (087#) からの変更点](#1-前回-codex-レビュー-087-からの変更点)
2. [システム概要 (更新版)](#2-システム概要)
3. [コンポーネント稼働状況](#3-コンポーネント稼働状況)
4. [全データ統計概要](#4-全データ統計概要)
5. [多次元分析結果](#5-多次元分析結果)
6. [致命的発見事項](#6-致命的発見事項)
7. [シミュレーション結果](#7-シミュレーション結果)
8. [現在の設定 (fill_test.yaml)](#8-現在の設定)
9. [改善候補の整理](#9-改善候補の整理)
10. [Codex への質問事項](#10-codex-への質問事項)

---

## 1. 前回 Codex レビュー (087#) からの変更点

### 082# → 087# → 088# → 089# の流れ

| # | 内容 | Commit |
|---|------|--------|
| 082# | 初回 Codex レビュー用 fill_test 深掘り資料作成 | `8a2c589d3` |
| 087# | Codex からの回答: P0-1~P0-4, P1-1~P1-3, P2-1~P2-3 提案 | (受領のみ) |
| 088# | 087# 提案の実装 (P0-1~P0-4, P1-3) | `798adf21d` |
| 089# | time_filter 大幅削減 (16h → 6h per side) | `93ea3fd4c` |

### 088# で実装した内容

| 提案 | 実装内容 | テスト |
|------|---------|--------|
| P0-1 SkipGate 動的較正 | `adaptive_threshold=true`, 目標 skip 率ベースで閾値自動調整 | 12 tests ✅ |
| P0-2 Sell ハードガード | `sell_guard.max_spread_jpy=4000`, `offset_floor=0.08` | 5 tests ✅ |
| P0-3 Status unknown リトライ | 1 → 3 回リトライ、指数バックオフ | 5 tests ✅ |
| P0-4 データ品質 | `run_id`/`git_sha` を全 FillRecord 早期リターンパスに付与 | 3 tests ✅ |
| P1-3 Side 別適応 | `param_adapter` に buy/sell 分離ロジック追加 | 追加済 |

### 088# で保留した提案 (理由付き)

| 提案 | 却下理由 |
|------|---------|
| P0-3 cancel_reason 正規化 | 既に `cancel_reason` フィールドで正規化済み |
| P1-1 OB 特徴量復元 | SkipGate 自体が機能不全、OB データ取得コスト高 |
| P1-2 SkipGate 再学習パイプライン | まず現モデルの有効性確認が先 |
| P2 系全般 | 優先度低、効果不明 |

### 089# time_filter 削減

**旧**: buy 16/24h ブロック, sell 16/24h ブロック → 両方 open 4/24h (**17%**)  
**新**: buy 6/24h ブロック, sell 6/24h ブロック → 両方 open 13/24h (**54%**)  
**基準**: `mean ≤ -2.0 bps AND n ≥ 5` の統計的に確実な悪時間帯のみ残す

---

## 2. システム概要

### 取引環境
| 項目 | 値 |
|------|-----|
| 取引所 | Coincheck (日本) |
| 通貨ペア | BTC/JPY |
| BTC 価格帯 | ≈ ¥14,700,000 |
| ロットサイズ | 0.001 BTC (≈ ¥14,700) |
| 注文方式 | Maker limit order (指値注文) |
| 手数料 | Maker: 0% (無料) |
| 1 bps 換算 | ≈ ¥1.47 |
| 現在残高 | JPY 2,794 + BTC 0.002 |

### 戦略コンセプト
- スプレッド内にメイカー指値注文を配置し、AS を制御しつつ fill を獲得
- `spread × offset_ratio` でオフセット算出 → mid price から buy/sell 方向にずらす
- fill 後 30 秒の mid 価格変動 (`post_fill_30s_pnl`) を評価指標
- 目的: Tick 単位の微小エッジを高頻度で回収

### データセット (本分析対象)
| 項目 | 値 |
|------|-----|
| 計測期間 | 2026-02-13 09:39 ~ 02-16 ≈17:00 UTC |
| 総レコード | 512 |
| Fill 数 | 387 (75.6%) |
| Cancel 数 | 125 (24.4%) |
| PnL 付き Fill | 387 |
| Run 数 | 9 (複数の再起動あり) |

### コアファイル
- **`scripts/v460/run_fill_test.py`** (2,618 行): メイン実行ランナー
- **`configs/v460/fill_test.yaml`** (184 行): 設定ファイル
- **`ztb/metrics/fill_quality.py`**: FillRecord 定義・分析ライブラリ
- **`scripts/v460/ml/skip_gate.py`** (690 行): SkipGate ML フィルタ

### 1 サイクルの実行フロー

```
1. Ticker 取得 (best_bid, best_ask → spread 計算)
2. Time Filter 判定 (UTC hour × side で skip)
3. Regime Detection (trending/ranging 判定)
4. Offset 計算:
   a. base = spread_offset_ratio (0.05)
   b. sell 側: ratio を side_offset.sell (0.12) に置換 (乗算ではない)
   c. sell_guard: offset_floor 0.08 で最低保証 (b の結果と max)
   d. trending: × trending_offset_boost (1.5)
   e. spread_adaptive: narrow(< 10 bps) なら ×2.0, wide(> 25 bps) なら ×0.5
   f. 091# sell floor 事後再適用: e で floor 割れした場合に再保証
   g. 最終 offset = spread × effective_ratio
5. SkipGate ML 判定 (AS 確率推定 → skip 可否)
6. Maker limit order 発注 (最大 2 回リトライ)
7. Fill 監視 (最大 300 秒待機, 5 秒ポーリング)
8. Fill 後: mid_at_fill 記録, 30 秒後の mid 価格で PnL 計算
9. Early Exit: fill 後に ±5.0 bps 検知 → 次サイクルで反転注文 (当該注文キャンセルではない)
10. Fast Fill Defense: queue_wait ≤ 5s で次サイクル offset ×2
11. 結果を JSONL に保存
```

---

## 3. コンポーネント稼働状況

| コンポーネント | 設定 | 実態 | 判定 |
|---------------|------|------|------|
| **Spread Offset** | `0.05` | 全 fill で適用 | ✅ 稼働 |
| **Side Offset** | sell: `0.12` | sell 側保守化 | ✅ 稼働 |
| **Time Filter** | buy 6h, sell 6h ブロック | 089# で大幅削減 | ✅ 稼働 |
| **Regime Detection** | enabled | ranging 77%, trending 13%, unknown 10% | ⚠️ 判定改善 |
| **SkipGate ML** | AS mode, adaptive | **P(AS) 0.49-0.55, 0 skip** | ❌ 機能不全 |
| **Early Exit** | threshold 5.0 bps | 0 回トリガー | ❌ 未発火 |
| **Adaptation** | 50 cycle interval | offset 変動記録少 | ⚠️ 不明 |
| **Spread Adaptive** | narrow ×2, wide ×0.5 | 稼働中 | ✅ 稼働 |
| **Fast Fill Defense** | enabled, 5s threshold | 稼働扱い (効果不明) | ⚠️ 検証必要 |
| **Sell Guard** | max_spread 4000, offset_floor 0.08 | 088# 新規追加 | ✅ 新規 |
| **Imbalance** | disabled | 071# で無効化 | ⛔ 無効 |
| **Smart Side** | disabled | 071# で無効化 | ⛔ 無効 |

---

## 4. 全データ統計概要

### 4.1 PnL サマリー

| 指標 | 値 |
|------|-----|
| mean | **-0.638 bps** |
| median | -0.138 bps |
| stddev | 5.572 bps |
| positive rate | 46.5% (180/387) |
| t-stat | **-2.25** (有意に負) |
| min | -31.4 bps |
| max | +19.3 bps |

**統計的に有意な負のエッジ** — 現在の設定は平均で損失を生んでいる。

### 4.2 Side 別 PnL

| Side | n | mean | median | positive% |
|------|---|------|--------|---------|
| buy | 199 | **-0.340** | +0.138 | 50.3% |
| sell | 188 | **-0.954** | -0.344 | 42.6% |

Sell 側が 2.8 倍悪い。Side offset 0.12 (buy の 2.4 倍) を適用しても不十分。

### 4.3 Cancel 理由

| 理由 | n | 割合 |
|------|---|------|
| timeout | 54 | 10.5% |
| api_error | 34 | 6.6% |
| unknown | 26 | 5.1% |
| status_unknown | 10 | 2.0% |
| time_filter | (多数) | - |

- `api_error` 34 件 (6.6%) はリトライでも回復しなかったケース
- `status_unknown` 10 件は 088# で 3 回リトライに強化済み (直近 run では 0)

### 4.4 AS (Adverse Selection) 率

| 区分 | n | AS 件数 | AS 率 |
|------|---|---------|-------|
| 全体 | 387 | 151 | **39.0%** |
| buy | 199 | 79 | 39.7% |
| sell | 188 | 72 | 38.3% |

AS 率 39% は、「注文が約定した方向と逆に 2.5 bps 以上動く」確率が高いことを示す。

---

## 5. 多次元分析結果

### 5.1 スプレッド別 PnL

| Spread (JPY) | n | mean (bps) | AS% |
|--------------|---|-----------|-----|
| [0-2000) | 251 | **-1.20** | 高 |
| [2000-3000) | 52 | **-1.22** | 中 |
| [3000-4000) | 22 | +0.39 | - |
| [4000+) | 56 | +0.61 | - |

**狭スプレッド (< 3000 JPY) が損失源**。スプレッドが広がると利益化する傾向。

### 5.2 Queue Wait 別 PnL

| Wait Time | n | mean (bps) |
|-----------|---|-----------|
| < 10s (fast fill) | 140 | -0.79 |
| 10-30s | 81 | **-1.24** |
| 30-60s | 61 | -0.30 |
| **60-120s** | 51 | **+0.63** |
| 120s+ | 54 | -0.85 |

**60-120 秒待ちが唯一のプラスバケット**。即約定と長時間待ちは両方とも損失傾向。

### 5.3 Offset 別 PnL

| Offset Ratio | n | mean (bps) | AS% |
|--------------|---|-----------|-----|
| [0.03-0.04) | 32 | **-1.63** | 41% |
| [0.04-0.05) | 39 | -0.51 | 26% |
| [0.05-0.06) | 95 | -1.07 | 31% |
| [0.10-0.15) | 7 | -1.40 | 43% |

低 offset (< 0.04) が最悪。ただし 0.10-0.15 も悪い (n=7 で信頼性低)。

> **注**: 多くの旧データ (a9320c9 等) は `spread_offset_ratio` 未記録のため上記に含まれない (n=173 のみ)。

### 5.4 レジーム別 PnL

| Regime | n | mean (bps) |
|--------|---|-----------|
| ranging | 258 | -0.504 |
| trending | 81 | **-1.209** |
| unknown | 48 | **-1.538** |

- Trending が ranging の 2.4 倍悪い (offset ×1.5 では不十分)
- Unknown は旧バージョン (regime detector 導入前) のデータ

### 5.5 Version 別 PnL

| Git SHA | n | mean (bps) | positive% | 期間 |
|---------|---|-----------|---------|------|
| a9320c9 | 118 | **+0.366** | **54%** | 02/13 18:39-02/14 03:25 UTC |
| 51c02be | 36 | -1.075 | 42% | - |
| b4a9b96 | 10 | -3.091 | 30% | - |
| 68a13aa | 18 | -1.126 | 44% | - |
| 1de6bcf | 79 | -0.895 | 45% | - |
| 85a160d | 87 | -0.855 | 44% | - |
| 798adf2 | 31 | -0.764 | 42% | 088# 以降 |
| 他 | 8 | various | - | - |

**唯一プラスのバージョン `a9320c9` の正体**:
- Commit 内容は `docs: 022#` (ドキュメントのみ、コード変更なし)
- 初期の非常に単純な設定 (offset 固定, time_filter なし, SkipGate なし)
- **夜間セッション (UTC 18-03 = JST 03-12)** に集中
- 結論: **コード差異ではなく市場環境 (夜間低ボラ) がプラスの原因**

### 5.6 Consecutive Same-Side PnL

| consec | n | mean (bps) |
|--------|---|-----------|
| 1 | 387 | -0.60 |
| 2 | (subset) | **-1.56** |

連続同サイド 2 回目は 2.6 倍悪化。

### 5.7 Round-Trip 分析

| 指標 | 値 |
|------|-----|
| 往復数 | 186 |
| mean PnL (bps) | **-1.290** |
| positive% | 45.7% |

---

## 6. 致命的発見事項

### 🔴 CRITICAL-1: SkipGate が完全に機能不全

**状態**: モデルは存在し稼働しているが、**全サイクルで P(AS) = 0.49~0.55 を出力**し、閾値 0.65 を一度も超えない。

```
最新 run (1771258270_75e34201):
  Records with skip_gate_as_prob: 11
  P(AS) range: 0.492 - 0.545
  Skip count: 0
  All decisions: "pass"
```

**根本原因の推定**:
- モデルサイズ 4,247 bytes → 極小 (特徴量がほぼ効いていない)
- 全入力に対して P(AS) ≈ 0.50 → **ランダム分類器と等価** (識別能力ゼロ)
- 088# の `adaptive_threshold` は閾値を動的調整するが、モデル出力が一定なら調整しようがない
- 入力特徴量: 価格ベースのみ (OB 無効化中, `use_ob_features: false`)

**影響**: AS 率 39% のうち、skip すべき高 AS 確率の注文を一切フィルタリングできていない。仮に上位 20% を正しく skip できれば、PnL は大幅改善の可能性。

### 🔴 CRITICAL-2: Sell Fast Fill が壊滅的

| 区分 | n | mean PnL (bps) | AS% |
|------|---|-------------|-----|
| sell fast (< 10s) | 67 | **-1.66** | 39% |
| sell slow (≥ 10s) | 121 | -0.56 | 38% |
| buy fast (< 10s) | 73 | +0.00 | **49%** |
| buy slow (≥ 10s) | 126 | -0.54 | 34% |

- Sell fast fill は **全バケットで最悪** (-1.66 bps)
- Buy fast fill は AS 49% だが PnL ≈ 0 (offset が効いている)
- Sell fast fill のワースト 5: -31.4, -26.2, -19.2, -12.1, -12.0 bps

**構造的仮説**: 売り側で即約定 = 価格が急上昇中に約定 → AS 発生。Fast fill defense (`enabled: true, threshold_sec: 5`) は次サイクルの offset を 2 倍にするだけで、**当該サイクルの損失は防げない**。

### 🟡 WARNING-1: Early Exit

> **091# 修正**: Early Exit は「当該注文をキャンセル」ではなく「次サイクルの反転注文を高速化」する仕組み (`rapid_exit_pending` フラグ + `rapid_exit_interval_sec`)。fill 後の PnL が ±5.0 bps を超えると発火し、次サイクルの cycle_interval を 10 秒に短縮して反対売買を急ぐ。ログ上は発火実績あり (2026-02-17 07:06:48)。

設定 `threshold_bps: 5.0` で発火はしているが、fill 後の Maker 注文である以上、即時的な損失回避には限界がある。

### 🟡 WARNING-2: データ品質 (歴史的問題)

149/512 レコード (29%) が `run_id` / `git_sha` 未記録。これは 088# P0-4 以前のデータ。今後の新規データでは解消済み。

### 🟡 WARNING-3: PnL 改善トレンドなし

| 時系列 5 分位 | mean PnL (bps) |
|--------------|--------------|
| Q1 (最古) | -1.10 |
| Q2 | +0.24 |
| Q3 | -0.03 |
| Q4 | -1.51 |
| Q5 (最新) | -0.78 |

082# → 088# と改善を重ねてきたが、**PnL の時系列改善が見られない**。各種フィルタやガードを追加しても、根本的なエッジが存在しない可能性がある。

---

## 7. シミュレーション結果

既存データに対するフィルタしたの仮想 PnL。

### 7.1 単一フィルタ

| フィルタ | n (残存) | mean PnL (bps) | 改善 |
|---------|---------|---------------|------|
| ベースライン | 387 | -0.638 | - |
| sell fast fill 除外 | 320 | -0.424 | +0.214 |
| spread ≥ 2500 のみ | 81 | -1.026 | -0.388 |
| offset ≥ 0.045 のみ | 109 | -1.076 | -0.438 |

- **Sell fast fill 除外が唯一のプラス改善** (+0.214 bps)
- Spread や offset のフィルタリングは逆効果 (サンプル偏り)

### 7.2 複合フィルタ

| フィルタ | n | mean PnL | positive% |
|---------|---|---------|---------|
| sell fast 除外 + offset ≥ 0.045 | 91 | -0.623 | 42.9% |

複合でも -0.623 で依然マイナス。

### 7.3 Sell Fast Fill x Spread

| Spread (JPY) | n | mean PnL (bps) |
|--------------|---|---------------|
| [0-2000) | 55 | **-1.67** |
| [2000-3000) | 7 | -0.72 |
| [3000-4000) | 4 | -3.36 |
| [4000+) | 1 | -1.25 |

Sell fast fill は全スプレッド帯で損失。

---

## 8. 現在の設定 (fill_test.yaml)

```yaml
# 基本設定
symbol: btc_jpy
order_quantity: 0.001
cycle_interval_sec: 120.0
order_timeout_sec: 300.0
post_fill_wait_sec: 30.0
start_side: buy

# スプレッド比例オフセット
spread_offset_ratio: 0.05
min_offset_jpy: 1.0

# リトライ
max_order_retries: 2

# AS 判定
as_deadzone_bps: 2.5

# 適応
adaptation:
  enabled: true
  interval_cycles: 50
  min_fill_rate: 0.80
  max_as_ratio: 0.15
  step_ratio: 0.01
  min_offset_ratio: 0.01
  max_offset_ratio: 0.30

# レジーム検知
regime:
  enabled: true
  trend_threshold_pct: 0.5
  trending_offset_boost: 1.5
  min_confidence: 0.3

# 時間帯フィルター (089# 削減済み)
time_filter:
  enabled: true
  skip_utc_hours: [16]                          # グローバル
  skip_utc_hours_buy: [1, 2, 12, 16, 18, 21]   # buy 6h blocked
  skip_utc_hours_sell: [4, 8, 13, 14, 16, 17]   # sell 6h blocked

# side 別 offset
side_offset:
  sell: 0.12

# 即約定防御
fast_fill_defense:
  enabled: true
  threshold_sec: 5.0
  offset_boost: 2.0

# 安全設計
safety:
  loss_cap_auto: true
  loss_cap_ratio: 0.05
  soft_loss_cap_ratio: 0.02

# SkipGate ML フィルター
skip_gate:
  enabled: true
  mode: as
  model_path: models/v460/skip_gate_as.pkl
  as_threshold: 0.65
  as_threshold_sell: 0.60
  adaptive_threshold: true
  target_skip_rate_buy: 0.10
  target_skip_rate_sell: 0.20
  adaptive_window: 50
  adaptive_min_samples: 20

# sell ハードガード (088#)
sell_guard:
  max_spread_jpy: 4000.0
  offset_floor: 0.08

# Early Exit
early_exit:
  enabled: true
  threshold_bps: 5.0

# Spread Adaptive
spread_adaptive:
  enabled: true
  narrow_spread_bps: 10.0
  narrow_spread_boost: 2.0
  wide_spread_bps: 25.0
  wide_spread_ratio: 0.5
```

---

## 9. 改善候補の整理

### 優先度 S (構造根幹)

#### S-1: SkipGate モデル再学習 (または代替ロジック)

**問題**: 現モデルは P(AS) ≈ 0.50 で識別能力ゼロ。  
**選択肢**:
1. **387 件の実データで再学習**: `adverse_selected` をラベルに、spread/offset/regime/wait_time/hour 等を特徴量に。ただしサンプル少 (正例 151, 負例 236)
2. **ルールベースに回帰**: ML を捨てて、統計的に AS 率が高い条件 (sell + fast fill + low spread) を直接ガードする
3. **ハイブリッド**: ルールベースの hard skip + ML の soft skip を併用

**推奨**: 選択肢 3 — ルールベースで sell fast fill を即ガード + ML をバックグラウンドで再学習

#### S-2: Sell Fast Fill 即時対策

**問題**: sell fast fill (-1.66 bps, n=67) が全体 PnL を大きく引き下げ。  
**対策案**:
1. **sell 側の fast fill 検知時にキャンセル** (約定前にキャンセルが間に合えば)
2. **sell 側 offset を更に引き上げ** (0.12 → 0.15-0.20) — fill 率低下とトレードオフ
3. **sell 側 min_spread_jpy 導入** — 狭スプレッド時は sell を skip
4. **cycle_interval を短縮しつ、sell 側の即約定を mid 更新で回避**

### 優先度 A (高効果)

#### A-1: Trending レジーム対策強化

**問題**: trending PnL = -1.209 (ranging -0.504 の 2.4 倍悪い)。  
**対策案**:
1. `trending_offset_boost` を 1.5 → 2.0-3.0 に引き上げ
2. Trending 時は sell 側のみ skip する
3. Trending 方向を判定し、順方向の注文のみ許可

#### A-2: Smart Side Selection 再有効化

**問題**: 連続同サイド 2 回目が -1.56 bps (1 回目 -0.60)。  
**対策案**: `smart_side.enabled: true`, `max_consecutive_same: 1`  
**懸念**: 071# で無効化した理由 (板情報依存) を回避する実装が必要

#### A-3: Queue Wait ベースの動的対応

**問題**: 60-120s 待ちのみ +0.63 bps (他は全て負)。  
**仮説**: 短すぎる wait → AS に遭う。長すぎる wait → スプレッド変動リスク。  
**対策案**: 60s 以内の fill 後にポジションを即ヘッジするか、offset を動的に厚くする

### 優先度 B (中効果)

#### B-1: 狭スプレッド sell ガード強化

**データ**: sell fast fill で spread < 2000 → -1.67 bps (n=55)  
**対策**: `sell_guard.min_spread_jpy: 2000-2500` (sell のみ最低スプレッドフィルタ)

#### B-2: PnL の長期トレンド変化検出

**問題**: Q1-Q5 で改善トレンドがないが、各 Q のサンプル数は 77 件程度。  
**対策**: 直近 50 件 moving average が -1.0 bps 以下になったら自動停止 (circuit breaker)

#### B-3: 評価期間の延長

**問題**: 30 秒後の PnL は BTC の短期ノイズが大きい (σ = 5.57 bps)。  
**対策**: E3 で収集中の 120s/300s PnL を評価し、より長い horizon のエッジを確認

---

## 10. Codex への質問事項

### Q1: 根本的な問い — エッジは存在するか？

全期間 mean = -0.638 bps (t-stat = -2.25) で有意に負。512 サイクル × 3 日間のデータで一貫して損失。

- **この戦略にエッジがあるとすれば、どの条件下か？**
- Version a9320c9 (夜間) のプラスは市場環境か、それとも当時の単純な設定が皮肉にも良かったのか？
- Maker 指値注文で AS を回避しつつ fill を取る、という戦略自体が Coincheck の流動性水準で成立するか？

### Q2: SkipGate モデルの設計

現在のモデルは識別能力ゼロ (P(AS) ≈ 0.50)。

- **387 件のデータで意味のある AS 分類器は学習可能か？** (正例 151, 負例 236)
- どの特徴量が最も重要か？ (spread, offset, regime, hour, queue_wait, consecutive_same_side)
- ルールベース vs ML vs ハイブリッド、n < 500 の状況ではどれが適切か？

### Q3: Sell 側の構造的劣勢

Sell mean = -0.954 (buy -0.340 の 2.8 倍)。Side offset 0.12 (通常の 2.4 倍) でも改善不十分。

- **Sell 側を完全に無効化して buy only で運用すべきか？**
- 代替: sell 側のみ offset 0.20+ にする場合、fill 率がどこまで下がるか？
- 片側運用のリスク (ポジション片寄り) はどう管理すべきか？

### Q4: Fast Fill Defense の設計改善

Current: `threshold_sec: 5.0, offset_boost: 2.0`。次サイクルの offset を 2 倍にするだけ。

- **当該サイクルの損失を防ぐには？** (約定前キャンセルは間に合うか？)
- 5 秒閾値は適切か？データでは < 10s 全体が問題。
- Sell 側のみ fast fill defense を厳しくする (例: threshold 15s) は有効か？

### Q5: Cycle Interval と市場環境

120 秒サイクルで 5.2 fill/時。Queue wait 60-120s のみプラス。

- **サイクル間隔を 180-240 秒に延長すべきか？** (fill 品質 vs fill 頻度のトレードオフ)
- 夜間 (UTC 18-03) に集中した a9320c9 がプラスだったのは、ボラティリティが低いからか？
- 夜間限定運用は検討に値するか？

### Q6: 追加すべきデータ収集

現在の FillRecord に不足しているフィールドは何か？

- 約定時の板状態 (depth, imbalance) — 現在 OB 無効化中
- 直前 N 秒の価格変動 (momentum indicator)
- 他取引所 (bitFlyer, Binance) の BTC/JPY or BTC/USDT 価格差 (cross-exchange signal)

### Q7: 損益分岐点の推定

- 1 bps ≈ ¥1.47 のロットで、**何 fill/日で損益分岐できるか？**
- ロットを 0.005 BTC に増やす場合、スリッページリスクはどう変わるか？
- 現在の残高 (JPY 2,794 + BTC 0.002) で継続する意味はあるか？

---

## 付録: 分析に使用したデータファイル

```
results/v460/fill_test/fill_records_*.jsonl
```

全 9 ファイル、512 レコード (JSONL 形式)。

## 付録: 最近 5 run のパフォーマンス

| Run ID | n (filled) | mean PnL (bps) | AS% |
|--------|-----------|---------------|-----|
| 1771071431_2dfec424 | 24 | -1.322 | 29% |
| 1771095285_383ebf85 | 82 | -0.927 | 30% |
| 1771227270_eca8d769 | 2 | -2.200 | 50% |
| 1771228012_ec0d219b | 4 | -0.632 | 25% |
| 1771258270_75e34201 | 8 | -1.099 | 38% |

---

## 091# 修正記録

> 以下は 091# レビューにより修正された事項。本ドキュメントの分析データ (512件) は
> 088#/089# 適用前のデータが大半であり、「事前仮説」として位置づける。
> 088#/089# 適用後の clean n ≥ 200 を満たした時点で再評価が必要。

| # | 修正内容 |
|---|---------|
| 1 | §2 offset 計算フロー: sell 側は「乗算」ではなく「ratio を 0.12 に置換」に修正 |
| 2 | §2 offset 計算フロー: 091# sell floor 事後再適用ステップ追加 |
| 3 | §6 WARNING-1: Early Exit は「即キャンセル」ではなく「次サイクル高速反転」に修正。発火実績あり |
| 4 | 本分析は 088#/089# 効果未反映データが大半であることを明記 |

---

*以上。レビュアーは自由に追加質問してください。*
