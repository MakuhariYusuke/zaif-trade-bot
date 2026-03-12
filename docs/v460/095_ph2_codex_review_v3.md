# 095# fill_test Codex レビュー v3 — 構造損失の根本原因と状態管理バグ

> **作成日**: 2025-02-18  
> **対象**: v460 "Microstructure Edge" BTC/JPY 自動取引 fill_test  
> **目的**: 外部 AI コーディングエージェント (Codex) にシステム全体をレビューしてもらい、見落とし・設計欠陥・改善余地を特定する  
> **Git HEAD**: `c3dbaed55` (094# stale order cancel-replace)  
> **前回 Codex レビュー**: 090# (`93ea3fd4c`) → 091# で回答受領・実装済  
> **データ期間**: 2025-02-14 〜 02-18 (稼働中)

---

## 目次

1. [前回レビュー (091#) からの変更点](#1-前回レビュー-091-からの変更点)
2. [システム概要](#2-システム概要)
3. [パフォーマンス実績](#3-パフォーマンス実績)
4. [致命的発見事項](#4-致命的発見事項)
5. [新発見の設計欠陥](#5-新発見の設計欠陥)
6. [データ異常・ノイズ源](#6-データ異常ノイズ源)
7. [改善候補の優先順位整理](#7-改善候補の優先順位整理)
8. [現在の設定 (fill_test.yaml)](#8-現在の設定)
9. [Codex への質問事項](#9-codex-への質問事項)

---

## 1. 前回レビュー (091#) からの変更点

### 091# → 094# の流れ

| # | 内容 | Commit | 成果 |
|---|------|--------|------|
| 091# | 090# Codex レビュー回答の実装 (P0-1〜P0-4, P1-3) | `a61256061` | SkipGate 動的較正, sell_guard, status_unknown リトライ |
| 092# | 実装ギャップ分析 — E1〜E7 KPI の測定方法を確認 | `a61256061` | テスト修正, E3 sampling 検証 |
| 093# | spread_adaptive / fast_fill_defense の side 別パラメータ化 | `4e5b7c9f2` | buy/sell 独立閾値・boost 倍率, 17 tests pass |
| 094# | stale order cancel-replace — 価格乖離注文の自動検出・再発注 | `c3dbaed55` | 30s 後 5bps 乖離で reprice, max 2 回/cycle |

### 091# で実装した Codex 提案の状況

| 提案 | 実装状態 | 効果評価 |
|------|---------|---------|
| P0-1 SkipGate 動的較正 | ✅ 実装・稼働中 | ❌ **効果なし** — P(AS) が AS/非 AS で同一分布 |
| P0-2 Sell ハードガード | ✅ 実装・稼働中 | ⚠️ 効果不明 — sell_guard 発動回数未計測 |
| P0-3 Status unknown リトライ | ✅ 実装・稼働中 | ⚠️ status_unknown: 10 件 → 改善度不明 |
| P0-4 データ品質 | ✅ 実装・稼働中 | ✅ quarantine 149 件が分離成功 |
| P1-3 Side 別適応 | ✅ 実装・稼働中 | ❌ **適応が振動** — 下記 §5 参照 |

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
| 現在残高 | JPY 2,827 + BTC 0.002 |

### アーキテクチャ

```
run_fill_test.py (2,819行)
├── FillTestConfig (80+ params, YAML 外部化)
├── _compute_maker_price()      ← offset 計算チェーン
│   ├── base: spread × offset_ratio
│   ├── side_offset: sell=0.12 (buy=共通 0.05)
│   ├── sell_guard: offset_floor=0.08, max_spread=4000JPY
│   ├── regime: trending → 1.5× boost
│   ├── spread_adaptive: narrow(<10bps) → buy 1.5×/sell 2.0×, wide(>25bps) → 0.5×
│   └── imbalance: disabled
├── run_single_cycle()          ← 1 回の注文サイクル
│   ├── SkipGate ML 判定
│   ├── 注文 → ポーリング (5s 間隔, 300s timeout)
│   ├── stale order 検出 (094#: 30s 後 5bps 乖離で reprice)
│   └── 約定後 PnL 計測 (30s + E3 sampling 50% で 60s/120s)
├── run_continuous()            ← メインループ
│   ├── time_filter (side 別 UTC 時間帯ブロック)
│   ├── balance check → side 自動切替
│   ├── fast_fill_defense (即約定防御)
│   ├── param_adapter (方策 A: offset 自動適応, 50 cycle 毎)
│   └── batch save → JSONL
└── 補助モジュール
    ├── param_adapter.py (方策 A: offset 自動調整)
    ├── skip_gate.py (ML AS 分類器: LogReg + SelectKBest)
    └── fill_quality.py (FillRecord, g1_1_judgment)
```

### offset 計算の流れ (実コード忠実)

```python
# 1. base offset
raw = spread_jpy * spread_offset_ratio   # e.g. 2000 × 0.05 = 100 JPY

# 2. side-specific override (sell のみ)
if sell: raw = spread_jpy * spread_offset_ratio_sell  # 2000 × 0.12 = 240 JPY

# 3. sell_guard floor
if sell: raw = max(raw, spread_jpy * sell_offset_floor)  # min 0.08

# 4. sell_guard max_spread cap
if sell and spread > 4000JPY: SKIP

# 5. regime trending boost
if regime == trending: raw *= 1.5

# 6. spread_adaptive
if spread_bps < 10: raw *= 1.5 (buy) / 2.0 (sell)
if spread_bps > 25: raw *= 0.5

# 7. post-adaptive sell floor re-apply
if sell: raw = max(raw, spread_jpy * sell_offset_floor)

# 8. imbalance boost (DISABLED)

# 9. min_offset_jpy clamp
raw = max(raw, 1.0)
```

---

## 3. パフォーマンス実績

### 3.1 全体サマリ

| 指標 | 値 | 備考 |
|------|-----|------|
| 総レコード | 533 | quarantine 149 含む |
| クリーン | 384 | git_sha 付き |
| 約定 (fill) | 317 | fill_rate = 82.6% |
| タイムアウト | 67 | 17.4%, avg wait 249.0s |
| SkipGate スキップ | **0** | 完全に機能していない |
| **30s PnL 合計** | **-135.7 JPY** | |
| 30s PnL (buy) | -25.8 JPY | |
| 30s PnL (sell) | -109.9 JPY | sell は buy の 4.3 倍悪い |
| AS rate (buy) | 33.3% (53/159) | |
| AS rate (sell) | 34.2% (54/158) | |

### 3.2 SHA 別パフォーマンス

| SHA | 内容 | fills | 30s PnL | AS rate | 備考 |
|-----|------|-------|---------|---------|------|
| a9320c9 | 089# time_filter 削減 | 118 | **+43.1** | 38% | 唯一の黒字 SHA |
| 51c02be | 091# Codex 実装 | 82 | -76.0 | 30% | |
| ca1bcae | 088# SkipGate 動的較正 | 60 | -65.8 | 33% | |
| 68a13aa | 090# Codex 資料作成 | 24 | -31.7 | 29% | |
| **c3dbaed** | **094# stale order** | **11** | **+15.2** | **18%** | 最新, 低 AS |
| 1de6bcf | ? | 8 | -8.8 | 38% | |
| 4e5b7c9 | 093# side params | 8 | -4.8 | 38% | |
| b4a9b96 | ? | 6 | -6.9 | 33% | |

**注目**: c3dbaed (094#) は最新 11 fills で PnL +15.2、AS 18% と好成績だが、n が少なくまだ統計的に不十分。

### 3.3 時間帯別 PnL (JST)

| JST 時間 | fills | 30s PnL 合計 | 平均 PnL | 評価 |
|----------|-------|-------------|----------|------|
| JST05 | 36 | **+35.7** | +0.99 | ★ 最良 |
| JST07 | 19 | +12.1 | +0.64 | ★ 好調 |
| JST12 | 17 | +15.6 | +0.92 | ★ 好調 |
| JST09 | 25 | +8.4 | +0.34 | 〇 |
| JST14 | 29 | +3.6 | +0.13 | 〇 |
| JST06 | 26 | -42.8 | **-1.65** | ✗ 悪い |
| JST17 | 15 | **-57.1** | **-3.81** | ✗✗ 最悪 |
| JST11 | 16 | -20.4 | -1.27 | ✗ |
| JST10 | 27 | -22.0 | -0.81 | ✗ |
| JST04 | 12 | -15.4 | -1.28 | ✗ |

**JST17 (UTC08)** は平均 -3.81 で突出して悪い。sell 側は UTC08 で既にブロック済みだが、buy 側は未ブロック。

### 3.4 約定待ち時間帯別 PnL

| Wait Band | fills | 30s PnL | AS rate | 損失シェア |
|-----------|-------|---------|---------|-----------|
| 5-10s | 94 | -17.8 | 36% | 13% |
| **10-30s** | **95** | **-87.9** | **39%** | **65%** |
| 30-60s | 55 | -10.9 | 36% | 8% |
| 60-120s | 32 | +4.0 | **16%** | 0% (黒字) |
| 120-300s | 39 | -22.1 | 28% | 16% |

**10-30s 帯が全損失の 65% を占める。** AS rate 39% は最悪。仮説: この帯域は informed trader のテイクが集中するゾーン。

### 3.5 E3 (120s PnL) — 平均回帰の証拠

| 指標 | 値 |
|------|-----|
| 120s データ保有 | 41/317 fills (12.9%) |
| 120s PnL 合計 | +22.0 JPY |
| 120s PnL (buy) | -3.0 (n=17) |
| 120s PnL (sell) | **+25.0** (n=24) |
| 30s↔120s 符号一致率 | 65.9% (27/41) |
| 30s 負 → 120s 正 (反転) | **8/41** (19.5%) |

**sell の 30s PnL は -109.9 で壊滅的だが、120s では +25.0 に反転。**
つまり sell の逆選択の多くは **一時的なもので、120s 後には有利側に回復**している。
→ 「sell 側は損切りが早すぎる」か「ポジション保持時間を伸ばすべき」という示唆。

---

## 4. 致命的発見事項

### 4.1 SkipGate は完全なランダム分類器 [★★★★★]

SkipGate (AS 分類器) は AS イベントと非 AS イベントの区別が**全くできていない**。

```
SkipGate P(AS) データ (n=27):
  AS events:   n=8,  median=0.5113, range=[0.4947, 0.5449]
  non-AS:      n=19, median=0.5100, range=[0.4809, 0.5449]
```

- P(AS) の中央値が AS / 非 AS でほぼ同一 (0.5113 vs 0.5100)
- 全出力が 0.48〜0.55 の狭いバンドに収束 → **事実上コイントス**
- adaptive_threshold も意味がない (元のスコアが弁別力ゼロ)
- **結果**: 384 サイクルで **0 件のスキップ** (threshold が 0.60-0.65 だが P(AS) は常に 0.50 前後)

**根本原因の仮説**:
- SelectKBest k=8 で選択される特徴量が AS を捕捉していない
- LogisticRegression C=0.01 (強い正則化) がモデルをフラットにしている
- 訓練データの AS ラベル自体が弱い (AS が構造的・文脈的でなく、タイミングノイズに近い)

**影響**: 悪条件でもフィルタリングが一切されず、全損失がスルーされている。

### 4.2 Param Adapter の振動問題 [★★★★★ NEW]

param_adapter (方策 A) が offset を頻繁に変更しており、しかも **短時間で激しく振動**している。

```
02-17 のオフセット変動 (33 回の変更):
  05:47  0.1000 → 0.2400 (UP)     ← fast_fill boost?
  05:52  0.2400 → 0.1000 (DOWN)   ← fast_fill deactivate
  14:01  0.1000 → 0.2400 (UP)
  14:02  0.2400 → 0.0750 (DOWN)   ← pre-boost 値が 0.075 に変わっている!
  14:07  0.0750 → 0.2400 (UP)
  14:10  0.2400 → 0.0750 (DOWN)
  14:18  0.0750 → 0.2400 (UP)
  14:22  0.2400 → 0.1875 (DOWN)   ← 0.075 × 2.5 = 0.1875
  ... (3-5 分間隔で UP/DOWN を繰り返す)
```

**offset 値ごとの PnL**:

| offset | fills | 30s PnL 合計 | 平均 PnL | AS rate |
|--------|-------|-------------|----------|---------|
| 0.03 | 32 | -52.2 | **-1.63** | **62%** |
| 0.04 | 39 | -19.9 | -0.51 | 54% |
| 0.05 | 95 | -101.5 | -1.07 | 56% |
| 0.10 | 7 | -9.8 | -1.40 | 71% |
| 0.19 | 5 | -6.1 | -1.22 | 60% |
| **0.24** | **15** | **+8.4** | **+0.56** | **40%** |

**offset=0.24 のみが黒字 (AS 40%)。** 小さい offset ほど AS rate が高く損失が大きい。
→ 現在の offset (0.05) は小さすぎる可能性が高い。

---

## 5. 新発見の設計欠陥

### 5.1 Fast Fill Defense と Param Adapter の状態競合 [CRITICAL]

**発見**: `fast_fill_defense` と `param_adapter` はどちらも `self.config.spread_offset_ratio` を **直接変更**しており、相互の変更を上書きする競合状態がある。

**メカニズム**:

```python
# fast_fill_defense 発動時:
self._pre_boost_offset = self.config.spread_offset_ratio  # 保存
self.config.spread_offset_ratio *= boost                   # 直接変更

# --- この間に param_adapter が走ると ---
# param_adapter: self.config.spread_offset_ratio += step_ratio
# → boosted 値をさらに変更

# fast_fill_defense 解除時:
self.config.spread_offset_ratio = self._pre_boost_offset   # 旧値に復元
# → param_adapter の変更が消失！
```

**証拠**: 02-17 のログで `0.2400 → 0.0750` への復帰が確認されるが、これは param_adapter が 0.10 → 0.075 に下げた後の pre_boost_offset 値。つまり:
1. まず adapter が 0.10 → 0.075 に変更
2. 次に fast_fill が pre_boost=0.075 を保存し 0.24 にブースト
3. fast_fill 解除で 0.24 → 0.075 に復元
4. しかし adapter が次の 50 cycle で別の値を提案して 0.24 にブースト
5. ループが永続する

**修正案**: offset の「基準値」と「一時的ブースト」を分離する (e.g. `base_offset` + `boost_multiplier`)。

### 5.2 Param Adapter が全レコードを使って判断 [HIGH]

`_try_auto_adapt()` は毎回 `load_fill_records_glob()` で **全 clean レコード** (384 件) を読み込み、そこから fill_rate / AS_ratio を算出する。

**問題点**:
- 古い設定 (offset=0.03 など) で収集したデータが支配的
- SHA 間で環境やパラメータが異なるのに混合して判断
- 適応結果が「現在の設定での成績」ではなく「全期間の平均」に基づく
- → 最新設定の効果がデータに薄まり、正しい適応判断ができない

**例**: offset=0.24 (最新) は PnL +8.4 だが、全体 AS rate=34% (>15% 閾値) のため adapter は offset を下げようとする → offset=0.24 の良い成績が失われる。

**修正案**: recency window を導入 (e.g. 直近 N 件のみ、または直近 M 時間のみ)。

### 5.3 Fast Fill Defense の buy 側閾値が実効性ゼロ [MEDIUM]

| Side | 閾値 | ≤閾値の fills | 判定 |
|------|------|-------------|------|
| buy | 5.0s | **0** | 永久に発動しない |
| sell | 15.0s | 68 (neg_edge=36) | 発動の可能性あり |

buy 側の `fast_fill_threshold_sec=5.0` は **0 件の約定** しかカバーしない。
→ buy 側は fast_fill_defense が事実上 **無効**。

### 5.4 Regime Detector の有効性 [MEDIUM]

| Regime | 件数 | 割合 |
|--------|------|------|
| None | 206 | 53.6% |
| unknown | 83 | 21.6% |
| ranging | 73 | 19.0% |
| trending | 21 | 5.5% |

**75.2% が None/unknown** で、有用なラベルが付くのは 4 件に 1 件。
trending (n=21) に 1.5× offset boost が適用されるが、サンプル数が少なすぎて効果検証ができない。

---

## 6. データ異常・ノイズ源

### 6.1 E3 Sampling の実効率

| SHA | fills | 60s PnL 保有 | 実効率 |
|-----|-------|-------------|--------|
| a9320c9 | 118 | 0 | 0% |
| ca1bcae | 60 | 0 | 0% |
| 68a13aa | 24 | 0 | 0% |
| 51c02be | 82 | 26 | **32%** |
| c3dbaed (最新) | 11 | 5 | **45%** |
| 4e5b7c9 | 8 | 2 | 25% |
| 1de6bcf | 8 | 5 | 62% |
| b4a9b96 | 6 | 2 | 33% |

初期 SHA (a9320c9, ca1bcae, 68a13aa) は E3 機能が存在しなかったため 60s/120s データがゼロ。
現状の有効データは **41/317 = 12.9%** にとどまる。配信が 50% なのに 32-45% 止まりなのは、E3 サンプリングが確率的にスキップされるケースがあるため。

### 6.2 status_unknown の特徴

10 件の status_unknown (注文状態不明) の特徴:
- 全件 `queue_wait_sec ≈ 5.5-6.2s` (最初のポーリングで発生)
- 7/10 件が sell 側
- 特定 SHA に集中しない (3 SHA にまたがる)
- → Coincheck API の応答遅延やレースコンディションの可能性

### 6.3 effective_offset_used の track gap

`effective_offset_used` フィールドは 094# 前後で追加されたため、**33/317 fills (10.4%)** にしか存在しない。残りの 284 fills は実際に使用された offset が `spread_offset_ratio` カラムからの推定しかできない。

**問題**: `spread_offset_ratio` は fast_fill_defense / param_adapter / spread_adaptive の全てが事後的に変更するため、**実際の注文で使われた offset と一致しない可能性がある**。

---

## 7. 改善候補の優先順位整理

### 優先度マトリクス

| # | 施策 | 影響度 | 実装難度 | 優先度 |
|---|------|--------|---------|--------|
| **M1** | **SkipGate 抜本的見直し** — 特徴量再設計 or 別アルゴリズム or 撤去 | ★★★★★ | ★★★★☆ | **P0** |
| **M2** | **Fast Fill / Adapter 状態分離** — base_offset + boost_multiplier | ★★★★★ | ★★☆☆☆ | **P0** |
| **M3** | **Adapter に recency window 導入** — 直近 N 件のみ使用 | ★★★★☆ | ★★☆☆☆ | **P0** |
| M4 | Order timeout 短縮 (300s → 120-150s) | ★★★☆☆ | ★☆☆☆☆ | P1 |
| M5 | 10-30s wait band の対策 (早期キャンセル or offset 引き上げ) | ★★★★☆ | ★★★☆☆ | P1 |
| M6 | JST17 buy 側ブロック追加 | ★★☆☆☆ | ★☆☆☆☆ | P1 |
| M7 | Sell 120s 平均回帰の活用 (保持時間延長?) | ★★★☆☆ | ★★★★☆ | P2 |
| M8 | Regime detector の改善 or 撤去 | ★★☆☆☆ | ★★★☆☆ | P2 |
| M9 | effective_offset_used の全レコード化 | ★★☆☆☆ | ★☆☆☆☆ | P2 |
| M10 | Offset 全体引き上げ (0.05 → 0.10-0.15?) | ★★★★☆ | ★☆☆☆☆ | **要検討** |

### M10 について (offset 引き上げの検討)

データが示唆するのは **「offset が大きいほど成績が良い」** ということ:

| offset | avg PnL | AS rate |
|--------|---------|---------|
| 0.03 | -1.63 | 62% |
| 0.05 | -1.07 | 56% |
| 0.10 | -1.40 | 71% ← サンプル少 |
| 0.24 | **+0.56** | **40%** |

ただし offset=0.24 は fast_fill_defense のブースト値であり、**意図的な設定値ではない**。
また offset を上げると fill_rate が低下するトレードオフがある。
→ **最適 offset の体系的な探索** が必要。

---

## 8. 現在の設定

<details>
<summary>fill_test.yaml (クリックで展開)</summary>

```yaml
symbol: btc_jpy
order_quantity: 0.001
cycle_interval_sec: 120.0
order_timeout_sec: 300.0
poll_interval_sec: 5.0
post_fill_wait_sec: 30.0
spread_offset_ratio: 0.05
min_offset_jpy: 1.0
as_deadzone_bps: 2.5

adaptation:
  enabled: true
  interval_cycles: 50
  min_fill_rate: 0.80
  max_as_ratio: 0.15
  step_ratio: 0.01
  min_offset_ratio: 0.01
  max_offset_ratio: 0.30
  min_samples: 50

lot_sizing:
  enabled: false

regime:
  enabled: true
  window: 20
  trend_threshold_pct: 0.5
  high_vol_multiplier: 2.0
  hysteresis_count: 3
  min_confidence: 0.3
  trending_offset_boost: 1.5

time_filter:
  enabled: true
  skip_utc_hours: [16]
  skip_utc_hours_buy: [1, 2, 12, 16, 18, 21]
  skip_utc_hours_sell: [4, 8, 13, 14, 16, 17]

e3:
  sampling_ratio: 0.50

side_offset:
  sell: 0.12

fast_fill_defense:
  enabled: true
  threshold_sec: 5.0
  threshold_sec_buy: null   # buy側: 5.0s → 0件発動
  threshold_sec_sell: 15.0  # sell側: 15.0s
  offset_boost: 2.0
  offset_boost_buy: null
  offset_boost_sell: 2.5

stale_order:
  enabled: true
  check_after_sec: 30.0
  drift_bps: 5.0
  max_reprice: 2
  cooldown_sec: 10.0

safety:
  loss_cap_auto: true
  loss_cap_ratio: 0.05
  soft_loss_cap_ratio: 0.02
  loss_cap_jpy: 10000.0

spread_adaptive:
  enabled: true
  narrow_spread_bps: 10.0
  narrow_spread_boost: 2.0
  narrow_spread_boost_buy: 1.5
  narrow_spread_boost_sell: 2.0
  wide_spread_bps: 25.0
  wide_spread_ratio: 0.5

skip_gate:
  enabled: true
  mode: as
  model_path: models/v460/skip_gate_as.pkl
  as_threshold: 0.65
  as_threshold_buy: null
  as_threshold_sell: 0.60
  adaptive_threshold: true
  target_skip_rate_buy: 0.10
  target_skip_rate_sell: 0.20
  adaptive_window: 50
  adaptive_min_samples: 20

sell_guard:
  max_spread_jpy: 4000.0
  offset_floor: 0.08
```

</details>

---

## 9. Codex への質問事項

### Q1: SkipGate の根本的改善方針

現在の SkipGate は **LogisticRegression (C=0.01) + SelectKBest (k=8)** だが、完全にランダム分類器になっている。

(a) この設定 (強い正則化 + 少数特徴量) が原因か、それとも特徴量自体が AS を捕捉していないのか？  
(b) AS ラベルの品質 (30s 後の price movement + deadzone 2.5bps) は妥当か？AS の定義を変えるべきか？  
(c) SkipGate を撤去して、代わりにルールベース (e.g. spread 幅 + 時間帯 + wait_time) で AS フィルタリングする方が有効か？  
(d) もし ML を維持するなら、どのようなモデル / 特徴量 / 訓練パイプラインを推奨するか？

### Q2: Offset 最適化戦略

データは offset=0.24 で黒字 (avg +0.56, AS 40%) を示すが、これは意図的設定ではなく fast_fill boost の副産物。

(a) offset を 0.05 → 0.10〜0.15 に引き上げるべきか？ fill_rate 低下のトレードオフをどう評価するか？  
(b) buy/sell で最適 offset が異なるはず。体系的な探索方法 (grid search? A/B test?) の推奨は？  
(c) 現在の param_adapter (fill_rate vs AS_ratio ベース) は正しいアプローチか？別の目的関数 (e.g. PnL 直接最適化) が良いか？

### Q3: Fast Fill Defense / Param Adapter 状態管理

§5.1 で述べた状態競合の解消方法として:

(a) `base_offset` + `temp_boost_multiplier` の分離設計は妥当か？  
(b) param_adapter と fast_fill_defense の実行順序・排他制御はどう設計すべきか？  
(c) 両方を同時に動かすこと自体が危険か？どちらかを停止すべきか？

### Q4: 10-30s Wait Band の異常 AS rate

10-30s 帯が全損失の 65% を占め AS rate 39%。

(a) この帯域の高 AS rate は「informed trader のテイク」と解釈して正しいか？  
(b) 対策として「10-30s で約定したら次サイクルの offset を引き上げる」は有効か？  
(c) そもそも Coincheck の maker order は 10-30s で約定するのが構造的な傾向なのか？

### Q5: Sell 平均回帰パターンの活用

Sell 30s PnL = -109.9 だが 120s PnL = +25.0。30s→120s で符号反転するケースが 19.5%。

(a) これは「sell の一時的 AS は構造的なオーバーシュート (行き過ぎ戻り) である」と解釈して良いか？  
(b) 具体策: 30s PnL が negative でも即諦めず、保持時間を 60-120s に延長して反転を待つべきか？  
(c) この戦略のリスク (テイルロスの拡大) はどう管理するか？

### Q6: 全体アーキテクチャへの疑問

(a) offset 計算の 8 段階チェーンは複雑すぎないか？相互作用が予測困難になっていないか？  
(b) 「spread × ratio」方式の offset 算出自体が最善か？代替 (e.g. ATR ベース, VWAP ベース, machine learning ベース) はあるか？  
(c) 現在の cycle_interval=120s は長すぎるか短すぎるか？  
(d) **最も優先すべき 1-2 個の改善は何か？** (本文書の M1-M10 のうち)

---

## 補足: コードの主要関数位置

| 関数 | ファイル | 行 | 概要 |
|------|---------|-----|------|
| `_compute_maker_price()` | run_fill_test.py | 690 | offset 計算チェーン |
| `run_single_cycle()` | run_fill_test.py | 1060 | 1 サイクル実行 |
| `run_continuous()` | run_fill_test.py | 1724 | メインループ |
| `_try_auto_adapt()` | run_fill_test.py | 2348 | param_adapter 実行 |
| `fast_fill_defense` | run_fill_test.py | 2022 | 即約定防御 |
| `compute_adaptation()` | param_adapter.py | 67 | offset 増減ロジック |
| `train_and_save_as_skip_gate()` | skip_gate.py | 603 | SkipGate 訓練 |
| `FillTestConfig` | run_fill_test.py | 60 | 全パラメータ定義 |
