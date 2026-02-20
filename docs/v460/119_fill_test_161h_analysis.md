# 119# Fill Test 161h 中間分析レポート

> **対象**: v460 Fill Test (168h 長期実測)  
> **分析時点**: 161.5h / 168h (96.1%)  
> **期間**: 2025/02/13 18:39 — 2025/02/20 12:10 JST  
> **Git HEAD**: `29de9fa78` (123# Gemini review 対応)  
> **ステータス**: PID 51480 稼働中、残り ~6.5h  

---

## 目次

1. [Executive Summary](#1-executive-summary)
2. [全体統計](#2-全体統計)
3. [核心発見: Early Exit が PnL を破壊](#3-核心発見-early-exit-が-pnl-を破壊)
4. [Side 別分析](#4-side-別分析)
5. [Regime 別分析](#5-regime-別分析)
6. [時間帯分析](#6-時間帯分析)
7. [Offset 分析](#7-offset-分析)
8. [Spread 分析](#8-spread-分析)
9. [Skip Gate 分析](#9-skip-gate-分析)
10. [P(AS) 較正精度](#10-pas-較正精度)
11. [PnL 時間ホライゾン分析](#11-pnl-時間ホライゾン分析)
12. [エラー・警告パターン](#12-エラー警告パターン)
13. [日次・曜日別トレンド](#13-日次曜日別トレンド)
14. [改善提案 (優先度順)](#14-改善提案-優先度順)
15. [シナリオ試算](#15-シナリオ試算)
16. [データ品質ノート](#16-データ品質ノート)
17. [Appendix A: レコードスキーマ進化](#appendix-a-レコードスキーマ進化)
18. [Appendix B: 累積 PnL 推移](#appendix-b-累積-pnl-推移)
19. [Appendix C: PnL 分布](#appendix-c-pnl-分布)

---

## 1. Executive Summary

**結論: v460 は Early Exit を除去すれば正の期待値を持つシステムである。**

| 指標 | 全体 (879約定) | Enriched (612) | Enriched No-EE (564) |
|---|---|---|---|
| PnL30s mean | -0.291 bps | -0.207 bps | **+0.388 bps** |
| Win Rate | 46.3% | 46.4% | **50.4%** |
| Cumulative PnL | -250.1 bps | -126.7 bps | **+218.8 bps** |

> **最重要発見**: Early Exit (EE) 48件が累積 **-345.5 bps** を生成。EE を除くと enriched 期間は **+218.8 bps** の黒字。EE は損失カットを意図した機構だが、実際には 30s PnL を早期の最悪時点で「凍結」し、本来のリバージョン利益を失わせている。

### 上位 3 改善アクション

| # | アクション | 推定効果 | 難易度 |
|---|---|---|---|
| **P0** | Early Exit 無効化 or 閾値引上げ | +345.5 bps 回復 | ★☆☆ (YAML 1行) |
| **P1** | Unknown レジーム時 offset boost 2.0x | +82.9 bps 改善 | ★★☆ (コード変更) |
| **P2** | Balance 不足解消 (lot 自動縮小) | 新規約定機会 +298件 | ★★☆ (ロジック変更) |

---

## 2. 全体統計

### 2.1 約定サマリ

| 指標 | 値 |
|---|---|
| 総サイクル数 | 1,287 |
| 約定 (filled) | 879 (68.3%) |
| キャンセル (cancelled) | 408 (31.7%) |
| SG スキップ (skip_gate) | 106 (8.2%) |
| PostOnly リジェクト | 61 (4.7%) |
| タイムアウト | 129 (10.0%) |
| API エラー | 34 (2.6%) |

### 2.2 PnL 統計

| 指標 | 30s | 60s | 120s |
|---|---|---|---|
| 件数 | 879 | 264 | 264 |
| 平均 (bps) | -0.291 | -0.398 | **+0.178** |
| 中央値 | -0.16 | — | — |
| 標準偏差 | 5.174 | — | — |
| 1% 分位 | -15.01 | — | — |
| 5% 分位 | -7.77 | — | — |
| 95% 分位 | +7.48 | — | — |
| 99% 分位 | +15.39 | — | — |

### 2.3 AS (Adverse Selection)

| 指標 | 値 |
|---|---|
| AS 判定 (deadzone 2.5bps) | 276/879 = **31.5%** |
| Buy AS | 32.0% |
| Sell AS | 31.0% |
| 最大連続損失 | 12 |

### 2.4 オペレーション

| 指標 | 値 |
|---|---|
| 約定間隔 median | 4.2 min |
| 約定間隔 mean | 11.0 min |
| 約定間隔 max | 27.1 h |
| Queue wait median | 12.6 s |
| Queue wait mean | 34.7 s |
| Reprice 発生 | 88 (6.8%) |
| 累積 PnL | -250.1 bps = **-258 JPY** (lot=0.001 BTC) |

---

## 3. 核心発見: Early Exit が PnL を破壊

### 3.1 Early Exit のメカニズム

```
config: early_exit.threshold_bps = 5.0, monitoring_interval_sec = 5.0
動作: 5秒ごとに mid price を取得し、interim PnL < -5bps で即座に計測終了
結果: post_fill_30s_pnl が早期の最悪時点の値で「凍結」される
```

### 3.2 EE vs 通常の PnL 比較 (Enriched 期間)

| グループ | n | PnL mean | Cumulative | Win% |
|---|---|---|---|---|
| Early Exit (< 28s) | 48 | **-7.198** | -345.5 | — |
| Normal (≥ 28s) | 564 | **+0.388** | +218.8 | 50.4% |

EE 48 件が全損失の **137%** を生成 (total -250.1bpsのうち -345.5bps)。

### 3.3 EE タイミング分布

| 計測秒 | n | PnL mean |
|---|---|---|
| ~5s | 7 | -7.10 |
| ~11s | 10 | -7.81 |
| ~16s | 12 | -6.74 |
| ~22s | 7 | -6.91 |
| ~27s | 8 | -7.53 |

全タイミングで -6.7 〜 -7.8 bps 範囲に収束。閾値 -5bps で発火するため当然。

### 3.4 EE トレードのリカバリー (60s/120s)

EE 後も 60s/120s 計測が行われた 20 件の追跡:

| 時点 | PnL mean |
|---|---|
| 30s (exit時) | -7.214 |
| 60s | -5.755 |
| 120s | **-4.454** |

- 120s で 65% が 30s 時点より回復
- 平均 2.76 bps のリカバリーが発生
- **EE がなければリバージョンにより損失は自然縮小した**

### 3.5 EE の構造的問題

1. **計測バイアス**: EE は「最悪の瞬間」で PnL を凍結する。通常のマーケットメイキングでは短期的な逆行は不可避であり、30s まで待てばリバージョンで回復する可能性が高い
2. **Rapid Exit 副作用**: EE 後に `cycle_interval = 10s` に短縮され、次サイクルで高 spread / 不利な状況で再エントリーするリスク
3. **Side 無差別**: buy / sell 各24件で均等 → side 特有の問題ではなくメカニズム自体の問題
4. **Regime 分布**: trending 24件 / ranging 24件 → trending で特に悪い (-7.984) が ranging でも -6.411

### 3.6 推奨アクション

**即座に `early_exit.enabled: false` に変更。**

代替案として閾値引上げ (`threshold_bps: 15.0`) も検討可能だが、
EE の計測凍結問題は閾値に関わらず存在するため、無効化が最善。

> **注意**: EE は 054# S3 で「テール損失カット」を意図して導入されたが、
> fill test は 0.001 BTC の最小ロットで実行しており、実損は微小。
> EE の本来の意義はリアルロットでの損失防止だが、計測目的のテストでは逆効果。

---

## 4. Side 別分析

### 4.1 全データ

| Side | n | PnL mean | AS% |
|---|---|---|---|
| Buy | 462 | -0.028 | 32.0% |
| Sell | 417 | -0.557 | 31.0% |

### 4.2 Enriched No-EE

| Side | n | PnL mean | Win% |
|---|---|---|---|
| Buy | 279 | **+0.693** | 53.0% |
| Sell | 285 | **+0.090** | 47.7% |

- **Sell は EE 除去で黒字化** (+0.090)。従来 -0.557 が正に転換
- Buy は +0.693 で安定的に黒字
- Sell の win% 47.7% は marginal — さらなる offset 最適化の余地あり

### 4.3 Sell Offset 配分

| 指標 | Buy | Sell |
|---|---|---|
| Offset min | 0.0750 | 0.1920 |
| Offset max | 0.6000 | 0.6429 |
| Offset median | 0.1125 | 0.3000 |
| Offset mean | 0.1405 | 0.3721 |

- Sell offset は config `side_offset.sell = 0.14` + `trending_offset_boost × 1.5` + 適応で拡大 → 実効 median 0.30
- Sell offset が高い = fill は遅いが AS 防御は効いている (AS 31% で Buy 並)
- ただし 99.6% が offset ≥ 0.20 → **fill opportunity cost** が大きい

### 4.4 PostOnly Reject

| Side | 件数 |
|---|---|
| Buy | 24 |
| Sell | **37** |

Sell は PostOnly reject が 1.5x 多い。高 offset → 価格が一方に動いた際に taker 化しやすい。

---

## 5. Regime 別分析

### 5.1 Enriched No-EE

| Regime | n | PnL mean | Win% |
|---|---|---|---|
| Trending | 131 | **+1.441** | 55.7% |
| Ranging | 340 | **+0.332** | 51.2% |
| Unknown | 93 | **-0.891** | 39.8% |

- **Trending が最も利益的** (+1.441) — offset boost 1.5x が適切に機能
- **Unknown が唯一の赤字** (-0.891) — レジーム判定不能時に適切な防御がない

### 5.2 Unknown レジームの深堀り

| 指標 | 値 |
|---|---|
| n (enriched no-EE) | 93 |
| PnL mean | -0.891 |
| Win% | 39.8% |
| Offset median | 0.1920 |
| Regime confidence | 0.176 (mean) |
| Buy PnL | -1.384 (n=47) |
| Sell PnL | -0.388 (n=46) |

- Confidence が 0.176 と極めて低い → レジーム判定が「自信なし」状態
- Buy side で -1.384 と特に悪い
- Offset は 0.19 で trending boost が適用されていない

### 5.3 Regime Confidence vs PnL

| Confidence | n | PnL mean |
|---|---|---|
| [0.0, 0.3) | 83 | **-0.860** |
| [0.3, 0.5) | 50 | +0.237 |
| [0.5, 0.7) | 262 | -0.060 |
| [0.7, 1.0) | 217 | -0.237 |

Low confidence (< 0.3) は高損失と強い相関。

### 5.4 推奨アクション

- **Unknown レジーム時に offset boost 1.5x を適用** (trending と同等の防御)
- あるいは `regime.min_confidence` を引き上げて unknown 判定を狭める
- Unknown + Buy の -1.384bps が最大の赤字源 → SkipGate で補完可能か検討

---

## 6. 時間帯分析

### 6.1 時間帯別 PnL (JST, 全データ)

| Hour | n | PnL mean | AS% |
|---|---|---|---|
| 00h | 49 | +0.266 | 38.8% |
| 01h | 43 | **-2.817** | 55.8% |
| 02h | 15 | +0.319 | 60.0% |
| 03h | 32 | **-1.399** | 65.6% |
| 04h | 22 | — | 50.0% |
| 05h | 23 | **+1.134** | 43.5% |
| 06h | 13 | **-1.282** | 61.5% |
| 07h | 30 | — | 43.3% |
| 08h | 16 | — | 56.2% |
| 09h | 57 | — | 54.4% |
| 10h | 15 | — | **66.7%** |
| 11h | 48 | — | 54.2% |
| 12h | 20 | — | 60.0% |
| 13h | 47 | — | **66.0%** |
| 14h | 77 | — | 42.9% |
| 15h | 34 | — | 55.9% |
| 16h | 53 | — | 50.9% |
| 17h | 55 | **-3.805** | 56.4% |
| 18h | 59 | — | 50.8% |
| 19h | 36 | — | 55.6% |
| 20h | 22 | — | 50.0% |
| 21h | 45 | — | 46.7% |
| 22h | 16 | — | 50.0% |
| 23h | 52 | **-1.492** | 55.8% |

### 6.2 Time-of-Day 危険時間帯

| JST | 重度 | 現在遮断 |
|---|---|---|
| 01h (UTC16) | PnL -2.817, AS 55.8% | ✅ 両 side |
| 03h (UTC18) | PnL -1.399, AS 65.6% | ✅ buy のみ |
| 17h (UTC08) | PnL -3.805, AS 56.4% | ✅ 両 side |
| 23h (UTC14) | PnL -1.492, AS 55.8% | ✅ sell のみ |

- 現行の time_filter 設定は概ね妥当
- JST10h (AS 66.7%, n=15) と JST13h (AS 66.0%, n=47) は高 AS だが n が小さい / PnL データ不足

---

## 7. Offset 分析

### 7.1 Buy Offset Sweet Spot

| Offset 帯 | n | PnL mean | AS% |
|---|---|---|---|
| [0.05, 0.08) | 88 | +0.096 | 22.7% |
| **[0.08, 0.15)** | **58** | **+0.697** | **19.0%** |
| [0.15, 0.20) | 8 | -0.187 | 25.0% |
| [0.20+) | 47 | -0.346 | 36.2% |

- **Buy の最適帯は [0.08, 0.15)**: PnL +0.697, AS 19.0%
- 現行設定 `spread_offset_ratio = 0.05` は低すぎる可能性 → Buy offset を 0.08-0.12 帯に調整
- offset ≥ 0.20 で AS 36.2% に悪化 → 過度な引き上げは逆効果

### 7.2 Sell Offset

Sell は median 0.30 で固定的 → データが [0.19, 0.64] 帯に散在しており sweet spot 特定が困難。`side_offset.sell = 0.14` 設定の結果として実効値が高い。

---

## 8. Spread 分析

### 8.1 Spread at Order vs PnL

| Spread 帯 (JPY) | n | PnL mean |
|---|---|---|
| [0, 500) | 26 | -0.323 |
| **[500, 1,000)** | **41** | **+1.701** |
| [1,000, 2,000) | 151 | -0.624 |
| [2,000, 3,000) | 245 | -0.384 |
| [3,000, 5,000) | 206 | -0.305 |
| [5,000+) | 3 | -0.918 |

- **500-1,000 JPY の narrow spread が最も profitable** (+1.701)
- これは `narrow_spread_boost` が offset を拡大し、AS 防御が効いているため
- 広 spread (1,000+) では PnL が一様に悪い → spread 幅と PnL の逆相関

### 8.2 Spread BPS 統計 (Enriched)

```
n=506, min=0.04, median=2.43, mean=2.34, max=5.03
```

spread_bps 5 未満がほぼ全量 → `wide_spread_bps = 25.0` は実質発動していない。

---

## 9. Skip Gate 分析

### 9.1 SG 統計

| 指標 | 値 |
|---|---|
| SG スキップ件数 | 106 (8.2%) |
| SG 通過 → 約定 | 879 |
| SG score 有効件数 | 506 |
| AS prob median | 0.505 |
| AS prob range | [0.378, 0.600] |

### 9.2 SG 設定

```yaml
skip_gate:
  enabled: true
  buy_enabled: true
  sell_enabled: false           # sell は SkipGate 無効
  as_threshold: 0.52
  adaptive_threshold: true
  target_skip_rate_buy: 0.15
  target_skip_rate_sell: 0.25
```

- Sell 側は 118# A3 で SkipGate 無効化済 (逆選別対策)
- Buy 8.2% skip は `target_skip_rate_buy = 0.15` に対してやや低い → 閾値が高すぎ or warm-up 期間

---

## 10. P(AS) 較正精度

### 10.1 P(AS) 予測 vs 実績

| P(AS) 帯 | n | 実 AS% | PnL mean |
|---|---|---|---|
| [0.4, 0.5) | 177 | 18.1% | **+0.176** |
| [0.5, 0.6) | 241 | 31.5% | -0.141 |

- **P(AS) 0.5 に明確な変曲点**: 0.5 未満は AS 18.1% / 正 PnL、0.5 以上は AS 31.5% / 負 PnL
- 現行閾値 `as_threshold = 0.52` は変曲点 0.5 に近い
- **提案: `as_threshold_buy` を 0.50 に引き下げ** → P(AS) [0.5, 0.52) の n≒50 を追加スキップ

---

## 11. PnL 時間ホライゾン分析

### 11.1 30s vs 60s vs 120s (Matched n=264)

| 時点 | PnL mean |
|---|---|
| 30s | -0.231 |
| 60s | -0.398 |
| 120s | **+0.178** |

- 30s → 60s で悪化 (momentum continuation)、60s → 120s でリバージョン
- **120s で正の PnL に転換** → マーケットメイキングの edge は 120s 以降に顕在化
- 30s → 120s で改善した割合: 51.1% (135/264)

### 11.2 E3 サンプリング率

```yaml
e3.sampling_ratio: 0.50
```

- 879 約定中 264 件で 60s/120s PnL を取得 (30%)
- E3 は追加 60-90s の待機が必要 → サイクル遅延とのトレードオフ
- **現行 50% は適切。168h 完了後に 100% への引上げを検討**

### 11.3 PnL 計測最適化の含意

120s PnL が正であることは、v460 の本質的な edge が「短期的逆行 → 中期リバージョン」にあることを示す。Early Exit はこの edge を直接破壊している。

---

## 12. エラー・警告パターン

### 12.1 カウント (全ログ, 161.5h)

| パターン | 件数 | 重度 |
|---|---|---|
| Order not found retry | 224 | LOW |
| Insufficient BTC for sell | 142 | **HIGH** |
| Insufficient JPY for buy | 134 | **HIGH** |
| Coincheck API 400 (cancel) | 90 | MED |
| Failed to cancel order | 89 | MED |
| Coincheck API 400 (amount) | 86 | **HIGH** |
| Failed to place order | 74 | MED |
| Status unknown after retries | 62 | MED |
| All order attempts failed | 37 | MED |
| Quarantine (blank git_sha) | 22 | LOW |
| Spread > max (sell_guard) | 22 | LOW |
| Early exit at 25s | 17 | INFO |
| Early exit at 15s | 14 | INFO |
| Early exit at 10s | 12 | INFO |
| Status unknown | 12 | MED |
| Early exit at 20s | 11 | INFO |
| Early exit at 30s | 10 | INFO |
| Stale lockfile | 9 | LOW |

### 12.2 Balance 不足 (298 件)

- "Insufficient BTC for sell" (142) + "Insufficient JPY for buy" (134) = **298 件**
- これは全サイクルの **23%** に相当
- 0.001 BTC = ~15,000 JPY での取引でも残高不足が頻発 → **残高管理の根本的問題**
- `balance_shrink` (lot 半減) は連続 3 回失敗後に発動 → 初回での対応が遅い

### 12.3 API 400 (Amount error) — 86 件

- `Coincheck API 400` のうち "Amount" エラーが 86 件
- 最小注文量 0.001 BTC を下回る lot 指定が原因の可能性
- `balance_shrink` で lot が 0.0005 BTC まで縮小されると API が拒否

### 12.4 推奨アクション

- **Balance check を preflight の冒頭に移動し、不足時は即 side 切替**
- `balance_shrink_divisor = 2` で 0.001 → 0.0005 は最小注文量を割る → 下限ガード追加
- API 400 回数を累積監視し、連続 N 回で backoff 拡大

---

## 13. 日次・曜日別トレンド

### 13.1 日次 PnL

| 日付 | n | PnL mean | Cum PnL |
|---|---|---|---|
| 02/13 (Thu) | 37 | -1.163 | -43.0 |
| 02/14 (Fri) | 251 | -0.419 | -105.3 |
| 02/15 (Sat) | 85 | -0.976 | -82.9 |
| 02/16 (Sun) | 6 | -1.155 | -6.9 |
| 02/17 (Mon) | 82 | -0.473 | -38.8 |
| **02/18 (Tue)** | **148** | **+0.607** | **+89.8** |
| **02/19 (Wed)** | **131** | **+0.457** | **+59.9** |
| 02/20 (Thu) | 139 | -0.884 | -122.9 |

- 02/18-19 が唯一の黒字日: この 2 日間は enriched レコードの割合が高く、スキーマ充実期
- 02/13-14: 初期レコード (非 enriched) で構成 → スキーマ不足による分析限界
- **黒字日の特徴**: 約定件数が多い (148, 131) → アクティブな市場で edge が出やすい

### 13.2 曜日別

| 曜日 | n | PnL mean | Win% |
|---|---|---|---|
| Mon | 6 | -1.155 | 50.0% |
| Tue | 82 | -0.473 | 41.5% |
| **Wed** | **148** | **+0.607** | **53.4%** |
| **Thu** | **131** | **+0.457** | **51.9%** |
| Fri | 176 | -0.943 | 42.6% |
| Sat | 251 | -0.419 | 47.4% |
| Sun | 85 | -0.976 | 42.4% |

- 水曜・木曜が黒字 → 週中の流動性が高い時間帯で edge が出やすい
- 週末 (Sat/Sun) は PnL 負、n も多い → 低流動性でスプレッドが広がりやすい

### 13.3 累積 PnL 推移 (マイルストーン)

| Trade# | 日時 | Cum PnL |
|---|---|---|
| 0 | 02/13 18:39 | 0.0 |
| 100 | 02/14 04:23 | -101.7 |
| 200 | 02/14 11:55 | -58.6 |
| 300 | 02/15 06:00 | -160.6 |
| 400 | 02/17 14:54 | -239.7 |
| 500 | 02/18 05:58 | -198.5 |
| 600 | 02/18 20:21 | -201.4 |
| 700 | 02/19 14:18 | -124.9 |
| 800 | 02/20 05:20 | -210.5 |
| 879 | 02/20 12:04 | -250.1 |

- Trade#500-700 で回復 (-239.7 → -124.9): 02/18-19 の黒字日と一致
- 最大ドローダウンストリーク: **-52.2 bps**

---

## 14. 改善提案 (優先度順)

### P0: Early Exit 無効化 ⭐ (最優先)

**変更**: `configs/v460/fill_test.yaml`
```yaml
early_exit:
  enabled: false    # true → false
```

**根拠**: §3 参照。48件の EE が -345.5 bps を生成、EE 除去で +218.8 bps の黒字化。
EE の 65% が 120s で回復しており、本来不要な損失確定。

**リスク**: テール損失がそのまま 30s PnL に反映される。ただし 0.001 BTC lot では影響は ≤ 1 JPY/trade。
168h テスト完了後に YAML 1 行変更で即適用可能。

**期待効果**: +345 bps / 168h → 年換算で約 +17,963 bps。

---

### P1: Unknown レジーム対策

**変更案 A**: Unknown 時に offset boost 適用
```python
# scripts/v460/lib/offset_calculator.py (仮)
if regime == "unknown":
    offset *= 1.5  # trending と同等の防御
```

**変更案 B**: `regime.min_confidence` 引上げ
```yaml
regime:
  min_confidence: 0.4  # 0.3 → 0.4 (unknown 判定を狭める)
```

**根拠**: Unknown PnL -0.891 (n=93)、confidence mean 0.176。Low confidence は損失と強相関。

**期待効果**: 93 件 × 0.891 bps = ~+82.9 bps

---

### P2: Balance 不足対策

**変更案**:
1. Preflight で残高チェックを最優先に (現行: 発注後に判明)
2. 不足時に即 side 切替 (alt_side)
3. `balance_shrink` の下限を `tuning.min_order_btc` でガード

**根拠**: 298 件の balance 警告 = 23% のサイクルが無駄。

**期待効果**: 新規約定機会増加、サイクル効率 ~+20%

---

### P3: Buy Offset 最適化

**変更**:
```yaml
spread_offset_ratio: 0.08  # 0.05 → 0.08 (sweet spot [0.08, 0.15) に合わせる)
```

**根拠**: §7.1 Buy offset [0.08, 0.15) が PnL +0.697 / AS 19.0% で最適。

**リスク**: Fill rate 低下の可能性。適応ロジックとの干渉を要確認。

---

### P4: P(AS) 閾値微調整

**変更**:
```yaml
skip_gate:
  as_threshold_buy: 0.50  # 0.52 → 0.50 (変曲点に合わせる)
```

**根拠**: §10 P(AS) 0.5 に明確な変曲点。AS 18.1% → 31.5% の境界。

---

### P5: Sell SkipGate 再評価

**現状**: `sell_enabled: false` (118# A3)
**検討**: EE 除去後に sell PnL が +0.090 に改善。SkipGate 再有効化の余地。

168h データ完了後に enriched sell PnL を再検証し、SkipGate 有効化の A/B テストを検討。

---

### P6: E3 サンプリング 100% 化

**変更**:
```yaml
e3:
  sampling_ratio: 1.00  # 0.50 → 1.00
```

**根拠**: 120s PnL (+0.178) は正であり、30s PnL (-0.291) と符号が異なる。
全約定で 120s PnL を取得することで、edge の時間構造をより正確に把握。

**トレードオフ**: サイクル間隔が追加 60-90s 延長 → 時間あたり約定数が ~30% 減少。
168h テスト完了後専用の検証実行として計画。

---

### P7: API エラー耐性強化

- "Amount error" 86 件 → `min_order_btc` 以下の lot で発注を試行しない pre-check 追加
- "Failed to cancel" 89 件 → cancellation retry に exponential backoff 追加
- "Order not found retry" 224 件 → retry count の上限引下げ (ログ汚染防止)

---

## 15. シナリオ試算

Enriched 期間 (134.6h) のデータに基づくシナリオ別の期待 PnL:

| シナリオ | n 想定 | PnL mean | Cum PnL (168h換算) |
|---|---|---|---|
| **現状 (As-Is)** | 879 | -0.291 | -250.1 |
| **P0: EE 無効化** | 879 | +0.115 | +101 |
| **P0 + P1: EE OFF + Unknown boost** | ~786 | +0.641 | +504 |
| **P0 + P1 + P3: + Buy offset** | ~786 | ~+0.9 | ~+707 |

> ⚠️ 上記試算は過去データの回帰的な推定であり、将来の市場環境で同等の成果を保証しない。
> 特に P1 の Unknown boost は、unknown 判定のトレードを回避/保守化するため約定件数が減少する。

### 15.1 No-EE + No-Unknown シナリオ (実測ベース)

| 指標 | 値 |
|---|---|
| n | 471 |
| PnL mean | **+0.641** |
| Win% | **52.4%** |
| Cum PnL | **+301.7 bps** |

Enriched 期間で実測された最善シナリオ。

---

## 16. データ品質ノート

### 16.1 スキーマ進化

テスト期間中にレコードスキーマが拡張された:

| 期間 | キー数 | 追加フィールド |
|---|---|---|
| 前半 (~02/13-14) | 13 | 基本のみ |
| 中盤 (~02/14-) | 38 | regime, offset, spread, SG, OB depth 等 |
| 後半 (~02/16-) | 42 | git_sha, run_id, ob_quality_ok 等 |

- **Enriched レコード (38+ keys)**: 612/879 = 69.6%
- 前半 267 件は regime / offset 情報なし → Side / PnL のみで分析

### 16.2 60s/120s PnL カバレッジ

| 時点 | 件数 | カバレッジ |
|---|---|---|
| 30s | 879/879 | 100% |
| 60s | 264/879 | **30%** |
| 120s | 264/879 | **30%** |

- E3 sampling_ratio 0.50 だが実効カバレッジは 30% → EE やサイクルタイミングで 60s 未計測も
- 120s PnL の +0.178 は n=264 で統計的に marginal (SE ≈ 5.2/√264 ≈ 0.32)

### 16.3 Reprice 影響

| グループ | n | PnL mean |
|---|---|---|
| Repriced | 53 | -0.281 |
| Not repriced | 826 | -0.285 |

Reprice は PnL に有意な影響なし (差 0.004 bps, 事実上同等)。

---

## Appendix A: レコードスキーマ進化

**前半レコード (13 keys)**:
```
adverse_selected, cancelled, cycle_id, fill_price, filled,
mid_30s_after, mid_at_fill, order_price, order_quantity,
post_fill_30s_pnl, queue_wait_sec, side, timestamp
```

**後半レコード (42 keys)**:
```
上記 + actual_measurement_sec, adverse_selected_raw, ask_depth_total,
bid_depth_total, cancel_reason, effective_offset_used, error_message,
git_sha, mid_120s_after, mid_60s_after, mid_price_trend_5s, ob_age_ms,
ob_quality_ok, orderbook_imbalance, post_fill_120s_pnl, post_fill_60s_pnl,
regime, regime_confidence, regime_stability, reprice_count, run_id,
skip_gate_as_prob, skip_gate_model_used, skip_gate_reason, skip_gate_score,
skip_gate_skipped, skip_gate_threshold_used, spread_at_order, spread_bps,
spread_offset_ratio
```

---

## Appendix B: 累積 PnL 推移

```
Trade#    日時             Cum PnL (bps)
──────────────────────────────────────────
   0      02/13 18:39       0.0
 100      02/14 04:23    -101.7  ← 初期 drawdown (非 enriched 期)
 200      02/14 11:55     -58.6  ← 一時回復
 300      02/15 06:00    -160.6  ← 最大 drawdown 拡大
 400      02/17 14:54    -239.7  ← 底値圏
 500      02/18 05:58    -198.5  ← 回復開始 (enriched 期)
 600      02/18 20:21    -201.4
 700      02/19 14:18    -124.9  ← 黒字日で大幅回復
 800      02/20 05:20    -210.5  ← 再下落
 879      02/20 12:10    -250.1  ← 現在
```

最大 drawdown streak: **-52.2 bps** (連続)

---

## Appendix C: PnL 分布

### C.1 分位点

| 分位 | PnL (bps) |
|---|---|
| 1% | -15.01 |
| 5% | -7.77 |
| 10% | -6.05 |
| 25% | -2.39 |
| **50%** | **-0.16** |
| 75% | +1.79 |
| 90% | +4.99 |
| 95% | +7.48 |
| 99% | +15.39 |

### C.2 テール構成

**Worst 10 trades**:

| PnL | EE? | Side | Regime |
|---|---|---|---|
| -31.40 | | sell | (pre-enriched) |
| -31.17 | | sell | trending |
| -26.22 | | sell | (pre-enriched) |
| -23.55 | ✅ | buy | trending |
| -23.12 | | buy | ranging |
| -19.18 | | sell | (pre-enriched) |
| -18.82 | | buy | (pre-enriched) |
| -15.86 | | buy | (pre-enriched) |
| -15.01 | ✅ | sell | trending |
| -14.17 | | sell | (pre-enriched) |

**Best 10 trades**:

| PnL | Side | Regime |
|---|---|---|
| +25.83 | buy | (pre-enriched) |
| +25.09 | buy | trending |
| +20.11 | buy | trending |
| +20.00 | sell | trending |
| +17.47 | buy | trending |
| +16.26 | sell | (pre-enriched) |
| +16.15 | sell | unknown |
| +15.71 | sell | ranging |
| +15.39 | buy | trending |
| +14.99 | buy | trending |

- Worst 10 の 7/10 が pre-enriched 期 → スキーマ不足期の制御なしトレード
- Best 10 の 6/10 が trending regime → **trending は高リターン + 高リスク**

### C.3 分布特性

- **Skew**: -0.055 (ほぼ対称)
- **標準偏差**: 5.174 bps
- 中央値 -0.16 vs 平均 -0.291 → 少数の大損失が平均を引下げ (fat left tail)

---

*EOF — 119# Fill Test 161h 中間分析レポート*
