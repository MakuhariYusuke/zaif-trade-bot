# 155# 後知恵フィルター分析レポート

> Phase C dry-run 10 日間の fill_records (2,407 レコード) を後知恵で再評価し、
> 「こうしたら儲かったのに」を定量化する。

---

## §1 分析概要

| 項目 | 値 |
|------|-----|
| 分析期間 | 2026-02-13 〜 2026-02-23 |
| 総レコード | 2,407 |
| 有効分析 (price > 0) | 1,948 |
| 価格タイムライン | 5,448 ポイント |
| 分析スクリプト | `scripts/v460/analysis/hindsight_filter.py` |
| 結果 JSON | `results/v460/hindsight_analysis.json` |

### 手法

- 全レコードの `order_price` + filled の `mid_Xs_after` で**価格タイムライン**を構築
- 各レコードの 30s/60s/120s 後の補間価格から**後知恵 PnL** を計算
- 「もし実行していたら」「もし逆サイドだったら」を定量化

---

## §2 カテゴリ別後知恵 PnL

| カテゴリ | N | avg 30s | avg 120s | 利益率 30s | 見逃し利益 30s | 見逃し利益 120s |
|----------|---|---------|----------|-----------|---------------|----------------|
| **H1: skip_gate** | 228 | **+0.038** | +0.152 | 53.1% | 132.90 bps | 531.60 bps |
| **H2: timeout** | 261 | **+0.406** | **+1.622** | **72.4%** | 92.26 bps | 369.05 bps |
| H5: balance_forced | - | - | - | - | - | - |
| H6: technical | 127 | +0.456 | +1.823 | 58.7% | 119.77 bps | 479.07 bps |
| H7: other | 57 | +0.793 | +3.026 | 53.1% | 70.83 bps | 273.15 bps |
| **filled (実績)** | 1,275 | **-0.805** | -0.713 | 42.9% | - | - |

### 重要な発見

1. **filled の avg PnL が -0.805 bps**: 実行したトレードは平均で負けている
2. **skip_gate でスキップしたものは avg +0.038 bps**: ほぼゼロ → skip_gate は「良い機会」と「悪い機会」を均等に弾いている
3. **timeout が avg +0.406 bps で 72.4% 勝率**: **タイムアウトした注文は実は価格方向が正しかった**。板に並べなかっただけで方向判断は合っていた

---

## §3 H2: タイムアウト — 最大の機会損失

### 3.1 タイムアウト上位 5 件

| 日時 (UTC) | Side | pnl 30s | pnl 120s |
|------------|------|---------|----------|
| 2026-02-18 15:32 | buy | +5.804 | N/A |
| 2026-02-20 05:29 | buy | +5.230 | N/A |
| 2026-02-18 05:26 | sell | +3.402 | N/A |
| 2026-02-19 00:46 | sell | +2.656 | N/A |
| 2026-02-18 22:28 | buy | +2.487 | N/A |

### 3.2 原因分析

タイムアウトは**板に並べたが時間内に約定しなかった**ケース。
- 価格方向は正しい (72.4% 勝率) → **指値のオフセットが保守的すぎる**
- avg +0.406 bps が得られたはず → **fill rate 向上が最も ROI の高い改善**

### 3.3 推奨アクション

| ID | 内容 | 見込み改善 |
|----|------|-----------|
| **T-1** | 指値オフセットの微調整 (spread_offset_ratio をレジーム別に最適化) | timeout 50% 減 → +50 bps |
| **T-2** | reprice 戦略の改善 (stale 検出間隔を短縮) | timeout 30% 減 → +30 bps |
| **T-3** | postonly_reject 時の即再投入 | timeout 20% 減 → +20 bps |

---

## §4 H1: skip_gate — 閾値キャリブレーション

### 4.1 AS 確率帯別分析 (skip された注文)

| AS 帯 | N | avg PnL | 利益率 | profit | loss |
|--------|---|---------|--------|--------|------|
| [0.50-0.55) | 45 | +0.036 | 53.3% | +43.75 | -42.14 |
| [0.55-0.60) | 68 | -0.252 | 51.5% | +46.66 | -63.81 |
| [0.60-0.65) | 3 | +1.453 | 66.7% | +4.72 | -0.36 |
| [0.65+) | 1 | +0.114 | 100% | +0.11 | 0.00 |

### 4.2 閾値シミュレーション

| 閾値 | 実行 | スキップ | avg PnL | total PnL |
|------|------|---------|---------|-----------|
| 0.50 | 409 | 455 | **-0.133** | **-54.44** |
| 0.55 | 731 | 133 | -0.241 | -176.09 |
| 0.60 | 857 | 7 | -0.320 | -273.84 |
| 0.65 | 862 | 2 | -0.312 | -268.97 |

### 4.3 考察

- **閾値を下げる (0.50) ほど avg PnL が改善** → skip_gate はある程度機能している
- 現在の閾値 (~0.545) は妥当だが、**0.50 まで下げるとさらに -54 vs -176 = +122 bps の改善**
- ただし AS[0.50-0.55) のスキップ分は avg +0.036 でほぼゼロ → **これらを実行しても大きな損失にはならないが、利益にもならない**
- **結論: 閾値は現状維持が妥当。利益最大化は skip_gate でなく fill rate 改善で実現すべき**

---

## §5 H3: Side 逆転分析

| Side | Filled | 逆が良い | 逆良率 | avg 実績 | avg 逆 |
|------|--------|---------|--------|---------|--------|
| **buy** | 643 | 338 | 52.6% | -0.081 bps | +0.612 bps |
| **sell** | 632 | 370 | **58.5%** | **-0.516 bps** | **+1.001 bps** |

### 5.1 考察

- **Sell が著しく弱い**: avg -0.516 bps、逆 (buy) なら +1.001 bps
- sell → buy にした場合の改善幅: **+1.517 bps/trade × 632 trades = +959 bps**
- **Buy も弱い**: avg -0.081 bps だが、逆 sell にしても +0.612 bps
- → **市場全体が上昇トレンドだった可能性**。sell するタイミングの精度が低い

### 5.2 推奨アクション

| ID | 内容 | 見込み改善 |
|----|------|-----------|
| **S-1** | sell 判定モデルの改善 (sell-specific 閾値) | sell PnL +0.5 bps → +316 bps |
| **S-2** | マーケットレジーム別 side バイアス (trending up → buy 優先) | sell 損失 30% 削減 |
| **S-3** | sell timeout を短縮 (売りは速い撤退が有利) | sell 損失 20% 削減 |

---

## §6 H4: 時間帯別分析 (JST)

### ベスト時間帯 (skip でも利益が出た)

| 時間 (JST) | skip 数 | skip avg PnL | 勝率 |
|------------|---------|-------------|------|
| JST09 | 35 | **+0.900** | 77.1% |
| JST10 | 16 | **+0.887** | 81.3% |
| JST17 | 2 | +1.060 | 50.0% |
| JST01 | 5 | +2.713 | 60.0% |

### ワースト時間帯 (skip で損失回避)

| 時間 (JST) | skip 数 | skip avg PnL | 勝率 |
|------------|---------|-------------|------|
| JST05 | 27 | **-0.433** | 44.4% |
| JST02 | 29 | **-0.384** | 41.4% |
| JST20 | 27 | -0.085 | 59.3% |

### 6.1 考察

- **JST09-10 (東京市場前場)**: skip_gate がスキップした中で +0.9 bps。この時間帯は skip 閾値を緩和すべき
- **JST02, JST05 (深夜)**: skip が正解。深夜はボラティリティが低く、spread 負けしやすい
- **時間帯別 skip_gate 閾値** は有効な改善策

---

## §7 総合: 「こうしたら儲かったのに」ランキング

### 7.1 改善施策の ROI 順

| 優先 | 施策 | 根拠 | 見込み改善 (10日間) | 実装工数 |
|------|------|------|-------------------|---------|
| **P0** | **T-1: fill rate 向上 (offset 最適化)** | timeout avg +0.406, 72.4% 勝率 | **+50〜100 bps** | 0.3 日 |
| **P0** | **S-1: sell 判定精度改善** | sell avg -0.516, 逆なら +1.001 | **+100〜300 bps** | 0.5 日 |
| P1 | H4: 時間帯別 skip 閾値 | JST09-10 で +0.9 bps の機会損失 | +20〜50 bps | 0.2 日 |
| P1 | T-2: reprice 戦略改善 | stale 検出→再投入の高速化 | +20〜40 bps | 0.3 日 |
| P2 | S-2: レジーム別 side バイアス | trending up → buy 優先 | +30〜80 bps | 0.5 日 |
| P2 | S-3: sell timeout 短縮 | 売りは速い撤退 | +10〜30 bps | 0.1 日 |

### 7.2 即実行可能な quick win

1. **spread_offset_ratio のレジーム別チューニング** (現在固定 7.5%)
   - ranging: 5% に下げる → fill rate UP
   - trending: 10% に上げる → adverse selection 防止

2. **JST09-10 の skip_gate 閾値を 0.50 に緩和**
   - この時間帯は 77-81% 勝率 → スキップ不要

3. **sell 側に時間ベースの早期撤退** (sell_timeout を現行より 10-20% 短縮)

---

## §8 変更履歴

| 日付 | 内容 |
|------|------|
| 2026-02-24 | 初版: 後知恵フィルター分析完了 |
| 2026-02-23 | Codex追記: ログ再点検・盲点補正・追加収益機会の特定 |

---

## §9 Codex追記レビュー (2026-02-23 UTC スナップショット)

### 9.1 155本文へのコメント

| 観点 | 判定 | コメント |
|------|------|----------|
| H2 timeout を最優先とする判断 | 妥当 | 最新再集計でも `timeout=261` 件で機会損失が大きい。fill rate 改善優先は正しい。 |
| sell 側が弱いという指摘 | 妥当 | filled 30s は `buy=-0.069 bps`, `sell=-0.512 bps`。sell 側の劣化は継続。 |
| skip_gate 閾値を主戦場にしない判断 | 概ね妥当 | `skip_gate`単独最適化より execution 改善の方が期待値が高い。 |
| H5 が空欄である点 | 要補正 | `hindsight_filter.py` が `order_price<=0` を除外しており、H5/H6の一部が分析対象外。 |

### 9.2 追加ログ分析で見えた盲点

レビュー時点スナップショット:
- `rows=2434, filled=1278, nonfilled=1156`
- nonfilled 上位: `balance_forced_skip=314 (27.2%)`, `timeout=261 (22.6%)`, `skip_gate=230 (19.9%)`, `orderbook_error=156 (13.5%)`
- `order_price=0` の nonfilled が `480/1156 = 41.5%`

この 41.5% は現行 hindsight 集計から漏れているため、機会損失の上限を過小評価している。

特に重要:
1. `balance_forced_skip` は **2/22:131件, 2/23:183件** に集中しており、実質的な停止時間を作っている。  
2. `orderbook_error=156件` も全件 `order_price=0` で、H6 の真の損失が見えない。  
3. fill 待機時間は `15-30s` が最悪 (`avg -0.563 bps`)。リプライス/撤退の境界が遅い可能性。  
4. レジーム×side では `trending buy +0.594 bps` に対し `trending sell -0.687 bps`。トレンド時 sell が逆噴射。  

### 9.3 「儲かる箇所」追加提案 (優先順)

| 優先 | 施策 | 根拠ログ/定量 | 期待効果 |
|------|------|---------------|----------|
| **P0** | `balance_forced` 救済モード実装 | `balance_forced_skip=314` が最大クラス、2/22-2/23で集中 | 停止時間削減、機会損失の直接回収 |
| **P0** | sell 条件の非対称化 (trending時のsell抑制) | `trending sell -0.687 bps`、`sell全体 -0.512 bps` | 逆行トレード削減、即効で損失圧縮 |
| **P1** | `15-30s` で強制 reprice / cancel-replace | 待機帯別で最悪 `-0.563 bps` | timeout・不利約定の削減 |
| **P1** | `orderbook_error` 時のフォールバック執行 | `orderbook_error=156` が非filled 13.5% | 欠損サイクル縮小 |
| **P2** | 時間帯×side の局所制御 | JST06 sell, JST10-11 buy で大幅マイナス | 無駄打ち削減、時間帯適応 |

### 9.4 実装前に最低限やるべき計測補強

現状のままでは「見えない機会損失」が多い。以下を先に入れると判断精度が上がる。

1. `order_price=0` 時でも `timestamp` 基準の補間価格を疑似参照価格として保存し、H5/H6にも後知恵PnLを出す。  
2. `balance_forced_skip` 連続回数、連続秒数、解除条件(残高回復・約定発生)を fill_records に追加。  
3. `queue_wait_sec` が 15s を超えた注文の再価格調整有無をログへ明示し、PnL 差分を比較可能にする。  

### 9.5 次アクション (短期)

1. `balance_forced` を「完全スキップ」から「低リスク救済執行」に変更する A/B テスト。  
2. sell 側のみ `timeout短縮 + 閾値強化` を適用した 24h 比較。  
3. `orderbook_error` フォールバックの有無で fill rate と 30s PnL を比較。  

上記3点は、現状ログから見て **即効性が高く、かつ実装工数が小さい順** のため優先度が高い。

---

## §10 §9 レビュー対応 (実装結果)

### 10.1 hindsight_filter.py 修正

| 対応 | 内容 | 根拠 |
|------|------|------|
| **§9.4 #1: price=0 補間** | `order_price<=0` のレコードをスキップせず、`_interpolate_price(timeline, ts)` で疑似参照価格を取得し分析対象に含める。`HindsightResult.interpolated_ref` フラグで追跡。 | 41.5% の nonfilled (balance_forced=314, orderbook_error=156) が分析漏れしていた |
| **§9.2 #3: 待機時間帯分析** | `_analyze_wait_bands()` 追加: queue_wait_sec を 0-5s/5-15s/15-30s/30-60s/60s+ にバンド分割し avg PnL を出力 | 15-30s が最悪 (-0.563 bps) という指摘に対応 |
| **§9.2 #4: regime×side 分析** | `_analyze_regime_side()` 追加: regime×side のクロス集計で filled の avg PnL を出力 | trending sell -0.687 bps という指摘の定量化 |
| **§9.4 #1: 補間統計** | `_analyze_interpolated_stats()` 追加: 補間参照価格で分析したレコード群の統計 | 補間精度の可視化 |
| **H8 カテゴリ** | `trending_sell_skip`, `unknown_regime_buy_skip`, `sell_dynamic_kill` を `H8_regime_guard` として分類 | レジームガード系の集約 |

### 10.2 trending sell 抑制 (§9.3 P0 #2)

| 項目 | 内容 |
|------|------|
| 新設定 | `skip_sell_trending: bool` (default=False, YAML で true に設定) |
| 対象 | `regime_detector.current_regime == "trending"` かつ `side == "sell"` |
| 動作 | `cancel_reason=trending_sell_skip` でレコード生成しサイクルをスキップ |
| 根拠 | trending sell avg -0.687 bps、trending buy +0.594 bps → sell が逆噴射 |

### 10.3 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/analysis/hindsight_filter.py` | price=0 補間、wait_bands、regime_side、interpolated_stats、H8 カテゴリ |
| `scripts/v460/lib/cancel_reasons.py` | `TRENDING_SELL_SKIP` 定数 + AUDIT frozenset 更新 |
| `scripts/v460/lib/fill_config.py` | `skip_sell_trending` フィールド + YAML ローダー |
| `scripts/v460/run_fill_test.py` | trending sell スキップブロック (P0-09 と P0-10 の間) |
| `configs/v460/fill_test.yaml` | `skip_sell_trending: true` |
| `tests/unit/v460/test_155_hindsight_review.py` | 12 テスト (補間、バンド、regime×side、カテゴリ、定数、設定) |
| `tests/unit/v460/test_145_structural_fixes.py` | TRENDING_SELL_SKIP を frozenset 期待値に追加 |

### 10.4 テスト結果

```
tests/unit/v460/test_155_hindsight_review.py — 12 passed
tests/unit/v460/test_145_structural_fixes.py::TestCancelReasons — 6 passed
tests/unit/v460/test_152_parallel_tasks.py — 12 passed
tests/unit/v460/test_154_deadlock_prevention.py — 15 passed
```

### 10.5 残課題 (§9 未対応分)

| §9 項目 | ステータス | 備考 |
|---------|-----------|------|
| §9.4 #2: balance_forced 連続回数/秒数を fill_records に追加 | 未対応 | run_fill_test.py の FillRecord 拡張が必要 |
| §9.4 #3: queue_wait_sec 15s超の reprice 有無ログ | 一部対応 | wait_bands 分析は入ったが、reprice ログ連携は未実装 |
| §9.5 #1: balance_forced 低リスク救済 A/B | 未対応 | Phase D (次期運用) で検証予定 |
| §9.5 #3: orderbook_error フォールバック | 未対応 | 板取得失敗時の代替価格源が必要 |
