# 162# phg — Fill Test 10日間ログ分析レポート

> **日付**: 2026-02-25  
> **種別**: rpt (ログ分析)  
> **前提**: 161# (コード品質改善) + Fill Test 10日間データ  
> **テーマ**: 実測ログからの改善機会特定

---

## §1 データ概要

| 指標 | 値 |
|---|---|
| 分析期間 | 2026-02-13 09:39 UTC ~ 2026-02-24 15:09 UTC (約11日) |
| 総レコード数 | 3,144 |
| Fill (約定) 数 | 1,422 |
| Fill Rate | 45.2% |
| 日別データファイル | 10本 (`fill_records_20260215.jsonl` ~ `20260224.jsonl`) |

---

## §2 重大発見事項

### 🔴 F1: Fill Rate の急激な低下

| 日付 | Orders | Filled | Fill Rate | avg_pnl30 | sum_pnl30 |
|---|---|---|---|---|---|
| 02/13 | 211 | 163 | **77%** | -0.44bps | -71.8bps |
| 02/14 | 220 | 161 | **73%** | -0.72bps | -116.5bps |
| 02/15 | 60 | 49 | **82%** | -0.88bps | -42.9bps |
| 02/17 | 205 | 137 | **67%** | +0.45bps | +61.5bps |
| 02/18 | 277 | 149 | 54% | +0.35bps | +52.6bps |
| 02/19 | 250 | 176 | 70% | -0.55bps | -97.2bps |
| 02/20 | 217 | 132 | 61% | -0.20bps | -26.2bps |
| 02/21 | 377 | 164 | 44% | -0.60bps | -99.0bps |
| 02/22 | 401 | 127 | **32%** | -0.18bps | -23.1bps |
| 02/23 | 592 | 52 | **9%** | -0.37bps | -19.5bps |
| 02/24 | 313 | 98 | **31%** | +0.78bps | +76.1bps |

**傾向**: Fill Rate が 77-82% → 9-32% に急落。特に 2/22-23 で `balance_forced_skip` が集中。  
**原因仮説**: 残高枯渇 or ガード条件の過度な厳格化 or コードバージョン変更。

### 🔴 F2: Sell サイドの構造的不利

| Side | Orders | Filled | Fill Rate | avg_pnl30 | Profitable | AS率 |
|---|---|---|---|---|---|---|
| buy | 1,222 | 717 | **58.7%** | +0.05bps | 49.0% | 27.5% |
| sell | 1,922 | 705 | **36.7%** | -0.50bps | 44.4% | 26.7% |

- Sell は Buy の **1.57倍** のオーダー数ながら Fill Rate は 22pt 低い
- Sell の avg_pnl30 がマイナス、profitable rate も 4.6pt 低い
- Sell ガード系キャンセルが合計 **341件** (`trending_sell_skip` 233 + `sell_dynamic_kill` 92 + `sell_guard_reject` 16)

### 🔴 F3: Adverse Selection が収益性を支配

| 区分 | 件数 | avg_pnl30 |
|---|---|---|
| AS (逆選択) | 385 (27.1%) | **-5.29bps** |
| Non-AS | 1,037 (72.9%) | **+1.65bps** |

AS を完全に排除できれば avg_pnl30 は -0.23bps → +1.65bps に改善。  
SkipGate の AS 検知精度向上が最もレバレッジの高い改善ポイント。

### 🟡 F4: Cancel Reason 分布

| Reason | 件数 | 比率 | 対処方針 |
|---|---|---|---|
| `balance_forced_skip` | 377 | 21.9% | 残高管理改善 / ポジションサイジング見直し |
| `skip_gate` | 288 | 16.7% | SkipGate が正しく機能中（期待通り） |
| `timeout` | 276 | 16.0% | 正常（注文が約定しなかった） |
| `trending_sell_skip` | 233 | 13.5% | sell 側トレンドフィルタの閾値調整 |
| `orderbook_error` | 161 | 9.3% | 多くは `sell_guard: spread > max` |
| `sell_dynamic_kill` | 92 | 5.3% | 動的 sell キル条件の緩和検討 |
| `postonly_reject` | 68 | 3.9% | PostOnly リジェクト（取引所仕様） |
| `spread_too_narrow` | 50 | 2.9% | 正常（マージン不足） |
| `api_error` | 34 | 2.0% | API接続安定性 |
| `stale_skip_gate_blocked` | 27 | 1.6% | SkipGate staleness 閾値調整 |

### 🟡 F5: Retrain Scheduler の停滞

| 指標 | 値 |
|---|---|
| 総 retrain history エントリ | 98 |
| skipped (データ不足等) | **69 (70%)** |
| rejected (品質ゲート不合格) | 13 (13%) |
| skipped_trigger (トリガー未到達) | 12 (12%) |
| deployed_verified (成功) | **4 (4%)** |

- Walk-Forward 評価で **pnl30_improvement が頻繁にマイナス** → モデル改善ができていない
- 直近 (2/24 13:54 以降) は `Insufficient data for WF eval` でブロック
- sell モデル (`skip_gate_lgbm_pnl120_sell.pkl`) は score=-0.187 でもデプロイ（初期訓練なので）

---

## §3 詳細分析

### 3.1 Regime 別パフォーマンス

| Regime | Orders | Filled | Fill Rate | avg_pnl30 |
|---|---|---|---|---|
| ranging | 1,106 | 793 | **71.7%** | -0.21bps |
| None | 973 | 267 | 27.4% | -0.46bps |
| trending | 521 | 236 | 45.3% | -0.04bps |
| null (未検出) | 355 | **0** | **0.0%** | N/A |
| unknown | 116 | 93 | 80.2% | -0.89bps |
| trending_down | 40 | 30 | 75.0% | **+2.16bps** |
| trending_up | 33 | 3 | 9.1% | -0.56bps |

- **`null` regime が 355件で全キャンセル**: regime 検出失敗時にオーダーが全て無駄に
- **`trending_down` が唯一の正PnL**: buy オーダーが有利な局面
- **`None` regime も低 fill rate**: regime 検出器の改善余地

### 3.2 時間帯別パフォーマンス (UTC)

| 時間帯 | 特に悪い | avg_pnl30 | profitable |
|---|---|---|---|
| 08h (17:00 JST) | ⚠️ | **-3.81bps** | 40% |
| 14h (23:00 JST) | ⚠️ | **-1.72bps** | 38% |
| 16h (01:00 JST) | ⚠️ | **-2.82bps** | 33% |
| 21h (06:00 JST) | ⚠️ | **-1.36bps** | 37% |

| 時間帯 | 特に良い | avg_pnl30 | profitable |
|---|---|---|---|
| 01h (10:00 JST) | ✅ | **+1.22bps** | 56% |
| 20h (05:00 JST) | ✅ | **+0.85bps** | 56% |
| 11h (20:00 JST) | ✅ | **+0.31bps** | 57% |

**示唆**: 時間帯フィルタ ("休むも相場") の導入検討。特に 08/14/16/21h UTC はエッジがほぼない。

### 3.3 git_sha 別パフォーマンス

| git_sha | Orders | Filled | Fill Rate | avg_pnl30 |
|---|---|---|---|---|
| `8ba101953` | 122 | 74 | 60.7% | **+1.35bps** |
| `a7e5d0b82317` | 351 | 115 | 32.8% | **+0.54bps** |
| `361c67f4e` | 338 | 191 | 56.5% | +0.24bps |
| `10c68dba6` | 408 | 12 | **2.9%** | +0.32bps |
| `3959424ef883` | 225 | 3 | **1.3%** | -0.04bps |
| `a573a3be6` | 453 | 209 | 46.1% | -0.48bps |
| `ce5c6f8d6` | 245 | 179 | 73.1% | -0.60bps |

- `10c68dba6` と `3959424ef883` は Fill Rate 1-3% → コードバージョンにバグまたは過剰ガードの可能性
- `8ba101953` が +1.35bps で最高パフォーマンス → 何が差分だったか要調査

### 3.4 追加統計

| 項目 | 値 | 評価 |
|---|---|---|
| Spread (at order) | Mean 2.38bps, Median 2.42bps | 狭い → PostOnly戦略に合致 |
| Queue Wait | Mean 31.0s, Median 13.0s, p90 70.7s | 待ち時間はやや長い |
| Reprice | 89/1422 filled (6.3%), avg drift 7.44bps | Reprice後の PnL は -0.42bps (微量損) |
| VG Triggered | 291/2632 (11.1%) | ボラティリティガード発火率は適正 |
| FFD Boost Active | 78/1035 (7.5%) | FastFillDefense 稼働中 |
| balance_forced by day | 02/22: 131, 02/23: 246 | 残高枯渇が集中 |

---

## §4 改善提案 (優先度順)

### P0: 収益直結

| ID | 提案 | 期待効果 | 根拠 |
|---|---|---|---|
| **A1** | SkipGate AS 検知精度向上 | avg_pnl30 → +1.65bps（AS排除時） | F3: AS が -5.29bps で支配的損失 |
| **A2** | `balance_forced_skip` 根本対策 | Fill Rate +10-15pt | F4: 21.9% が残高不足キャンセル |
| **A3** | sell 側ガード条件の緩和・再調整 | Sell Fill Rate 向上 | F2: sell 36.7% vs buy 58.7% |

### P1: 中優先

| ID | 提案 | 期待効果 | 根拠 |
|---|---|---|---|
| **B1** | 時間帯フィルタ導入 (08/14/16/21h UTC OFF) | 損失回避 -2~-4bps/h 削減 | §3.2 |
| **B2** | `null` regime 時のフォールバック戦略 | 355件の無駄なキャンセル回避 | §3.1 |
| **B3** | Retrain データ蓄積量の増加策 | モデル更新成功率 4% → 向上 | F5 |
| **B4** | git_sha `8ba101953` 差分分析 | +1.35bps の要因特定 | §3.3 |

### P2: 構造改善

| ID | 提案 | 期待効果 | 根拠 |
|---|---|---|---|
| **C1** | `orderbook_error` の sell_guard 閾値見直し | 161件のキャンセル削減 | F4 |
| **C2** | Reprice ロジックのチューニング | drift 7.44bps → PnL改善 | §3.4 |
| **C3** | stale_skip_gate_blocked の閾値調整 | 27件の不要ブロック削減 | F4 |

---

## §5 Retrain Scheduler 詳細

### 5.1 最新デプロイ履歴

| 時刻 | モデル | Score | pnl30_imp |
|---|---|---|---|
| 02/21 20:01 | `skip_gate_lgbm_pnl120.pkl` | — | — |
| 02/21 20:56 | `skip_gate_lgbm_pnl120.pkl` | — | — |
| 02/24 05:34 | `skip_gate_lgbm_pnl30_buy.pkl` | 0.3552 | +0.0177 |
| 02/24 05:34 | `skip_gate_lgbm_pnl120_sell.pkl` | -0.1874 | -0.4087 |

### 5.2 問題パターン

1. **Walk-Forward データ不足**: 直近4回連続で `Insufficient data for WF eval` → WF Window が小さすぎ
2. **pnl30_improvement が頻繁にマイナス**: 新モデルが旧モデルより悪い → 特徴量の改善が必要
3. **Quality Gate REJECT 繰り返し**: `improvement=-0.3552 < -0.05. Keeping existing model.` — 既存モデルとの差が大きすぎ
4. **Correlated features 除去**: `depth_imbalance_ob`, `side_aligned_tfi`, `side_aligned_velocity`, `trade_flow_imbalance_60s` が冗長として除去 → 特徴量設計の見直し

---

## §6 結論

**現状評価**: Fill Test 全体の avg_pnl30 = -0.23bps でわずかにマイナス圏。ただし Non-AS のみでは +1.65bps と正のエッジが存在。

**最大レバレッジ**: Adverse Selection の排除精度向上 (SkipGate) が収益改善の最大テコ。27.1% の AS 率を半減できれば、全体 avg_pnl30 を +0.5bps 以上に押し上げられる。

**次ステップ**: A1 (SkipGate AS精度向上) → A2 (balance_forced対策) → B1 (時間帯フィルタ) の順で実装を推奨。
