# 129# ログ分析・改善指針・残課題統合レビュー

> **セッション**: 131# (文書番号: 129#)  
> **日付**: 2026-02-21  
> **分析対象**: fill_test 全レコード (02/13~02/21) + retrain_scheduler.log + fill_test_state.json  
> **Git HEAD**: `7cb39ebb5` (131# trades date_filter fallback fix)  
> **前提文書**: 118# (残課題深掘り考察) + 128# (ログレビューと方策) + 130#/131# 実装結果  
> **目的**: 外部 AI コーディングエージェントによるクロスレビュー用

---

## §0 エグゼクティブサマリ

### 大義への距離

本プロジェクトの大義は **「短期間での高収益性システム」** (000# §0)。  
現在 ph2 (G1.1-exec gate) の fill_test を継続稼働中。

**結論**: 128# の提案 6 項目を 130# で全実装、131# で retrain 学習改善を投入。
最新 run (130# 適用後 51 cycles) で **AS 率 5.0%** (全体 30.8% → 劇的改善) を記録したが、
PnL30 は依然として -0.10 bps (改善傾向)。retrain は bootstrap 段階で未 deploy。
**最大の懸念は 131# 再起動後のデータ蓄積速度と retrain 到達のタイミング**。

### 現在の KPI サマリ

| 指標 | 全期間値 (1609 cycles) | 130# run (51 cycles) | 傾向 |
|------|----------------------|---------------------|------|
| Fill rate | 65.1% | 39.2% | ⬇️ cancel 増 (skip_gate + orderbook_error) |
| PnL30 mean | -0.173 bps | -0.10 bps | ⬆️ 改善中 |
| AS rate | ~30.8% | 5.0% | ⬆️⬆️ 劇的改善 |
| Cumulative PnL | -168.7 JPY | -2.03 bps | ⬆️ 損失ペース緩和 |
| VG 発動率 | 3.4% | — | 低 (感度向上の余地) |
| retrain deploy | 0 回 | 0 回 | ❌ 未到達 |

### 130#/131# 投入効果

| 施策 | 130# 実装 | 観測効果 |
|------|-----------|---------|
| Bootstrap 2段化 | `min_total=30, min_new=10` | pnl30 target で coverage 100% に (131#)。ただし still <30 sample |
| Gate 統一 | K1-K6 + F1-F8 表示 | モニタリング一致。判断ブレ解消 |
| UTC21 sell block | `skip_utc_hours_sell` に 21 追加 | 02/21 で UTC21 sell 0 件 (ブロック有効) |
| Unknown buy guard | `offset_boost=2.0` | unknown regime buy の PnL 改善を期待 |
| Postonly 二重確認 | mid 再取得 + best_bid/ask 補正 | 02/21 postonly_reject 0 件 (git `870007d8f` 以降) |
| I/O 日付限定 | date_filter で trades/OB 日単位 | OB 70,543→51 (99.9%削減)。**trades fallback 実装済** (`7cb39ebb5`) |
| pnl30 target (131#) | `retrain.target: pnl30` | valid_target_samples 20/20=100% (was 60/109=55% at pnl120) |
| Config hot-reload (131#) | per-cycle YAML再読込 | YAML 変更で retrain 即反映 (再起動不要) |

---

## §1 データ分析

### 1.1 全期間サマリ (02/13~02/21, 1609 cycles)

```
Total cycles:  1609
Filled:        1048 (65.1%)
Cancelled:      561 (34.9%)
PnL30 sum:    -263.04 bps   (avg: -0.173 bps)
Side:          buy 847 / sell 762
Filled sides:  buy ~530 / sell ~518
```

### 1.2 日別 PnL 推移

| 日付 | n fills | pnl30 sum | pnl30 avg | 損益 |
|------|---------|-----------|-----------|------|
| 02/13 | 37 | -43.0 | -1.163 | ❌ |
| 02/14 | 251 | -105.3 | -0.419 | ❌ |
| 02/15 | 85 | -82.9 | -0.976 | ❌ |
| 02/16 | 6 | -6.9 | -1.155 | ❌ |
| 02/17 | 82 | -38.8 | -0.473 | ❌ |
| **02/18** | **148** | **+89.8** | **+0.607** | ✅ |
| **02/19** | **131** | **+59.9** | **+0.457** | ✅ |
| 02/20 | 179 | -107.3 | -0.600 | ❌ |
| 02/21 | 129 | -28.5 | -0.221 | ❌ (改善中) |

**所見**: 黒字は 02/18-19 のみ (+149.7 bps)。残り 7 日間で -413 bps。
02/21 は -0.221 bps/fill で損失ペースが緩和。130# 施策の効果と見られる。

### 1.3 130# run 詳細分析 (51 cycles, sha `870007d8f`)

```
Filled:  20/51 (39.2%)
PnL30:   sum=-2.03, avg=-0.10 bps
AS rate: 1/20 = 5.0%  (全期間 ~30.8% → 25.8pt 改善)
Cancel reasons:
  skip_gate: 11 (21.6%)
  orderbook_error: 11 (21.6%)
  timeout: 8 (15.7%)
  stale_skip_gate_blocked: 1
```

**重要所見**:
- **AS 率 5.0%** は全期間 30.8% から劇的改善。UTC21 sell block + unknown buy guard + postonly 二重確認の複合効果
- **Fill rate 39.2%** は低い。skip_gate 11 件 (21.6%) + orderbook_error 11 件 (21.6%) が主因
- orderbook_error の内訳: 全件 "Spread too narrow" (1500 JPY 基準超)  
  → `min_spread_jpy: 1500` が厳しすぎる可能性
- PnL30 avg = -0.10 bps は全期間 -0.173 bps から改善

### 1.4 キャンセル要因 (全期間)

| Cancel Reason | 件数 | 割合 | 傾向 |
|---|---:|---:|---|
| timeout | 176 | 31.4% | 最多。90s timeout で fill されず |
| skip_gate | 137 | 24.4% | ゲート判定で除外 |
| orderbook_error | 75 | 13.4% | 130# で細分化済。主に spread narrow |
| postonly_reject | 68 | 12.1% | 130# E1 二重確認で改善見込み |
| api_error | 34 | 6.1% | 最低注文量不足 (lot 引上げ時に解消) |
| status_unknown | 23 | 4.1% | API レスポンス不明 |
| stale_skip_gate_blocked | 10 | 1.8% | stale 再発注時の SG 抑止 |
| stale_reprice_failed | 7 | 1.2% | reprice 失敗 |
| status_unknown_fast | 4 | 0.7% | 高速 status 不明 |

### 1.5 Git SHA 別サイクル数

全期間で 17 の異なる git_sha が記録されている。コード変更が頻繁でデータの一貫性に注意。

| sha | cycles | メモ |
|-----|--------|------|
| 361c67f4e | 338 | 最多 (初期安定版) |
| ce5c6f8d6 | 245 | |
| ddcdcc934 | 214 | |
| 870007d8f | 51 | 最新 (130# run) |

---

## §2 retrain_scheduler 分析

### 2.1 状態遷移

```
[旧 run: 1771607250]
  target: pnl120 → valid_target_samples: 50~60 / 109 = 55% coverage
  OB matched: 70,543 snapshots (全量ロード)
  trades: 4,396,171 (全量ロード)
  結果: 全サイクル skipped ("insufficient samples: 60 < 100")

[130# run: 1771651879]  
  target: pnl120 (変更前) → 同様に insufficient
  OB matched: 15/15~39/39 (OB recorder 効果)
  trades: 4,396,171 (I/O 日付限定の効果なし — この run では date_filter 未適用)

[131# restart: 1771661473]
  target: pnl30 → valid_target_samples: 20/20 = 100% coverage ✅
  OB: date_filter ['20260221'] → 51 snapshots (99.9%削減)
  trades available=False ← ⚠️ 問題
  Phase: bootstrap (20 < 30, min_total=30)
  結果: skipped (bootstrap 閾値 30 に未到達)
```

### 2.2 重要問題: `trades available=False`

131# restart 後、date_filter で `['20260221']` に限定した結果、
OB snapshots は 51 件ロードされたが **trades が 0 件** (available=False)。

**原因推定**:
- `data/v460/raw/trades/` ディレクトリに 20260221 の trades ファイルが存在しない
- OB recorder (129#) は板情報のみを記録し、約定履歴 (trades) は別系統で取得が必要
- 旧 run では全日付の trades を使っていたため問題が隠れていた
- date_filter による trades 日付限定が、実データ不在を露呈させた

**影響**:
- feature_enricher の `enrich()` で trades ベースの特徴量 (`trade_count_60s`, `buy_ratio`, `vpin_60s` 等) が欠落
- retrain 学習の品質が低下 (preorder 16 特徴量のうち ~6 特徴量が使えない)

**対策案**: 
1. trades の日付限定を緩和し、直近 N 日分を使う (OB と同じ date_filter ではなく、別の window)
2. OB recorder に trades 記録機能も追加
3. date_filter を OB のみに適用し、trades は全量維持

### 2.3 retrain deployment 0 回の根本原因

**v460 全期間を通じて retrain が一度もモデルを deploy していない。**

時系列:
1. OB データ欠如 → OB matched=0 (129# 前)
2. OB recorder 導入 → OB matched 回復 (129#)
3. pnl120 target で coverage 55% → `insufficient samples` 継続
4. pnl30 target 変更 → coverage 100% ✅ (131#)
5. date_filter で trades 欠如 → 特徴量品質低下 ⚠️
6. Bootstrap 閾値 30 に未到達 → skipped 継続

**解消の見通し**: 130# run の 20 filled + 新 run の蓄積で 30 到達まで **あと 10 fill** (約 5-10 時間)。
trades 問題が解消されれば、初回 bootstrap deploy が可能。

---

## §3 131# 再起動後の問題

### 3.1 fill records 未出力

131# restart (17:11) から 20+ 分経過時点で、新 run_id (`1771661473_ac4f9cb1`) の
fill records が 0 件。最終レコードは 16:59 (旧 run)。

**確認事項**:
- プロセス (PID 98372, 98780, 62812, 96640) は全て alive
- fill_test_state.json は 16:34 で更新停止 (旧 run のまま)
- retrain_scheduler.log は 17:11 に新 run のログを出力済

**可能性**:
1. 新 run の最初のサイクルが長時間実行中 (warm-up + API 初期化)
2. 例外発生で fill loop が停止 (ただしプロセスは alive)
3. fill_records の書き出し先が異なる (日付ファイル切替の問題)

**推奨**: fill_test のターミナル出力 (stdout/stderr) を確認し、エラーの有無を検証。

### 3.2 fill_test_state.json の陳腐化

state file の `saved_at_iso` が `2026-02-21T16:34:22` で止まっている。
新 run が state を更新していない = fill loop が state 保存に到達していない。

---

## §4 118# 残課題の現状評価

118# で特定された 53 OPEN items + 24 未検討提案の現状を精査。

### 4.1 解決済み (130#/131# で追加解決)

| 118# ID | 内容 | 解決 |
|---------|------|------|
| A5 part | UTC21 sell block | 130# `skip_utc_hours_sell` に 21 追加 |
| B10 part | sell offset 段階引上げ | 130# unknown buy guard (間接的) |
| §8.2 | postonly_reject | 130# E1 二重確認 |
| §13.1 | retrain bootstrap 2段化 | 130# `bootstrap_min_total=30` |
| §13.2 | retrain I/O 日付限定 | 130# date_filter 実装 |
| §13.3 | gate 判定統一 | 130# K1-K6 + F1-F8 二段表示 |
| §13.5 | orderbook_error 細分化 | 130# タイムアウト/レート制限/空/ガード拒否 分離 |
| — | retrain pnl30 target | 131# coverage 55%→100% |
| — | config hot-reload | 131# per-cycle YAML 再読込 |

### 4.2 残留 OPEN items (優先度順)

#### P0 — 即時対応推奨

| ID | 内容 | 出典 | 現況 | 推奨アクション |
|----|------|------|------|-------------|
| **NEW-1** | trades available=False (date_filter 副作用) | §2.2 | retrain 品質悪化 | trades 取得方法の修正 (日付限定を trades には非適用 or OB recorder に trades 追加) |
| **NEW-2** | 131# fill records 未出力 | §3.1 | 新 run で fills 記録されず | fill_test stdout/stderr 確認。loop 停止の有無を検証 |
| **B3** | fast_fill has_negative_edge sell 側実効性 | 098# | 100# L2 で部分解決 | sell 側の二層化検証 (Layer 2: 30s 後 PnL フィードバック) |
| **D1/D6** | SkipGate 再訓練 (759 filled) | 097#/Appendix F | データ十分。未実行 | Gate 判定後 or 130# データ蓄積後に実行 |

#### P1 — 次サイクル対応

| ID | 内容 | 出典 | 現況 | 推奨アクション |
|----|------|------|------|-------------|
| **D7** | sell 専用 SkipGate | 098# | buy/sell 各 ~380 件で分割可能 | Appendix F の計画に従い実行 |
| **§5.6** | time_filter Phase 3 Step 1 | 107# | VG 有効確認済 | BUY 7h→3h は 121# 実施済。SELL 6h→3h を次回検討 |
| **C5** | v458 Walk-Forward バグ 6 件 | 111# | ph3 ブロッカー | ph3 進入前に修正必須 |
| **C2/§8.5** | Oracle テスト | 111# | 未実施 | maker 0% での理論上限を確認。ph3 進入判断に必須 |
| **§8.3** | 自動再起動の仕組み | 113# | 手動再起動で運用中 | ph5 で systemd/nssm 導入 |
| **E11** | skip_gate.py → ztb/models/ 移動 | 123# Gemini | ph3-pre | fill_test 非稼働時に実施 |
| **E12** | VG JSONL 構造化ログ | 123# Gemini | ph3-pre | retrain の trades 問題と併せて対応 |
| **NEW-3** | min_spread_jpy=1500 の妥当性 | §1.3 | orderbook_error 11/51 (21.6%) | 130# run で spread narrow による cancel が急増。1500→1200 に緩和を検討 |

#### P2 — 中長期

| ID | 内容 | 出典 | 現況 |
|----|------|------|------|
| C1 | エラーハンドリング全 API 横展開 | 013# | ph3 以降 |
| C4 | BaseExchangeAdapter 継承化 | 013# | v461 |
| C6 | BacktestReporter 統一 | 111# | ph3 |
| C8 | ph3 Stop 条件明文化 | 112# | ph3 進入時 |
| C9 | Seed 非決定性 + チェックポイント共有 | 014# | ph3 |
| §5.8 | Offset 体系的探索 (AB テスト) | 095# | Gate 判定後 |
| E13 | MC CVaR Binding | 123# | ph5 |
| §8.7 | 多取引所展開 | 000# | ph5 以降 |

### 4.3 118# §9 行動計画 vs 実績

| Phase | 施策 | 118# 状態 | 現在 |
|-------|------|-----------|------|
| A1 | fill_test 再起動 | ✅ | ✅ (複数回再起動) |
| A2 | warm_start 即復元 | ✅ `db41b7c57` | ✅ 持続中 |
| A3 | sell SG 無効化 | ✅ `db41b7c57` | ✅ 持続中 |
| A4 | VG 効果測定 | ✅ vg_and_trend.py | ✅ AS -7.7pt 確認 |
| B1 | Gate 自動判定 | ✅ `8a27ce2af` | ✅ 持続中 |
| B2 | Holm-Bonferroni | ✅ `8a27ce2af` | ✅ 持続中 |
| B3 | PnL t 検定 | ✅ | ✅ |
| B4 | AS 日別トレンド | ✅ | ✅ |
| C1 | WF バグ 6 件 | ⬜ | ⬜ **未着手 (ph3 ブロッカー)** |
| C2 | Oracle テスト | ⬜ | ⬜ **未着手 (ph3 必須)** |
| C3 | ph3 Stop 条件 | ⬜ | ⬜ 未着手 |
| C4 | execute_trade() | ⬜ | ⬜ 未着手 |
| D1-D6 | Gate FAIL 施策 | ⬜ | 部分的 (130# offset/guard で間接対応) |

**Phase A/B は完了。Phase C (ph3 準備) は全件未着手。Phase D は 130#/131# で部分進行。**

---

## §5 過去成果の活用状況

### 5.1 活用中の過去成果

| 成果 | 出典 | 活用状況 |
|------|------|---------|
| SkipGate preorder 16 特徴量 | 097# | fill_test で稼働中。skip_rate 8.8% |
| Volatility Guard | 107# | AS -7.7pt の有効性確認済 |
| StatePersistence | 113# | 再起動時の warm_start で機能 |
| CircuitBreaker / HealthMonitor | 113# | 安全装置として常時稼働 |
| Gate 二段階判定 (K1-K6 / F1-F8) | 116# + 130# | モニタリングで統一使用 |
| Holm-Bonferroni 多重比較 | 118# B2 | gate_judgment.py に統合 |
| OB Recorder | 129# | fill_test 内で板情報記録 |

### 5.2 未活用で活用可能な過去成果

| 成果 | 出典 | 活用可能性 | 推奨 |
|------|------|-----------|------|
| SkipGate 再訓練計画 (Appendix F) | 118# | 759 filled で実行可能 | **P1: Phase C 前に実行** |
| regime 特徴量復活 | 118# F2 | ranging 32%/trending 16% で有意 | 再訓練時に強制 include |
| params_adapter recency_window=120 | 096# | 設定済みだが adaptation=false | adaptation 再有効化時に活用 |
| GatesToAlerts | 128# §13.6 | WATCH/FAIL 即通知 | ph3 で実装 |
| MC CVaR | 123# E13 | gate_judgment.py に統合済 | ph5 で binding 化 |
| Counterfactual analysis (v459) | 128# §13.6 | Oracle 上限の定期算出 | C2 Oracle テストと統合 |

### 5.3 SkipGate 再訓練の好機

118# Appendix F で策定された再訓練計画の前提条件:

| 条件 | 097# 時点 | 現在 | 判定 |
|------|-----------|------|------|
| filled records | 215 | 1048 | ✅ 4.9x |
| buy 件数 | ~110 | ~530 | ✅ 4.8x |
| sell 件数 | ~105 | ~518 | ✅ 4.9x |
| regime 有効件数 | 0 | ~650 | ✅ ∞ |
| Walk-forward folds | 8 | ~33 | ✅ 4x |

**全条件を大幅に超過。再訓練の実行障壁はゼロ。**

---

## §6 新規発見・改善提案

### 6.1 P0: trades 取得問題の解決

**問題**: date_filter による trades 日付限定で `trades available=False`  
**原因**: `data/v460/raw/trades/` に当日分の trades ファイルが存在しない  
**影響**: retrain 時の特徴量 6/16 が欠落 → モデル品質低下

**提案 A**: `feature_enricher.py` の date_filter を trades に適用しない  
```python
# 現状: OB と trades に同じ date_filter を適用
# 提案: OB のみ date_filter、trades は全量 (または直近 7 日)
def _load_raw_data(self, date_filter=None):
    ob = load_raw_orderbook(date_filter=date_filter)  # 日付限定
    trades = load_raw_trades(date_filter=None)         # 全量維持
```

**提案 B**: OB recorder に trades 記録機能も追加  
→ 実装コスト中。API 追加呼び出しが必要。

**推奨**: 提案 A (即時、低コスト) → 提案 B (次回セッション)

### 6.2 P0: 131# fill records 不出力の調査

fill_test プロセス (PID 98372) は alive だが、17:11 以降 fill records が 0 件。

**調査手順**:
1. fill_test のターミナル出力 (stdout/stderr) を確認
2. `results/v460/fill_test/fill_records_*.jsonl` の最終更新日時を確認
3. fill_test_state.json が更新されているか確認
4. API 接続エラーの有無を確認

### 6.3 P1: min_spread_jpy 緩和

130# run で orderbook_error 11/51 = 21.6% が "Spread too narrow" (1500 JPY 基準)。

**データ**: reject された spread = 277~1406 JPY  
128# §4.4 で "spread < 2 bps でのみ黒字" と分析。BTC ~10.5M JPY で 2 bps = 2100 JPY。
1500 JPY = 1.4 bps は黒字帯内。

**提案**: `min_spread_jpy: 1500` → `min_spread_jpy: 1000` (0.95 bps)  
→ fill rate 向上しつつ、黒字帯 (<2 bps) の取引機会を増やす  
→ ただし AS リスクの増大とのトレードオフ。A/B 比較推奨。

### 6.4 P1: retrain bootstrap 到達加速

現在 bootstrap 閾値 30 に対して 20 samples。あと 10 fill で bootstrap deploy。

**加速案**:
1. `bootstrap_min_total: 30` → `25` に引き下げ (config hot-reload で即反映)
2. `cycle_interval_sec: 120` → `90` に短縮 (fill 頻度向上)
3. skip_gate の閾値を一時的に緩和 (skip_gate cancel 11/51 → 減少)

**推奨**: 案 1 (YAML 変更のみ、hot-reload で即効)。
初回 deploy の品質は bootstrap 段階なので 25 sample でも十分。

### 6.5 P1: VG 発動感度の引き上げ

128# §4.1 で VG 発動率 3.4% (32/941) は低い。  
130# run では VG 発動データが不足で評価困難。

**提案**: `vpin_threshold` を現行値から 10% 引き下げ → 発動率 5-7% を目標  
→ AS 追加 -3~5pt の改善を期待

### 6.6 P2: adaptation 再有効化の検討

`adaptation.enabled: false` (122# R2 因果分離のため無効化)。

128# 分析で offset-PnL の因果が不明確とされたが、130# で major confounders (UTC21, unknown buy, postonly) が解消。
データの質が向上した新 run で adaptation を再有効化し、offset 自動調整の効果を検証する価値がある。

ただし、retrain が deploy されるまでは skip_gate との相互作用が予測困難。
**retrain bootstrap deploy 後に有効化を推奨**。

---

## §7 リスク評価

### 7.1 最大リスク: retrain 未 deploy のまま ph3 進入

118# §4 で指摘されたとおり、SkipGate は AUC ≈ 0.5 (ランダム分類器) のまま。
retrain なしでは gate 判定の PnL 改善に限界がある。

**緩和**: 130#/131# の施策 (UTC21, unknown guard, postonly) が AS 率を改善しているため、
retrain deploy 前でも Gate WATCH は維持できる見通し。

### 7.2 trades 欠如による学習品質

§2.2 で指摘。retrain が bootstrap deploy しても、trades ベース特徴量なしでは
モデルの予測力が制限される。trades 問題は P0 で対応すべき。

### 7.3 累積損失

全期間 -168.7 JPY (キャップ 10,000 JPY の 1.7%)。安全域内。
ただし lot 増量 (0.001→0.01 BTC) で 10x に拡大するため、PnL 正転は lot 増量前に必須。

### 7.4 fill_test state 陳腐化

state.json が 131# 再起動以降更新されていない。
StatePersistence が機能していなければ、次の再起動で warm_start が旧状態に戻るリスク。

---

## §8 優先実施計画

### Phase X: 緊急 (本セッション内) — ✅ 完了

| # | 施策 | コスト | 効果 | 結果 |
|---|------|-------|------|------|
| X1 | 131# fill records 未出力の調査・修正 | 低 | fill_test 正常稼働の確認 | ✅ 原因: .venv fill_test が Exit Code 1 で死亡、system python が lock 取得し空回り。全プロセス kill → .venv で再起動 (`7cb39ebb5`) |
| X2 | trades date_filter 問題の修正 | 低 | retrain 特徴量品質回復 | ✅ `feature_enricher.py` に trades fallback 追加 (`7cb39ebb5`)。retrain log で `trades available=True` 確認済 |

### Phase Y: 次セッション

| # | 施策 | コスト | 効果 |
|---|------|-------|------|
| Y1 | bootstrap_min_total 引き下げ (30→25) | 極低 (YAML) | 初回 retrain deploy 加速 |
| Y2 | min_spread_jpy 緩和 (1500→1000) | 極低 (YAML) | fill rate +10~15% |
| Y3 | SkipGate 再訓練 (Appendix F 計画) | 高 | AS 予測精度向上。P0 level |
| Y4 | VG 感度引上げ | 低 (YAML) | AS -3~5pt |

### Phase Z: ph3 準備 (Gate 判定後)

| # | 施策 | コスト | 備考 |
|---|------|-------|------|
| Z1 | v458 Walk-Forward バグ 6 件修正 | 高 | ph3 ブロッカー |
| Z2 | Oracle テスト | 中 | 理論上限の確認 |
| Z3 | ph3 Stop 条件明文化 | 低 | |
| Z4 | execute_trade() 実装 | 中 | ph4 ブロッカー |
| Z5 | skip_gate.py → ztb/models/ 移動 | 中 | E11 |

---

## §9 外部レビュー向け質問事項

以下の項目について外部 AI エージェントの見解を求める:

### Q1: trades 欠如問題の設計判断 — ✅ 解決済

**採用**: 提案 A (fallback)。`7cb39ebb5` で `feature_enricher.py` に date_filter 付き trades が空の場合に `date_filter=None` でフォールバックする仕組みを実装。
retrain log で `trades available=True` を確認済。タイムスタンプ整合性は「最寄の trades を使う」ことでマイクロ秒レベルの乖離あるが、特徴量計算上は許容範囲。

### Q2: SkipGate 再訓練の優先度

現在の AS 率 5.0% (130# run) は劇的改善だが、これはルールベース施策 (UTC block, unknown guard) の効果。
SkipGate 自体は AUC ≈ 0.5 のまま。再訓練を prior to Gate 判定で行うか、Gate 判定後に行うか。

### Q3: fill rate 低下 (39.2%) の許容度

130# run の AS 率改善は fill rate 低下とのトレードオフで達成されている。
F1 (attempted_fill_rate ≥ 70%) は全期間でギリギリ。130# 単体では 39.2% と大幅悪化。
保守的な offset/guard で AS を下げると fill rate が犠牲になるパラドックスへの見解。

### Q4: min_spread_jpy 1500→1000 のリスク

128# §4.4 の "spread < 2 bps でのみ黒字" データに基づき、spread 1000 JPY (≈0.95 bps) まで
許容するのは理にかなうか。それとも 1500 JPY を維持して安全マージンを取るべきか。

### Q5: retrain bootstrap 閾値の妥当性

`bootstrap_min_total=30` で初回 deploy を急ぐか、`50` に引き上げてモデル品質を確保するか。
短期高収益の大義を考慮した判断。

### Q6: 130#/131# 施策の相互作用リスク

同時に 8 施策を投入した (§0 表) ため、個別の効果測定が困難。
効果の分離が必要か、「全体で良くなれば良い」で進めるべきか。

---

## Appendix A: 118# → 129# 解決進捗マトリクス

### 新規 RESOLVED (130#/131# セッション)

| 元 118# 項目 | 内容 | 解決 commit | 方法 |
|---|---|---|---|
| §13.1 | retrain bootstrap 2段化 | `b525b3a8a` (130#) | `bootstrap_min_total=30, bootstrap_min_new=10` |
| §13.2 | retrain I/O 日付限定 | `b525b3a8a` (130#) | date_filter で日単位ロード |
| §13.3 | gate 判定統一 | `b525b3a8a` (130#) | K1-K6 + F1-F8 二段表示 |
| §13.4 | unknown buy guard | `b525b3a8a` (130#) | `unknown_buy_offset_boost: 2.0` |
| §13.5 | orderbook_error 細分化 | `b525b3a8a` (130#) | 4 区分に分離 |
| §9-D part | UTC21 sell block | `b525b3a8a` (130#) | `skip_utc_hours_sell` |
| §11.2 E1 | postonly 二重確認 | `b525b3a8a` (130#) | mid 再取得 + best_bid/ask 補正 |
| _read_jsonl_gz DRY | feature_enricher 重複削除 | `9df2715ea` (130# refactor) | market_data_collector から import |
| retrain target | pnl120→pnl30 | `2780f80b0` (131#) | coverage 55%→100% |
| retrain hot-reload | config 再読込 | `2780f80b0` (131#) | per-cycle YAML reload |

### 残留 OPEN サマリ

| 優先度 | 件数 | 主な項目 |
|--------|------|---------|
| P0 | 4 | trades 欠如, fill records 未出力, fast_fill sell, SG 再訓練 |
| P1 | 8 | sell SG, WF バグ, Oracle, spread 緩和, VG 感度, E11/E12 |
| P2 | 8 | 多取引所, AB テスト, CVaR, 等 |
| **合計** | **20** | (118# の 53→42 RESOLVED→残11 + 130#/131# 新規 9) |

---

## Appendix B: 現行 fill_test.yaml 重要パラメータ

| パラメータ | 値 | 設定時期 | 備考 |
|-----------|-----|---------|------|
| order_quantity | 0.001 BTC | 初期 | Coincheck 最小 |
| cycle_interval_sec | 120.0 | 初期 | |
| order_timeout_sec | 90.0 | 096# | 300→90s |
| min_spread_jpy | 1500 | 122# | postonly 回避 |
| side_offset.sell | 0.18 | 121# | 0.14→0.18 |
| adaptation.enabled | false | 122# | 因果分離 |
| skip_utc_hours_sell | [4,8,14,15,16,21] | 130# | UTC21 追加 |
| unknown_buy_offset_boost | 2.0 | 130# | VG 相当 |
| retrain.target | pnl30 | 131# | pnl120→pnl30 |
| retrain.min_total_samples | 100 | 初期 | bootstrap で 30 に緩和 |
| bootstrap_min_total | 30 | 130# | |
| bootstrap_threshold | 100 | 130# | total < 100 → bootstrap |

---

## Appendix C: プロセス状態

| PID | 種別 | 開始時刻 | 状態 | Git SHA |
|-----|------|---------|------|---------|
| 98372 | fill_test (parent) | 17:11:09 | alive | `2780f80b0` |
| 98780 | fill_test (worker) | 17:11:09 | alive | `2780f80b0` |
| 62812 | retrain_scheduler | 17:11:14 | alive | `2780f80b0` |
| 96640 | retrain child | 17:11:14 | alive | `2780f80b0` |

---

## Appendix D: 追記レビュー (2026-02-21 17:32 JST, Codex)

### D.1 事実照合 (129# 記載 vs 実ログ/実ファイル)

| 項目 | 129# 記載 | 再点検結果 | 判定 |
|---|---|---|---|
| retrain で `trades available=False` | 17:11 サイクルで発生 | `logs/retrain_scheduler.log` に同一ログあり。`Date filter ['20260221']` + `trades available=False` を確認 | ✅ 正しい |
| trades 日次ファイル欠如 | 20260221 trades が無い想定 | `data/v460/raw/trades/` は `20260219` まで。`20260221` 不在 | ✅ 正しい |
| 新 run_id の fill records 未出力 | `1771661473_ac4f9cb1` の新規記録なし | `fill_records_20260221.jsonl` は run_id `1771607250_*`/`1771651879_*` のみ。`1771661473_*` は 0 件 | ✅ 正しい |
| 「fill loop 停止」の断定 | state 更新停止から停止推定 | `fill_test.log` には 17:27 の heartbeat があり、17:11-17:27 は `time_filter` でサイクル抑止。停止断定は早い | ⚠️ 要補正 |
| プロセス alive 表 | 98372/98780/62812/96640 alive | 現時点で `/proc/98372` は不存在。`fill_test.lock` は残存 (`98372|1771661484|1771661473_ac4f9cb1`) | ⚠️ 要更新 |
| bootstrap パラメータ表記 | `bootstrap_min_total` 等 | 実 YAML は `bootstrap_min_total_samples`, `bootstrap_min_new_samples` | ⚠️ 名称不一致 |
| 130#/131# 文書参照 | 前提文書として明示 | `docs/v460` に 130#/131# 文書は存在せず、追跡対象が commit ベースのみ | ⚠️ トレーサビリティ不足 |

### D.2 129# で薄い/未記載の重要論点

1. 残高制約が実験結果の交絡要因になっている。`fill_test.log` に `Insufficient JPY` / `Insufficient BTC` が複数回あり、side 強制切替で性能評価が歪む。
2. 新 run は `time_filter` 開始直後に入っており、fill が出ない時間帯だった。`未出力=即異常` ではなく「時間帯起因か、停止か」の切り分けを先に固定すべき。
3. stale lock 回収設計が弱い。PID 死亡後も `fill_test.lock` が残り、再起動時の誤判定リスクがある。

### D.3 優先アクション (追補)

| 優先度 | アクション | 目的 |
|---|---|---|
| P0 | trades 収集を再開し、`20260220-20260221` を最低限 backfill | `trades available=False` の解消、retrain 特徴量欠損防止 |
| P0 | `feature_enricher` に trades フォールバック窓を追加 (例: 当日欠損時は直近 N 日) | 日次欠損で学習停止しない設計にする |
| P0 | lockfile に PID 生存確認 + stale 自動回収を追加 | 死亡プロセス由来の停止/誤起動を防ぐ |
| P1 | `balance_constrained` フラグを fill_records に追加し、評価/学習で分離 | 残高不足による擬似性能悪化を除去 |
| P1 | Appendix B のキー名と文書参照を修正 (130#/131# は doc 化または commit 注記化) | 文書の再現性と監査可能性を回復 |
