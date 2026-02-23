# 132# fill_test 9日間ログ解析 + YAML外部化 + 改善計画

> **前提**: 131# (Appendix A-D) が肥大化したため独立文書として起票。
> 別 AI コーディングエージェントによるレビュー対象。

> **Git HEAD**: `0642682a8` (131# Appendix D コミット後)
> **対象期間**: 2026-02-13 〜 2026-02-21 (9日間, 1722 レコード)
> **BTC 価格帯**: ~10,550,000 JPY

---

## §0 エグゼクティブサマリ

### 現況スナップショット (fill_test_state.json)

| 指標 | 値 | 目標 | 判定 |
|------|-----|------|------|
| cycle_count | 1,700 | — | — |
| fill_rate | 63.7% (1,096/1,722) | ≥ 70% | **NG** |
| PnL 30s | -0.293 bps (mean) | ≥ 0 | **NG** |
| PnL 120s | +0.178 bps (mean) | ≥ 0 | OK (but n=371) |
| AS rate | 不明 (分析バグ) | ≤ 25% | **計測不能** |
| cumulative_pnl_jpy | -198.96 JPY | ≥ 0 | **NG** |
| regime_confirmed | ranging | — | 安定 (stability=9) |
| retrain deploys | 0/26 | ≥ 1 | **NG** |
| win_rate 30s | 46.9% | ≥ 50% | **NG** |
| run_id 数 | 21 (9日間) | ≤ 5 | **NG** (過剰再起動) |

### 発見された重大問題

| # | 分類 | 概要 | 影響 |
|---|------|------|------|
| **F1** | **バグ** | 分析スクリプト: フィールド名 4 件ずれ → AS/FFD/VG 統計が全て 0 | 意思決定の土台崩壊 |
| **F2** | **バグ** | retrain: `new_samples` が負値 (-831) → deploy 永久不能 | モデル改善停止 |
| **F3** | **性能** | feature_enricher: trades 全件 (440万行) 毎時ロード | retrain 毎に ~30s の I/O waste |
| **F4** | **設計** | SkipGate/TimeFilter/VG スキップが FillRecord に未記録 | 分析に欠損 |
| **F5** | **設計** | YAML 未外部化のマジックナンバー 15 件以上 | チューニング不能 |
| **F6** | **傾向** | sell PnL が -0.554 bps (buy の 15 倍悪い) | 損失の主要因 |
| **F7** | **傾向** | 最新 run の fill_rate=46%, PnL=-1.24 bps | 直近で急速劣化 |
| **F8** | **傾向** | unknown レジームの PnL=-0.891 bps (最悪) | D1 ガードは実装済みだが未有効 |

---

## §1 ログ解析結果

### §1.1 全体統計

```
Total records: 1,722      (9 files, 2026-02-13 ~ 2026-02-21)
Filled: 1,096  Skipped: 626
Fill rate: 63.6%
```

### §1.2 PnL タイムフレーム別

| TF | n | mean (bps) | win_rate |
|----|---|-----------|----------|
| 30s | 1,096 | **-0.293** | 46.9% |
| 60s | 371 | **-0.149** | 48.0% |
| 120s | 371 | **+0.178** | 49.6% |

**考察**: 短期 (30s) は負だが 120s で回復傾向。mean reversion の種は存在するが、
30s 計測で判断すると負に見える。target=pnl30 の retrain は短期ノイズを学習している可能性。

### §1.3 Side 別

| side | n | PnL 30s (bps) | fill_rate |
|------|---|---------------|-----------|
| buy | 553 | **-0.036** | 63.3% |
| sell | 543 | **-0.554** | 64.0% |

**sell PnL = -0.554 bps は buy の 15 倍悪い**。sell_offset=0.18 でも改善不十分。

### §1.4 レジーム別

| regime | n (全体) | n (filled) | PnL 30s (bps) |
|--------|---------|-----------|---------------|
| ranging | 722 (41.9%) | 533 | **-0.177** |
| n/a | 355 (20.6%) | 267 | **-0.462** |
| trending | 273 (15.9%) | 203 | **-0.100** |
| None | 256 (14.9%) | — | (init期間) |
| unknown | 116 (6.7%) | 93 | **-0.891** |

**unknown: -0.891 bps はレジーム有の 5 倍悪い**。131# D1 のレジームガードは必須。

### §1.5 時間帯 (UTC) 最悪 6

| UTC | JST | n | PnL 30s (bps) | 備考 |
|-----|-----|---|--------------|------|
| 08 | 17 | 15 | **-3.805** | skip_utc_hours_buy に含まれる |
| 16 | 01 | 15 | **-2.817** | skip_utc_hours に含まれる |
| 14 | 23 | 21 | **-2.566** | skip_utc_hours_sell に含まれる |
| 21 | 06 | 42 | **-1.136** | skip_utc_hours_sell に含まれる |
| 18 | 03 | 23 | **-1.132** | skip_utc_hours_buy に含まれる |
| 13 | 22 | 40 | **-0.887** | **未遮断** (n=40, 要検討) |

**UTC13 (JST22)**: n=40, -0.887 bps。統計的に有意 (n 十分)。
skip_utc_hours_sell に含めるか、SkipGate 委譲を検討。

### §1.6 日別推移

| 日付 | total | filled | FR | PnL 30s (bps) |
|------|-------|--------|-----|---------------|
| 02-13 | 211 | 163 | 77% | -0.441 |
| 02-14 | 220 | 161 | 73% | -0.724 |
| 02-15 | 60 | 49 | 82% | -0.875 |
| 02-16 | 21 | 14 | 67% | -1.123 |
| **02-17** | **205** | **137** | **67%** | **+0.449** |
| **02-18** | **277** | **149** | **54%** | **+0.353** |
| 02-19 | 250 | 176 | 70% | -0.552 |
| 02-20 | 217 | 132 | 61% | -0.198 |
| 02-21 | 261 | 115 | 44% | -0.563 |

**02-17, 02-18 が唯一のプラス日**。それ以外は全てマイナス。
最新日 02-21 は fill_rate=44% まで悪化 — run_id 断片化 (21 runs) の影響が疑われる。

### §1.7 Run ID 別 (上位 5)

| run_id | n | FR | PnL 30s (bps) | 備考 |
|--------|---|-----|--------------|------|
| `3afba87b` | 122 | 61% | **+1.353** | ベスト |
| `4f513c12` | 136 | 87% | **+0.366** | 高 FR |
| `b7d09bbf` | 338 | 57% | **+0.241** | 最多サンプル |
| `481369d6` (最新) | 85 | 46% | **-1.243** | ワースト |
| `a9646841` | 27 | 52% | +0.259 | 小サンプル |

**最新 run が最悪** — ロジック変更かタイミングか要調査。

### §1.8 その他指標

| 指標 | 値 |
|------|-----|
| Queue wait (median) | 12.8 秒 |
| Fast fill (≤5s) | 0/1,096 (0%) |
| Repriced (stale order) | 67/1,096 (6.1%) |
| Max consecutive same side | 10 |
| balance_forced_switch | 16/1,722 (0.9%) |

**fast_fill = 0%**: threshold_sec=5.0 以下の即約定は皆無
→ fast_fill_defense は事実上不発 (threshold 引き上げ or 閾値見直し)。

---

## §2 発見バグ一覧

### F1: 分析スクリプトのフィールド名不一致 (P0)

分析コードと `FillRecord` データクラスの属性名が 4 箇所でずれており、
skip_reason / AS / FFD / VG の統計が全て誤値 (0) を返す。

| 分析コードの参照名 | FillRecord 実属性 | 影響 |
|-------------------|------------------|------|
| `adverse_selection` | `adverse_selected` | AS 率が常に 0% と表示 |
| `skip_reason` | `cancel_reason` | skip 理由が全て "unknown" |
| `fast_fill_defense_active` | `ffd_boost_active` | FFD 発動件数 0 |
| `volatility_guard` | `vg_triggered` | VG 発動件数 0 |

**修正方針**: 分析スクリプト 2 本のフィールド参照を修正。

### F2: retrain new_samples 負値バグ (P0)

```python
# retrain_scheduler.py L1008
new_samples = len(X_valid) - prev_n_samples  # = 27 - 858 = -831
```

`latest_run_only=true` の場合、現 run のサンプル数 (27) が
前回モデル学習時のサンプル数 (858, 別 run) を下回り **負値**。
→ `insufficient new samples: -831 < 10` で永久スキップ。

**修正方針**: `latest_run_only` 時は `prev_n_samples` を現 run の前回学習時の値に限定するか、
`new_samples = max(0, len(X_valid) - prev_n_samples)` として負値を防止。

### F3: trades 全件ロード (P1)

```
130# F7: trades empty with date_filter, falling back to recent ±1 days
130# F7: still empty after ±day fallback, loading all trades
Loaded 4396171 trades (days=all)
```

date_filter → ±1日 → 全件という 3 段フォールバックで最終段に到達。
4.4M 行のロードに ~25 秒、毎時発生。

**修正方針**: (a) 全件フォールバック無効化オプション追加、(b) max_trades_limit、
(c) trades ファイルの日付不一致を修正 (根本原因)。

### F4: SkipGate/TimeFilter スキップの FillRecord 未記録 (P2)

time_filter / SkipGate でスキップされたサイクルは FillRecord を生成せず `continue`。
→ 626 件の "skipped" は `filled=False` + 不明な `cancel_reason` で、
skip 種別が分析上不可視。

**修正方針**: スキップ時も `filled=False, cancel_reason="time_filter_buy"` 等の
FillRecord を記録し、skip 種別ごとの集計を可能にする。

---

## §3 YAML 外部化すべき設定項目

### §3.1 高優先度 (retrain パイプライン)

| # | ファイル | 現在のハードコード | 提案 YAML キー | 値 |
|---|---------|------------------|---------------|-----|
| Y1 | retrain_scheduler.py L366 | `subsample=0.8` | `retrain.lgbm_subsample` | `0.8` |
| Y2 | retrain_scheduler.py L367 | `colsample_bytree=0.8` | `retrain.lgbm_colsample_bytree` | `0.8` |
| Y3 | retrain_scheduler.py L368 | `reg_alpha=1.0` | `retrain.lgbm_reg_alpha` | `1.0` |
| Y4 | retrain_scheduler.py L369 | `reg_lambda=1.0` | `retrain.lgbm_reg_lambda` | `1.0` |
| Y5 | retrain_scheduler.py L370 | `random_state=42` | `retrain.lgbm_random_state` | `42` |
| Y6 | retrain_scheduler.py L371 | `n_jobs=1` | `retrain.lgbm_n_jobs` | `1` |
| Y7 | retrain_scheduler.py L530,682 | `np.percentile(preds_test, 20)` | `retrain.skip_percentile` | `20` |
| Y8 | retrain_scheduler.py L131 | `absolute_min_score: -0.10` | (既にYAML) → **bootstrap用別値** | `retrain.bootstrap_absolute_min_score: -0.50` |

### §3.2 中優先度 (feature_enricher, regime, etc.)

| # | ファイル | 現在のハードコード | 提案 YAML キー | 値 |
|---|---------|------------------|---------------|-----|
| Y9 | feature_enricher.py L26 | `_TRADE_WINDOW_SEC = 60` | `retrain.trade_window_sec` | `60` |
| Y10 | feature_enricher.py L27 | `_OB_MATCH_TOLERANCE_SEC = 5` | `retrain.ob_match_tolerance_sec` | `5` |
| Y11 | feature_enricher.py L254 | `[30, 300]` | `retrain.multi_tf_windows` | `[30, 300]` |
| Y12 | ob_recorder.py L22 | `_FLUSH_INTERVAL_SEC = 60` | `ob_recorder.flush_interval_sec` | `60` |
| Y13 | skip_gate_evaluator.py L34 | `_HOT_RELOAD_CHECK_INTERVAL_SEC = 120` | `skip_gate.hot_reload_interval_sec` | `120` |
| Y14 | side_selector.py L138 | `freeze_side(side, cycles=3)` | `side_selector.freeze_cycles` | `3` |

### §3.3 低優先度 (regime 信頼度パラメータ)

| # | ファイル | 現在のハードコード | 提案 YAML キー |
|---|---------|------------------|---------------|
| Y15 | regime_detector.py L200 | HIGH_VOL 信頼度 `0.6 + excess * 0.4` | `regime.confidence_high_vol_base/scale` |
| Y16 | regime_detector.py L207 | TRENDING 信頼度 `0.5 + excess * 0.3` | `regime.confidence_trending_base/scale` |
| Y17 | regime_detector.py L213 | RANGING 信頼度 `0.4 + proximity * 0.4` | `regime.confidence_ranging_base/scale` |
| Y18 | regime_detector.py L122 | buffer `window * 3` | `regime.buffer_multiplier` |

---

## §4 改善計画

### Phase 1: 計測基盤修復 (必須・先行)

> **目的**: 正確な計測なくして改善は不可能。F1-F4 の修正が全ての前提。

| # | タスク | 影響度 | 工数 |
|---|--------|--------|------|
| **A1** | F1 修正: 分析スクリプト 4 フィールド名修正 | 高 | 小 |
| **A2** | F2 修正: new_samples 負値バグ | 高 | 小 |
| **A3** | F4: skip 時の FillRecord 記録拡充 (cancel_reason 設定) | 中 | 中 |
| **A4** | 分析スクリプトを正しいフィールドで再実行、真の AS/FFD/VG 統計取得 | 高 | 小 |

### Phase 2: YAML 外部化 + retrain 有効化

> **目的**: retrain を deploy 可能にし、モデル改善ループを始動。

| # | タスク | 影響度 | 工数 |
|---|--------|--------|------|
| **B1** | §3.1 Y1-Y8: retrain LGBM ハイパラ + skip percentile YAML化 | 中 | 小 |
| **B2** | §3.2 Y9-Y14: enricher/OB/SG/side_selector YAML化 | 低 | 中 |
| **B3** | F2 修正後の retrain 再実行 → 初回 deploy 目標 | 高 | 小 |
| **B4** | F3: trades 全件ロード抑制 (max_rows or 全件FB無効化) | 中 | 小 |
| **B5** | bootstrap_absolute_min_score 導入 (bootstrap段階で緩い品質ゲート) | 中 | 小 |

### Phase 3: PnL 改善施策

> **目的**: PnL 30s を ≥ 0 bps に持ち上げ、方策 B (動的ロット) 有効化条件を達成。

| # | タスク | 根拠 | 期待効果 |
|---|--------|------|---------|
| **C1** | sell offset 再調整: 0.18 → 0.22 or sell 時間帯追加遮断 | §1.3: sell -0.554bps | sell PnL 改善 +0.2~0.3 bps |
| **C2** | UTC13 (JST22) sell 遮断追加 | §1.5: n=40, -0.887bps | PnL +0.05 bps (全体) |
| **C3** | retrain target 検討: pnl30 → pnl120 再評価 | §1.2: 120s は +0.178bps | 長期リバージョン活用 |
| **C4** | unknown regime ガード有効化 (131# D1) + 動的ロット有効化条件監視 | §1.4: unknown -0.891bps | PnL +0.05~0.1 bps |
| **C5** | fast_fill_defense 閾値見直し: 0%発動 → 実効性調査 | §1.8: fast_fill=0 | (調査のみ) |

### Phase 4: 安定性・運用改善

| # | タスク | 根拠 |
|---|--------|------|
| **D1** | run_id 断片化抑制: 再起動の原因特定 (21 runs/9日 → 目標 ≤ 3 runs/9日) | §1.7: 過剰再起動 |
| **D2** | E3 サンプリング比率見直し: 0.50 → 0.70 (120s PnL カバレッジ拡大) | §1.2: 120s n=371 |
| **D3** | §3.3 Y15-Y18: regime 信頼度パラメータ YAML化 | §3.3 |

---

## §5 実装優先順位

```
Phase 1 (A1-A4) ─→ Phase 2 (B1-B5) ─→ Phase 3 (C1-C5) ─→ Phase 4 (D1-D3)
     計測修復          retrain始動          PnL改善            安定性
     [即実行]         [P1完了後]          [P2完了後]          [並行可]
```

### 依存関係

```
A1 (フィールド名修正) → A4 (真の統計) → C1-C5 (PnL施策根拠)
A2 (new_samples修正) → B3 (retrain deploy) → C3 (target再評価)
B1 (YAML化) → B5 (bootstrap緩和) → B3 (deploy)
```

### 見積工数

| Phase | タスク数 | 見積 | テスト |
|-------|---------|------|--------|
| Phase 1 | 4 | 1h | 既存 + 分析再実行 |
| Phase 2 | 5 | 2h | YAML読込テスト |
| Phase 3 | 5 | 3h | fill_test ドライラン |
| Phase 4 | 3 | 2h | 運用監視 |

---

## §6 レビュー依頼事項

本ドキュメントを別 AI コーディングエージェントにレビューして頂く際、以下の観点での批評を期待:

1. **分析の妥当性**: §1 の統計解釈に誤りや見落としはないか
2. **バグ優先度**: §2 の P0/P1/P2 の判定は妥当か
3. **YAML 外部化**: §3 の項目選定は適切か、過不足はないか
4. **改善計画**: §4 の施策順序・依存関係に論理的欠陥はないか
5. **見落とし**: ログから読み取れるが本文書で言及されていない改善機会はないか
6. **リスク**: 提案施策の副作用・リスクの見落としはないか
7. **大義との整合**: 「短期間での高収益性システム」という大義に対し、本計画は最短経路か

### 批判的にレビューすべき点

- PnL 30s が負なのに retrain target を pnl30 にしている矛盾 (C3)
- sell offset 0.18 でも -0.554bps → offset 以外の根本原因はないか
- 21 runs/9日の再起動原因が不明のまま Phase 3 に進む是非
- unknown レジーム 6.7% で -0.891bps → ガード有効化だけで十分か
- trades 全件ロードが日付不一致の症状なら、根本原因は何か

---

## §7 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `scripts/v460/analysis/analyze_fill_records.py` | **新規**: fill_records 包括分析 |
| `scripts/v460/analysis/analyze_fill_detail.py` | **新規**: 詳細分析 (daily/run_id/regime) |
| `docs/v460/132_ph2_rpt_fill_test_log_analysis.md` | **新規**: 本ドキュメント |

---

## Appendix A: fill_test.yaml 全設定キー一覧 (現行)

<details>
<summary>クリックで展開 (336行)</summary>

セクション一覧:
- 基本設定 (symbol, order_quantity, cycle_interval_sec, etc.)
- スプレッド比例オフセット (CM-1)
- リトライ (CM-2)
- AS 判定 (CM-3)
- 保存 (batch_size, max_save_retries)
- ログ (progress_log_interval, log_max_bytes)
- 方策 A: パラメータ適応 (adaptation)
- 方策 B: 動的ロットサイジング (lot_sizing)
- レジーム検知 (regime)
- 時間帯フィルター (time_filter)
- E3 サンプリング (e3)
- side 別 offset (side_offset)
- 即約定防御 (fast_fill_defense)
- stale order (stale_order)
- 安全設計 (safety)
- Orderbook Imbalance (imbalance) — **無効**
- Smart Side Selection (smart_side) — **無効**
- テール損失カット (early_exit) — **無効**
- Spread 適応型 Offset (spread_adaptive)
- SkipGate ML フィルター (skip_gate)
- Volatility Guard (volatility_guard)
- sell ガード (sell_guard)
- チューニング (tuning)
- SkipGate 再学習 (retrain)

**合計**: ~90 設定キー (有効), ~15 無効セクション内のキー

</details>

## Appendix B: 提案 YAML 追加キー (§3 まとめ)

```yaml
# ---- §3.1 (B1): retrain LGBM 追加ハイパラ ----
retrain:
  lgbm_subsample: 0.8
  lgbm_colsample_bytree: 0.8
  lgbm_reg_alpha: 1.0
  lgbm_reg_lambda: 1.0
  lgbm_random_state: 42
  lgbm_n_jobs: 1
  skip_percentile: 20
  bootstrap_absolute_min_score: -0.50

# ---- §3.2 (B2): enricher / OB / SG / side_selector ----
retrain:
  trade_window_sec: 60
  ob_match_tolerance_sec: 5
  multi_tf_windows: [30, 300]

ob_recorder:
  flush_interval_sec: 60

skip_gate:
  hot_reload_interval_sec: 120

side_selector:
  freeze_cycles: 3
```
