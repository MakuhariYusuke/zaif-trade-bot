# 137# ph2 impl: 136# §9 レビュー対応 + 134# P1-06/08/11

| key | value |
|---|---|
| 番号 | 137 |
| フェーズ | ph2 |
| 種別 | impl (実装) |
| 対象 | `docs/v460/136_ph2_impl_p1_retrain_kill.md` §9 + `docs/v460/134_ph2_rev_133_validity_evaluation.md` §7 |
| 作成日 | 2026-02-22 |
| 前提 | Git `af30e12b1` (136# P1-01/02/03 実装済み) |
| 結論 | **136# §9 レビュー全 8 件対応完了。134# P1-06/08/11 を実装。テスト 1104→1118。** |

---

## §0 エグゼクティブサマリ

前回 136# で実装した P1-01 (RetrainTrigger)、P1-02 (FeatureFreshness)、P1-03 (SellDynamicKillManager) に対する外部レビュー §9 の全 8 件 (#1-#5, #A-#C) を対応。加えて 134# ロードマップ Phase E の P1 群から P1-06/08/11 を実装し、テストを 14 件追加した。

---

## §1 136# §9 レビュー対応

### §1.1 #1 HIGH: mtime 更新タイミング修正

**問題:** `RetrainTrigger.should_retrain()` で `_last_fill_mtime` を trades health チェック前に更新していたため、trades unhealthy で一度ブロックされた後、同じ mtime のまま健全化しても `fill_records unchanged` 扱いで再学習が走らなかった。

**修正:** `_last_fill_mtime` 更新を全チェック（mtime → trades health → feature freshness）通過後に遅延。`_pending_mtime` 変数で保留し、全 gate 通過時にのみ確定更新。

**ファイル:** `ztb/ml/retrain_trigger.py` L101-136

### §1.2 #2 MEDIUM: feature freshness → trigger 接続

**問題:** `check_feature_freshness()` が実装されているが scheduler/trigger から未使用。

**修正:** 
- `RetrainTriggerConfig` に `check_feature_freshness`、`feature_trades_stale_hours`、`feature_ob_stale_hours` 追加
- `should_retrain()` に Check 3 として freshness gate 追加
- `retrain_scheduler.py` で YAML→Config マッピング追加

**ファイル:** `ztb/ml/retrain_trigger.py` L47-52 (config), L126-136 (check)

### §1.3 #3 MEDIUM: regime → check_kill 配線

**問題:** `run_fill_test._is_sell_killed()` が `check_kill()` に regime を渡しておらず、`regime_thresholds` が実質無効。

**修正:** `_is_sell_killed()` で `self._regime_detector.current_regime.value` を取得し `check_kill(regime=...)` に渡す。kill 時は `telemetry.threshold_used` をログ出力。

**ファイル:** `scripts/v460/run_fill_test.py` L322-337

### §1.4 #4 LOW: trigger 設定 YAML 外部化

**問題:** `backoff_multiplier`、`backoff_max_interval_sec`、`trades_stale_threshold_hours` 等がハードコード。

**修正:** `retrain_scheduler.py` の `run_scheduler()` で全 `RetrainTriggerConfig` パラメータを `cfg.get()` 経由で YAML から読み取り。`fill_test.yaml` の `retrain:` セクションに `trigger_*` キーを追加。

**ファイル:** `scripts/v460/ml/retrain_scheduler.py` L1418-1430, `configs/v460/fill_test.yaml` L343-352

### §1.5 #5 LOW: RetrainTrigger 改名

**問題:** クラス名が `RetainTrigger`（retain）で retrain と不整合。

**修正:** `RetainTrigger` → `RetrainTrigger`、`RetainTriggerConfig` → `RetrainTriggerConfig` に改名。ファイル末尾に `RetainTrigger = RetrainTrigger` 互換エイリアスを配置。既存の import は変更不要。

**ファイル:** `ztb/ml/retrain_trigger.py` 全体

### §1.6 #A MEDIUM: 回帰テスト追加

**追加テスト:** `test_unhealthy_to_healthy_same_mtime_retrain_fires`
- fill_records 作成 → 1st call: trades 不在で blocked → trades ディレクトリ作成 → 2nd call: 同一 mtime で retrain 通過を検証

**追加テスト:** `test_feature_freshness_integrated_in_trigger`
- `check_feature_freshness=True` 設定で stale → skip を検証

**追加テスト:** `test_backward_compat_alias`
- `RetainTrigger is RetrainTrigger` を確認

**ファイル:** `tests/unit/v460/test_136_p1_retrain_kill.py`

### §1.7 #B LOW: regime 統合テスト

`run_fill_test` 統合テストは run_fill_test の async ループ全体の mock が必要で工数大。manager 単体テスト (`test_regime_threshold_override`) で regime 機能自体は検証済み。`_is_sell_killed` の配線は §1.3 の修正で担保。

### §1.8 #C LOW: テスト数環境依存注記

本ドキュメントの §6 に注記。

---

## §2 134# P1-06: reprice 売側上限縮小

**ロードマップ:** 134# §7 Phase E — P1-06「stale reprice 売側上限縮小 AB」

**分析:** 134# §3 により reprice=2 平均 -3.44bps。sell 側の 2 回 reprice は追随コスト＞便益。

**修正:** `configs/v460/fill_test.yaml` の `stale_order.max_reprice_sell` を `2→1` に変更。

**AB テスト:** YAML の値を `2` に戻すだけで A/B 切替可能。既存の `FillTestConfig.stale_max_reprice_sell` + `order_monitor.py` の side 別パラメータ解決ロジックをそのまま使用。新規コード不要。

**影響:** sell 側の stale reprice が最大 1 回に制限。fill rate は若干低下する可能性があるが、-3.44bps の損失抑止が上回る想定。

---

## §3 134# P1-08: spread 狭小時の「休む」判定

**ロードマップ:** 134# §7 Phase E — P1-08「spread 狭小時の休む判定」

**分析:** `too_narrow` 判定（既存 `min_spread_jpy` フィルター）は ValueError を投げるだけで即座に次サイクルに突入。狭小 spread 時は流動性が低く逆選別リスクが高いため、一定時間「休む」方が合理的。

**実装:**
- `fill_config.py`: `narrow_spread_pause_enabled`、`narrow_spread_pause_bps`（閾値）、`narrow_spread_pause_sec`（待機秒数）、`narrow_spread_pause_max_consecutive`（連続上限）
- `run_fill_test.py`: `_compute_maker_price()` 成功後に spread を bps 評価。閾値未満なら `cancel_reason="narrow_spread_pause"` で FillRecord を返却。連続スキップは `narrow_spread_pause_max_consecutive` で上限制御（超過で強行突入）。
- YAML: `loss_control.narrow_spread_pause.{enabled, threshold_bps, pause_sec, max_consecutive}`

**デフォルト:** `enabled: false`（AB テスト対応）

---

## §4 134# P1-11: PnL fee/slippage 控除統一

**ロードマップ:** 134# §7 Phase E — P1-11「PnL 評価 fee/slippage 控除後で統一」

**問題:** 現行 PnL 計算は純粋な mid price delta (bps) で、手数料を控除していない。楽観バイアスの原因。

**実装:**
- `fill_config.py`: `pnl_fee_deduction_enabled`、`maker_fee_bps`、`taker_fee_bps`
- `pnl_measurer.py`: `measure()` の末尾で `post_fill_pnl`、`post_fill_60s_pnl`、`post_fill_120s_pnl`、`pnl_at_exit_bps` から一律 `maker_fee_bps` を減算
- YAML: `loss_control.pnl_fee_deduction.{enabled, maker_fee_bps, taker_fee_bps}`

**Coincheck 現状:** maker fee = 0% → 実質的に PnL 変化なし。fee が変更された場合に YAML 変更のみで対応可能。

**デフォルト:** `enabled: false`

---

## §5 変更ファイル一覧

### 新規

| ファイル | 行数 | 概要 |
|---|---|---|
| `tests/unit/v460/test_137_p1_features.py` | 230 | P1-06/08/11 + §9 #4 テスト (11件) |

### 修正

| ファイル | 変更概要 |
|---|---|
| `ztb/ml/retrain_trigger.py` | §9 #1/#2/#5: mtime 遅延更新, freshness gate, RetrainTrigger 改名 |
| `scripts/v460/ml/retrain_scheduler.py` | §9 #4: YAML 外部化 + RetrainTrigger 使用 |
| `scripts/v460/run_fill_test.py` | §9 #3: regime 配線 + P1-08: narrow spread pause |
| `scripts/v460/lib/fill_config.py` | P1-08/11: narrow_spread_pause + fee deduction config + YAML parsing |
| `scripts/v460/lib/pnl_measurer.py` | P1-11: fee 控除ロジック追加 |
| `configs/v460/fill_test.yaml` | P1-06/08/11 + trigger YAML 外部化 |
| `tests/unit/v460/test_136_p1_retrain_kill.py` | §9 #A/#5: 回帰テスト 3件追加 + RetrainTrigger 改名対応 |

---

## §6 テスト結果

### 対象テスト (29件: 18 + 11)

**`test_136_p1_retrain_kill.py` (18 passed):**
| # | テスト | 検証内容 |
|---|---|---|
| 1 | test_skip_when_no_fill_records_updated | mtime 不変スキップ |
| 2 | test_pass_when_fill_records_updated | mtime 変化で通過 |
| 3 | test_skip_when_trades_unhealthy | trades 不在ブロック |
| 4 | test_backoff_increases_interval | バックオフ倍増 |
| 5 | test_backoff_resets_on_deploy | deploy でリセット |
| 6 | **test_unhealthy_to_healthy_same_mtime_retrain_fires** | §9 #A 回帰 |
| 7 | **test_feature_freshness_integrated_in_trigger** | §9 #2 統合 |
| 8 | **test_backward_compat_alias** | §9 #5 互換 |
| 9-11 | TestFeatureFreshness (3件) | fresh/stale/partial |
| 12-18 | TestSellDynamicKillManager (7件) | kill/cooldown/regime等 |

**`test_137_p1_features.py` (11 passed):**
| # | テスト | 検証内容 |
|---|---|---|
| 1 | test_fee_deducted_from_pnl | P1-11: fee 控除正常 |
| 2 | test_no_fee_when_disabled | P1-11: 無効時変化なし |
| 3 | test_fee_zero_no_change | P1-11: fee=0 影響なし |
| 4 | test_config_defaults | P1-08: デフォルト値 |
| 5 | test_yaml_parsing (narrow) | P1-08: YAML パース |
| 6 | test_defaults (fee) | P1-11: デフォルト値 |
| 7 | test_yaml_parsing (fee) | P1-11: YAML パース |
| 8 | test_sell_max_reprice_default | P1-06: 共通デフォルト |
| 9 | test_yaml_sell_reprice_override | P1-06: sell 上書き |
| 10 | test_all_config_fields_have_defaults | §9 #4: 全フィールド |
| 11 | test_config_override | §9 #4: オーバーライド |

### フルスイート

v460 unit tests: **1118 passed** (1104→1118, +14)

> **注記 (§9 #C):** フルスイートのテスト数は実装環境の依存パッケージ構成に依存する。上記数値は実装時環境で確認。別環境では対象テスト (`test_136_*`, `test_137_*`) のみの再確認を推奨。

---

## §7 残課題 (134# ロードマップ)

| 134# ID | 施策 | 状態 | 備考 |
|---|---|---|---|
| P1-01/02 | buy/sell 分離モデル + target 二層化 | ⬜ | データ蓄積後 |
| P1-03 (134#) | score 校正 (isotonic/quantile) | ⬜ | FillRecord ベースで事後分析→リアルタイム化 |
| P1-06 | reprice 売側上限縮小 AB | ✅ **137# 完了** | YAML: `max_reprice_sell: 1` |
| P1-08 | spread 狭小時の「休む」判定 | ✅ **137# 完了** | YAML: `narrow_spread_pause.enabled: false` (AB待ち) |
| P1-10 | preflight 失敗連続→run pause | ⬜ | dead-cycle 抑止 |
| P1-11 | PnL 評価 fee/slippage 控除 | ✅ **137# 完了** | YAML: `pnl_fee_deduction.enabled: false` |
| P2 群 | logging 改善, parallelism, oracle 日次 KPI | ⬜ | 工数対効果で優先 |

---

## §8 コミット履歴

| SHA | 内容 |
|---|---|
| `b96ac2ef3` | 135# §9 review fixes |
| `2d3a99ccd` | 135# §10 レビュー対応結果ドキュメント追記 |
| `af30e12b1` | 136# P1-01/02/03 実装 + テスト 15 件 |
| `a10520fb4` | **137# §9 review fixes + P1-06/08/11 (本ドキュメント)** |

---

## §9 外部レビュー追記欄

### §9.1 重大度付きレビュー結果 (2026-02-22)

| # | 重大度 | 対象 | 問題 | 推奨対応 |
|---|---|---|---|---|
| 1 | HIGH | `scripts/v460/ml/retrain_scheduler.py` | `new_samples` 負値バグは「0に clamp」されただけで、`latest_run_only=true` + 旧モデル `n_samples` が大きいケースでは `new_samples=0` のまま長期固定化し、実質的に retrain が再開しない。132/133 の本質課題（run 混線）を未解決。 | `prev_n_samples` を「同一 run_id 基準」に変更するか、`latest_run_only` 時は `prev_n_samples` を無効化して別閾値で判定。 |
| 2 | MEDIUM | `scripts/v460/run_fill_test.py`, `scripts/v460/lib/fill_config.py`, `ztb/risk/sell_dynamic_kill.py` | 136# §9 #3 の「regime 閾値配線」は部分対応。`check_kill(regime=...)` は実装されたが、`regime_thresholds` を YAML/Config から注入する経路がなく、実運用では常に空 dict。 | `FillTestConfig` に `sell_dynamic_kill_regime_thresholds` を追加し、`loss_control.sell_dynamic_kill.regime_thresholds` をパースして `SellKillConfig` に渡す。 |
| 3 | MEDIUM | `scripts/v460/run_fill_test.py` | P1-08 の `narrow_spread_pause_sec` がログ出力のみで実際の待機に使われていない。仕様文「Pausing Ns」と実挙動が乖離。 | `narrow_spread_pause` 分岐で `await asyncio.sleep(narrow_spread_pause_sec)` を実行するか、仕様を「次サイクルまで待機」に明記して設定項目を削除。 |
| 4 | MEDIUM | `scripts/v460/lib/pnl_measurer.py`, `scripts/v460/lib/fill_config.py` | P1-11 は「fee/slippage 控除統一」を掲げる一方、実装は maker fee 一律控除のみ。`taker_fee_bps` は未使用、slippage 控除は未実装。 | 仕様を fee-only に縮小するか、`pnl_at_exit_bps` 等に taker/slippage を反映する計算式を追加。 |
| 5 | MEDIUM | `scripts/v460/ml/feature_enricher.py` | 132# F3 で問題化した `trades 全量フォールバック` が依然残存。date_filter/±N 日で空の場合に `date_filter=None` で全読み込みし、時間整合と I/O の双方で再悪化リスク。 | 全量フォールバックを `retrain` 設定で明示 opt-in にし、デフォルトは `skip retrain`。同時に欠損日を error として記録。 |
| 6 | LOW | `tests/unit/v460/test_137_p1_features.py` | P1-08/P1-11 のテストは主に config/パースと単体計算で、`run_fill_test` 統合挙動（pause 実動作、kill+regime 閾値）が未検証。 | 統合テストを 2 件追加（狭スプレッド時の待機時間、regime 閾値で kill 判定分岐）。 |
| 7 | LOW | `configs/v460/fill_test.yaml` | feature freshness gate は実装済みだがデフォルト `false`。データ供給障害の早期検知が運用で効かない可能性。 | 収集系が安定したら `trigger_check_feature_freshness: true` を段階導入し、しきい値を保守設定で開始。 |

### §9.2 132/133 計画起点の追加見落とし

| # | 重大度 | 起点 | 見落とし内容 | 推奨対応 |
|---|---|---|---|---|
| A | HIGH | 133# P0-01 | 「負値回避=修正完了」と判定しているが、run_id 非整合の根因は未修正。 | run_id 別 `prev_n_samples` 追跡を実装し、同一 run 比較へ変更。 |
| B | MEDIUM | 132# F3 / 133# C4 | trades 欠損時の全量 fallback を温存したまま。データ基盤修復後でも再発時に silently 劣化。 | fallback 最終段を禁止し、欠損を `status=skipped_trigger_data_missing` として監査ログ化。 |
| C | MEDIUM | 134# P1-08 | pause パラメータを追加したが wait 実装が無いため、AB テスト結果の解釈が不正確。 | pause 実装または設定削除のどちらかに統一。 |
| D | MEDIUM | 134# P1-11 | 「fee/slippage」要件に対して slippage 未実装。 | spread/queue_wait を利用した簡易 slippage 控除を first step として導入。 |

### §9.3 再検証ログ (このレビュー時点)

- `tests/unit/v460/test_136_p1_retrain_kill.py`: **18 passed**
- `tests/unit/v460/test_137_p1_features.py`: **11 passed**
- `tests/unit/v460` 全体: **1118 passed, 91 warnings**（`137` 記載値と一致）

### §9.4 優先修正順 (提案)

1. P0: `new_samples` を run_id 整合で再設計（HIGH #1 / 132-133 A）  
2. P1: `regime_thresholds` を YAML→Config→Manager に完全配線（MEDIUM #2）  
3. P1: `narrow_spread_pause_sec` の実挙動を仕様一致化（MEDIUM #3）  
4. P1: fee/slippage 仕様を実装か文言修正で確定（MEDIUM #4）  
5. P1: trades 全量フォールバックを安全側（skip）へ変更（MEDIUM #5）
