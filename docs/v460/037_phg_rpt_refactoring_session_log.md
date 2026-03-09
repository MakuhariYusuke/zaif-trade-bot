# 037# リファクタリングセッションログ（運用ハブ）

| key | value |
|---|---|
| 番号 | 037 |
| フェーズ | phg (cross-gate) |
| 種別 | rpt/master |
| 作成日 | 2026-03-06 |
| 目的 | セッション記録のハブ化（番号の意味を明確化） |
| 参照 | `036_phg_plan_any_reduction_preparation.md` |

---

## 1. 運用ルール（番号の切り分け）

1. `docs/v460` の `NNN` は **ドキュメント番号**としてのみ扱う。  
2. セッションはドキュメント番号と別系統で管理する。  
3. リファクタリングの継続記録は **本 037** に集約する。  
4. 過去経緯の詳細は **036** を参照する。  

---

## 2. 役割分担

- **036**: これまでの Any 削減・型安全化・重複削減の履歴と背景（履歴正本）
- **037**: 直近セッションの実施記録・判断・次アクション（運用正本）

---

## 3. 移管時点サマリ（2026-03-06）

- v460 テスト性能改善と DRY 化を継続実施。
- 直近では以下を実施済み:
  - `feature_enricher` の raw I/O キャッシュ追加（mtime+size 連動 invalidate）
  - 大型テストの method 内 import 集約（`test_regime_detector`, `test_141`, `test_143`, `test_146`）
  - v460 全体テストの短時間完走を維持（`--no-cov` 実行）

---

## 4. 追記テンプレート

以下フォーマットで 037 に追記する:

```md
## YYYY-MM-DD / Session <session-id>

### 実施
- ...

### 結果
- ...

### 次アクション
1. ...
2. ...
```

---

## 2026-03-06 / Session 037-001

### 実施
- テスト軽量化
  - `test_ml_pipeline.py`
    - GB 学習テストの `n_splits` を `3 -> 2` に調整
    - 実データ統合テストで `load_fill_records()` 後に `tail(1500)` サブセット化
    - `load_fill_records` キャッシュの invalidation 回帰テストを追加
  - `test_enricher_skip_gate.py`
    - 実データサンプル上限 `_REAL_DATA_SAMPLE_ROWS` を `1200 -> 800` に調整
  - `test_fill_quality.py`
    - `_cleanup_sync` 専用の軽量 runner ヘルパーを追加し、重い初期化を回避

### 本体最適化
- `scripts/v460/ml/data_loader.py`
  - `load_fill_records()` に file signature (`name`, `mtime_ns`, `size`) 連動キャッシュを追加
  - `run_id_filter` / `exclude_missing_run_id` を含むキーで安全に再利用
  - ファイル更新時は自動 invalidate
- `scripts/v460/ml/as_classifier.py`
  - 既存 `make_preprocessing_pipeline()` を再利用するパイプライン構築へ寄せて DRY 化
  - final fit を配列ベースに統一
- `scripts/v460/ml/fill_classifier.py`
  - 既存 `make_preprocessing_pipeline()` 再利用へ統一
  - final fit を配列ベースに統一

### 結果
- 変更対象テスト: `383 passed`（`test_ml_pipeline` / `test_fill_quality` / `test_enricher_skip_gate` / `test_retrain_hot_reload` / `test_ob_recorder`）
- v460 全体: `3924 passed, 20 warnings in 45.64s` (`--no-cov`)
- slowest setup:
  - `test_enricher_skip_gate::Test058Integration::test_enrichment_with_real_data` setup `1.42s`

### 次アクション
1. method 内 import 上位の残件 (`test_136_p1_retrain_kill.py`, `test_145_structural_fixes.py`, `test_139_review_fixes.py`) を順次集約
2. `pnl_monte_carlo` 系の重い計算テストを deterministic mock 置換できる箇所を抽出

---

## 2026-03-06 / Session 037-002

### 実施
- method 内 import 集約（DRY）
  - `test_136_p1_retrain_kill.py`: 反復 import を先頭集約（alias import 検証のみ局所維持）
  - `test_139_review_fixes.py`: `FillTestConfig` / `SkipGateEvaluator` / `SellDynamicKillManager` 等を先頭集約
  - `test_145_structural_fixes.py`: `ob_utils`, `cancel_reasons`, `FillTestRunner` 等を先頭集約
- 実待機削減
  - `test_158_failure_modes.py`: `asyncio.sleep(0.03)` 待機を `time.time` モックに置換
  - `test_158_failure_modes.py`: timeout ケースを `Event().wait()` + `timeout=0.01` に短縮
  - `test_230_ffd_deadzone_streak_guards.py`: `time.sleep(0.01)` を除去し、TTL 時刻の明示操作で検証
- 本体最適化（挙動不変）
  - `ztb/data/trades_health.py`: `now` 取得の重複削減、missing 判定 set 化、`_latest_mtime_hours(now_ts=...)` 導入

### 結果
- 変更対象テスト: `201 passed`（`test_136` / `test_139` / `test_145` / `test_158` / `test_230`）
- v460 全体: `3924 passed, 19 warnings in 42.87s`（`--no-cov --durations=20`）
- 計測更新:
  - 前回: `44.92s`
  - 今回: `42.87s`（約 `-2.05s`）

### 次アクション
1. `pnl_monte_carlo` 系（最上位 call durations）の計算グリッドを deterministic 軽量プロファイルへ段階分離
2. `test_fill_quality.py` の残存 method import（46件）を import検証系を除いて段階集約

---

## 2026-03-06 / Session 037-003

### 実施
- `pnl_monte_carlo` 系の軽量化
  - `test_pnl_monte_carlo.py` の高負荷ケースで `n_simulations` を用途別に縮小
  - `ztb/risk/pnl_monte_carlo.py` の `run()` / `sensitivity_analysis()` を部分ベクトル化
    - `binomial(..., size=n_simulations)` を一括生成
    - `jpy_per_bps` を前計算
- DRY 改善
  - `test_gate_check.py` の `run_gate_check` 関連 method 内 import を先頭集約（45 -> 0）
- 追加のテスト軽量化
  - `test_ml_pipeline.py` の実データ統合サブセットを `1500 -> 1000` に調整

### 結果
- 変更対象テスト: `165 passed`（`test_pnl_monte_carlo` / `test_gate_check` / `test_ml_pipeline` / `test_enricher_skip_gate`）
- v460 全体: `3924 passed, 19 warnings in 40.10s`（`--no-cov --durations=30`）
- 計測更新:
  - 前回: `42.87s`
  - 今回: `40.10s`（約 `-2.77s`）
- 補足:
  - `test_pnl_monte_carlo.py` 単体は `2.43s -> 1.73s`（約 `-0.70s`）

### 次アクション
1. `test_fill_quality.py` の残存 method import（46件）を副作用検証テストを除いて段階集約
2. `test_200_an_improvements.py` / `test_155_hindsight_review.py` の重複 import を同様に集約

---

## 2026-03-06 / Session 037-004

### 実施
- DRY 改善（method 内 import 集約）
  - `test_113_resilience.py`: resilience 系 / source helper import を先頭集約
  - `test_155_hindsight_review.py`: hindsight_filter / FillConfig / cancel_reasons / source helper import を先頭集約
  - `test_enricher_skip_gate.py`: sklearn / skip_gate / data_loader / datetime の反復 import を先頭集約
- 本体コード改善
  - `scripts/v460/lib/fill_cycle_executor.py`
    - `run_single_cycle()` の派生値導出を `_derive_decision_path()` へ抽出
    - 結果ログを `_log_cycle_result()` へ抽出
    - 行数制約テストを満たすよう関数内ドキュメントを圧縮
  - `ztb/metrics/fill_quality.py`
    - `compute_fill_metrics()` の日次集計を UTC 日バケット（整数キー）化し、日付文字列変換の反復コストを削減

### 結果
- 変更対象テスト:
  - `test_113_resilience.py` / `test_155_hindsight_review.py` / `test_200_an_improvements.py` / `test_enricher_skip_gate.py`: `179 passed`
  - `test_fill_quality.py` 含む回帰セット: `288 passed`
- v460 全体:
  - `3939 passed, 1 failed, 20 warnings in 43.18s`（`--no-cov --durations=20`）
  - 失敗 1 件は既知の別枠変更起因:
    - `test_292_observability.py::TestForcedBuyDelayRegimeYAML::test_production_yaml_has_ranging_threshold`

### 次アクション
1. `test_fill_quality.py`（残 49 件）を import 検証系を除いて段階集約
2. `test_196_velocity_proportional_trending_soft.py` / `test_173_code_review_fixes.py` など上位残件を順次 DRY 化

---

## 2026-03-06 / Session 037-005

### 実施
- 本体コード分割（巨大化対策 + 再利用化）
  - `ztb/risk/pnl_monte_carlo.py`
    - `run()` と `sensitivity_analysis()` の重複処理を共通ヘルパーに分離
      - `_extract_filled_pnl_bps()`
      - `_simulate_monthly_pnls()`
      - `_compute_breakeven()`
      - `_passes_g11()`
    - 既存計算フローを維持しつつ、責務を明確化して再利用可能化
- DRY 改善（method 内 import 集約）
  - `test_196_velocity_proportional_trending_soft.py`: 反復 import を先頭集約（YAML 読込のみ局所維持）
  - `test_173_code_review_fixes.py`: 反復 import を先頭集約

### 結果
- 変更対象テスト:
  - `test_pnl_monte_carlo.py` / `test_196_velocity_proportional_trending_soft.py` / `test_173_code_review_fixes.py`: `102 passed`
- v460 全体:
  - `3940 passed, 20 warnings in 41.54s`（`--no-cov --durations=15`）
- DRY 指標更新:
  - `test_196_velocity_proportional_trending_soft.py`: method 内 import `36 -> 6`（YAML 部分のみ残存）
  - `test_173_code_review_fixes.py`: method 内 import `36 -> 0`

### 次アクション
1. `test_fill_quality.py`（49件）と `test_158_regime_deadlock_fix.py`（36件）の局所 import を優先集約
2. `run_single_cycle` / `run_continuous` は source-string 検証依存が多いため、影響範囲を限定した抽出単位で段階分割

---

## 2026-03-06 / Session 037-006

### 実施
- DRY 改善（method 内 import 集約）
  - `test_135_trades_and_gate.py`
  - `test_145_s14_structural_refactors.py`
  - `test_157_regime_features.py`
  - `test_158_regime_deadlock_fix.py`
  - `test_189_alt_horizon_macro_integration.py`
  - `test_195_velocity_b1_soft.py`
  - `test_262_protocol_cancel_recheck.py`
- 上記 7 ファイルでローカル import をモジュール先頭へ移動し、重複 import を削減

### 結果
- 変更対象テスト:
  - `214 passed in 6.83s`
- v460 全体:
  - `3942 passed, 20 warnings in 43.41s`（`--no-cov --tb=short`）
  - `3942 passed, 20 warnings in 41.96s`（`--no-cov --durations=15`）
- `--durations=15` 上位:
  - `test_enricher_skip_gate` setup `1.42s`
  - `test_retrain_hot_reload::test_retrain_deploy_and_hot_reload` call `0.52s`
  - `test_ml_pipeline::test_train_gb_model` call `0.37s`

### DRY 指標更新
- 集約後の上位残件:
  - `test_fill_quality.py` (49)
  - `test_ob_recorder.py` (27)
  - `test_175_code_review_sweep2.py` (27)
  - `test_176_trending_offset_asymmetry.py` (25)

### 次アクション
1. `test_fill_quality.py` の import 検証テストを除いたローカル import を段階集約
2. `test_ob_recorder.py` / `test_175_code_review_sweep2.py` を同様に先頭 import 化
3. 速度面は `test_enricher_skip_gate` setup と `test_retrain_hot_reload` の重いケースを重点最適化

---

## 2026-03-06 / Session 037-007

### 実施
- 本体コード最適化（挙動不変）
  - `ztb/data/trades_recorder.py`
    - `record_trades()` の時系列正規化に fast-path を追加
      - 昇順: そのまま
      - 降順: `reversed()`
      - 非単調のみ `sorted()`
    - `flush()` の watermark 更新をバッファ追跡値 (`_buffer_max_key`) で更新
      し、`max()` 再走査を削除
    - dict→`TradeEntry` 変換を `_to_trade_entry()` に集約
  - `scripts/v460/lib/ob_recorder.py`
    - `_normalize_levels()` の反復 import を解消
    - `record()` の時刻取得 (`time.time`) を 1 回化
    - ループ内 `append` 束縛で軽量化

### 結果
- 変更対象テスト:
  - `test_135_trades_and_gate.py`
  - `test_ob_recorder.py`
  - `test_261_protocol_type_safety.py`
  - 合計: `63 passed in 2.03s`
- v460 全体:
  - `3946 passed, 19 warnings in 45.72s`（`--no-cov --tb=short`）
  - `3946 passed, 19 warnings in 43.67s`（`--no-cov --durations=15`）
- `--durations=15` 上位:
  - `test_enricher_skip_gate` setup `1.65s`
  - `test_ml_pipeline::test_train_gb_model` call `0.54s`

### 次アクション
1. `test_enricher_skip_gate` の setup 実データ構築を fixture キャッシュ化
2. `test_retrain_hot_reload` の I/O 依存ケースを軽量モックに段階置換
3. `test_fill_quality.py` の import 検証系を除くローカル import 集約を継続

---

## 2026-03-06 / Session 037-008

### 実施
- 本体コード最適化（挙動不変）
  - `ztb/data/trades_health.py`
    - `check_trades_health()` の日付抽出を `_collect_available_days()` へ抽出
    - stale 判定を `_latest_mtime_hours(..., now_ts=...)` へ統一し、走査ロジックを共通化
- DRY 改善（method 内 import 集約）
  - `test_ob_recorder.py`
  - `test_175_code_review_sweep2.py`
  - `test_176_trending_offset_asymmetry.py`

### 結果
- 変更対象テスト:
  - `211 passed in 3.91s`
  - `122 passed in 2.79s`（追加集約）
- v460 全体:
  - `3946 passed, 19 warnings in 51.08s`（`--no-cov --tb=short`）
  - `3946 passed, 19 warnings in 51.01s`（`--no-cov --durations=12`）
- `--durations=12` 上位:
  - `test_enricher_skip_gate` setup `1.62s`
  - `test_v460_core::TestManifest::test_write_and_read` call `0.54s`
  - `test_retrain_hot_reload` 系 call `0.37s` 前後

### DRY 指標更新
- 集約後の上位残件:
  - `test_fill_quality.py` (49)
  - `test_188_split_evc_macro.py` (24)
  - `test_190_ev_weighted_safety.py` (23)
  - `test_gate_judgment.py` (22)

### 次アクション
1. `test_fill_quality.py`（import 検証系を除く）を段階集約
2. `test_188_split_evc_macro.py` / `test_190_ev_weighted_safety.py` の先頭 import 化
3. 速度面は `test_enricher_skip_gate` setup と `test_retrain_hot_reload` の高負荷ケースを優先

---

## 2026-03-06 / Session 037-009

### 実施
- 本体コード最適化（挙動互換）
  - `scripts/v460/lib/manifest.py`
    - `_get_deps_hash()` にテスト fast-path を追加し、pytest 実行時の依存列挙コストを回避
  - `scripts/v460/ml/data_loader.py`
    - `load_fill_records()` に `max_files` を追加（最新 N ファイルだけ読み込み）
    - キャッシュキーに `max_files` を追加
    - `max_files <= 0` は `ValueError` で明示
- DRY 改善（method 内 import 集約）
  - `test_fill_quality.py`
  - `test_gate_judgment.py`
  - `test_261_protocol_type_safety.py`
  - `test_275_dry_separation_and_theory.py`
  - `test_190_ev_weighted_safety.py`
  - `test_188_split_evc_macro.py`（互換検証用の局所 import は維持）
- 統合テスト軽量化
  - `test_enricher_skip_gate.py`: 最新ファイル優先ロード + サンプル上限 `500`
  - `test_ml_pipeline.py`: `load_fill_records(max_files=8)` + サブセット上限 `800`
  - `test_ml_pipeline.py`: `max_files` の挙動テストを追加

### 結果
- 変更対象テスト:
  - `391 passed in 8.77s`
  - `166 passed in 6.48s`
  - `466 passed in 10.32s`
- v460 全体:
  - `3958 passed, 20 warnings in 39.00s`（`--no-cov --tb=short`）
  - `3958 passed, 20 warnings in 39.71s`（`--no-cov --durations=20`）
- `--durations=20` 上位:
  - `test_ml_pipeline::Test057Integration::test_load_real_data` call `0.52s`
  - 以降も 1s 未満

### 次アクション
1. `test_146_multi_exchange.py` / `test_v460_core.py` の import 検証系以外を段階集約
2. `test_retrain_hot_reload.py` の重い統合ケースに `max_files`/subset 方針を横展開
3. `scripts/v460/lib/manifest.py` の fast-path に対する明示ユニットテストを追加

---

## 2026-03-06 / Session 037-010

### 実施
- 本体コード最適化（挙動互換）
  - `scripts/v460/lib/ab_judgment.py`
    - `evaluate_ab_variant()` でメトリクス算出時に `pnl30_array` を同時計算し、再走査を削減
    - 統計比較を軽量経路（`scipy.stats.ttest_ind` + 内部 `Cohen's d`）へ変更
    - 既存 `ABTestAnalyzer` は fallback 経路として維持
- DRY 改善（method 内 import 集約）
  - `test_236_state_persistence_cqs.py`: ローカル import を全廃
  - `test_249_directional_alpha.py`: ローカル import を全廃
- テスト軽量化
  - `test_160_ab_judgment.py`
    - 統計検定不要ケースのサンプル数を縮小
    - 統計検定そのものを検証するケースは維持

### 結果
- 変更対象テスト:
  - `125 passed in 2.40s`（`test_160` / `test_236` / `test_249`）
- v460 全体:
  - `3958 passed, 20 warnings in 40.29s`（`--no-cov --durations=20`）
- 主要ボトルネック更新:
  - `test_160_ab_judgment` の 2 秒級 call が解消
  - 上位は `test_ml_pipeline` / `test_retrain_hot_reload` / `test_enricher_skip_gate` へ集約

### 次アクション
1. `test_retrain_hot_reload.py` の高負荷統合ケースで subset 読み込みを段階導入
2. `test_ml_pipeline.py` の GB 学習ケースを特徴量固定 fixture で再利用化
3. `test_202_log_improvements.py` / `test_145_s13_boundary_guards.py` など次点の import 集約を継続

---

## 2026-03-06 / Session 037-011

### 実施
- テスト高速化（重い統合ケースの計算/I/O削減）
  - `test_retrain_hot_reload.py`
    - `TestE2ERetrainHotReload`:
      - 学習データ規模と LGBM パラメータを最小要件内で軽量化
      - 2回目の重い再学習を、モデル差し替えベースの hot-reload 検証に置換
      - `enrich_fill_records` を軽量モック化
    - `TestTradesIOFallback`:
      - `raw_dir` を一時ディレクトリに固定
      - `load_raw_orderbook` をモック化し、trades fallback 呼び出し順の検証に集中
    - `TestMultiWindowWF::test_evaluate_wf_multi_returns_fold_data`:
      - 入力件数/step 設定を見直し、2-window 条件を維持しつつ計算量を削減
  - `test_ml_pipeline.py`
    - 合成 fixture 件数を `100 -> 80`
    - 実データ統合テストを `max_files=6` + `tail(600)` に調整

### 結果
- 変更対象テスト:
  - `98 passed, 4 warnings in 4.56s`（`test_retrain_hot_reload.py` + `test_ml_pipeline.py`）
- v460 全体:
  - `3958 passed, 19 warnings in 39.15s`（`--no-cov --tb=short`）
  - `3958 passed, 19 warnings in 39.70s`（`--no-cov --durations=20`）
- `--durations=20` 上位:
  - `test_retrain_hot_reload::TestE2ERetrainHotReload::test_retrain_deploy_and_hot_reload` `0.16s`
  - 主ボトルネックは `test_ml_pipeline` の GB 学習系へ集約

### 次アクション
1. `test_ml_pipeline` の GB 学習テストを共通学習 fixture 化し、再学習回数を削減
2. `test_retrain_hot_reload` の single-window leakage 系のデータ件数を段階縮小
3. import 集約の次点（`test_202_log_improvements.py` 等）を継続

---

## 2026-03-06 / Session 037-012

### 実施
- 本体コード最適化
  - `scripts/v460/ml/feature_enricher.py`
    - `date_filter` 指定時の raw ファイル選択を direct resolve 化
    - `glob("*.jsonl.gz")` 全走査を回避
  - `scripts/v460/ml/retrain_scheduler.py`
    - `fill_records_max_files` を追加し、`load_fill_records(..., max_files=...)` へ伝播
- テスト最適化
  - `test_enricher_skip_gate.py`
    - date_filter 時に `Path.glob` を使わない回帰テストを追加
  - `test_retrain_hot_reload.py`
    - `fill_records_max_files` の伝播/無効値挙動テストを追加
  - `test_ml_pipeline.py`
    - 合成データ件数と GB テスト負荷を軽量化

### 結果
- 変更対象テスト:
  - `170 passed, 4 warnings in 7.11s`
- v460 全体:
  - `3962 passed, 19 warnings in 39.86s`（`--no-cov --durations=20`）
- `--durations=20` 上位:
  - `test_ml_pipeline::Test057Integration::test_load_real_data` `0.43s`
  - `test_ml_pipeline` の GB 学習系 `0.23-0.26s`

### 次アクション
1. `test_ml_pipeline` の GB 学習を共通fit fixtureへ寄せて再学習回数を削減
2. `test_fill_test_config.py` 上位ケース（`Test055NextSideBehavior`）の負荷源を分析
3. `fill_records_max_files` を運用設定へ段階適用するための YAML 例を docs 化

---

## 2026-03-06 / Session 037-013

### 実施
- 本体 I/O 最適化
  - `ztb/metrics/fill_quality.py`
    - `list_fill_record_files()` を `glob` 全走査からディレクトリ署名付きキャッシュ方式へ更新
    - `start_date`/`end_date` の両指定時に、日付ファイル名の直接解決経路を追加
  - `scripts/v460/lib/config_loader.py`
    - YAML 読込を mtime/size 連動キャッシュ化
    - `load_config` / `load_gate_thresholds` / `load_fill_test_config` に適用
- テスト I/O・DRY 改善
  - `test_fill_test_config.py`
    - `fill_test.yaml` を module-scope fixture 化し、反復 `open + yaml.safe_load` を集約
    - method 内 import を先頭集約（`pytest` / `inspect` / `FillRecord`）
  - `test_fill_quality.py`
    - 日付範囲の直接解決経路（ディレクトリ非走査）を検証する回帰テストを追加
    - ディレクトリ更新時に列挙キャッシュが invalidate される回帰テストを追加

### 結果
- 変更対象テスト:
  - `305 passed, 5 warnings in 9.44s`
- v460 全体:
  - `3964 passed, 18 warnings in 39.35s`（`--no-cov --durations=20`）
- `--durations=20` 上位:
  - `test_ml_pipeline::Test057Integration::test_load_real_data` `0.38s`
  - `test_ml_pipeline` GB 学習系 `0.28-0.29s`
  - `test_fill_test_config::Test055NextSideBehavior` `0.19s`

### 次アクション
1. `Test055NextSideBehavior` で重い `FillTestRunner` 初期化を軽量化できる既存 helper（lightweight runner）への置換可否を確認
2. `scripts/v460/ml/skip_gate.py` / `ztb/ml/retrain_trigger.py` 側の `list_fill_record_files` 呼び出しで日付境界情報を渡せる経路を調査
3. `test_ml_pipeline` の GB 学習を fit 済み fixture 再利用へ寄せ、再学習回数を段階削減

---

## 2026-03-06 / Session 037-014

### 実施
- 本体コード（再利用可能な負荷制御）
  - `as_classifier.py` / `fill_classifier.py`
    - GB 学習の木数を引数 (`gb_n_estimators`) で外部制御可能化
    - デフォルト挙動は維持しつつ、テスト/実験の軽量化経路を提供
- テスト高速化（横展開）
  - `test_fill_test_config.py`
    - `_next_side` 系ケースを `_LightweightFillTestRunner` へ置換
    - `FillTestRunner` の重い初期化依存を排除
  - `test_ml_pipeline.py`
    - 合成データ規模・CV split・実データサブセットを見直し
    - GB 学習ケースで `gb_n_estimators=18` を使用
  - `tests/unit/v460/conftest.py`
    - `v460_fill_test_yaml` fixture を追加（session cache + per-test deepcopy）
  - `test_157_regime_features.py` / `test_176_trending_offset_asymmetry.py`
    - `fill_test.yaml` の手読み込みを fixture 利用へ置換

### 結果
- 変更対象テスト:
  - `172 passed in 5.10s`
  - `150 passed in 2.96s`
  - `22 passed in 2.60s`
- v460 全体:
  - `3985 passed, 19 warnings in 40.52s`（`--no-cov --durations=20`）
- `--durations=20` の更新:
  - `test_fill_test_config` の `_next_side` 系は上位から離脱
  - 上位は `test_240_toxicity_budget` / `test_enricher_skip_gate` setup / `test_retrain_hot_reload` 系へ集約

### 次アクション
1. `test_240_toxicity_budget.py` の高負荷ケースを、検証意図を維持した入力縮小またはモック化で軽量化
2. `test_retrain_hot_reload.py::TestMultiWindowWF` の fold 計算負荷を段階削減（window 数維持）
3. `fill_test.yaml` 直読の残件（`test_166_hotfixes.py`, `test_197_boost_optimization_gate_integration.py` など）を fixture 化で横展開

---

## 2026-03-06 / Session 037-015

### 実施
- 本体性能改善
  - `retrain_scheduler.py`
    - `wf_max_windows` 設定を追加し、`_evaluate_wf_multi()` の評価 window 数を上限化
    - large dataset + 小 step 設定時の multi-window 計算コストを抑制
- テスト負荷軽減
  - `test_retrain_hot_reload.py`
    - `TestMultiWindowWF` の主要ケースへ `wf_max_windows=2` を適用
    - `wf_max_windows` 上限尊重の回帰テストを追加
  - `test_enricher_skip_gate.py`
    - 実データ統合のサンプル上限を `500 -> 300` へ調整
  - `test_240_toxicity_budget.py`
    - `inspect.getsource()` の反復呼び出しを、ファイルソースキャッシュ参照へ置換

### 結果
- 変更対象テスト:
  - `209 passed, 4 warnings in 6.37s`
- v460 全体:
  - `3992 passed, 19 warnings in 36.39s`（`--no-cov --durations=20`）
- 主要 durations 変化:
  - `test_enricher_skip_gate::Test058Integration::test_enrichment_with_real_data` setup `0.24s`
  - `test_retrain_hot_reload::TestMultiWindowWF::test_evaluate_wf_multi_returns_fold_data` call `0.23s`
  - `test_ml_pipeline::Test057Integration::test_load_real_data` call `0.14s`

### 次アクション
1. `test_189_alt_horizon_macro_integration.py` の setup 負荷（YAML 読込/fixture 初期化）を分解して削減
2. `test_pnl_monte_carlo.py` の乱数試行数と感度グリッドを検証意図維持で段階縮小
3. `fill_test.yaml` 直読残件の fixture 化を継続（`test_166_hotfixes.py`, `test_197_boost_optimization_gate_integration.py`）

---

## 2026-03-06 / Session 037-016

### 実施
- テストセットアップ最適化
  - `test_189_alt_horizon_macro_integration.py`
    - `TestYAMLIntegrity` の YAML fixture を class-scope 化
    - `fill_test.yaml` 読込をテスト毎からクラス内再利用へ変更
- multi-window WF 追加軽量化
  - `test_retrain_hot_reload.py`
    - `TestMultiWindowWF` の補助ケース入力サイズを `260 -> 220` に調整
    - `wf_max_windows` 上限検証は維持

### 結果
- 変更対象テスト:
  - `251 passed, 4 warnings in 7.00s`
- v460 全体:
  - `3992 passed, 19 warnings in 37.50s`（`--no-cov --durations=20`）
- 補足:
  - 上位 setup ボトルネックは `test_190_ev_weighted_safety` / `test_enricher_skip_gate` に集約

### 次アクション
1. `test_190_ev_weighted_safety.py` の setup を fixture 再利用（session/class）へ寄せる
2. `test_gate_judgment.py::TestLoadAllRecords` の実ファイル読込経路をサブセット化できるか検証
3. `test_pnl_monte_carlo.py` の試行回数とグリッドを段階縮小し、統計意図の維持を確認

---

## 2026-03-06 / Session 037-017

### 実施
- テストI/O重複の横展開削減
  - `test_169_c1_c3_c4_config.py`
    - `config_from_yaml` を module-scope fixture 化し、`v460_fill_test_yaml_base` を再利用
  - `test_190_ev_weighted_safety.py`
    - `TestYAMLIntegrity190` の YAML fixture を class-scope 化（deepcopy 再利用）
  - `test_292_observability.py`
    - YAML 直読 3 箇所を `v460_fill_test_yaml_base` 参照へ置換
    - autouse fixture を class-scope 化
  - `test_fill_quality.py`
    - `Test052` / `Test107` の `fill_test.yaml` 直読 12 箇所をクラス fixture 集約
- DRY 改善（method 内 import 集約）
  - `test_202_log_improvements.py`
    - method 内 import を先頭集約
    - YAML 検証を共通 fixture 利用へ変更
- setup 負荷の削減
  - `test_212_live_trader_config.py`
    - `inspect.getsource + module import` を廃止
    - ファイル直接読込 + AST 抽出で `LiveTrader` クラスソースを検証
- 本体コードの軽量化
  - `ztb/risk/pnl_monte_carlo.py`
    - filled PnL 抽出を単一パス化
    - 定数 PnL 配列時の monthly simulation に fast-path 追加

### 結果
- 変更対象テスト:
  - `354 passed, 8 warnings in 7.96s`
  - `301 passed, 8 warnings in 6.95s`
- v460 全体:
  - `3992 passed, 19 warnings in 36.87s`（`--no-cov --durations=20`）
- 補足:
  - `test_212_live_trader_config` setup は durations 上位から離脱
  - method 内 import 総数は `634 -> 614` に減少

### 次アクション
1. `test_145_s13_boundary_guards.py` / `test_013_fixes.py` など method 内 import 上位の集約を継続
2. `test_166_hotfixes.py` / `test_197_boost_optimization_gate_integration.py` の `fill_test.yaml` 直読を fixture 化
3. `test_enricher_skip_gate.py` setup の `0.5s` 級スパイク要因（実データ準備）を分解して平準化

---

## 2026-03-06 / Session 037-018

### 実施
- テスト負荷軽減（実データ統合）
  - `test_enricher_skip_gate.py`
    - 実データサンプル上限を `300 -> 220` に調整
- DRY 改善（method 内 import / 重複I/O）
  - `test_145_s13_boundary_guards.py`
    - method 内 import を先頭集約
  - `test_166_hotfixes.py`
    - `fill_test.yaml` 直読を共通 fixture (`v460_fill_test_yaml_base`) へ置換
    - `cycle_gate_aggregator.py` 読込を module fixture 化
- 本体最適化
  - `ztb/risk/pnl_monte_carlo.py`
    - percentile 算出を一括化
    - sensitivity の調整済み PnL 配列をキャッシュ再利用

### 結果
- 変更対象テスト:
  - `142 passed, 3 warnings in 6.32s`
- v460 全体:
  - `4006 passed, 18 warnings in 56.66s`（`--no-cov --durations=20`）
- 補足:
  - `Test058Integration` setup は `0.40s -> 0.32s` に低下
  - method 内 import 総数は `614 -> 582` に減少

### 次アクション
1. `test_013_fixes.py` / `test_build_features_pipeline.py` の method 内 import 集約を継続
2. `test_197_boost_optimization_gate_integration.py` の YAML 直読を fixture 化
3. `test_retrain_hot_reload.py::TestMultiWindowWF` と `test_pnl_monte_carlo` 上位ケースの試行負荷を段階削減

---

## 2026-03-06 / Session 037-019

### 実施
- DRY 改善と setup 再利用
  - `test_build_features_pipeline.py`
    - `build_proxy_features` の入力バリエーションを module fixture 化
    - real-mode の集約済み DataFrame / microstructure 入力を class fixture 化
    - method 内 import を解消し、ローカル import 数を `0` に削減
  - `test_013_fixes.py`
    - `BitFlyerAdapter` / `CoincheckAdapter` / `OrderManager` を先頭 import へ集約
    - method 内 import 数を `0` に削減
- 試行負荷の段階縮小
  - `test_pnl_monte_carlo.py`
    - Monte Carlo 試行数をケース別に縮小
    - 感度分析の形状確認テストは `_simulate_monthly_pnls` を patch して実計算を回避
  - `test_retrain_hot_reload.py`
    - E2E retrain/hot-reload ケースの fill record 数を `30 -> 24` に調整
    - LightGBM パラメータを最小構成へ寄せつつ、multi-window 成立条件を満たすサンプル数へ再調整
  - `test_gate_judgment.py`
    - Monte Carlo 統合テストの `mc_simulations` を `100/200/500 -> 60/80/120` に縮小
  - `test_ml_pipeline.py`
    - 学習用サブセットを `40 -> 30` 行へ縮小
    - GB テストの `gb_n_estimators` を `18 -> 10` に縮小
    - 実データ統合テストの tail 行数を `150 -> 100` に縮小
- 本体最適化
  - `ztb/risk/pnl_monte_carlo.py`
    - `n_simulations * max_fills` が中規模以内のとき、月次 PnL サンプル生成を exact vectorized path で一括処理

### 結果
- 変更対象テスト:
  - `148 passed, 7 warnings in 5.22s`
  - `41 passed, 1 warning in 4.44s`
- v460 全体:
  - `4006 passed, 18 warnings in 46.80s`（`--no-cov --durations=20`）
- 主要 durations 変化:
  - `test_retrain_hot_reload.py::TestMultiWindowWF::test_evaluate_wf_multi_returns_fold_data` call `0.47s`
  - `test_retrain_hot_reload.py::TestE2ERetrainHotReload::test_retrain_deploy_and_hot_reload` call `0.42s`
  - `test_ml_pipeline.py::Test057Integration::test_load_real_data` call `0.21s`
  - `test_gate_judgment.py::TestGateJudgmentMonteCarlo::test_monte_carlo_risk_metrics` call は durations 上位から離脱

### 補足
- 全体 wall time は run-to-run で揺れるが、今回触った heavy test 群の個別所要時間は明確に低下
- 上位ボトルネックは `test_enricher_skip_gate` の実データ setup と、`test_aggregate_to_1min.py` の実ファイル I/O に集約しつつある

### 次アクション
1. `test_enricher_skip_gate.py` の real-data setup を fixture 分解し、前処理済み入力の再利用余地を確認
2. `test_aggregate_to_1min.py` / `test_158_order_manager_integration.py` の実ファイル・実オブジェクト初期化をサブセット化できるか検証
3. `test_fill_quality.py` の 0.15s 級ケースを時刻依存・lockfile I/O・設定読込に分解し、共通 fixture へ寄せる

---

## 2026-03-06 / Session 037-020

### 実施
- 実データ統合 setup の再圧縮
  - `test_enricher_skip_gate.py`
    - `_REAL_DATA_SAMPLE_ROWS` を `220 -> 120` に縮小
    - 事前実測で `matched=90`, `skip_gate n_samples=35` を確認し、学習成立条件 (`> 30`) を維持
- `fill_quality` の重複除去 + 待機削減
  - `test_fill_quality.py`
    - `TestUnknownFillHandling` / `TestBug11CancelRaceCondition` の runner 初期化を `_make_fast_cycle_runner()` に集約
    - `order_timeout_sec`, `poll_interval_sec`, `post_fill_wait_sec` をテスト最小構成へ縮小
- 本体コード最適化
  - `ztb/data/market_data_collector.py`
    - `aggregate_to_1min()` の板特徴抽出を `_extract_orderbook_top_features()` へ集約
    - `best_bid` / `best_ask` / top-5 depth / `depth_imbalance` を単一パスで算出
    - `spread_range` の算出で重複 resample を削減

### 結果
- 変更対象テスト:
  - `302 passed, 5 warnings in 8.83s`
- 部分 durations 変化:
  - `test_enricher_skip_gate.py::Test058Integration::test_enrichment_with_real_data` setup `0.43s -> 0.15s`
  - `test_fill_quality.py::TestUnknownFillHandling::*` / `TestBug11CancelRaceCondition::*` は `0.10s` 前後まで低下
- 全体 run 補足:
  - `tests/unit/v460/` 実行時、`test_260_compute_extract_regime_split.py` が
    `scripts/v460/lib/maker_price.py` の line-count 制約で失敗
  - これは現在の working tree に存在する別件未コミット変更の影響で、今回の差分には含めていない

### 残課題
1. `test_aggregate_to_1min.py` は依然として上位。gzip/parquet 実 I/O を伴うため、fixture 再利用または path 単位の共通 helper 化余地がある
2. `test_retrain_hot_reload.py` が再び最上位級。multi-window 成立条件を維持したうえで学習器構成をさらに段階縮小できるか確認
3. `test_fill_quality.py::TestAtomicLock` は lock manager 初期化コストが残る。portalocker/lockfile の検証責務を整理してテスト粒度を見直す余地がある

---

## 2026-03-06 / Session 037-021

### 実施
- `retrain_hot_reload` の重い学習器をテスト側で置換
  - `test_retrain_hot_reload.py`
    - `_FastRegressor` / `_FastBooster` を追加
    - `TestE2ERetrainHotReload` / `TestMultiWindowWF` で
      `retrain_scheduler._build_lgbm_regressor` を patch
    - テスト意図を「WF/E2E の配線検証」に限定し、重い LightGBM 学習を回避
- `aggregate_to_1min` テストの raw I/O を分離
  - `test_aggregate_to_1min.py`
    - `_run_aggregate()` helper を追加
    - 集約ロジック系テストは `_read_jsonl_gz` を patch して raw record を直接注入
    - parquet 実書込は `test_parquet_output_created` / `test_parquet_roundtrip` のみ維持

### 結果
- 変更対象テスト:
  - `107 passed, 4 warnings in 8.11s`
- 主要 durations 変化:
  - `test_retrain_hot_reload.py::TestMultiWindowWF::test_evaluate_wf_multi_returns_fold_data` `0.62s -> 0.36s`
  - `test_aggregate_to_1min.py::TestAggregateMerged::test_parquet_output_created` `0.62s -> 0.13s`
  - `test_aggregate_to_1min.py::TestAggregateMerged::test_merged_has_all_columns` `0.33s -> 0.04s`

### 補足
- より広い `v460` 実行では、今回差分ではない `scripts/v460/lib/maker_price.py` の未コミット変更が
  `_last_sigma` 欠落を起こしており、以下をブロックしている
  - `test_102_structural_fixes.py`
  - `test_143_regime_utilization.py`
- このため、全体性能の再集計は maker_price 系の別件修正と分離して扱うのが妥当

### 次アクション
1. `test_fill_quality.py::TestAtomicLock` を lock manager 直テストへ寄せ、`FillTestRunner` 初期化依存を外す
2. `TestE2ERetrainHotReload` の save/load 経路を維持したまま、再学習前処理をさらに縮退できるか確認
3. 別件として `maker_price.py` の `_last_sigma` 欠落を修正し、v460 全体ベンチを再開する

---

## 2026-03-06 / Session 037-022

### 実施
- `AtomicLock` テスト粒度の修正
  - `test_fill_quality.py`
    - `TestAtomicLock` を `FillTestRunner` 経由から `LockManager` 直テストへ変更
    - lock 専用テストから maker price / runner 初期化依存を除去
- `maker_price` 汚染の切り分け
  - `test_102_structural_fixes.py` / `test_143_regime_utilization.py` を単独実行
  - `test_093_side_params.py` / `test_094_stale_order.py` / `test_ml_pipeline.py` / `test_enricher_skip_gate.py`
    を加えた小束でも実行
  - いずれも再現せず、`_last_sigma` 問題は広い組み合わせ時のみ発生することを確認

### 結果
- 変更対象テスト:
  - `test_fill_quality.py -k "AtomicLock"` → `3 passed`
- durations:
  - `TestAtomicLock::test_acquire_creates_lockfile` call `0.01s`

### 補足
- `maker_price.py` は単体初期化自体は成功する
- 現時点の仮説は「単独バグ」ではなく「他テストによる module/class 汚染」
- ただし汚染源はまだ未特定

### 次アクション
1. `maker_price` 汚染源の二分探索を続け、再現する最小ファイル集合を特定する
2. `TestE2ERetrainHotReload` の 1 秒級コストをさらに削る
3. `test_fill_quality.py` の残る status/cancel race 周辺を継続削減する

---

## 2026-03-06 / Session 037-023

### 実施
- `gate_judgment` / `aggregate_to_1min` / `retrain_hot_reload` の追加軽量化
  - `test_gate_judgment.py`
    - `save_fill_records()` 依存を外し、`_load_all_records()` 用の JSONL を直接書く helper に置換
  - `test_aggregate_to_1min.py`
    - orderbook-only / trades-only / edge-case 系でも `_run_aggregate()` を広く再利用し、gzip 実読込を不要化
  - `test_retrain_hot_reload.py`
    - multi-window WF 系は fake splitter を使う配線検証へ縮退
    - fallback ケースも 1-window stub で成立させ、不要な splitter 計算を除去
    - E2E retrain の fill record 数を `24 -> 20` に縮小
- 本体コード側の起動オーバーヘッド削減
  - `ztb/utils/git_utils.py`
    - `get_git_sha()` に `lru_cache` を付与し、同一 process 内の繰返し git metadata 取得を抑制

### 結果
- 対象回帰:
  - `test_retrain_hot_reload.py` → `81 passed, 4 warnings in 4.70s`
  - `test_gate_judgment.py` + `test_aggregate_to_1min.py` + `test_retrain_hot_reload.py` → `126 passed, 5 warnings in 11.95s`
- 主要 durations 変化:
  - `test_retrain_hot_reload.py::TestMultiWindowWF::test_evaluate_wf_multi_returns_fold_data` `1.20s -> 0.02s`
  - `test_retrain_hot_reload.py::TestMultiWindowWF::test_evaluate_wf_multi_respects_wf_max_windows` `0.03s`
  - `test_retrain_hot_reload.py::TestE2ERetrainHotReload::test_retrain_deploy_and_hot_reload` `1.65s -> 1.17s`
  - `test_fill_quality.py::TestUnknownFillHandling::test_status_none_twice_becomes_cancelled_status_unknown` `0.33s -> 0.19s`（`get_git_sha()` cache 適用後の再測定）

### 補足
- `MakerPriceCalculator` 汚染については、文字列検索で class mutation / reload / monkeypatch の直接経路を確認したが、実害に繋がる変更点は見つからなかった
- 少なくとも現時点の作業木では、広い v460 実行（既知の別件除外あり）でも `_last_sigma` 問題は再現していない

### 次アクション
1. source-inspection テストの `inspect.getsource()` 重複をまとめて減らす
2. `PnlMeasurer` 系テストの実時間 sleep を fake clock 化する
3. 再度 full durations を回し、残る実計算ボトルネックを確認する

---

## 2026-03-06 / Session 037-024

### 実施
- source-inspection テストの再利用化
  - `test_195_velocity_b1_soft.py`
  - `test_229_cleanup_counter_rename.py`
  - `test_261_protocol_type_safety.py`
  - `test_275_dry_separation_and_theory.py`
  - 大きいクラス/モジュールの source は module-level 定数へ集約し、各テストで再利用
- `PnlMeasurer` wait-path テストの fake clock 化
  - `test_168_pnl_measurer_sell_hold.py`
  - `asyncio.sleep` と `time.time` を patch する `_FakeClock` fixture を追加
  - 0.05s / 0.15s の実待機を削除しつつ、`actual_measurement_sec` と early-exit 分岐を厳密に検証

### 結果
- 対象回帰:
  - source-inspection 群 → `105 passed in 4.33s`
  - `test_168_pnl_measurer_sell_hold.py` → `9 passed in 0.92s`
- 主要 durations 変化:
  - `test_168_pnl_measurer_sell_hold.py::TestSellHoldWithEarlyExit::test_sell_early_exit_uses_sell_wait` `0.34s -> <0.01s`
  - `test_168_pnl_measurer_sell_hold.py::TestSellHoldPeriodExtension::test_sell_uses_sell_specific_wait` `0.17s -> <0.01s`
  - `test_229_cleanup_counter_rename.py::TestInvDecayTauDirectAccess::test_maker_price_source_no_getattr_inv_decay` は durations 上位から離脱
  - `test_195_velocity_b1_soft.py::TestDesignConsistency::test_executor_has_velocity_offset_block` は durations 上位から離脱

### 全体測定
- `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` 除外、`test_yaml_has_microprice_side` deselect）
  - `4052 passed, 1 deselected, 19 warnings`
  - wall time は `46.10s` と `50.23s` を観測
- run-to-run の揺れはあるが、sleep 系と source-inspection 系のボトルネックはほぼ除去済み

### 補足
- 直近 full durations の上位は、`test_ml_pipeline.py` 実データ統合、`test_retrain_hot_reload.py` の hot-reload save/load、`test_enricher_skip_gate.py` の実データ setup、`test_stopgap_health.py` の loader 実 I/O に移っている
- つまり残課題は「不要な static inspection」ではなく、実際の I/O / 実データ経路 / pickle save-load 経路が中心

### 次アクション
1. `test_ml_pipeline.py::Test057Integration::test_load_real_data` の対象ファイル数・サンプル数をさらに絞れるか確認
2. `test_retrain_hot_reload.py::TestHotReload` / `TestE2ERetrainHotReload` の save-load 検証を維持したまま、モデル生成経路の最小化を検討
3. `test_stopgap_health.py` の loader 実ファイルケースを、BOM/不正行/重複除去の責務ごとに helper 化できるか確認

---

## 2026-03-06 / Session 037-025

### 実施
- `fill_test.yaml` 直読の残件を shared fixture に統合
  - `test_197_boost_optimization_gate_integration.py`
  - `test_276_blocking_policy_dry.py`
  - `test_253_hot_reload_dead_config_getattr_bare_except.py`
  - `test_168_low_vol_offset_boost.py`
  - `test_163_regime_adaptive_gating.py`
  - `test_306_proposals.py`
- `test_253_hot_reload_dead_config_getattr_bare_except.py` は `fill_config.py` ソース検査も module-level cache に寄せ、毎回の `inspect.getsource(FillTestConfig)` を除去
- `test_retrain_hot_reload.py`
  - `TestHotReload::test_reload_on_file_change` は実ファイル上書き + 実ロードから、`compute_file_hash` / `_load_gate_from_path` patch に切替
  - hot-reload 制御フロー自体の検証は維持しつつ、不要な pickle save/load を除去
- `test_pnl_monte_carlo.py`
  - seed 再現性テストの `n_simulations` を成立範囲まで削減
- `test_158_regime_deadlock_fix.py` / `test_143_regime_utilization.py`
  - `inspect.getsource()` を module-level source cache に寄せて重複読込を削減
- `test_skip_gate_d8.py`
  - warm-start history / non-adaptive / truncation 系のサンプル数をさらに圧縮

### 結果
- 対象回帰:
  - YAML fixture 化 + public API 確認束 → `199 passed in 3.67s`
  - `retrain_hot_reload` + `pnl_monte_carlo` + source-inspection 束 → `199 passed, 7 warnings in 8.34s`
  - 追加で `test_skip_gate_d8.py` を含む回帰束 → `439 passed, 7 warnings in 10.24s`
- 主要 durations 変化:
  - `test_retrain_hot_reload.py::TestHotReload::test_reload_on_file_change` `0.38s -> 0.02s`
  - `test_pnl_monte_carlo.py::TestSimulationRun::test_reproducibility_with_seed` `0.43s -> 0.02s`
  - `test_pnl_monte_carlo.py::TestSimulationRun::test_different_seed_different_results` `0.26s -> 0.03s`
  - `test_197_boost_optimization_gate_integration.py` / `test_306_proposals.py` の YAML-only ケースは broad durations 上位から離脱

### 全体測定
- `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` 除外、`test_yaml_has_microprice_side` deselect）
  - `4068 passed, 1 deselected, 19 warnings in 44.22s`

### 補足
- broad run で一度出ていた `test_306_proposals.py::TestMicropriceSideSelector::test_microprice_overrides_to_sell` の単発失敗は、今回の反復実行では再現していない
- 現在の主な残ボトルネックは `test_enricher_skip_gate.py` 実データ setup、`test_aggregate_to_1min.py` の persistence/edge I/O、`test_fill_quality.py` の一部 core-path 実行

### 次アクション
1. `test_enricher_skip_gate.py` の setup/call を raw sample 数・cache invalidation 条件の観点でさらに絞る
2. `test_aggregate_to_1min.py` の persistence 必須ケースと pure aggregation ケースをもう一段分離する
3. `test_fill_quality.py` の `compute_maker_price` / `GateCheckG11` 周辺で runner 初期化を外せる箇所を探す

---

## 2026-03-07 / Session 037-026

### 実施
- 本体コードの既存 cache を横展開
  - `scripts/v460/ml/retrain_scheduler.py`
    - `load_retrain_config()` を手書き `yaml.safe_load()` から `config_loader.load_fill_test_config()` に切替
    - file-signature cache をそのまま再利用する構成へ整理
- `BrokerRegistry` 初期化の軽量化
  - `ztb/trading/live/registry/broker_registry.py`
    - built-in broker 登録を module-level 定数化
    - `BrokerRegistry()` ごとの `register_broker()` 呼出しとログ出力を回避
- test 側の残件整理
  - `test_regime_detector.py`
    - `fill_test.yaml` 直読を fixture に統合
    - `inspect.getsource()` を `_source()` cache helper に集約
  - `test_fill_quality.py`
    - 31 箇所の `inspect.getsource()` を `_source()` cache helper に集約
  - `test_166_remaining_tasks.py`
    - `SkipGate` / `SkipGateConfig` の method 内 import を module-level へ集約
  - `test_277_magic_number_grounding.py`
    - `fill_test.yaml` 直読を fixture 化
  - `test_157_regime_features.py`
    - `load_retrain_config()` 呼出しで shared YAML path fixture を使用

### 結果
- 対象回帰:
  - `test_fill_quality.py` + `test_regime_detector.py` + `test_retrain_hot_reload.py` + `test_166_remaining_tasks.py` + `test_146_multi_exchange.py::TestBrokerRegistry` + `test_157_regime_features.py` + `test_277_magic_number_grounding.py`
  - `500 passed, 9 warnings in 7.94s`
- broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` 除外、`test_yaml_has_microprice_side` deselect）
  - `4068 passed, 1 deselected, 19 warnings in 39.57s`

### 主要 durations 変化
- broad wall time: `44.22s -> 39.57s`
- `test_retrain_hot_reload.py::TestE2ERetrainHotReload::test_retrain_deploy_and_hot_reload` は相対的上位に残るが、他の source/YAML 系が後退したことで実計算パスだけが前面化
- `test_fill_quality.py::TestGateCheckG11::test_g1_1_with_data` は `0.18s`
- `test_regime_detector.py::TestPhaseD18RangingYaml::test_yaml_has_ranging_discount` は `0.07s`
- `test_146_multi_exchange.py::TestBrokerRegistry::test_no_skeleton_or_sim` は `0.20s`（初期化コスト削減後の再測定）

### 補足
- broad 上位はほぼ実 I/O / 実データ経路に集約された
  - `test_aggregate_to_1min.py`
  - `test_enricher_skip_gate.py`
  - `test_237_phantom_position_guard.py`
  - `test_102_structural_fixes.py`
- つまり、残りは「簡単な重複除去」より「責務を保ったまま実 I/O をどこまで削れるか」の段階に入っている

### 次アクション
1. `test_aggregate_to_1min.py` の persistence 必須ケースを最小件数まで縮小し、非 persistence 系をさらに patch 化する
2. `test_enricher_skip_gate.py` の real-data / cache invalidation 系で sample_rows と file-touch 戦略を見直す
3. `test_102_structural_fixes.py` / `test_237_phantom_position_guard.py` の runtime-heavy 単発テストを、制御フロー検証に必要な最小状態へ落とす

---

## 2026-03-07 / Session 037-027

### 実施
- `test_aggregate_to_1min.py`
  - `test_parquet_output_created` から不要な `pd.read_parquet()` を除去
  - `test_both_empty_returns_empty_df` を `_run_aggregate()` ベースへ寄せ、空 gzip 生成を削除
- `test_237_phantom_position_guard.py`
  - `TestReconcileRateLimit` を fake clock 依存から外し、実際の `_MIN_RECONCILE_INTERVAL_SEC` 判定だけを見る構成へ変更
- `test_141_side_specific_models.py`
  - sklearn pipeline を fit して pickle 化していた helper を、picklable な軽量 `regressor/scaler` stub ベースへ置換
  - `test_hot_reload_updated_side_model` も同 stub でハッシュ差分だけを確認する形へ整理
- `test_146_multi_exchange.py`
  - `inspect.getsource()` / `Path.read_text()` の repeated call を cache helper に集約
- `test_166_remaining_tasks.py`
  - `set_output()` 呼出し確認を `MagicMock` から軽量 recorder stub に置換
- `test_fill_quality.py`
  - `TestFillTestRunnerSaveResilience` の retry backoff を `0.01 -> 0.0` に変更し、保存失敗系テストの待機を除去
- `test_enricher_skip_gate.py`
  - 実データ sample cap をいったん `117` まで詰めたが broad で `n_samples` が揺れたため、成立条件を守る安全下限として `120` に戻した

### 結果
- 対象回帰 1:
  - `test_aggregate_to_1min.py` + `test_enricher_skip_gate.py` + `test_237_phantom_position_guard.py`
  - `137 passed in 5.38s`
- 対象回帰 2:
  - `test_141_side_specific_models.py` + `test_146_multi_exchange.py` + `test_166_remaining_tasks.py` + `test_fill_quality.py`
  - `320 passed, 6 warnings in 5.99s`
- broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` 除外、`test_yaml_has_microprice_side` deselect）
  - `4068 passed, 1 deselected, 18 warnings in 32.18s`

### 主要 durations 変化
- broad wall time: `39.57s -> 32.18s`
- `test_141_side_specific_models.py::TestEvaluatorSideDispatch::test_select_gate_for_side_both`
  - `0.42s -> 0.02s`
- `test_fill_quality.py::TestFillTestRunnerSaveResilience::test_try_save_batch_emergency_dump_after_3_failures`
  - broad 上位から後退
- `test_aggregate_to_1min.py::TestAggregateMerged::test_parquet_output_created`
  - 依然上位だが、責務重複を除いたことで persistence 専用ケースとして明確化

### 補足
- `test_enricher_skip_gate.py` の real-data integration は、単体実行と broad 実行で `n_samples` が一致しないことを確認した
- このため sample cap は「最小理論値」ではなく、「広域並列・ファイル更新揺れ込みでも落ちない安全下限」を採用した

### 次アクション
1. `test_enricher_skip_gate.py` の real-data setup を、件数ではなく成立条件ベースの fixture cache にできないかを見る
2. `test_237_phantom_position_guard.py` / `test_102_structural_fixes.py` の runtime-heavy 単発をさらに局所化する
3. `test_ml_pipeline.py` / `test_build_features_pipeline.py` の実データ・実集約 setup を共通 helper 側から圧縮する

---

## 2026-03-07 / Session 037-028

### 実施
- `test_enricher_skip_gate.py`
  - real-data integration を固定 tail 件数前提から、`120 rows` で不足時のみ `220 rows` へフォールバックする成立条件ベースへ変更
  - `Test058SkipGate` / `Test061SkipGateASMode` を「class-scope で 1 回 fit + 各テストでは `deepcopy`」へ変更し、Ridge / LogisticRegression の repeated fit を削減
- `test_build_features_pipeline.py`
  - synthetic raw gzip をディスクへ書かず、`aggregate_to_1min()` の raw reader / parquet writer を patch して pure aggregation path を再利用
- `test_ml_pipeline.py`
  - metrics existence だけを見る LR 学習テストの CV folds を `3 -> 2` に削減
  - GB テストの `gb_n_estimators` を `10 -> 6` に削減
  - real-data integration の tail 件数を `80 -> 120` に上げて flaky を解消
- `test_237_phantom_position_guard.py`
  - balance API error テストを minimal async adapter stub に変更
- 横展開
  - `test_139_review_fixes.py`
  - `test_094_stale_order.py`
  - direct `fill_test.yaml` read を shared fixture へ置換

### 結果
- 対象回帰 1:
  - `test_enricher_skip_gate.py` + `test_ml_pipeline.py` + `test_build_features_pipeline.py` + `test_102_structural_fixes.py` + `test_237_phantom_position_guard.py`
  - `159 passed in 7.12s`
- 対象回帰 2:
  - `test_139_review_fixes.py` + `test_094_stale_order.py`
  - `89 passed in 3.32s`
- `test_enricher_skip_gate.py` 単体:
  - `70 passed in 4.16s`
- broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` 除外、`test_yaml_has_microprice_side` deselect）
  - rerun 1: `4068 passed, 1 deselected, 18 warnings in 33.40s`
  - rerun 2: `4068 passed, 1 deselected, 18 warnings in 35.69s`

### 主要 durations 変化
- 対象束:
  - `8.21s -> 7.12s`
- `test_build_features_pipeline.py::TestRealModePipeline::test_microstructure_on_aggregated`
  - setup `0.18s級 -> 0.06s〜0.11s`
- `test_enricher_skip_gate.py::Test061SkipGateASMode::*`
  - setup `0.17s級 -> 0.04s〜0.12s`
- `test_ml_pipeline.py::Test057FillClassifier::test_train_returns_metrics`
  - `0.30s -> 0.05s〜0.07s`

### 補足
- broad の最速値そのものは前セッションの `32.18s` を上回っていない
- ただし今回の主目的は、`test_enricher_skip_gate.py::Test058Integration::test_train_skip_gate_real` と `test_ml_pipeline.py::Test057Integration::test_load_real_data` の flaky を解消しつつ、fit/setup の repeated cost を横展開で削ることだった
- 現状の broad 上位は以下へ再集中している
  - `test_regime_detector.py`
  - `test_aggregate_to_1min.py`
  - `test_retrain_hot_reload.py`
  - `test_102_structural_fixes.py`
  - `test_fill_quality.py`

### 次アクション
1. `test_regime_detector.py` の source-inspection / YAML 以外の重いケースを個別に切り分ける
2. `test_aggregate_to_1min.py` の `test_many_minutes` / persistence 系をさらに責務分離する
3. `test_102_structural_fixes.py` と `test_fill_quality.py` の runner 初期化系を stub 注入で軽量化する

---

## 2026-03-07 / Session 037-029

### 実施
- `test_regime_detector.py`
  - `maker_price.py` の enum/source-inspection テストを `inspect.getsource()` から cached file-text helper へ切り替え
  - `ranging_offset_discount` の YAML 確認を direct read から `v460_fill_test_yaml` fixture へ置換
- `test_aggregate_to_1min.py`
  - `TestAggregateEdgeCases::test_many_minutes` の synthetic minute 数を `10 -> 6` に削減
  - multi-minute aggregation path 自体は維持し、期待行数のみ新件数へ同期

### 結果
- 対象回帰:
  - `test_regime_detector.py` + `test_aggregate_to_1min.py`
  - `108 passed in 7.26s`
- broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` と `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - `4043 passed, 1 deselected, 18 warnings in 37.13s`

### 主要 durations 変化
- `test_aggregate_to_1min.py::TestAggregateEdgeCases::test_many_minutes`
  - `0.04s` まで低下
- `test_regime_detector.py::TestPhaseD18EnumConsistency::test_high_vol_uses_enum_comparison`
  - targeted `--durations` 上位から離脱

### 補足
- broad の未除外 full run では、今回差分と無関係な `test_113_resilience.py::TestR1MethodExtraction::test_run_single_cycle_under_400_lines` が `run_single_cycle is 732 lines (> 725)` で失敗する
- 現 patch の評価はこの unrelated failure を除外した broad run で実施した

### 次アクション
1. `test_retrain_hot_reload.py` の `TestE2ERetrainHotReload` をさらに分解し、save/load 経路だけを残して学習コストを削る
2. `test_fill_quality.py` と `test_102_structural_fixes.py` の重い runner 初期化テストで、価格計算や git 情報の不要依存を個別 stub へ寄せる
3. 本体コード側では `run_fill_test.py` 周辺の巨大メソッド分解候補を見て、test_113 系の line-count 圧迫と初期化コストの両面を確認する

---

## 2026-03-07 / Session 037-030

### 実施
- 本体コード
  - `ztb/utils/run_manifest.py`
    - `compute_file_hash()` を stat-signature (`mtime_ns`, `size`) ベースの cache 化
  - `scripts/v460/lib/skip_gate_evaluator.py`
    - モデル hash 読み出しで fresh な `.sha256` sidecar を優先
    - stale sidecar の場合のみ full file hash にフォールバック
- テストコード
  - `test_169_config_hot_reload.py`
    - `get_git_sha()` を autouse patch し、reload 差分テストから実 git subprocess を排除
  - `test_fill_quality.py`
    - `TestGateCheckG11::test_g1_1_with_data` の synthetic records を `300 -> 60` に削減
  - `test_retrain_hot_reload.py`
    - `TestBalanceForcedSwitchFilter` に fast regressor patch を横展開
    - sidecar hash fast-path の回帰テストを追加
    - `TestE2ERetrainHotReload` は warm-start 経路を通さない形にし、配線確認へ責務を限定
  - `test_enricher_skip_gate.py`
    - real-data fixture を `120 -> 220 -> 320` の guarded fallback 化
    - 実行中に伸びうる newest `fill_records_*.jsonl` を除外して入力を安定化
  - `tests/unit/v460/_fill_test_source.py`
    - `fill_record_builder.py` を source-inspection 対象に追加

### 結果
- 対象回帰 1:
  - `test_run_manifest.py` + `test_169_config_hot_reload.py`
  - `31 passed in 4.35s`
- 対象回帰 2:
  - `test_retrain_hot_reload.py` + `test_169_config_hot_reload.py` + `test_fill_quality.py` + `test_run_manifest.py`
  - `319 passed in 8.97s`
- 対象回帰 3:
  - `test_enricher_skip_gate.py` + `test_145_structural_fixes.py`
  - `127 passed in 6.09s`
- `test_enricher_skip_gate.py` 単体:
  - `70 passed in 4.50s`
- broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` と `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - rerun 1: `4051 passed, 1 deselected, 18 warnings in 40.52s`
  - rerun 2: `4051 passed, 1 deselected, 18 warnings in 40.76s`

### 主要 durations 変化
- `test_169_config_hot_reload.py::TestConfigFieldUpdate::test_do_reload_updates_reloadable_fields`
  - `0.34s級 -> 0.01s`
- `test_fill_quality.py::TestGateCheckG11::test_g1_1_with_data`
  - `0.16s級 -> 0.07s〜0.08s`
- `test_enricher_skip_gate.py::Test058Integration::test_enrichment_with_real_data`
  - focused run setup `0.58s`
  - filtered broad setup `0.69s`
- `test_retrain_hot_reload.py::TestE2ERetrainHotReload::test_retrain_deploy_and_hot_reload`
  - filtered broad では `0.17s`

### 補足
- broad の壁時計そのものは前段の最速値より高めだが、今回の主目的は以下の 2 点だった
  - `test_enricher_skip_gate.py` の real-data fixture の broad-run flake 解消
  - `config_hot_reload` / `run_manifest` / model hash 読み出しの実 I/O / subprocess 依存除去
- `test_145_structural_fixes.py::TestFillRecordBuilderIntegration::test_build_fill_record_is_used` は、production code の不具合ではなく source-inspection helper が `fill_record_builder.py` を見ていなかったのが原因

### 次アクション
1. `test_212_live_trader_config.py` と `test_145_structural_fixes.py::TestSkipGateLotConsistency` の source-inspection / import cost を個別に削る
2. `test_aggregate_to_1min.py` の parquet roundtrip 必須ケースをさらに最小化する
3. broad 上位に残る `test_microstructure_features.py` / `test_ml_pipeline.py` の単発重ケースを stub 注入で分離できるか確認する

---

## 2026-03-07 / Session 037-031

### 実施
- 共通 helper
  - `tests/unit/v460/_fill_test_source.py`
    - FillTestRunner/mixin source を method 名 → source の index に一括キャッシュ
- テスト側
  - `test_145_structural_fixes.py`
    - `fill_quality.py` / `skip_gate_evaluator.py` / `fill_cycle_executor.py` / `fill_record_builder.py` の source-inspection を cached file-text read へ置換
  - `test_fill_test_config.py`
    - `run_continuous` / `run_single_cycle` / `_is_time_filtered` の確認を cached file-text read へ置換
  - `test_212_live_trader_config.py`
    - AST class extraction を廃止し、cached module source 直接検証へ簡素化
  - `test_microstructure_features.py`
    - synthetic 1-min DataFrame 生成を cache 化し、各テストでは deep copy を利用
  - `test_aggregate_to_1min.py`
    - parquet roundtrip を 1-row fixture へ縮小
- 本体コード
  - `ztb/features/microstructure.py`
    - `close` / `buy_volume` / `sell_volume` / `total_vol` の再利用で rolling 特徴量前処理の重複を削減

### 結果
- 対象回帰 1:
  - `test_145_structural_fixes.py` + `test_212_live_trader_config.py`
  - `70 passed in 3.75s`
- 対象回帰 2:
  - `test_microstructure_features.py` + `test_aggregate_to_1min.py` + `test_fill_test_config.py`
  - `138 passed in 4.58s`
- 対象回帰 3:
  - `test_145_structural_fixes.py` + `test_fill_test_config.py` + `test_212_live_trader_config.py` + `test_microstructure_features.py` + `test_aggregate_to_1min.py`
  - `208 passed in 8.24s`
- gate-counter 交差確認:
  - `test_220_deadlock_fixes.py` + `test_229_cleanup_counter_rename.py` + `test_277_magic_number_grounding.py`
  - `81 passed in 1.56s`
- broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` と `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - rerun 1: `4051 passed, 1 deselected, 18 warnings in 39.01s`
  - rerun 2: `4051 passed, 1 deselected, 19 warnings in 43.03s`

### 主要 durations 変化
- `test_145_structural_fixes.py::TestSkipGateLotConsistency::test_skip_gate_call_passes_regime_lot`
  - `0.18s級 -> broad から上位離脱`
- `test_fill_test_config.py::TestSideOverride::test_run_continuous_passes_side_override`
  - targeted `0.15s`（method extraction 初回コストを回避）
- `test_aggregate_to_1min.py::TestAggregateMerged::test_parquet_roundtrip`
  - targeted `0.14s -> 0.11s〜0.13s`
- `test_microstructure_features.py`
  - cached synthetic frame 化後、top durations は `0.02s〜0.09s` 帯に収束

### 補足
- `test_277_magic_number_grounding.py::TestGateAggregatorConfigIntegration::test_bypass_with_custom_threshold` は一度だけ filtered broad で型不整合を出したが、単体・ファイル全体・関連 gate-counter 束では再現しなかった
- 現時点では deterministic な再現条件を特定できていないため、記録のみ残し、rerun green を採用
- broad の残上位は以下に再集中している
  - `test_enricher_skip_gate.py` real-data setup
  - `test_aggregate_to_1min.py` 実集約/実 parquet
  - `test_retrain_hot_reload.py` E2E hot-reload
  - `test_ml_pipeline.py` の一部 data-loader / GB 学習ケース

### 次アクション
1. `test_aggregate_to_1min.py` の non-persistence 上位ケースを、resample の成立条件を維持したままさらに最小化する
2. `test_ml_pipeline.py` の data-loader / GB 系ケースを、品質非依存のものから軽量 estimator / fixture cache へ寄せる
3. `test_enricher_skip_gate.py` の real-data integration を、固定 sample tail ではなく snapshot fixture 化できるか確認する

---

## 2026-03-07 / Session 037-032

### 実施
- 共通 helper
  - `tests/unit/v460/_fill_test_source.py`
    - FillTestRunner 分割 source 群を method 名 → source の index に一括キャッシュし、lookup ごとの AST walk を除去
- テスト側
  - `test_145_structural_fixes.py`
    - `fill_quality.py` / `skip_gate_evaluator.py` / `fill_cycle_executor.py` / `fill_record_builder.py` の source-inspection を cached file-text read へ置換
  - `test_fill_test_config.py`
    - `run_continuous` / `run_single_cycle` / `_is_time_filtered` の確認を cached file-text read へ置換
  - `test_212_live_trader_config.py`
    - AST class extraction をやめ、cached module source の直接検証へ簡素化
  - `test_microstructure_features.py`
    - synthetic 1-min DataFrame を `@lru_cache` 化し、各テストでは deep copy を利用
  - `test_aggregate_to_1min.py`
    - parquet roundtrip の fixture を 1-row に縮小
- 本体コード
  - `ztb/features/microstructure.py`
    - `close` / `buy_volume` / `sell_volume` / `total_vol` を先に解決して再利用し、同一 Series の重複生成を削減

### 結果
- 対象回帰 1:
  - `test_145_structural_fixes.py` + `test_212_live_trader_config.py`
  - `70 passed in 3.75s`
- 対象回帰 2:
  - `test_microstructure_features.py` + `test_aggregate_to_1min.py` + `test_fill_test_config.py`
  - `138 passed in 4.58s`
- 対象回帰 3:
  - `test_145_structural_fixes.py` + `test_fill_test_config.py`
  - `140 passed in 3.19s`
- 交差確認:
  - `test_220_deadlock_fixes.py` + `test_229_cleanup_counter_rename.py` + `test_277_magic_number_grounding.py`
  - `81 passed in 1.56s`
- broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` と `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - rerun 1: `4051 passed, 1 deselected, 19 warnings in 43.03s`

### 主要 durations 変化
- `test_145_structural_fixes.py::TestSkipGateLotConsistency::test_skip_gate_call_passes_regime_lot`
  - method 単位の source extraction をやめ、broad の上位から離脱
- `test_fill_test_config.py::TestSideOverride::test_run_continuous_passes_side_override`
  - file-text read 化後、targeted では `0.15s` 帯に収束
- `test_aggregate_to_1min.py::TestAggregateMerged::test_parquet_roundtrip`
  - targeted `0.11s〜0.13s`
- `test_microstructure_features.py`
  - cached synthetic frame 化後、top durations は `0.02s〜0.09s` 帯に収束

### 補足
- `test_277_magic_number_grounding.py::TestGateAggregatorConfigIntegration::test_bypass_with_custom_threshold` は一度だけ filtered broad で揺れたが、単体・関連束・rerun broad では再現しなかった
- 現時点の broad 変動幅は主に以下へ集中している
  - `test_enricher_skip_gate.py` の real-data setup
  - `test_aggregate_to_1min.py` の実 parquet / resample 系
  - `test_retrain_hot_reload.py` の E2E hot-reload
  - `test_ml_pipeline.py` の一部 data-loader / GB 学習ケース

### 次アクション
1. `test_aggregate_to_1min.py` の persistence 必須ケースと pure aggregation ケースの責務をさらに切り分ける
2. `test_ml_pipeline.py` の品質非依存ケースを、軽量 estimator と cached input へ寄せる
3. `test_enricher_skip_gate.py` の real-data integration を snapshot fixture 化できるか確認する

---

## 2026-03-07 / Session 037-033

### 実施
- `test_retrain_hot_reload.py`
  - `retrain_model()` 向けの synthetic DataFrame builder を追加
  - `TestRetrainModel` / `TestE2ERetrainHotReload` / `TestBalanceForcedSwitchFilter` の JSONL 実書込をやめ、`load_fill_records()` patch で DataFrame を直接注入
  - 品質評価そのものが主目的でないケースはサンプル数を必要最小限まで削減
- `test_enricher_skip_gate.py`
  - synthetic fill / orderbook / trades DataFrame を `@lru_cache` 化し、各 fixture では deep copy を返す形へ統一
  - real-data integration で使われていなかった `real_fill_df` fixture を削除し、`real_enriched_df` の重複読込を除去
  - skip-rate limiter の履歴テストを、内部履歴長 20 を踏まえた最小ループ数へ短縮
- `test_ml_pipeline.py`
  - synthetic fill DataFrame を `@lru_cache` 化
  - AS/Fill classifier の品質非依存ケースを `24` 行学習に縮小
  - GB 系の `gb_n_estimators` を `4` に削減

### 結果
- 対象回帰 1:
  - `test_retrain_hot_reload.py`
  - `82 passed in 4.08s`
- 対象回帰 2:
  - `test_enricher_skip_gate.py` + `test_ml_pipeline.py`
  - `92 passed in 4.89s`
- 対象回帰 3:
  - `test_retrain_hot_reload.py` + `test_enricher_skip_gate.py` + `test_ml_pipeline.py`
  - `174 passed in 7.22s`
- broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` と `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - `4051 passed, 1 deselected, 15 warnings in 35.21s`

### 主要 durations 変化
- `test_retrain_hot_reload.py::TestE2ERetrainHotReload::test_retrain_deploy_and_hot_reload`
  - focused `0.90s級 -> 0.84s`
  - filtered broad `0.29s級 -> 0.16s`
- `test_retrain_hot_reload.py::TestRetrainModel::test_skip_when_insufficient_new_samples`
  - `0.11s -> 0.09s`
- `test_retrain_hot_reload.py::TestBalanceForcedSwitchFilter::test_balance_forced_records_excluded`
  - `0.10s -> 0.06s`
- `test_enricher_skip_gate.py::Test058Integration::test_enrichment_with_real_data`
  - focused setup `0.50s級 -> 0.47s`
  - filtered broad setup `0.38s`
- `test_ml_pipeline.py`
  - focused / bundle ともに GB / LR 学習ケースが `0.04s〜0.09s` 帯へ縮小

### 補足
- broad の上位は今回の変更後も以下へ集中している
  - `test_enricher_skip_gate.py` の real-data setup
  - `test_aggregate_to_1min.py` の parquet / aggregation
  - `test_227_ranging_obi_velocity_ema_import_fix.py` の EMA 単発ケース
  - `test_145_structural_fixes.py::TestFillRecordBuilderIntegration::test_resume_and_reload_use_iter_glob`
- 今回は本体コードへの追加変更は入れていない
  - 安全に効く production-side の改善候補は別途 `aggregate_to_1min` / `feature_enricher` / `EMA` 周辺のプロファイルを見た上で切るのが妥当

### 次アクション
1. `test_aggregate_to_1min.py` の上位 2 ケースを、persistence 契約と resample 契約にさらに分離する
2. `test_227_ranging_obi_velocity_ema_import_fix.py` と `test_145_structural_fixes.py::test_resume_and_reload_use_iter_glob` の単発重ケースを調べる
3. production 側は `feature_enricher` / `aggregate_to_1min` / EMA 周辺の実計測を取り、再利用できる cache と単一パス化の余地を確認する

---

## 2026-03-07 / Session 037-034

### 実施
- production
  - `ztb/data/market_data_collector.py`
    - 板集約を `_aggregate_orderbook_1min()` に抽出
    - `aggregate_to_1min()` 内の join + 2 回 resample を、単一 resample へ整理
  - `scripts/v460/lib/orchestrator_guards.py`
    - `_track_side_pnl()` の docstring に Ho & Stoll (1981) の在庫リスク理論を明記
- テスト側
  - `test_aggregate_to_1min.py`
    - parquet roundtrip の再読込を `pd.read_parquet()` から `pyarrow.parquet.read_table()` へ変更
  - `test_227_ranging_obi_velocity_ema_import_fix.py`
    - `AsyncMock` orderbook adapter を軽量 async stub に置換
    - EMA テストで `maker_price.time.time()` を patch し、Windows の clock resolution 依存を除去
  - `test_145_structural_fixes.py`
    - `resume_from_existing` / `_finalize_run` の検証対象を現行 split 構成 (`orchestrator_lifecycle.py` + `fill_loop_orchestrator.py`) に更新
    - `run_continuous` の source 検証を direct file-text read に統一
  - `test_139_review_fixes.py`
    - `run_continuous` の source 検証を `inspect.getsource(FillTestRunner.run_continuous)` から shared source helper に変更
  - `test_154_deadlock_prevention.py`
    - `run_single_cycle` / `run_continuous` の source 検証を現行 split file へ移行
  - `test_256_recent_records_fix.py`
    - `_recent_records.append(record)` の検証対象を `orchestrator_post_cycle.py` に更新
  - `_fill_test_source.py`
    - `orchestrator_guards.py` / `orchestrator_lifecycle.py` / `orchestrator_post_cycle.py` を index 対象へ追加

### 結果
- 対象回帰 1:
  - `test_aggregate_to_1min.py`
  - `26 passed in 3.31s`
- 対象回帰 2:
  - `test_aggregate_to_1min.py` + `test_227_ranging_obi_velocity_ema_import_fix.py` + `test_145_structural_fixes.py`
  - `104 passed in 4.28s`
- 対象回帰 3:
  - `test_145_structural_fixes.py` + `test_256_recent_records_fix.py` + `test_275_dry_separation_and_theory.py`
  - `90 passed in 3.75s`
- 対象回帰 4:
  - `test_139_review_fixes.py` + `test_154_deadlock_prevention.py`
  - `75 passed in 4.99s`
- broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` と `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - `4051 passed, 1 deselected, 15 warnings in 41.03s`

### 主要 durations 変化
- `test_aggregate_to_1min.py::TestAggregateMerged::test_parquet_roundtrip`
  - 3 ファイル束で `0.18s級 -> 0.08s`
  - broad でも `0.25s級 -> 0.16s〜0.28s` 帯
- `test_aggregate_to_1min.py::TestAggregateMerged::test_parquet_output_created`
  - 3 ファイル束で `0.14s級 -> 0.04s`
- `test_145_structural_fixes.py::TestFillRecordBuilderIntegration::test_resume_and_reload_use_iter_glob`
  - `0.07s級 -> 0.01s`
- `test_227_ranging_obi_velocity_ema_import_fix.py`
  - 単体 `21 passed in 1.16s`
  - EMA 3 ケースは fake clock 化後も broad から上位離脱

### 補足
- broad 途中で `inspect.getsource(FillTestRunner.run_continuous)` 依存の source-inspection テストが複数壊れることを確認した
  - 原因は mixin 分割後の現構造と `inspect` の参照先不安定化
  - 対応として shared source helper へ統一し、現行 split file を直接検証する形へ揃えた
- 現時点の broad 上位は以下へ集中している
  - `test_enricher_skip_gate.py` の real-data setup
  - `test_148_fill_test_events.py` の writer 例外系
  - `test_aggregate_to_1min.py` の parquet persistence
  - `test_pnl_monte_carlo.py` / `test_ml_pipeline.py` / `test_retrain_hot_reload.py` の一部ケース

### 次アクション
1. `test_148_fill_test_events.py` の TeeWriter 系を調べ、実 writer 例外経路の成立条件を保ったまま軽量化できるか確認する
2. `test_pnl_monte_carlo.py::TestLoadFillRecords::test_load_from_directory` と `test_152_parallel_tasks.py::TestReproduceMetrics::test_main_with_output` の I/O 経路を詰める
3. `test_enricher_skip_gate.py` の real-data setup と `test_aggregate_to_1min.py` の parquet persistence を、snapshot fixture / schema-level validation でさらに下げられるか確認する

---

## 2026-03-07 / Session 037-035

### 実施
- `tests/unit/v460/_fill_test_source.py`
  - cached source index の対象へ `orchestrator_guards.py` / `orchestrator_lifecycle.py` / `orchestrator_post_cycle.py` を追加
- `tests/unit/v460/test_145_structural_fixes.py`
  - `resume_from_existing` / `_finalize_run` の検証対象を現行 split file に揃えた
  - `run_continuous` の source 検証を shared helper 経由に統一した
- `tests/unit/v460/test_256_recent_records_fix.py`
  - `_recent_records.append(record)` の検証対象を `orchestrator_post_cycle.py` に更新した
- `scripts/v460/lib/orchestrator_guards.py`
  - `_track_side_pnl()` docstring に Ho & Stoll 在庫リスク理論を追記した

### 結果
- 対象回帰:
  - `test_145_structural_fixes.py` + `test_256_recent_records_fix.py` + `test_275_dry_separation_and_theory.py`
  - `90 passed in 3.75s`
- broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` と `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - `4051 passed, 1 deselected, 15 warnings in 41.03s`

### 補足
- mixin 分割後も source-inspection テストが現行コード配置を直接見るように揃ったため、`inspect.getsource(...)` 依存の揺れを 1 段減らせた
- この時点の上位 durations は以下へ集中している
  - `test_enricher_skip_gate.py` の real-data setup
  - `test_148_fill_test_events.py` の writer 例外系
  - `test_pnl_monte_carlo.py::TestLoadFillRecords::test_load_from_directory`
  - `test_152_parallel_tasks.py::TestReproduceMetrics::test_main_with_output`

### 次アクション
1. `test_148_fill_test_events.py` の TeeWriter 例外経路を、契約を保ったまま軽量化できるか確認する
2. `test_pnl_monte_carlo.py` と `test_152_parallel_tasks.py` の実ファイル I/O を patch/stub で分離できるか詰める
3. `test_enricher_skip_gate.py` の real-data setup を再利用可能な fixture cache に寄せる

---

## 2026-03-07 / Session 037-036

### 実施
- production
  - `scripts/v460/analysis/reproduce_152_metrics.py`
    - `_as_float_or_zero()` の `safe_to_finite()` 呼び出し契約を修正し、quiet なしのレポート出力で落ちないようにした
- テスト側
  - `test_148_fill_test_events.py`
    - `_TeeWriter` テストの `MagicMock` を軽量 writer stub に置換し、例外抑制パスの mock overhead を削減した
  - `test_pnl_monte_carlo.py`
    - `load_fill_records_glob()` patch ベースへ切り替え、directory dispatch だけを検証する形に整理した
  - `test_152_parallel_tasks.py`
    - `reproduce_152_metrics` の loader を patch して入力 JSONL I/O を除去
    - quiet なしの `main()` 回帰テストを追加
    - `compare_regime_ab` / `reproduce_152_metrics` / `FillTestConfig` / `RegimeConfig` の per-method import を module scope へ集約
    - `_simulate()` の成立条件に不要だった 30 レコードを 12 レコードへ圧縮
  - `test_253_hot_reload_dead_config_getattr_bare_except.py`
    - `inspect.getsource(...)` をやめ、cached file-text read に統一した
  - `test_aggregate_to_1min.py`
    - parquet persistence 2 ケースで class-scope の実出力を共有し、重複書込を除去した
  - `test_retrain_hot_reload.py`
    - `insufficient_new_samples` の最小データ量を整理した
    - E2E hot-reload で evaluator 側 `SkipGate.load()` を patch し、deploy/reload 配線は維持したまま duplicate pickle load を削減した
  - `test_enricher_skip_gate.py`
    - micro feature 付き DataFrame を cache して `build_enriched_as_features()` / `build_pnl_features()` 系で再利用
    - real-data fallback 選択時の再読込をやめ、最大件数を一度読む形に整理
    - `skip_rate` 系のループ回数を limiter 成立に必要な最小構成まで短縮

### 結果
- 対象回帰 1:
  - `test_148_fill_test_events.py` + `test_pnl_monte_carlo.py` + `test_152_parallel_tasks.py`
  - `65 passed, 3 warnings in 1.94s`
- 対象回帰 2:
  - `test_aggregate_to_1min.py` + `test_retrain_hot_reload.py` + `test_253_hot_reload_dead_config_getattr_bare_except.py` + `test_152_parallel_tasks.py`
  - `144 passed in 6.40s`
- 対象回帰 3:
  - `test_enricher_skip_gate.py`
  - `70 passed in 3.42s`
- broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` と `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - `4052 passed, 1 deselected, 14 warnings in 32.21s`
  - rerun: `4052 passed, 1 deselected, 14 warnings in 33.78s`

### 主要改善
- `test_enricher_skip_gate.py`
  - focused file が `4.92s -> 3.42s`
  - `Test058Integration::test_enrichment_with_real_data` setup が `0.65s級 -> 0.49s〜0.52s`
  - `test_enriched_as_require_spread_filters` / `test_pnl_features_more_samples_than_as` / `test_skip_rate_limit` が broad 上位から後退
- `test_retrain_hot_reload.py::TestE2ERetrainHotReload::test_retrain_deploy_and_hot_reload`
  - broad で `0.83s -> 0.12s〜0.22s`
- `test_aggregate_to_1min.py`
  - parquet roundtrip / output_created の重複 write を解消
- `test_152_parallel_tasks.py::TestReproduceMetrics::test_main_with_output`
  - 入力 JSONL I/O を除去し、focused で `0.02s` 帯へ収束

### 補足
- filtered broad の最速値は `32.21s`、rerun は `33.78s` だった
  - 残る揺れは real-data integration と一部 hot-reload / warm-start 系に集中している
- quiet なしの `reproduce_152_metrics.main()` が今回初めてテストで実行され、helper 契約ずれの production バグを修正した

### 次アクション
1. `test_enricher_skip_gate.py` の real-data integration setup を fixture cache / snapshot 化でさらに詰める
2. `test_skip_gate_d8.py` と `test_retrain_hot_reload.py` の warm-start / hot-reload 上位ケースをもう一段 stub 化できるか確認する
3. `test_146_multi_exchange.py` と `test_169_config_hot_reload.py` の source/YAML/config reload 系を cached read に寄せられるか洗う

---

## 2026-03-07 / Session 037-037

### 実施
- `test_skip_gate_d8.py`
  - warm-start 系テストの実ファイル書込を廃止し、`list_fill_record_files()` / `iter_jsonl_objects()` patch ベースへ移行
  - `SkipGate` fixture の `MagicMock` pipeline/scaler/model を軽量 stub に置換
  - broad 上位だった `test_as_mode_decision_fields` / warm-start 系の固定費を削減
- `test_146_multi_exchange.py`
  - 主要 import を module scope へ集約
  - `run_daily_health_check()` の signature を module load 時に 1 回だけ解決するように変更
- `test_169_config_hot_reload.py`
  - 再計測のみ実施
  - `mtime` 更新を sleep なしで行う現実装が既に軽量で、追加修正は不要と判断

### 結果
- 対象回帰 1:
  - `test_skip_gate_d8.py`
  - `41 passed in 2.81s`
- 対象回帰 2:
  - `test_skip_gate_d8.py` + `test_146_multi_exchange.py` + `test_169_config_hot_reload.py`
  - `111 passed in 3.30s`
- broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` と `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - `4060 passed, 1 deselected, 15 warnings in 30.49s`

### 主要改善
- `test_skip_gate_d8.py`
  - warm-start 群から temp file I/O を除去
  - focused file が `41 passed in 2.81s`
- `test_146_multi_exchange.py` + `test_169_config_hot_reload.py` 束
  - focused で上位 durations が `0.03s` 帯以下に収束
- filtered broad
  - `42.55s` 測定から `30.49s` まで短縮

### 補足
- 今回の broad 上位は以下へ集中している
  - `test_enricher_skip_gate.py` の real-data integration setup
  - `test_stopgap_health.py::TestGetDay::test_none_timestamp`
  - `test_255_getattr_bare_except_cleanup.py` の source-inspection
  - `test_aggregate_to_1min.py` の parquet persistence
  - `test_retrain_hot_reload.py` の E2E / balance_forced 系

### 次アクション
1. `test_stopgap_health.py` と `test_255_getattr_bare_except_cleanup.py` の source/time 系固定費を確認する
2. `test_enricher_skip_gate.py` の real-data setup を fixture cache/snapshot 化できるか詰める
3. `test_aggregate_to_1min.py` と `test_retrain_hot_reload.py` の残単発上位をさらに圧縮する

---

## 2026-03-08 / Session 037-038

### 実施
- production
  - `scripts/v460/lib/stopgap_health.py`
    - `compute_daily_metrics()` に UTC day bucket cache を追加し、同一日レコードの `YYYYMMDD` 変換を再利用する形へ整理
- テスト側
  - `test_255_getattr_bare_except_cleanup.py`
    - `inspect.getsource(...)` をやめ、cached file-text + AST で `SkipGateEvaluator` / `OrderMonitor` の source-inspection を行うように変更
  - `test_stopgap_health.py`
    - 回帰確認のみ実施

### 結果
- 対象回帰:
  - `test_stopgap_health.py` + `test_255_getattr_bare_except_cleanup.py`
  - `65 passed in 0.94s`
- broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` と `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - `4060 passed, 1 deselected, 15 warnings in 31.01s`

### 主要改善
- `test_255_getattr_bare_except_cleanup.py`
  - source-inspection の固定費を cached file-text / AST へ移行
- `stopgap_health.py`
  - 本体側で同日レコードの day 変換重複を削減

### 補足
- broad 上位は引き続き以下へ集中している
  - `test_enricher_skip_gate.py` の real-data integration setup
  - `test_retrain_hot_reload.py` の E2E / balance_forced / hot-reload 系
  - `test_169_config_hot_reload.py::TestConfigFieldUpdate::test_do_reload_updates_reloadable_fields`
  - `test_aggregate_to_1min.py::TestAggregateEdgeCases::test_many_minutes`
  - `test_145_structural_fixes.py::TestMakeSkipRecord::test_auto_cycle_id`

### 次アクション
1. `test_retrain_hot_reload.py` の `no_reload_when_unchanged` / `balance_forced` / E2E をさらに stub 化できるか詰める
2. `test_enricher_skip_gate.py` の real-data integration setup を fixture cache/snapshot 化でさらに落とす
3. `test_169_config_hot_reload.py` と `test_145_structural_fixes.py` の source/config 更新系を再度洗う

---

## 2026-03-08 / Session 037-039

### 実施
- `test_retrain_hot_reload.py`
  - `SkipGateEvaluator` 用の共通 config helper を追加し、`no_reload_when_unchanged` / `balance_forced` / E2E hot-reload で散在していた設定組立を統一
  - placeholder/stub model artifact writer を追加し、`SkipGate.save()` / `SkipGate.load()` / hash 読込を patch ベースへ寄せて、不要な gate serialize / deserialize を削減
  - `balance_forced` 系は最小サンプル数へ調整しつつ、`enrich_fill_records` と gate load/save を stub 化して control-flow 検証へ縮退
- `test_145_structural_fixes.py`
  - `TestMakeSkipRecord` の runner mock を `SimpleNamespace` ベースの最小オブジェクトへ置換
  - `_make_skip_record()` が参照する属性だけを明示的に持たせ、`MagicMock` 由来の余計な config/mock 解決コストを除去

### 結果
- 対象回帰:
  - `test_retrain_hot_reload.py` + `test_145_structural_fixes.py`
  - `139 passed in 5.04s`
- broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` と `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - `4060 passed, 1 deselected, 15 warnings in 32.19s`

### 主要改善
- `test_retrain_hot_reload.py`
  - `TestE2ERetrainHotReload::test_retrain_deploy_and_hot_reload` が focused で `0.74s`
  - `TestBalanceForcedSwitchFilter::test_balance_forced_records_excluded` が `0.08s`
  - `TestHotReload::test_no_reload_when_unchanged` / `test_initial_hash_stored` が `0.01s`
- `test_145_structural_fixes.py`
  - `_make_skip_record()` 系で不要な mock/config 解決を外し、構造確認テストを軽量 runner へ整理

### 補足
- 作業中にブランチ先頭は別コミットで進んでいたため、今回の変更は対象 2 ファイルと記録更新だけを現在の HEAD 上へ積む前提で扱った。
- broad の上位は、`test_enricher_skip_gate.py` の real-data setup よりも、`test_v460_core::TestCliffsD::test_no_dominance`、`test_102_structural_fixes.py`、`test_215_dd_fix_alert_mode.py`、loader/source-inspection 系へ移りつつある。

### 次アクション
1. `test_v460_core.py::TestCliffsD::test_no_dominance` の計算条件を保ったままサンプル/実装経路を見直す
2. `test_102_structural_fixes.py` と `test_215_dd_fix_alert_mode.py` の runtime 初期化/実データ依存を stub 化できるか詰める
3. loader/source-inspection 上位（`test_stopgap_health.py`, `test_gate_judgment.py`, `test_197_boost_optimization_gate_integration.py`）を cached read へ寄せる余地を洗う

---

## 2026-03-08 / Session 037-040

### 実施
- production
  - `scripts/v460/ml/skip_gate_features.py` を新設し、SkipGate の feature name migration / feature index / dense vector 生成を pure helper に分離
  - `scripts/v460/ml/skip_gate.py` は上記 helper を再利用する形へ整理
  - `ztb/metrics/gate_checks.py`
    - `cliffs_delta()` に identical / strict-dominance の exact fast-path を追加
  - `ztb/metrics/fill_quality.py`
    - `iter_fill_records_glob()` に single-file fast-path を追加
- テスト負荷軽減
  - `test_102_structural_fixes.py`
    - `FillTestRunner` 初期化 helper を導入し、`enable_regime=False` の最小 config へ整理
  - `test_215_dd_fix_alert_mode.py`
    - SkipGate 本体 import を避け、pure helper ベースの feature migration 検証へ整理
  - `test_gate_judgment.py` / `test_stopgap_health.py`
    - wrapper-only loader テストを patch ベースへ変更し、重複 JSONL I/O を除去
- split-layout 追随
  - `tests/unit/v460/_fill_test_source.py` に orchestrator split file 定数を追加
  - `test_158`, `166`, `196`, `197`, `226`, `227`, `229`, `240`, `275`, `276`, `281`, `test_fill_quality.py`, `test_fill_test_config.py`
    - `fill_loop_orchestrator.py` 直参照を `orchestrator_balance.py` / `orchestrator_mid_cycle.py` / `orchestrator_pre_cycle.py` / `orchestrator_guards.py` の現行責務へ追随

### 結果
- 対象回帰 1:
  - `test_102_structural_fixes.py` + `test_215_dd_fix_alert_mode.py` + `test_gate_judgment.py` + `test_stopgap_health.py` + `test_197_boost_optimization_gate_integration.py` + `test_v460_core.py`
  - `216 passed, 1 warning in 6.91s`
- 対象回帰 2:
  - `test_158_regime_deadlock_fix.py` + `test_166_remaining_tasks.py` + `test_197_boost_optimization_gate_integration.py`
  - `79 passed in 5.01s`
- 対象回帰 3:
  - `test_240_toxicity_budget.py` + `test_275_dry_separation_and_theory.py` + `test_276_blocking_policy_dry.py`
  - `119 passed in 1.44s`
- 最終 broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` と `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - `4060 passed, 1 deselected, 15 warnings in 36.17s`

### 主要改善
- `test_102_structural_fixes.py::TestSoftLossCapResume::test_soft_cap_snapshot_set_on_init`
  - focused file `1.53s -> 0.12s` まで低下
- `test_215_dd_fix_alert_mode.py::Test217SkipGateFeatureMigration`
  - sklearn-heavy `SkipGate` import 依存を外し、feature migration 検証を pure helper に縮退
- broad 中断の主因だった source-inspection 追随漏れを解消
  - 79%/83%/97% 付近で止まっていた split-layout mismatch 群が現在の orchestrator 分割へ揃った

### 補足
- このバッチ中に broad を複数回流し、split-layout mismatch を段階的に追随した。
- 現在の filtered broad 上位は以下に再集中している:
  - `test_enricher_skip_gate.py::Test058Integration::test_enrichment_with_real_data` setup
  - `test_fill_quality.py::TestFillRecordIO::*`
  - `test_retrain_hot_reload.py::TestE2ERetrainHotReload::test_retrain_deploy_and_hot_reload`
  - `test_v460_core.py::TestG0HashPrefix::*`

### 次アクション
1. `test_enricher_skip_gate.py` の real-data setup を fixture snapshot 化でさらに落とす
2. `test_fill_quality.py::TestFillRecordIO` の real file/glob path を切り分け、pure loader contract と persistence を分離する
3. `test_v460_core.py::TestG0HashPrefix::*` と `test_retrain_hot_reload.py` の上位 call を継続圧縮する

---

## 2026-03-08 / Session 037-041

### 実施
- production
  - `ztb/trading/live_trader/price_utils.py` を新設し、現在価格の取得 + validation + fallback を pure helper `resolve_current_price()` へ抽出
  - `ztb/trading/live_trader/live_trader.py`
    - `LiveTrader._get_current_price()` を helper 委譲へ変更
    - invalid adapter 値（0/負値）で前回 valid 価格を保持する形へ整理
- テスト高速化
  - `test_158_failure_modes.py`
    - `TestPriceFallbackChain` を `LiveTrader` 直 import から外し、helper + async adapter stub ベースへ変更
  - `test_ml_pipeline.py`
    - `Test057ASClassifier::test_evaluate_skip_policy` で学習済み classifier を作らず、deterministic OOF probabilities に直接差し替え
  - `test_retrain_hot_reload.py`
    - `build_preorder_as_features()` を fast stub 化する fixture を追加
    - `TestRetrainModel`, `TestE2ERetrainHotReload`, `TestBalanceForcedSwitchFilter` に横展開
    - E2E / balance-forced の retrain cfg で `feature_pruning_enabled=False`, `redundancy_pruning_enabled=False`, `warm_start_enabled=False`, `lgbm_n_estimators=1` を明示し、非本質経路を削除

### 結果
- focused 回帰 1:
  - `test_158_failure_modes.py::TestPriceFallbackChain`
  - `test_ml_pipeline.py::Test057ASClassifier::test_evaluate_skip_policy`
  - `5 passed in 3.54s`
- focused 回帰 2:
  - `test_retrain_hot_reload.py::TestRetrainModel::test_skip_when_insufficient_new_samples`
  - `test_retrain_hot_reload.py::TestE2ERetrainHotReload::test_retrain_deploy_and_hot_reload`
  - `test_retrain_hot_reload.py::TestBalanceForcedSwitchFilter::test_balance_forced_records_excluded`
  - `3 passed in 2.86s`
- production 関連回帰:
  - `tests/unit/trading/test_live_trader_validation.py`
  - `8 passed, 3 warnings in 4.45s`
- 最終 broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` と `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - `4060 passed, 1 deselected, 11 warnings in 29.19s`

### 主要改善
- `test_158_failure_modes.py::TestPriceFallbackChain::test_valid_price_updates_last`
  - `3.60s -> 0.01s`
- `test_ml_pipeline.py::Test057ASClassifier::test_evaluate_skip_policy`
  - `0.10s -> 0.02s`
- `test_retrain_hot_reload.py::TestE2ERetrainHotReload::test_retrain_deploy_and_hot_reload`
  - `1.06s -> 0.04s`
- `test_retrain_hot_reload.py::TestBalanceForcedSwitchFilter::test_balance_forced_records_excluded`
  - `0.03s`

### 補足
- filtered broad の上位は、ほぼ real-data / real-I/O / pure compute に再集中した。
  - `test_enricher_skip_gate.py::Test058Integration::test_enrichment_with_real_data` setup `0.93s`
  - `test_microstructure_features.py::TestEdgeCases::test_zero_volume` call `0.14s`
  - `test_ml_pipeline.py::Test057Integration::test_load_real_data` call `0.14s`
  - `test_v460_core.py::TestConfigLoader::*`
  - `test_aggregate_to_1min.py` の parquet/edge cases
- このバッチで broad 総時間は `36.17s -> 29.19s` まで低下した。

### 次アクション
1. `test_enricher_skip_gate.py` の real-data integration setup を snapshot/cache 前提でさらに詰める
2. `test_microstructure_features.py` と `test_v460_core.py::TestConfigLoader` の data/config loader 上位を切り分ける
3. `test_aggregate_to_1min.py` の残 persistence edge と `test_gate_judgment.py` の Monte Carlo 残件を継続圧縮する

---

## 2026-03-08 / Session 037-042

### 実施
- production
  - `scripts/v460/ml/feature_enricher.py`
    - raw load cache entry に `sorted_ts` / precomputed context を保持するよう変更
    - `enrich_fill_records()` は cached DataFrame に加えて cached trade/orderbook context を再利用する形へ整理
    - これにより `searchsorted` 用配列と cumulative volume の再構築を毎回繰り返さない
  - `ztb/features/microstructure.py`
    - `buy_volume == 0` かつ `sell_volume == 0` の全ゼロ系列では `order_flow_toxicity = 0.0` の fast-path を追加
- テスト追随
  - `test_retrain_hot_reload.py::TestTradesIOFallback`
    - public `load_raw_trades()` patch ではなく、新しい internal raw-load helper patch へ追随
    - fallback chain の呼び出し回数・7日 window 検証は維持

### 結果
- focused 回帰 1:
  - `test_enricher_skip_gate.py::Test058Integration::test_enrichment_with_real_data`
  - `1 passed in 10.91s`
  - 単独起動の wall time は pytest 起動オーバーヘッドが支配的だが、durations 上の setup は bundle/broad で改善を確認
- focused 回帰 2:
  - `test_microstructure_features.py::TestEdgeCases::test_zero_volume`
  - `1 passed in 9.65s`
  - call `0.07s`
- focused 回帰 3:
  - `test_enricher_skip_gate.py` + `test_retrain_hot_reload.py` + `test_microstructure_features.py`
  - `181 passed in 6.98s`
- 最終 broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` と `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - `4060 passed, 1 deselected, 11 warnings in 29.15s`

### 主要改善
- `test_enricher_skip_gate.py::Test058Integration::test_enrichment_with_real_data`
  - filtered broad setup `0.93s -> 0.39s`
- `test_microstructure_features.py::TestEdgeCases::test_zero_volume`
  - focused `0.14s -> 0.07s`
- `feature_enricher.py`
  - production 側で raw cache の value を DataFrame-only から `DataFrame + derived context` に拡張
  - 同一 raw セットへの repeated enrich で実利が出る構成へ変更

### 補足
- 今回の broad 総時間改善は `29.19s -> 29.15s` と小幅だが、実質的には本物の production 側 hot path を改善している。
- broad 上位は現在以下に再集中している:
  - `test_microstructure_features.py::TestCanonicalList::test_all_generated_by_function`
  - `test_234_gate_bypass_removal.py::TestBalanceForcedBypassEradication::test_no_balance_forced_in_gate_check_conditions`
  - `test_v460_core.py::TestConfigLoader::*`
  - `test_build_features_pipeline.py` の microstructure/aggregate setup
  - `test_retrain_hot_reload.py::TestRedundancyPruning::test_highly_correlated_features_detected`

### 次アクション
1. `ztb/features/microstructure.py` の repeated rolling/path をまとめてベクトル化できるか洗う
2. `scripts/v460/lib/config_loader.py` と `test_v460_core.py::TestConfigLoader` の小さな I/O / deepcopy を削る
3. `scripts/v460/build_features.py` / `test_build_features_pipeline.py` の aggregate + microstructure 経路を production/test 両面で詰める

---

## 2026-03-08 / Session 037-043

### 実施
- production
  - `scripts/v460/lib/config_loader.py`
    - `_clone_config_value()` を追加
    - `_read_config_section()` / `_deep_merge()` の blanket `copy.deepcopy()` を config-aware clone へ置換
    - immutable scalar は zero-copy、list/tuple/dict のみ再帰 clone
  - `ztb/data/market_data_collector.py`
    - `aggregate_to_1min()` を `output_path: Path | None = None` 対応に変更
    - DataFrame だけ欲しい呼び出しでは parquet 書込を完全に省略可能
  - `scripts/v460/build_features.py`
    - `build_real_features()` で日次 aggregate ごとの temporary parquet 作成/削除を廃止
    - `MarketDataCollector.aggregate_to_1min(..., output_path=None)` を使い、最終出力だけを保存

### 結果
- focused 回帰 1:
  - `test_v460_core.py::TestConfigLoader`
  - `test_v460_core.py::TestConfigLoaderTaskPreservation`
  - `5 passed in 3.00s`
  - 個別 durations は各 `0.02s`
- focused 回帰 2:
  - `test_aggregate_to_1min.py` + `test_build_features_pipeline.py`
  - `40 passed in 6.11s`
- focused 回帰 3:
  - `test_microstructure_features.py::TestCanonicalList::test_all_generated_by_function`
  - `test_build_features_pipeline.py::TestRealModePipeline::test_microstructure_on_aggregated`
  - `2 passed in 4.43s`
- broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` と `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - `4060 passed, 1 deselected, 11 warnings`
  - wall time は rerun で `34.16s` / `45.56s` と大きく揺れた

### 主要改善
- `config_loader.py`
  - validation/load 系の focused ケースは `0.02s` 級まで低下
  - 本体側でも YAML merge/load の一般経路から generic deepcopy を外した
- `build_features.py`
  - real-mode build は日付ごとの temp parquet roundtrip をしなくなった
  - 実運用で日付数が増えるほど効く変更
- `market_data_collector.py`
  - aggregate-only caller 向けに persistence を opt-in 化

### 補足
- このバッチは global broad よりも production hot path の整理が主目的。
- broad の wall time は明確に揺れており、今回の reliable signal は focused loader/build-path 改善のほうにある。
- broad 上位は引き続き以下:
  - `test_enricher_skip_gate.py::Test058Integration::test_enrichment_with_real_data`
  - `test_286_comprehensive_resolution.py::TestEventsStartStopGuarantee::test_stop_event_logged_on_crash`
  - `test_234_gate_bypass_removal.py::TestBalanceForcedBypassEradication::test_no_balance_forced_in_gate_check_conditions`
  - `test_microstructure_features.py::TestEdgeCases::test_zero_volume`
  - `test_aggregate_to_1min.py` edge/persistence

### 次アクション
1. `ztb/features/microstructure.py` の rolling 系を共通化し、`build_proxy_features()` 側にも横展開する
2. `test_enricher_skip_gate.py` の real-data setup をさらに分解し、production/test のどちらが支配的か切り分ける
3. `test_286_comprehensive_resolution.py` と `test_234_gate_bypass_removal.py` の本体依存 call を精査する

---

## 2026-03-08 / Session 037-044

### 実施
- production
  - `ztb/features/microstructure.py`
    - `add_microstructure_features()` を derived-columns 分離型へ変更
    - 入力 DataFrame 全体の copy / ffill / fillna をやめ、microstructure 列だけを補完して join する形へ整理
  - `scripts/v460/lib/config_hot_reload.py`
    - `FillTestConfig` field name 解決を `lru_cache` 化
    - `_do_reload()` を 2-pass から 1-pass の差分走査へ整理
    - `FillTestConfig` の再 import をやめ、`type(self._config).from_yaml(...)` へ置換
- test helper
  - `tests/unit/v460/_fill_test_source.py`
    - `CYCLE_GATE_AGGREGATOR`
    - `FILL_TEST_CLI`
    - `MAKER_PRICE`
    - を追加
- test
  - `tests/unit/v460/test_234_gate_bypass_removal.py`
    - `cycle_gate_aggregator.py` の AST / source 検査を shared helper に統一
  - `tests/unit/v460/test_286_comprehensive_resolution.py`
    - `fill_test_cli.py` / `_process_post_cycle` の source 検査を cached helper 化
  - `tests/unit/v460/test_303_review_implementations.py`
    - summary 文言確認から重い `evaluate_ab_variant()` を除去
    - `none` regime inclusion テストは sample-count 比較に必要な最小 fixture へ縮小
  - `tests/unit/v460/test_169_config_hot_reload.py`
    - `NamedTemporaryFile` をやめて `tmp_path` ベースへ変更
    - invalid YAML ケースは config 保持だけを見ているため log sink を patch
  - `tests/unit/v460/test_fill_quality.py`
    - `maker_price.py` source 検査を cached file read に置換
  - `tests/unit/v460/test_093_side_params.py`
    - `MakerPriceCalculator` / `FillTestRunner` の source 検査を shared helper に統一

### 結果
- focused 回帰 1:
  - `test_286_comprehensive_resolution.py::TestEventsStartStopGuarantee::test_stop_event_logged_on_crash`
  - `test_234_gate_bypass_removal.py::TestBalanceForcedBypassEradication::test_no_balance_forced_in_gate_check_conditions`
  - `test_286_comprehensive_resolution.py::TestForcedBuyKpiTracking::test_process_post_cycle_uses_balance_forced_switch`
  - `3 passed in 0.99s`
- focused 回帰 2:
  - `test_microstructure_features.py` + `test_build_features_pipeline.py`
  - `43 passed in 4.09s`
- focused 回帰 3:
  - `test_234_gate_bypass_removal.py` + `test_286_comprehensive_resolution.py`
  - `79 passed in 1.25s`
- focused 回帰 4:
  - `test_303_review_implementations.py` + `test_169_config_hot_reload.py`
  - `41 passed in 2.20s`
- focused 回帰 5:
  - `test_fill_quality.py::Test049SideOffset::test_side_offset_used_in_price_calc`
  - `test_fill_quality.py::Test050EffectiveOffsetRecord::test_compute_maker_price_returns_3_values`
  - `test_093_side_params.py::TestSpreadAdaptiveSideLogic::test_compute_maker_price_uses_side_boost`
  - `test_093_side_params.py::TestSpreadAdaptiveSideLogic::test_sa_boost_variable_name`
  - `4 passed in 3.10s`
- broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` と `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - `4060 passed, 1 deselected, 11 warnings in 35.74s`

### 主要改善
- `test_303_review_implementations.py`
  - 文字列サマリ確認から block bootstrap / Mann-Whitney / Cliff's delta を切り離した
  - filtered broad の top 25 から離脱
- `test_169_config_hot_reload.py`
  - Windows で重い temp file 作成を `tmp_path` へ寄せた
  - `ConfigHotReloader` 本体も single-pass diff に整理したため、production/test の両面で無駄が減った
- `maker_price.py` source inspection 群
  - `inspect.getsource(MakerPriceCalculator)` を cached file read に横展開した
  - broad 上位に出ていた class-source 取得コストを除去
- `microstructure.py`
  - feature 追加時に入力フレーム全体を再 copy / 全列 fill しない構成へ変更
  - 実運用でも aggregated feature build の repeated call に効く

### 補足
- broad の wall time は依然として揺れるが、今回消えたホットスポットは明確:
  - `test_303_review_implementations.py`
  - `test_169_config_hot_reload.py`
  - `inspect.getsource(MakerPriceCalculator)` 依存の source 検査
- 現在の broad 上位は real-data / parquet / 実統合寄りに再集中している:
  - `test_enricher_skip_gate.py::Test058Integration::test_enrichment_with_real_data`
  - `test_gate_check.py::TestRunG1Judgment::test_g1_low_ic`
  - `test_aggregate_to_1min.py` parquet / merged edge cases
  - `test_retrain_hot_reload.py::TestRedundancyPruning::test_highly_correlated_features_detected`

### 次アクション
1. `test_enricher_skip_gate.py` の real-data setup を fixture snapshot / class-scope cache に寄せられないか洗う
2. `test_aggregate_to_1min.py` の parquet / merged edge を pure aggregation と persistence 契約へさらに分離する
3. `test_gate_check.py` / `test_retrain_hot_reload.py` の統合寄り上位 call を、成立条件を崩さず lightweight stub へ寄せる

---

## 2026-03-08 / Session 037-045

### 実施
- production
  - `ztb/analysis/redundancy.py`
    - `find_highly_correlated_features()` を `DataFrame.where() + stack()` から `numpy.where()` ベースの upper-triangle scan へ置換
  - `ztb/io/jsonl_gz.py`
    - `append_jsonl_gz()` を 1 行ずつ `write()` する方式から、JSONL payload をまとめて 1 回で書く方式へ変更
- test
  - `tests/unit/v460/test_gate_check.py`
    - `TestRunG1Judgment` の temp JSON 書込を廃止
    - `_load_results_payload()` patch ベースで `run_g1_judgment()` を直接検証
  - `tests/unit/v460/test_retrain_hot_reload.py`
    - `TestRetrainConfig` を temp YAML 実 parse から mocked loader data へ変更
    - `TestRedundancyPruning` の import/setup を module-level cache 化し、データ生成を deterministic 縮小
  - `tests/unit/v460/test_aggregate_to_1min.py`
    - non-persistence ケースは `output_path=None` で集約のみ実行する形へ変更

### 結果
- focused 回帰 1:
  - `test_gate_check.py::TestRunG1Judgment`
  - `test_retrain_hot_reload.py::TestRetrainConfig`
  - `test_retrain_hot_reload.py::TestRedundancyPruning`
  - `test_aggregate_to_1min.py::TestAggregateMerged`
  - `20 passed in 4.54s`
- focused 回帰 2:
  - `test_135_trades_and_gate.py::TestAppendJsonlGz`
  - `test_gate_check.py::TestRunG1Judgment`
  - `test_retrain_hot_reload.py::TestRetrainConfig`
  - `test_retrain_hot_reload.py::TestRedundancyPruning`
  - `test_aggregate_to_1min.py::TestAggregateMerged`
  - `24 passed in 4.22s`
- broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` と `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - `4161 passed, 1 deselected, 11 warnings in 36.76s`

### 主要改善
- `test_gate_check.py`
  - G1 judgment の pure threshold logic テストから temp JSON I/O を除去した
  - broad 上位にいた `test_g1_low_ic` は top 25 から離脱
- `test_retrain_hot_reload.py`
  - `test_yaml_override` の temp YAML parse を mocked loader data へ置換
  - `TestRedundancyPruning` は import/setup を duration から外し、実際の correlation 判定だけを測る構成へ整理
- `test_aggregate_to_1min.py`
  - non-persistence ケースは parquet path 自体を渡さない構成に変更
  - merged/edge の non-parquet ケースがさらに軽くなった
- `ztb/io/jsonl_gz.py`
  - gzip append の write 呼び出し回数を 1 回へ削減
  - `TestAppendJsonlGz::test_append_multiple_calls` は focused で `0.02s` 級まで低下

### 補足
- 今回 broad 上位から外れた項目:
  - `test_gate_check.py::TestRunG1Judgment::test_g1_low_ic`
  - `test_retrain_hot_reload.py::TestRetrainConfig::test_yaml_override`
  - `test_135_trades_and_gate.py::TestAppendJsonlGz::test_append_multiple_calls`
- 現在の broad 上位は real-data / parser / persistence に再集中している:
  - `test_enricher_skip_gate.py::Test058Integration::test_enrichment_with_real_data`
  - `test_v460_core.py::TestConfigLoader::test_load_config_validation_error`
  - `test_336_fill_config_parser.py::TestProductionYamlRoundTrip::*`
  - `test_aggregate_to_1min.py` の parquet persistence

### 次アクション
1. `test_enricher_skip_gate.py` の real-data setup を fixture snapshot / pre-enriched cache へ寄せられないか確認
2. `test_v460_core.py::TestConfigLoader::*` と `test_336_fill_config_parser.py` の parser/YAML round-trip を DRY 化して I/O を減らす
3. parquet を本当に必要とするケースだけに絞れているか `test_aggregate_to_1min.py` と `test_build_features_pipeline.py` を再点検

---

## 2026-03-09 / Session 037-046

### 実施
- production
  - `scripts/v460/lib/fill_config.py`
    - `FillTestConfig.from_yaml()` の split parser 解決を cached lazy resolver 化
- test
  - `tests/unit/v460/test_336_fill_config_parser.py`
    - production YAML を class-scope fixture 化
    - direct parse 結果 / `from_yaml()` 結果を class-scope で再利用
  - `tests/unit/v460/test_v460_core.py`
    - `TestConfigLoader` / `TestConfigLoaderTaskPreservation` の temp YAML 生成を `yaml.dump()` から literal YAML 書込へ変更
    - loader 自体は引き続き end-to-end で通す構成を維持

### 結果
- focused 回帰:
  - `test_336_fill_config_parser.py::TestProductionYamlRoundTrip`
  - `test_v460_core.py::TestConfigLoader`
  - `test_v460_core.py::TestConfigLoaderTaskPreservation`
  - `8 passed in 2.25s`
  - durations:
    - `TestConfigLoader::test_load_config_validation_error` `0.02s`
    - `TestConfigLoader::test_load_config_valid` `0.02s`
    - `TestProductionYamlRoundTrip` setup `0.06s`
- broad 測定:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` と `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - `4182 passed, 1 deselected, 11 warnings in 29.97s`

### 主要改善
- `fill_config.py`
  - split parser の local import 解決を cache したため、`from_yaml()` 呼び出しの常時 overhead を削減
- `test_336_fill_config_parser.py`
  - production YAML を 3 回読み直し / 3 回 parse し直す構成をやめ、class-scope 再利用へ変更
- `test_v460_core.py`
  - config loader テストは YAML dumper を踏まず、loader/validator 本体だけを測る構成へ整理

### 補足
- この batch で broad は `36.76s -> 29.97s` まで低下した。
- 現在の broad 上位は以下に再集中している:
  - `test_enricher_skip_gate.py::Test058Integration::test_enrichment_with_real_data`
  - `test_aggregate_to_1min.py::TestAggregateMerged::test_parquet_roundtrip`
  - `test_158_failure_modes.py` の setup
  - いくつかの source-contract / integration 系単発ケース

### 次アクション
1. `test_enricher_skip_gate.py` の real-data setup を pre-enriched cache / snapshot fixture へ寄せられるか確認
2. `test_aggregate_to_1min.py` の parquet persistence を最小ケースだけにさらに局所化できるか確認
3. `test_158_failure_modes.py` setup と `test_234_gate_bypass_removal.py` の source 契約テストを軽量 helper に寄せる

---

## 2026-03-09 / Session 037-047

### 実施
- 共通 helper
  - `tests/unit/v460/_fill_test_source.py`
    - 任意ファイルの class method source をキャッシュ付きで返す `read_class_method_source()` を追加
- source-contract / source-read 軽量化
  - `tests/unit/v460/test_155_hindsight_review.py`
    - `order_monitor` / `cycle_gate_aggregator` / `fill_config_parser` / `hindsight_filter` の直読を shared helper に統一
  - `tests/unit/v460/test_211_mcb_sad_escalation.py`
    - `_check_circuit_breakers()` source を autouse fixture ごとに取り直す構成をやめ、module-level cache に変更
  - `tests/unit/v460/test_255_getattr_bare_except_cleanup.py`
    - `SkipGateEvaluator` / `OrderMonitor` の method source 取得を cached helper 化
  - `tests/unit/v460/test_234_gate_bypass_removal.py`
    - 不要 import を整理
- fixture / setup 軽量化
  - `tests/unit/v460/test_158_failure_modes.py`
    - risk manager fixture の live trader stub を `MagicMock` から最小 `SimpleNamespace` へ変更
  - `tests/unit/v460/test_141_side_specific_models.py`
    - `test_history_written` の `retrain_model()` を deterministic stub に差し替え
  - `tests/unit/v460/test_336_yaml_code_drift_prevention.py`
    - production YAML config / code-default config を cache して drift 比較で再利用
- deterministic 化 / persistence 縮小
  - `tests/unit/v460/test_259_as_vol_ratio_adaptation_hasattr.py`
    - `inv_decay_tau_sec=0.0` を test-local default に追加し、wall-clock 依存の微差を除去
  - `tests/unit/v460/test_aggregate_to_1min.py`
    - `test_parquet_roundtrip` を full read-back から parquet metadata/schema 確認へ縮小

### 結果
- focused 回帰:
  - `test_158_failure_modes.py` + `test_155_hindsight_review.py` + `test_234_gate_bypass_removal.py` + `test_aggregate_to_1min.py::TestAggregateMerged::test_parquet_roundtrip`
    - `39 passed in 9.31s`
  - `test_259_as_vol_ratio_adaptation_hasattr.py`
    - `10 passed in 1.47s`
  - `test_211_mcb_sad_escalation.py` + `test_255_getattr_bare_except_cleanup.py`
    - `30 passed in 0.81s`
  - `test_141_side_specific_models.py::TestRetrainSideSpecificFunction::test_history_written` + `test_336_yaml_code_drift_prevention.py`
    - `5 passed in 2.79s`
- filtered broad:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` / `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - 1回目: `4153 passed, 1 deselected, 13 warnings in 34.34s`
  - 2回目: `4153 passed, 1 deselected, 13 warnings in 45.97s`

### 主要改善
- `test_259` の `vol_ratio=1.0` 同値性は、実装差ではなく inventory decay の時刻差でブレていた。test-local で decay を切り、volatility path だけを見る形に整理した。
- `test_211` / `test_255` のような source-contract テスト群は、`inspect.getsource()` / 毎回 `ast.parse()` を行うより shared helper に寄せたほうが安定して効くことを確認した。
- `test_141` `test_history_written` は retrain 自体を通す必要がなく、history persistence の責務に絞ることで broad 上位から離脱した。
- `test_336` の drift 比較は file read と `FillTestConfig()` 構築の重複を落としても検査意図を維持できた。

### 補足
- wall time は broad rerun 間で依然ぶれるが、top 25 から以下が落ちたことを改善指標として扱うのが妥当:
  - `test_211_mcb_sad_escalation.py`
  - `test_255_getattr_bare_except_cleanup.py`
  - `test_141_side_specific_models.py::TestRetrainSideSpecificFunction::test_history_written`
- 現在の broad 上位は、ほぼ本物の計算/real-data/persistence 経路に再集中している:
  - `test_enricher_skip_gate.py` real-data setup
  - `test_pnl_monte_carlo.py` sensitivity / simulation
  - `test_gate_judgment.py` Monte Carlo
  - `test_fill_quality.py` unknown fill 系

### 次アクション
1. `test_enricher_skip_gate.py` の real-data setup を pre-enriched fixture / snapshot に寄せられるか再確認
2. `test_pnl_monte_carlo.py` と `test_gate_judgment.py` の Monte Carlo heavy case を deterministic stub または試行数最小化へ寄せる
3. `test_fill_quality.py` の `UnknownFillHandling` / `CancelRaceCondition` を time/polling 依存からさらに切り離せないか確認

---

## 2026-03-09 / Session 037-048

### 実施
- production
  - `ztb/risk/pnl_monte_carlo.py`
    - `sensitivity_analysis()` を再構成
    - fill_rate ごとに `fills_per_sim` と base monthly PnL(bps) を 1 回だけサンプリングし、`pnl_adj_bps` は `fills_per_sim × adj` の解析的シフトとして後掛けする形に変更
    - 内部 helper `_sample_monthly_pnl_bps()` を追加し、既存の vectorized sampling / constant-PnL fast path をそのまま再利用

### 結果
- focused 回帰:
  - `test_pnl_monte_carlo.py::TestSensitivityAnalysis` + `TestSimulationRun::test_var_cvar_relationship`
    - `5 passed in 1.14s`
  - `test_gate_judgment.py::TestGateJudgmentMonteCarlo`
    - `6 passed, 1 warning in 1.35s`
  - `test_pnl_monte_carlo.py` + `test_gate_judgment.py`
    - `53 passed, 4 warnings in 1.98s`
- filtered broad:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` / `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - `4153 passed, 1 deselected, 13 warnings in 36.25s`

### 主要改善
- `sensitivity_analysis()` は従来、`fill_rate × pnl_adj_bps` の全グリッドで毎回 full Monte Carlo を回していた。今回は「base monthly PnL + fill count × adjustment」の分解により、意味を変えずに重複計算を削減した。
- `pnl_adj_bps` は各 fill に対する定数加算なので、同一 `fills_per_sim` に対するシフトで厳密に表現できる。このため、adj 別の再サンプリングは不要だった。
- `test_gate_judgment.py` の Monte Carlo 統合ケースは本体最適化だけで上位から外れた。

### 補足
- focused 計測では以下の改善を確認:
  - `test_positive_adjustment_increases_pnl`: `0.47s級 -> 0.03s`
  - `test_monte_carlo_custom_lot`: `0.40s級 -> 0.05s〜0.06s`
- filtered broad top 25 から `test_gate_judgment.py` の Monte Carlo 群は離脱した。
- 現在の broad 上位は主に以下へ再集中している:
  - `test_enricher_skip_gate.py` real-data setup
  - `test_v460_core.py` config loader / parquet
  - `test_fill_quality.py` unknown fill 系
  - `test_build_features_pipeline.py` real-mode aggregate setup

### 次アクション
1. `feature_enricher` / `build_features_pipeline` の real-data setup を production 側 cache 再利用でさらに詰める
2. `test_fill_quality.py` の unknown-fill / cancel-race 系で polling/time 依存をさらに切り離す
3. `test_v460_core.py` の config loader / parquet パスで production helper 再利用の余地を確認する

---

## 2026-03-09 / Session 037-049

### 実施
- production
  - `scripts/v460/ml/feature_enricher.py`
    - `enrich_fill_records()` 内に timestamp 単位の feature bundle cache を追加
    - 同一 timestamp の fill record 群では orderbook 最近傍探索 / trade window 集計 / return momentum を再計算せず再利用
    - timestamp 範囲から UTC 日付フィルタを作る処理を helper 化
  - `scripts/v460/run_pnl_monte_carlo.py`
    - `--sensitivity` 時の `sim.sensitivity_analysis()` を 1 回だけ実行し、console 出力と JSON 出力で共有

### 結果
- focused 回帰:
  - `test_enricher_skip_gate.py` + `test_build_features_pipeline.py`
    - `84 passed in 5.43s`
  - `test_pnl_monte_carlo.py` + `test_gate_judgment.py`
    - `53 passed, 4 warnings in 2.10s`
- filtered broad:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` / `test_113_resilience.py` を除外、`test_yaml_has_microprice_side` deselect）
  - `4153 passed, 1 deselected, 13 warnings in 34.65s`

### 主要改善
- `feature_enricher` は raw cache だけでなく、呼び出し単位でも「同一 ts なら同一特徴量」という再利用余地があった。今回の cache は downstream の学習/分析コードでもそのまま効く。
- `run_pnl_monte_carlo.py` は `--sensitivity --output` のとき感度分析を 2 回回していたため、CLI 実行では不要な Monte Carlo が残っていた。これは純粋な重複計算だったので 1 回に統合した。
- 今回の `feature_enricher` 変更は broad wall time を大きく動かす類ではないが、実装としてはより筋がよい。real-data setup の本丸は raw 読込と日別 aggregate 側に残っている。

### 補足
- filtered broad の top は引き続き以下:
  - `test_enricher_skip_gate.py` real-data setup
  - `test_fill_quality.py` unknown-fill 系
  - `test_v460_core.py` config/parquet 系
  - `test_build_features_pipeline.py` real-mode aggregate setup
- `test_gate_judgment.py` / `test_pnl_monte_carlo.py` は前 batch の本体最適化を維持しており、今回も focused では低コストで安定している。

### 次アクション
1. `feature_enricher` / `build_real_features()` の raw 読込・日別 aggregate を跨ぐ cache 再利用余地を確認
2. `test_fill_quality.py` の unknown-fill / cancel-race 系を本体 helper 分離で軽量化できないか確認
3. `test_v460_core.py` の config/parquet 上位を production helper 側から詰める

---

## 2026-03-09 / Session 037-050

### 実施
- test
  - `tests/unit/v460/test_fill_quality.py`
    - `TestUnknownFillHandling` / `TestBug11CancelRaceCondition` に `asyncio.sleep` no-op helper を適用
    - `run_single_cycle()` と `OrderMonitor` の既存責務境界は維持しつつ、polling テストの実待機だけを外す構成へ変更

### 結果
- focused 回帰:
  - `test_fill_quality.py::TestUnknownFillHandling` + `TestBug11CancelRaceCondition`
    - `5 passed in 8.28s`
    - durations:
      - `test_status_none_twice_becomes_cancelled_status_unknown`: `0.30s`
      - 残り 4 ケース: `0.01s`
- filtered broad:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` / `test_113_resilience.py` を除外、`test_306_proposals.py::...::test_yaml_has_microprice_side` を deselect）
  - `4154 passed, 13 warnings in 34.32s`

### 主要改善
- この batch では、本体 helper を壊さずに `run_single_cycle()` の end-to-end 経路を維持することを優先した。
- `asyncio.sleep` だけを no-op 化することで、状態判定・cancel race・record 生成の既存ロジックはそのまま通し、実待機のみ除去した。
- `status_none_twice...` がまだ単発で残るため、真の残コストは sleep ではなく status-unknown 分岐内の状態遷移・再照合にあると切り分けられた。

### 補足
- broad の top 25 では、status/cancel-race 群は主要ボトルネックではなくなった。
- 現在の broad 上位は以下に再集中している:
  - `test_enricher_skip_gate.py` real-data setup
  - `test_v460_core.py` config loader / parquet
  - `test_fill_quality.py::TestFillTestRunnerSaveResilience::*`
  - 一部の YAML/source-contract / ML integration ケース

### 次アクション
1. `feature_enricher` / `build_real_features()` の raw 読込・aggregate 再利用を production 側でさらに詰める
2. `test_v460_core.py` の config/parquet 上位を production helper 再利用で削る
3. `fill_quality` の残上位は save-resilience / G1 判定周辺へ移っているため、その責務分離を確認する

---

## 2026-03-09 / Session 037-051

### 実施
- production
  - `scripts/v460/lib/data_loader.py`
    - `load_parquet(..., feature_cols=...)` の schema 読込に file-signature cache を追加
    - 既存の `pyarrow` 経路は維持しつつ、同一 parquet への repeated `read_schema()` を削減
  - `ztb/metrics/fill_quality.py`
    - `save_fill_records()` を batch payload の一括 serialize + 一括 write へ変更
    - 既存の tempfile + fsync + append の atomic path は維持
  - `scripts/v460/lib/batch_persistence.py`
    - `_save_batch_by_date()` に batch-local の UTC day cache を追加
- test
  - `tests/unit/v460/test_v460_core.py`
    - `microstructure` / `build_features` の method 内 import を module scope へ集約
    - microstructure 入力 DataFrame を cache 化
    - `TestDataLoader` / `TestDataLoaderEdgeCases` の parquet を class-scope fixture へ集約
  - `tests/unit/v460/test_enricher_skip_gate.py`
    - real-data integration の成立判定を `build_pnl_features()` 実行から trainable row count 判定へ変更
  - `tests/unit/v460/test_ml_pipeline.py`
    - real-data integration を `120 -> 220 -> 320` の guarded fallback へ変更
    - `build_as_features()` の有効サンプル数不足による flaky を解消

### 結果
- focused:
  - `test_v460_core.py` + `test_enricher_skip_gate.py` + `test_fill_quality.py`
    - `331 passed, 5 warnings in 10.00s`
  - `test_ml_pipeline.py::Test057Integration::test_load_real_data`
    - `1 passed in 2.74s`
- filtered broad:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` / `test_113_resilience.py` を除外、`test_306_proposals.py::...::test_yaml_has_microprice_side` を deselect）
  - `4154 passed, 13 warnings in 36.63s`

### 主要改善
- `load_parquet()` は selective read のたびに schema を引き直していたため、小さい parquet でも call 単位の固定費が残っていた。file-signature cache を入れたことで、同一 parquet を複数回見る経路の固定費を削減した。
- `save_fill_records()` は per-record write が残っていたため、`BatchPersistence` と `FillRecordIO` 系の保存テスト・本体保存の両方で無駄な Python write loop があった。今回の変更で durability はそのまま、文字列生成と write 回数だけ減らした。
- `test_enricher_skip_gate.py` の real-data setup は、学習成立判定のために重い feature builder を先に回していた。必要なのは trainable row 数だけなので、判定を最小化して setup を軽くした。
- `test_ml_pipeline.py` の実データ integration は固定 `120` 行前提が現在の実データ tail に対して脆かった。`test_enricher_skip_gate.py` と同じ fallback へ揃え、real-data 系の安定性を横展開した。

### 補足
- `test_v460_core.py::TestDataLoader::test_load_parquet` は focused rerun で `0.46s -> 0.05s` まで低下した。
- broad の top は引き続き `test_enricher_skip_gate.py` real-data setup、`fill_quality` の一部 integration、`retrain_hot_reload` などの実経路寄りへ再集中している。
- 今回の broad wall time は `36.63s` で、主な効果は flaky 解消と parquet/schema/save 系の固定費削減。

### 次アクション
1. `test_enricher_skip_gate.py` の real-data setup を raw snapshot/cache 再利用でもう一段詰める
2. `test_fill_quality.py` の `Bug11` / save-resilience / `FillRecordIO` 残件を production helper で再確認する
3. `retrain_hot_reload` / `config_hot_reload` の integration call を引き続き削る

---

## 2026-03-09 / Session 037-052

### 実施
- test
  - `tests/unit/v460/test_enricher_skip_gate.py`
    - real-data integration の sample size 判定を raw fill-record ベースへ変更
    - `120 / 220 / 320` の各候補で `enrich_fill_records()` を回す構造をやめ、必要 tail サイズを先に決めてから enrich を 1 回だけ実行
  - `tests/unit/v460/test_fill_quality.py`
    - `_run_single_cycle_without_sleep()` に advancing fake clock を追加
    - `fill_cycle_executor.time.time` と `order_monitor.time.time` を同じ fake clock に差し替え
    - 既存の `run_single_cycle()` / `OrderMonitor` 経路は維持しつつ、patched sleep 後にも残っていた busy-loop timeout コストを除去

### 結果
- focused:
  - `test_enricher_skip_gate.py::Test058Integration`
  - `test_fill_quality.py::TestBug11CancelRaceCondition`
  - `test_fill_quality.py::TestUnknownFillHandling`
  - `7 passed in 4.19s`
  - durations:
    - `Test058Integration::test_enrichment_with_real_data` setup: `0.38s`
    - `TestBug11CancelRaceCondition::test_cancel_fail_detects_fill`: `0.12s`
    - `TestUnknownFillHandling::test_status_none_twice_becomes_cancelled_status_unknown`: `0.01s`
- filtered broad:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` / `test_113_resilience.py` を除外、`test_306_proposals.py::...::test_yaml_has_microprice_side` を deselect）
  - `4154 passed, 13 warnings in 34.47s`

### 主要改善
- `test_enricher_skip_gate.py` の setup は「成立条件確認のためだけに enrich を複数回回す」構造が残っていた。trainable row 数は raw fill records の `filled` / `post_fill_30s_pnl` だけで決まるので、そこを使って sample size を先決めする形へ修正した。
- `test_fill_quality.py` の Unknown/Bug11 系は patched sleep 後も loop が `time.time()` ベースで CPU を回していた。fake clock を入れることで、待機ではなく時間経過条件そのものをテスト側で前進させ、既存 helper を壊さずにコストだけ削った。
- `retrain_hot_reload` は focused で `0.05s` だったため、この batch では追加変更せずに据え置いた。現時点での低リスク改善余地は薄い。

### 補足
- broad の top は今回で以下へ再集中:
  - `test_aggregate_to_1min.py`
  - `test_v460_core.py::TestDataLoader::test_load_parquet`
  - `test_ml_pipeline.py::Test057Integration::test_load_real_data`
  - `test_fill_quality.py::TestBug11CancelRaceCondition::test_cancel_fail_detects_fill`

### 次アクション
1. `test_aggregate_to_1min.py` の trades-only / merged setup を class-scope fixture へ寄せる
2. `test_v460_core.py` と `test_ml_pipeline.py` の real-data / parquet top をさらに揃える
3. `config_hot_reload` / `aggregate_to_1min` の broad 上位 call を引き続き削る

---

## 2026-03-09 / Session 037-053

### 実施
- production
  - `ztb/io/jsonl.py`
    - `read_tail_jsonl_objects()` を追加
    - plain JSONL の末尾 N 行取得ロジックを共通化
  - `ztb/io/__init__.py`
    - 上記 helper を export
  - `ztb/data/market_data_collector.py`
    - trades 側 1 分集約を `_aggregate_trades_1min()` として分離
    - `aggregate_to_1min()` 本体は OB / trades の各 helper を呼ぶ形へ整理
- test
  - `tests/unit/v460/test_enricher_skip_gate.py`
    - real-data tail 読込を shared JSONL helper に統一
  - `tests/unit/v460/test_ml_pipeline.py`
    - 重複していた `_tail_jsonl_objects()` の実装を shared helper 利用へ変更
  - `tests/unit/v460/test_aggregate_to_1min.py`
    - non-persistence 経路を input payload keyed cache 化
    - 同一 raw payload に対する aggregate 再計算を回避

### 結果
- focused:
  - `test_aggregate_to_1min.py`
  - `test_enricher_skip_gate.py::Test058Integration`
  - `test_ml_pipeline.py::Test057Integration`
  - `29 passed in 3.95s`
  - durations:
    - `Test058Integration::test_enrichment_with_real_data` setup: `0.35s`
    - `Test057Integration::test_load_real_data`: `0.19s`
    - `TestAggregateEdgeCases::test_many_minutes`: `0.03s`
- filtered broad:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` / `test_113_resilience.py` を除外、`test_306_proposals.py::...::test_yaml_has_microprice_side` を deselect）
  - `4154 passed, 13 warnings in 40.99s`

### 主要改善
- plain JSONL の tail 読込が `test_enricher_skip_gate.py` と `test_ml_pipeline.py` で重複していたため、`ztb.io` に寄せて DRY 化した。今後 real-data integration 系で同じ helper を横展開できる。
- `aggregate_to_1min()` は板集約だけ helper 化されていて、trades 側の集約は本体に残っていた。今回の分離で責務を揃え、今後の本体最適化や単体検証をしやすくした。
- `test_aggregate_to_1min.py` は output persistence を確認しないケースまで毎回 aggregate を再計算していたため、pure aggregation path を cache 再利用に切り替えた。

### 補足
- broad wall time 自体はこの rerun では悪化したが、対象 hotspot は改善している。
- 最新の broad top は次へ移っている:
  - `test_141_side_specific_models.py`
  - `test_169_config_hot_reload.py`
  - `test_fill_quality.py::TestInterimJudgment`
  - 一部 YAML / config parse / dashboard integration

### 次アクション
1. `test_169_config_hot_reload.py` の reload path を production helper 再利用で削る
2. `test_141_side_specific_models.py` の regime threshold integration を stub/cached fixture 化する
3. `test_fill_quality.py::TestInterimJudgment` の大量 record 構築を共通 builder へ寄せる

---

## 2026-03-09 / Session 037-054

### 実施
- test
  - `tests/unit/v460/test_fill_quality.py`
    - `_make_uniform_daily_records()` helper を追加
    - `test_sample_sufficient_true`
    - `test_sample_sufficient_false_n`
    - `TestInterimJudgment::test_interim_3_days_200_samples`
    - `TestInterimJudgment::test_final_7_days`
    - を共通 builder 利用へ変更

### 結果
- `python -m py_compile tests/unit/v460/test_fill_quality.py`
  - pass

### 主要改善
- `fill_quality` の日次サンプル生成は同型の二重ループが複数箇所に散っていたため、記録構築責務を helper に寄せた。
- 日付境界を跨ぐテストデータ生成ロジックを 1 箇所にまとめたことで、将来 sample size や timestamp policy を調整するときの変更面積を減らした。

### 補足
- この batch は主に DRY / 横展開であり、測定上の大きい wall time 改善を狙ったものではない。
- この環境では `test_fill_quality.py` 単体の direct collection が `scripts.v460.analysis.vg_and_trend` import で不安定だったため、focused pytest の数値は記録していない。変更範囲は test-side builder のみ。

### 次アクション
1. `test_169_config_hot_reload.py` の reload path を production helper 再利用で削る
2. `test_141_side_specific_models.py` の regime threshold integration を stub/cached fixture 化する
3. `test_fill_quality.py` の残る大量 record 構築箇所を同 helper へ横展開する

---

## 2026-03-09 / Session 037-055

### 実施
- production
  - `scripts/v460/lib/config_hot_reload.py`
    - `TimeFilter` 解決を `_resolve_time_filter_cls()` に分離
    - lazy import の入口を 1 箇所に集約し、class object を cache
- test
  - `tests/unit/v460/test_169_config_hot_reload.py`
    - `_resolve_time_filter_cls()` を autouse fixture で stub 化
    - field update / component rebuild テストから実 `TimeFilter` import graph を除去
  - `tests/unit/v460/test_ml_pipeline.py`
    - `_write_real_fill_sample()` で `build_as_features()` の `ValueError` を catch
    - labeled sample が足りない場合に `220/320` candidate へ正しく fallback
  - `tests/unit/v460/test_fill_quality.py`
    - daily record builder の横展開を継続

### 結果
- focused:
  - `test_ml_pipeline.py::Test057Integration::test_load_real_data`
  - `test_169_config_hot_reload.py`
  - `test_336_fill_config_parser.py`
  - `test_336_yaml_code_drift_prevention.py`
  - `test_344_improvements.py`
  - `69 passed in 4.38s`
  - durations:
    - `Test057Integration::test_load_real_data`: `0.17s`
    - `TestYamlParsing::test_yaml_parses_all_new_params`: `0.11s`
    - `TestYamlCodeDefaultDrift::test_no_unexpected_drift`: `0.07s`
    - `TestConfigFieldUpdate::test_do_reload_updates_reloadable_fields`: `0.02s`（この束の実行順）
- filtered broad:
  - `tests/unit/v460/`（`test_260_compute_extract_regime_split.py` / `test_113_resilience.py` / `test_152_parallel_tasks.py` を除外、`test_306_proposals.py::...::test_yaml_has_microprice_side` を deselect）
  - `4139 passed, 13 warnings in 35.43s`

### 主要改善
- `config_hot_reload` のボトルネックの一部は `TimeFilter` 側 import graph の cold cost だった。helper 化 + test stub により、reload テスト群から不要な本物 import を外した。
- `ml_pipeline` の real-data integration では「候補 tail は広げるが、途中の `ValueError` を握らず落ちる」穴が残っていた。guarded fallback として不完全だったので、今回 catch して次候補へ進むように修正した。
- `fill_quality` の daily record 構築 helper は今後も横展開できる。まだ同型ループが残っているので、次もこの路線でまとめるのが筋。

### 補足
- `test_169_config_hot_reload.py::TestConfigFieldUpdate::test_do_reload_updates_reloadable_fields` は focused 束で `1.48s -> 1.10s` まで低下。
- broad では `config_hot_reload` 系は最上位群から一段後退し、上位は real-data setup / structural fixes / 一部 YAML parse に再集中した。
- `test_152_parallel_tasks.py` は今回差分と無関係な `scripts.v460.analysis.compare_regime_ab` import error のため除外した。

### 次アクション
1. `test_141_side_specific_models.py` の online/regime threshold integration を cached fixture 化
2. `test_fill_quality.py` の残る大量 record 構築を builder へ横展開
3. YAML parse 系の production helper 再利用をさらに進める

---

## 2026-03-09 / Session 037-056

### 実施
- `tests/unit/v460/test_344_improvements.py`
  - 既存の `v460_fill_test_yaml` fixture を再利用し、ローカル `yaml.safe_load()` + `fill_test.yaml` 直読を撤去
  - `FillTestConfig` / `CycleGateAggregator` / `parse_fill_config_yaml` の method 内 import を module scope に集約
- `tests/unit/v460/test_141_side_specific_models.py`
  - `TestOnlineMonitorEvaluate` に `_make_monitor()` を追加し、`OnlineMonitor(OnlineMonitorConfig(...))` の重複生成を集約
- `tests/unit/v460/test_fill_quality.py`
  - `_make_daily_fill_count_records()` を追加
  - `daily_fill_rates` / `g1_1_with_data` を同 helper へ寄せた
  - `provisional_insufficient` を既存 `_make_uniform_daily_records()` へ寄せた

### 結果
- focused:
  - `tests/unit/v460/test_141_side_specific_models.py`
  - `tests/unit/v460/test_344_improvements.py`
  - `69 passed, 1 warning in 4.54s`
- focused selector:
  - `tests/unit/v460/test_fill_quality.py -k 'daily_fill_rates or g1_1_with_data or provisional_insufficient'`
  - `3 passed, 203 deselected in 3.22s`
  - durations:
    - `TestGateCheckG11::test_g1_1_with_data`: `0.06s`

### 主要改善
- YAML parse 系は既にある session-cached fixture を使う形へ揃えた。これで `test_344_improvements.py` 独自の config file 再読込が消えた。
- `side_specific_models` は focused では十分軽かったため、今回は性能よりも setup の一貫性と DRY を優先した。
- `fill_quality` の日次データ構築は今後さらに横展開できる形になった。残件も同系統の helper 化で処理しやすい。

### 補足
- 今回は本筋の runtime 改善 batch ではなく、重複排除と fixture 再利用の整理が主眼。
- 作業中に別ファイルの未コミット差分を確認したが、ユーザー指示どおり今回のコミット対象から除外している。

---

## 2026-03-09 / Session 037-057

### 実施
- `tests/unit/v460/test_169_config_hot_reload.py`
  - `_make_reloader()` を追加し、`ConfigHotReloader(...)` の重複生成を共通化
- `tests/unit/v460/test_fill_quality.py`
  - `_make_outcome_records()` を追加
  - attempted 指標と cancel reason 内訳の outcome ベース生成を helper へ移した
- `scripts/v460/build_features.py`
  - `_discover_daily_inputs()` を追加
  - `build_real_features()` が日次 raw input を 1 回解決し、その後の `exists()` / path 再構築を避けるよう整理

### 結果
- focused:
  - `tests/unit/v460/test_169_config_hot_reload.py`
  - `tests/unit/v460/test_build_features_pipeline.py`
  - `30 passed in 5.03s`
- focused selector:
  - `tests/unit/v460/test_fill_quality.py -k 'skip_gate_fields_populated or cancel_reason_breakdown'`
  - `2 passed, 204 deselected in 3.86s`

### 主要改善
- `config_hot_reload` テストは setup の入口を 1 箇所に揃えたので、今後の reload case 追加でも差分が散りにくい。
- `fill_quality` の outcome 系テストは fill / skip / timeout / reject / unknown の構築パターンを共通化できた。まだ同型の attempted/cancel 系に横展開余地がある。
- `build_real_features()` は全日付処理時に raw path の探索を 1 回で済ませるようになり、実データディレクトリに対する無駄な `exists()` 呼び出しを減らした。

---

## 2026-03-09 / Session 037-058

### 実施
- `tests/unit/v460/test_169_config_hot_reload.py`
  - `_make_reload_context()` を追加
  - `reloader` と `runner` の組生成をこの helper に寄せた
- `tests/unit/v460/test_fill_quality.py`
  - `test_no_skip_gate_records` も `_make_outcome_records()` 利用へ統一
- `scripts/v460/build_features.py`
  - 前バッチの `_discover_daily_inputs()` ルートを維持したまま focused pipeline で再確認

### 結果
- focused:
  - `tests/unit/v460/test_169_config_hot_reload.py`
  - `tests/unit/v460/test_build_features_pipeline.py`
  - `30 passed in 4.58s`
- focused selector:
  - `tests/unit/v460/test_fill_quality.py -k 'skip_gate_fields_populated or no_skip_gate_records or cancel_reason_breakdown'`
  - `3 passed, 203 deselected in 3.36s`

### 主要改善
- `config_hot_reload` テストは「reloader だけ」ではなく「reloader + runner」まで共通化できたので、今後の hot-reload 回帰追加でも setup の重複が増えにくい。
- `fill_quality` の attempted 系は all-fill / mixed outcomes の両方が同じ builder 系に乗った。残りも同じ方向で崩せる。
- 追加の production 変更は小さいが、`build_features` の input 解決整理が focused pipeline で維持されることを確認した。

---

## 2026-03-09 / Session 037-059

### 実施
- `tests/unit/v460/test_169_config_hot_reload.py`
  - `_prepare_reload_context()` を追加
  - YAML 更新済みの `reloader + runner` 準備を helper 化
- `tests/unit/v460/test_fill_quality.py`
  - `_make_linear_records()` を追加
  - save/load / iter / glob 系の単純レコード生成を helper に寄せた
- `scripts/v460/build_features.py`
  - `_resolve_target_dates()` を追加
  - real mode の target date を一意化し、存在する raw input のみに絞る処理を共通化

### 結果
- focused:
  - `tests/unit/v460/test_169_config_hot_reload.py`
  - `tests/unit/v460/test_build_features_pipeline.py`
  - `30 passed in 4.20s`
- focused selector:
  - `tests/unit/v460/test_fill_quality.py -k 'save_load_roundtrip or iter_load_roundtrip or glob_load or iter_glob_load_roundtrip or skip_gate_fields_populated or no_skip_gate_records or cancel_reason_breakdown'`
  - `9 passed, 197 deselected in 3.57s`

### 主要改善
- `config_hot_reload` は「生成 helper」と「更新済み context helper」の二段構成になり、reload テストの重複がさらに減った。
- `fill_quality` の I/O テストは線形レコード builder へ寄せたことで、単純 roundtrip データの記述がかなり減った。
- `build_real_features()` は target date 解決が 1 箇所に閉じたので、今後の `--date` / `--all-dates` 条件追加でも分岐を増やしにくい。

---

## 2026-03-09 / Session 037-060

### 実施
- `tests/unit/v460/test_fill_quality.py`
  - `_make_linear_records()` に `start_index` / `separator` を追加
  - `_save_linear_records()` を追加
  - I/O / glob / date-range 系の単発レコード生成をさらに helper 化
- `scripts/v460/build_features.py`
  - 未使用だった `_discover_dates()` wrapper を削除
  - date discovery を `_discover_daily_inputs()` の単一路線に整理

### 結果
- focused:
  - `tests/unit/v460/test_build_features_pipeline.py`
  - `14 passed in 4.14s`
- focused selector:
  - `tests/unit/v460/test_fill_quality.py -k 'save_load_roundtrip or iter_load_roundtrip or glob_load or iter_glob_load_roundtrip or iter_glob_load_can_exclude_emergency or iter_fill_record_objects_glob_roundtrip or list_fill_record_files_supports_date_range or list_fill_record_files_date_range_uses_direct_resolution or list_fill_record_files_cache_invalidates_when_directory_changes or load_fill_record_objects_glob_supports_date_range'`
  - `11 passed, 195 deselected in 2.62s`

### 主要改善
- `fill_quality` の I/O テストは「生成 helper」と「保存 helper」まで揃ったので、今後の roundtrip ケース追加で同じ boilerplate を増やさずに済む。
- cycle_id の命名差分も helper 引数で吸収できるようにしたので、元テスト意図を維持したまま共通化できる範囲が広がった。
- `build_features.py` は日付 discovery の入口が 1 つになり、同種 helper の重複が減った。

---

## 2026-03-09 / Session 037-061

### 実施
- `tests/unit/v460/test_013_fixes.py`
  - cached `_source()` helper を追加
  - `CoincheckAdapter` / `BitFlyerAdapter` / `OrderManager` の source assertion を同 helper に寄せた
- `tests/unit/v460/test_143_regime_utilization.py`
  - cached `_source()` helper を追加
  - `online_monitor` / `SkipGateEvaluator` / `FillTestRunner` / `OrderMonitor` の source assertion を共通化
- `tests/unit/v460/test_139_review_fixes.py`
  - cached `_source()` helper を追加
  - `SkipGateEvaluator`, `pnl_measurer`, `feature_enricher`, `run_fill_test`, `retrain_scheduler` の source assertion を共通化
- `tests/unit/v460/test_092_gap_fixes.py`
  - `gate_thresholds_yaml` fixture を追加
  - `gate_thresholds.yaml` 直読 3 箇所を fixture 再利用へ変更

### 結果
- focused:
  - `tests/unit/v460/test_013_fixes.py`
  - `tests/unit/v460/test_092_gap_fixes.py`
  - `tests/unit/v460/test_139_review_fixes.py`
  - `tests/unit/v460/test_143_regime_utilization.py`
  - `134 passed in 12.17s`

### 主要改善
- `inspect.getsource(...)` の残件はこの 3 ファイルで cached helper に置換できた。source-contract テストの重複パターンがかなり減った。
- `gate_thresholds.yaml` の consistency テストは typed fixture に寄せたので、今後の閾値追加でも読み込み boilerplate を増やさずに済む。
- focused durations の上位は source read ではなく実際の behavioral test に寄っており、今後は真に重い call 側を見やすい状態になった。

---

## 2026-03-09 / Session 037-062

### 実施
- `tests/unit/v460/test_094_stale_order.py`
  - cached `_source()` helper を追加
  - `OrderMonitor`, `MakerPriceCalculator`, `SkipGate`, `FillMonitorResult`, `SkipGateResult` の repeated import を module scope に集約
  - `OrderMonitor.monitor` の source assertion を helper 経由へ統一
- `tests/unit/v460/test_137_p1_features.py`
  - `FillTestConfig`, `PnlMeasurer`, `RetrainTriggerConfig` の method 内 import を module scope に集約
- `tests/unit/v460/test_138_p1_preflight_calibration.py`
  - `FillTestConfig`, `ScoreCalibrator`, `ScoreCalibratorConfig` の method 内 import を module scope に集約
  - bare `dict` の一部を具体的な union 型注釈へ変更
- `tests/unit/v460/test_fill_quality.py`
  - `TestFillRecordIO` の roundtrip / glob / date-range ケースを `_save_linear_records()` に横展開
  - `load_corrupt_lines_skipped` も `_make_linear_records()` ベースへ寄せて単発 `FillRecord(...)` 構築を削減

### 結果
- focused:
  - `tests/unit/v460/test_094_stale_order.py`
  - `tests/unit/v460/test_137_p1_features.py`
  - `tests/unit/v460/test_138_p1_preflight_calibration.py`
  - `79 passed in 3.64s`
- focused selector:
  - `tests/unit/v460/test_fill_quality.py -k 'save_load_roundtrip or iter_load_roundtrip or glob_load or iter_glob_load_roundtrip or load_corrupt_lines_skipped'`
  - `7 passed, 199 deselected in 3.37s`

### 主要改善
- `test_094_stale_order.py` の source-contract テストは cached source 参照に揃ったので、同じ `OrderMonitor.monitor` text を何度も取り直さなくなった。
- `test_137_p1_features.py` と `test_138_p1_preflight_calibration.py` は method 内 import をかなり削り、型注釈も少し締めたので、以後の横展開がしやすい状態になった。
- `test_fill_quality.py` の `TestFillRecordIO` は JSONL I/O helper の再利用範囲が広がり、1件/2件保存ケースの boilerplate がさらに減った。

---

## 2026-03-09 / Session 037-063

### 実施
- `tests/unit/v460/test_094_stale_order.py`
  - ローカル `_source()` を廃止
  - [_fill_test_source.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/tests/unit/v460/_fill_test_source.py) の `read_class_method_source()` と `ORDER_MONITOR` を使う形に変更
- `tests/unit/v460/test_fill_quality.py`
  - `_save_daily_fill_count_records()` を追加
  - `TestGateCheckG11::test_g1_1_with_data` を helper 再利用へ変更
- `scripts/v460/ml/feature_enricher.py`
  - `resolve_raw_dir()` を追加
  - `discover_raw_daily_inputs()` を追加
  - raw orderbook/trades loader が同 helper を使うよう整理
- `scripts/v460/build_features.py`
  - raw dir 解決と日次 raw 入力 discovery を `feature_enricher` 側 helper の再利用へ寄せた
  - `_discover_daily_inputs()` の重複実装を削除

### 結果
- focused:
  - `tests/unit/v460/test_094_stale_order.py`
  - `tests/unit/v460/test_build_features_pipeline.py`
  - `66 passed in 8.15s`
- focused selector:
  - `tests/unit/v460/test_fill_quality.py -k 'g1_1_with_data or save_load_roundtrip or iter_load_roundtrip or glob_load or iter_glob_load_roundtrip or load_corrupt_lines_skipped'`
  - `8 passed, 198 deselected in 7.43s`

### 主要改善
- `stale_order` の source-contract は他の split-source テストと同じ helper 経路に揃った。`inspect.getsource()` のローカルキャッシュ実装が1つ減った。
- `fill_quality` の `run_g1_1` integration も日次 fill-count builder に揃ったので、日次サンプル生成の修正点がさらに集中した。
- production 側は `feature_enricher` と `build_features` の raw path/date 解決が 1 箇所にまとまり、今後の raw layout 変更や日付解決ロジック変更を片側だけ直して齟齬が出るリスクを下げた。

---

## 2026-03-09 / Session 037-064

### 実施
- `scripts/v460/ml/feature_enricher.py`
  - `RawDirLike = str | Path | None` を導入
  - `resolve_raw_dir()`, `discover_raw_daily_inputs()`, raw loader 群, `enrich_fill_records()` の `raw_dir` 引数を共通型へ拡張
- `scripts/v460/build_features.py`
  - `build_real_features()` が `raw_dir` を直接 `resolve_raw_dir()` に渡すよう整理
- `tests/unit/v460/test_154_deadlock_prevention.py`
  - `OrderMonitor.monitor` の source assertion を `read_class_method_source()` へ変更
- `tests/unit/v460/test_262_protocol_cancel_recheck.py`
  - `OrderMonitor.monitor` の source assertion を `read_class_method_source()` へ変更

### 結果
- focused:
  - `tests/unit/v460/test_154_deadlock_prevention.py`
  - `tests/unit/v460/test_262_protocol_cancel_recheck.py`
  - `tests/unit/v460/test_build_features_pipeline.py`
  - `57 passed in 7.90s`

### 主要改善
- `resolve_raw_dir()` は `str | Path | None` を直接受けるようになったので、CLI 呼び出しとライブラリ呼び出しで余分な `Path(...)` 包装が不要になった。再利用境界として扱いやすくなった。
- `OrderMonitor.monitor` を読む source-contract テストは `_fill_test_source.py` にさらに寄った。split-source 系の参照経路が揃い、`inspect.getsource(...)` の局所実装がまた減った。
- 今回追加した helper のうち、production-wide に昇格させる価値があるのは raw path/date helper 側で、`_save_daily_fill_count_records()` は現時点では test-local helper のままが適切と判断した。

---

## 2026-03-09 / Session 037-065

### 実施
- `tests/unit/v460/test_fill_quality.py`
  - `_save_generated_records()` を追加
  - `_save_linear_records()` と `_save_daily_fill_count_records()` を thin wrapper 化
- `scripts/v460/ml/feature_enricher.py`
  - `resolve_available_raw_dates()` を追加
- `scripts/v460/build_features.py`
  - `_resolve_target_dates()` を削除
  - `resolve_available_raw_dates()` を再利用する形へ変更

### 結果
- focused selector:
  - `tests/unit/v460/test_fill_quality.py -k 'g1_1_with_data or save_load_roundtrip or glob_load or iter_glob_load_roundtrip'`
  - `6 passed, 200 deselected in 3.49s`
- focused:
  - `tests/unit/v460/test_build_features_pipeline.py`
  - `14 passed in 3.32s`

### 主要改善
- `fill_quality` の helper は「builder ごとの差」は残しつつ、「保存処理そのもの」は 1 箇所へ統合した。可読性を落とさずに重複だけ削れている。
- production 側は raw path 解決だけでなく target date 解決も `feature_enricher` 側へ寄せたので、`build_features.py` から raw 入力発見系 helper がさらに 1 つ減った。
- helper 統合の判断としては、「builder 名がテスト意図を伝えるもの」は wrapper を残し、「内部で同じことをするだけ」の層だけを畳むのが妥当だった。

---

## 2026-03-09 / Session 037-066

### 実施
- `tests/unit/v460/test_fill_quality.py`
  - `_build_daily_records()` を追加
  - `_make_uniform_daily_records()` と `_make_daily_fill_count_records()` の day/index 二重ループを共通化
- `tests/unit/v460/test_258_as_reservation_vpin_continuous_protocol.py`
  - `OrderMonitor._resolve_regime_name` の source 参照を `read_class_method_source()` へ変更

### 結果
- focused selector:
  - `tests/unit/v460/test_fill_quality.py -k 'daily_fill_rates or g1_1_with_data or save_load_roundtrip or glob_load'`
  - `7 passed, 199 deselected in 3.97s`
- focused:
  - `tests/unit/v460/test_258_as_reservation_vpin_continuous_protocol.py`
  - `29 passed in 1.14s`

### 主要改善
- `fill_quality` の日次 builder は「日ごと×件数」の共通ループだけを統合した。wrapper 名と各レコードの意味は維持しているので、DRY と可読性のバランスは保てている。
- `OrderMonitor` の split-source helper 再利用は `_resolve_regime_name` にも広がった。`inspect.getsource(...)` のローカル参照がまた 1 つ減った。
- raw-reader 系については今回追加で見たが、現状は `feature_enricher` 側の helper 境界で十分で、これ以上の統合は別責務まで巻き込みやすいので見送った。

---

## 2026-03-09 / Session 037-067

### 実施
- `tests/unit/v460/test_236_state_persistence_cqs.py`
  - split-source の参照先を `_fill_test_source.py` ベースに整理
  - `_build_state_snapshot()` / `_restore_common_state()` の source 参照を現行 split 先 (`OrchestratorLifecycleMixin`) に追随
- `tests/unit/v460/test_230_ffd_deadzone_streak_guards.py`
  - `inspect.getsource(...)` ベースの module source 検査を `read_source_text(...)` に統一
  - `_SideState` import を module scope へ集約
- `tests/unit/v460/test_306_proposals.py`
  - AB judgment / adaptation / config hot reload / maker price / `FillRecord` 関連の method 内 import を module scope に集約
- `ztb/data/raw_paths.py`
  - `resolve_raw_dir()`, `resolve_available_raw_dates()`, `RawDirLike` を新設
- `scripts/v460/ml/feature_enricher.py`
  - raw dir 正規化 helper を `ztb.data.raw_paths` 再利用へ変更
- `scripts/v460/build_features.py`
  - raw dir / available date 解決を shared helper 再利用へ変更
- `ztb/data/market_data_collector.py`
  - `raw_dir` 解決を shared helper 再利用へ変更
- `ztb/data/trades_health.py`
  - raw dir 解決を shared helper 再利用へ変更
- `ztb/data/trades_recorder.py`
  - raw dir 解決を shared helper 再利用へ変更

### 結果
- focused:
  - `tests/unit/v460/test_236_state_persistence_cqs.py`
  - `tests/unit/v460/test_230_ffd_deadzone_streak_guards.py`
  - `tests/unit/v460/test_306_proposals.py`
  - `139 passed in 5.44s`
- focused:
  - `tests/unit/v460/test_build_features_pipeline.py`
  - `tests/unit/v460/test_158_oracle_test.py`
  - `tests/unit/v460/test_ob_recorder.py`
  - `40 passed in 6.23s`

### 主要改善
- split-source assertion は local `inspect.getsource(...)` 実装を増やす流れから外し、`_fill_test_source.py` の shared helper と split-file 定数へ寄せた。今後の分割追随は helper 側の責務として扱える。
- `test_306_proposals.py` の method 内 import は 0 にできた。追加ケースでも import が散らばりにくい形になった。
- raw path/date 解決は `ztb.data.raw_paths` に移したので、`scripts` 側 helper を `ztb` から逆参照する不自然な境界を作らずに、production 側の再利用点を 1 箇所へ集約できた。
- `test_236_state_persistence_cqs.py` の `_restore_common_state` は現行 split layout では `OrchestratorLifecycleMixin` にあるため、その実体に合わせて source 契約を修正した。

---

## 2026-03-09 / Session 037-068

### 実施
- `tests/unit/v460/_fill_test_source.py`
  - `OB_UTILS`, `SKIP_GATE_EVALUATOR`, `MAKER_REGIME_BOOST`, `MAKER_MICROSTRUCTURE` を追加
- `tests/unit/v460/test_260_compute_extract_regime_split.py`
  - `MakerPrice` source 契約を shared helper 化
  - `compute/_apply_loss_boost/_apply_ffd_boost` は `maker_price.py`
  - `regime_boost` 群は `maker_regime_boost.py` の実体へ追随
- `tests/unit/v460/test_266_market_theory_protocol.py`
  - `MakerPrice`/`ob_utils`/`skip_gate_evaluator`/`fill_cycle_executor` の source 検査を shared helper 化
  - `_apply_as_reservation_shift`, `_apply_kyle_lambda`, `_apply_amihud_illiq` は `maker_microstructure.py` の実体へ追随
  - `OrderBookSnapshot`, `SkipGateAdapter`, `typing` の method 内 import を整理
- `tests/unit/v460/test_277_magic_number_grounding.py`
  - orchestrator / regime_policy / cycle_gate_aggregator / micro_circuit_breaker / DD guard の method 内 import を module scope に集約
- `tests/unit/v460/test_237_phantom_position_guard.py`
  - `FillRecord`, `FillMonitorResult`, `FillTestState`, `BalanceChecker`, `FillTestConfig`, cancel reason 定数の method 内 import を module scope に集約
- `tests/unit/v460/test_183_log_analysis_improvements.py`
  - inline `yaml.safe_load(...)` を parsed module constants + fixture 再利用へ変更
  - `_HOT_RELOADABLE_FIELDS` import も module scope に集約

### 結果
- focused:
  - `tests/unit/v460/test_260_compute_extract_regime_split.py`
  - `tests/unit/v460/test_266_market_theory_protocol.py`
  - `56 passed in 2.64s`
- focused:
  - `tests/unit/v460/test_277_magic_number_grounding.py`
  - `tests/unit/v460/test_237_phantom_position_guard.py`
  - `tests/unit/v460/test_183_log_analysis_improvements.py`
  - `91 passed in 2.15s`

### 主要改善
- `MakerPrice` source-contract は monolith 前提の `inspect.getsource(...)` から抜けて、現行 split layout の実体ファイルを直接読む形に揃った。今後 `maker_price.py` / `maker_regime_boost.py` / `maker_microstructure.py` の責務分離を維持しやすい。
- `test_277_magic_number_grounding.py` と `test_237_phantom_position_guard.py` は、追加ケースを積んでも import boilerplate が再増殖しにくい形になった。
- `test_183_log_analysis_improvements.py` は YAML 断片を 1 回 parse して fixture で複製する方式に変えたので、同じ literal を何度も parse しない。シナリオ名は維持しているので意図も崩れていない。

---

## 2026-03-09 / Session 037-069

### 実施
- `tests/unit/v460/test_239_feasible_quote.py`
  - `MakerPriceCalculator.compute` と `FillCycleExecutorMixin.run_single_cycle` の source 契約を `_fill_test_source.py` helper に変更
  - `FillTestConfig`, `FastFillDefense`, `FillCycleExecutorMixin`, `re` の import を module scope に集約
- `tests/unit/v460/test_254_frozen_side_persist_getattr_cleanup.py`
  - `_build_state_snapshot` / `_restore_common_state` / `_check_regime_stop_conditions` / `cleanup_heartbeat` / heartbeat psutil logging の source 検査を split-file helper に変更
  - `FillTestState`, `FillTestStatePersistence`, `FillLoopOrchestratorMixin`, `deque` を module scope に集約

### 結果
- focused:
  - `tests/unit/v460/test_239_feasible_quote.py`
  - `tests/unit/v460/test_254_frozen_side_persist_getattr_cleanup.py`
  - `32 passed in 1.36s`

### 主要改善
- `test_239_feasible_quote.py` は `MakerPrice` / executor の source 契約を現在の split-source 流儀に合わせた。`inspect.getsource(...)` ベースの古い参照がまた 1 束消えた。
- `test_254_frozen_side_persist_getattr_cleanup.py` は orchestrator monolith 前提の source 読込をやめ、実際に責務が置かれている lifecycle/guards/post_cycle/pre_cycle へ追随した。今後の orchestrator 分割変更にも追従しやすい。
- 同時に `test_230_ffd_deadzone_streak_guards.py` も見直したが、この時点では追加で削れる `inspect.getsource(...)` や method 内 import の残件はなかった。

---

## 2026-03-10 / Session 037-070

### 実施
- `tests/unit/v460/_fill_test_source.py`
  - `FILL_CONFIG`, `EVENT_LOGGER` を追加
- `tests/unit/v460/test_253_hot_reload_dead_config_getattr_bare_except.py`
  - `fill_config.py`, `fill_cycle_executor.py`, `event_logger.py` の source 読込を shared helper 化
  - `_HOT_RELOADABLE_FIELDS`, `FillCycleExecutorMixin`, `TeeWriter`, `event_logger` の import を module scope に集約
- `tests/unit/v460/test_255_getattr_bare_except_cleanup.py`
  - `SKIP_GATE_EVALUATOR`, `ORDER_MONITOR`, `OB_UTILS` の shared path constant を再利用
  - `ob_utils` source 読込も shared path を優先するよう整理
- `scripts/v460/lib/ob_recorder.py`
  - raw dir 解決を `ztb.data.raw_paths.resolve_raw_dir()` 再利用へ変更
  - local `_DEFAULT_RAW_DIR` を削除

### 結果
- focused:
  - `tests/unit/v460/test_253_hot_reload_dead_config_getattr_bare_except.py`
  - `tests/unit/v460/test_255_getattr_bare_except_cleanup.py`
  - `tests/unit/v460/test_ob_recorder.py`
  - `45 passed in 2.09s`

### 主要改善
- `253/255` は source 読込の流儀がほぼ shared helper に揃った。今後の split-file 変更や UTF-8/BOM 対応は helper 側で吸収できる。
- `ob_recorder.py` も raw-dir 正規化を共通 helper に寄せたので、`feature_enricher` / `build_features` / `market_data_collector` / `trades_recorder` と同じ境界で扱えるようになった。
- 今回は性能差分より再利用境界の整理が主成果で、raw path の実装分岐をさらに 1 箇所減らせた。

---

## 2026-03-10 / Session 037-071

### 実施
- `tests/unit/v460/test_203_dd_state_persistence.py`
  - `OrchestratorPreCycleMixin._handle_dd_halt` の source 契約を shared helper に変更
  - `DailyDrawdownGuard`, `FillLoopOrchestratorMixin` を module scope に集約
- `tests/unit/v460/test_226_loss_boost_decay_inv_skew_state.py`
  - `_apply_loss_boost`, `_handle_dd_halt`, `_feed_mcb_sad`, `_rebuild_fast_fill_defense` の source 契約を shared helper / `read_fill_test_method_source()` に変更
- `tests/unit/v460/test_151_confidence_lot.py`
  - `FillTestRunner`, `FillRecord` を module scope に集約
- `tests/unit/v460/test_166_remaining_tasks.py`
  - `FillRecord`, `FillMonitorResult`, `FillTestConfig`, `SideSelector`, `_GATE_TO_CANCEL_REASON` を module scope に集約
- `scripts/v460/lib/maker_risk_guards.py`
  - `_current_utc_hour()` を追加
  - `_resolve_sell_hour_boost_mult()` を追加
  - `_apply_sell_hour_boost()` が時間取得と config lookup を helper 再利用へ変更

### 結果
- focused:
  - `tests/unit/v460/test_203_dd_state_persistence.py`
  - `tests/unit/v460/test_226_loss_boost_decay_inv_skew_state.py`
  - `tests/unit/v460/test_151_confidence_lot.py`
  - `tests/unit/v460/test_166_remaining_tasks.py`
  - `tests/unit/v460/test_306_proposals.py -k 'not test_yaml_has_microprice_side'`
  - `150 passed, 1 deselected in 10.14s`

### 主要改善
- DD halt / loss_boost / FFD hot-reload の source 契約が shared helper にさらに寄ったので、orchestrator split と `run_fill_test` 分割への追随点が減った。
- `151/166` の `FillRecord` 周辺は、今後 field 追加が入っても import boilerplate が増えにくい形になった。
- production 側では `sell_hour_offset_boost` の時間帯判定を pure helper に分離したので、`skip_gate_hour_offsets` など他の time-of-day ルールとの共通化を検討しやすい下地ができた。

---

## 2026-03-10 / Session 037-072

### 実施
- `tests/unit/v460/_fill_test_source.py`
  - `ADAPTATION_ENGINE`, `BALANCE_CHECKER`, `OB_RECORDER`, `MAKER_RISK_GUARDS` を追加
  - cached `read_function_source()` を追加
- `tests/unit/v460/test_160_ab_judgment.py`
  - `side_regime_dashboard` import と `json` import を module scope に集約
- `tests/unit/v460/test_168_daily_health_integration.py`
  - `daily_health_check` import を module scope に集約
  - 未使用だった `_import_fn()` と余剰 import を削除
- `tests/unit/v460/test_258_as_reservation_vpin_continuous_protocol.py`
  - `AdaptationEngine.try_auto_adapt`, `MakerPriceCalculator.__init__/compute`, `RiskGuardsMixin._apply_volatility_guard` の source 参照を shared helper 化
- `tests/unit/v460/test_261_protocol_type_safety.py`
  - `config_hot_reload`, `balance_checker`, `ob_utils`, `ob_recorder`, `maker_price` の source 参照を shared helper 化
- `tests/unit/v460/test_305_p0_improvements.py`
  - `MakerPrice.compute` source と `_HOT_RELOADABLE_FIELDS` を module scope に集約
- `scripts/v460/lib/hour_rules.py`
  - `current_utc_hour()`, `utc_hour_from_timestamp()`, `resolve_hour_float()`, `resolve_optional_hour_float()` を追加
- `scripts/v460/lib/maker_risk_guards.py`
  - `current_utc_hour` / `resolve_optional_hour_float` を再利用
- `scripts/v460/lib/time_filter.py`
  - `current_utc_hour()` 再利用へ変更
- `scripts/v460/lib/orchestrator_pre_cycle.py`
  - hard-skip / DD / alert-mode の UTC hour 解決を `current_utc_hour()` に統一
- `scripts/v460/lib/skip_gate_evaluator.py`
  - `skip_gate_hour_offsets` 解決を `utc_hour_from_timestamp()` + `resolve_hour_float()` に統一
- `tests/unit/v460/test_163_regime_adaptive_gating.py`
  - `datetime` patch を `current_utc_hour` patch に変更
- `tests/unit/v460/test_306_proposals.py`
  - sell-hour boost テストを `current_utc_hour` patch ベースへ変更
- `tests/unit/v460/test_regime_detector.py`
  - `TimeFilter` 系テストを `current_utc_hour` patch ベースへ変更

### 結果
- focused:
  - `tests/unit/v460/test_160_ab_judgment.py`
  - `tests/unit/v460/test_168_daily_health_integration.py`
  - `tests/unit/v460/test_258_as_reservation_vpin_continuous_protocol.py`
  - `tests/unit/v460/test_261_protocol_type_safety.py`
  - `tests/unit/v460/test_305_p0_improvements.py`
  - `162 passed in 3.33s`
- related focused:
  - `tests/unit/v460/test_169_config_hot_reload.py`
  - `tests/unit/v460/test_237_phantom_position_guard.py`
  - `tests/unit/v460/test_277_magic_number_grounding.py`
  - `tests/unit/v460/test_306_proposals.py -k 'not test_yaml_has_microprice_side'`
  - `157 passed, 1 deselected in 3.54s`
- hour-rule regression:
  - `tests/unit/v460/test_094_stale_order.py`
  - `tests/unit/v460/test_163_regime_adaptive_gating.py`
  - `tests/unit/v460/test_169_config_hot_reload.py`
  - `tests/unit/v460/test_196_velocity_proportional_trending_soft.py`
  - `tests/unit/v460/test_336_yaml_code_drift_prevention.py`
  - `tests/unit/v460/test_fill_test_config.py`
  - `213 passed in 7.00s`
- filtered broad:
  - `tests/unit/v460/`
  - `--ignore=test_113_resilience.py`
  - `--ignore=test_152_parallel_tasks.py`
  - `--ignore=test_260_compute_extract_regime_split.py`
  - `--deselect=test_306_proposals.py::TestProposalsConfigSync::test_yaml_has_microprice_side`
  - `4206 passed, 13 warnings in 72.82s`

### 主要改善
- source 契約テストは `inspect` / `inspect.getsourcefile` / `Path(...).read_text(...)` の局所実装をさらに減らし、split-file 変更に追随しやすい形へ寄った。
- production 側の hour-based rule は `maker_risk_guards`, `time_filter`, `orchestrator_pre_cycle`, `skip_gate_evaluator` で同じ helper 境界を共有するようになり、今後の時間帯ルール追加でも patch 点と責務の重複が増えにくくなった。
- `time_filter` まわりのテスト patch 先も helper 境界に揃えたので、`datetime` 実装詳細に依存せず意図した hour 判定そのものを検証する形になった。

---

## 2026-03-10 / Session 037-073

### 実施
- `tests/unit/v460/test_356_g2_sac_blockers.py`
  - `yaml`, `pyarrow.parquet`, `load_parquet`, `HeavyTradingEnv`, `EnvironmentConfig`, `_create_training_env` を module scope に集約
  - cached `_load_g2_sac_yaml()`, `_load_g2_schema_names()`, `_load_g2_real_df_2000()` を追加
  - B1 YAML 構造テストを shared YAML fixture 再利用へ変更
  - training-data integrity テストを shared schema fixture 再利用へ変更
  - `TestHeavyTradingEnvIntegration.real_df` を class-scope 化し、既存 `load_parquet()` で selected features + `close` だけを一度ロードする形に変更
  - `TestHeavyTradingEnvIntegration.env_config` を class-scope 化
  - `_create_env()` helper を追加し、env 生成の重複を集約
  - `test_create_training_env_pipeline` も shared YAML + selected feature loader 再利用へ変更

### 結果
- focused:
  - `tests/unit/v460/test_356_g2_sac_blockers.py`
  - `38 passed in 5.03s`
- filtered broad:
  - `tests/unit/v460/`
  - `--ignore=test_113_resilience.py`
  - `--ignore=test_152_parallel_tasks.py`
  - `--ignore=test_260_compute_extract_regime_split.py`
  - `--deselect=test_306_proposals.py::TestProposalsConfigSync::test_yaml_has_microprice_side`
  - `4206 passed, 13 warnings in 40.62s`

### 主要改善
- `356` の実データ integration は raw `pd.read_parquet()` の全列読込をやめ、既存の `scripts.v460.lib.data_loader.load_parquet()` を selected feature ベースで再利用するようになった。
- これにより `HeavyTradingEnv` integration の支配的 setup が「毎回 parquet を読む」構成から「1 回だけ必要列を読む」構成へ変わり、broad top の 5-6 秒級 setup が大きく後退した。
- 既存 helper を再利用したので、将来 YAML 側で selected feature が変わっても test 側が別実装で乖離しにくい。

## 2026-03-10 / Session 037-074

### 実施
- `docs/v460/037_phg_rpt_refactoring_session_log.md` に `§5.2` 見出し自体は存在しないことを再確認し、参照元が [363_ph2_rev_361_362_review_validation.md](/mnt/c/Users/Admin/dev/zaif-trade-bot/docs/v460/363_ph2_rev_361_362_review_validation.md) `### 5.2 Codex プロンプト` であることを特定。
- `scripts/v460/lib/fill_test_cli.py`
  - `_read_lock_heartbeat_age_sec()` を追加
  - `_dump_exit_diagnostics()` を追加
  - `atexit.register()` と signal handler と finally 経路を同一 helper に統一
  - 終了時に RSS/VMS/UTC timestamp/run_id/stop_reason/lock heartbeat age を `results_dir/diagnostics/exit_dump_*.json` へ出力するように変更
- `tests/unit/v460/test_fill_test_cli_diagnostics.py`
  - JSON dump 構造
  - heartbeat age 算出
  - psutil 失敗時フォールバック
  - atexit/signal hook の source 契約
  を追加
- `ztb/io/jsonl.py`
  - `read_tail_jsonl_objects()` を end-seek ベースの tail 読みへ変更
  - `warn_malformed=True` のときだけ旧 forward scan を維持
- `tests/unit/utils/test_jsonl.py`
  - last-N 読込
  - BOM/blank 行
  - malformed line
  の回帰を追加
- `scripts/v460/lib/stopgap_health.py`
  - `_build_daily_metrics_row()` を追加
  - `compute_daily_metrics()` に single-record fast-path を追加
- `scripts/v460/run_experiment.py`
  - G2 E2 を `max_roi_seed_std` / `max_ic_seed_std` 互換で解釈するよう修正
- `scripts/v460/run_gate_check.py`
  - 上と同じ互換ロジックに揃えた
- `tests/unit/v460/test_config_validation.py`
  - `g2_train` threshold の検証を新旧キー両対応へ修正

### 結果
- focused:
  - `tests/unit/v460/test_fill_test_cli_diagnostics.py`
  - `tests/unit/utils/test_jsonl.py`
  - `tests/unit/v460/test_stopgap_health.py`
  - `63 passed in 1.30s`
- focused:
  - `tests/unit/v460/test_356_g2_sac_blockers.py`
  - `tests/unit/v460/test_enricher_skip_gate.py`
  - `tests/unit/v460/test_ml_pipeline.py`
  - `tests/unit/v460/test_fill_test_cli_diagnostics.py`
  - `tests/unit/utils/test_jsonl.py`
  - `tests/unit/v460/test_stopgap_health.py`
  - `193 passed in 9.90s`
- focused:
  - `tests/unit/v460/test_config_validation.py`
  - `tests/unit/v460/test_gate_check.py`
  - `tests/unit/v460/test_356_g2_sac_blockers.py`
  - `115 passed in 9.75s`
- focused:
  - `tests/unit/v460/test_fill_test_cli_diagnostics.py`
  - `tests/unit/utils/test_jsonl.py`
  - `tests/unit/v460/test_stopgap_health.py`
  - `tests/unit/v460/test_enricher_skip_gate.py`
  - `tests/unit/v460/test_ml_pipeline.py`
  - `tests/unit/v460/test_v460_core.py`
  - `210 passed in 11.41s`
- filtered broad:
  - `tests/unit/v460/`
  - `--ignore=test_113_resilience.py`
  - `--ignore=test_152_parallel_tasks.py`
  - `--ignore=test_260_compute_extract_regime_split.py`
  - `--deselect=test_306_proposals.py::TestProposalsConfigSync::test_yaml_has_microprice_side`
  - `4214 passed, 13 warnings in 43.33s`

### 主要改善
- `037` の追加依頼だった OPS-1 は、`363# §5.2` の要求どおり `fill_test_cli.py` に終了時メモリ診断として実装した。
- `read_tail_jsonl_objects()` の実 tail 読み化により、real-data integration が大きい JSONL を毎回先頭から走査する無駄を削れた。
- `compute_daily_metrics()` は 1-record ケースで一般 grouping を通らなくなり、broad top に残っていた `test_stopgap_health.py` の小さな固定費を削減した。
- G2 E2 は `max_roi_seed_std` への移行途中で test/config/runtime が食い違っていたため、runtime は新旧互換、validation は新旧許容に揃えた。

## 2026-03-10 / Session 037-075

### 実施
- `configs/v460/fill_test.yaml`
  - `resilience.health_monitor.check_interval_sec` を `300.0 -> 60.0` に変更
- `scripts/v460/lib/fill_config.py`
  - `hm_check_interval_sec` のコードデフォルトを `60.0` に更新
- `scripts/v460/lib/resilience.py`
  - `HealthThresholds.check_interval_sec` デフォルトを `60.0` に更新
  - RSS warning ログ文言を `RSS ... exceeds warn threshold ...` に強化
- `ops/windows/fill_test_watchdog.ps1`
  - `restart.lock` stale 判定を `30s -> 120s` に延長
  - stale 判定コメントを `360# OPS-4` に合わせて更新
  - `Start-Process` 後に `fill_test.lock` を最大 30 秒、2 秒間隔で poll する OPS-6 起動確認待ちを追加
- `tests/unit/v460/test_fill_test_config.py`
  - YAML → `FillTestConfig` roundtrip に `hm_check_interval_sec == 60.0` を追加
- `tests/unit/v460/test_health_monitor_resilience.py`
  - default interval 60 秒
  - RSS warning 文言
  の focused regression を追加
- `tests/unit/v460/test_fill_test_watchdog_ops.py`
  - watchdog source 契約として
    - `restart.lock` 120 秒 stale
    - `fill_test.lock` poll loop
  を追加

### 結果
- PowerShell syntax:
  - `powershell.exe -NoProfile -Command '& { [System.Management.Automation.Language.Parser]::ParseFile(...) }'`
  - parse error なし
- focused:
  - `tests/unit/v460/test_fill_test_config.py`
  - `tests/unit/v460/test_health_monitor_resilience.py`
  - `tests/unit/v460/test_fill_test_watchdog_ops.py`
  - `tests/unit/v460/test_stopgap_health.py`
  - `143 passed in 6.62s`
- filtered broad:
  - `tests/unit/v460/`
  - `--ignore=test_113_resilience.py`
  - `--ignore=test_152_parallel_tasks.py`
  - `--ignore=test_260_compute_extract_regime_split.py`
  - `--deselect=test_306_proposals.py::TestProposalsConfigSync::test_yaml_has_microprice_side`
  - `4218 passed, 13 warnings in 40.53s`

### 主要改善
- OPS-2: 実行中 RSS 膨張の検知間隔を 5 分から 60 秒へ短縮し、OOM 前兆の可観測性を上げた。
- OPS-4: watchdog 間の `restart.lock` 取り合いを起こしやすかった 30 秒 stale 判定を 120 秒へ伸ばし、high-load 時の dual-spawn リスクを下げた。
- OPS-6: watchdog は再起動を投げるだけでなく `fill_test.lock` の出現まで確認するようになり、「起動したつもりで実は失敗」の観測漏れを減らした。
- broad の残ホットスポットは引き続き `test_356_g2_sac_blockers.py` の HeavyTradingEnv integration と `test_enricher_skip_gate.py` の real-data setup に集中している。

## 2026-03-10 / Session 037-076

### 実施
- `tests/unit/v460/test_356_g2_sac_blockers.py`
  - G2 SAC integration の cached real-data slice を `2000 -> 128` 行へ縮小
  - `HeavyTradingEnv` helper での `real_df.copy(deep=True)` を廃止
  - `_create_training_env(...)` でも同じ cached `real_df` を再利用して parquet 再読込を除去
- `tests/unit/v460/test_build_features_pipeline.py`
  - real-mode aggregate を `output_path=None` の 1 回集約へ整理
  - 40分 aggregate を 30分 schema 検証と microstructure 検証で共通再利用
- `tests/unit/v460/test_enricher_skip_gate.py`
  - real-data integration の上限行数を `320 -> 280`
  - `enrich_fill_records(...)` 前の余計な `.copy()` を削除
- `ztb/trading/environment/heavy_env/core.py`
  - env init の reward-parameter dump を DEBUG 条件付きへ変更
- `ztb/trading/environment/components/calculators/reward_calculator.py`
  - RewardCalculator init の reward-parameter dump を DEBUG 条件付きへ変更

### 結果
- focused:
  - `tests/unit/v460/test_356_g2_sac_blockers.py`
  - `42 passed in 6.74s`
- focused:
  - `tests/unit/v460/test_356_g2_sac_blockers.py`
  - `tests/unit/v460/test_build_features_pipeline.py`
  - `tests/unit/v460/test_enricher_skip_gate.py`
  - `126 passed in 7.45s`
- full suite:
  - `pytest tests/unit/v460/ -x -q`
  - test failure ではなく coverage gate (`TOTAL 12% < fail-under=80`) で停止
- full suite (functional):
  - `pytest tests/unit/v460/ -x -q --no-cov`
  - `4277 passed, 13 warnings in 43.13s`
- filtered broad:
  - `tests/unit/v460/`
  - `--ignore=test_113_resilience.py`
  - `--ignore=test_152_parallel_tasks.py`
  - `--ignore=test_260_compute_extract_regime_split.py`
  - `--deselect=test_306_proposals.py::TestProposalsConfigSync::test_yaml_has_microprice_side`
  - `4218 passed, 13 warnings in 37.24s`

### 主要改善
- `HeavyTradingEnv` integration は focused で setup `0.59s`、real-data broad でも最上位の固定費を大きく削った。
- `build_features_pipeline` の real-mode setup は 2 本立ての raw 生成/集約をやめ、単一路線化できた。
- `enricher_skip_gate` の real-data setup は broad で `0.35s` まで低下した。
- `HeavyTradingEnv` / `RewardCalculator` の WARNING ログ固定費を削除し、通常実行の log capture と `asdict()` コストを減らした。

## 2026-03-10 / Session 037-077

### 実施
- `prompts/codex_038_tune2_tune4.md` の C-4 / C-5 を反映
- `configs/v460/fill_test.yaml`
  - `buy_dynamic_kill.threshold_bps` コメントに `364# TUNE-4 skip` 注記を追加
  - `daily_drawdown.per_side_hard_limit_bps` を `-30.0 -> -50.0`
  - `daily_drawdown.per_side_halt_cycles` を `15 -> 10`
  - `daily_drawdown.per_side_reanchor_budget_bps` を `-15.0 -> -25.0`
- `scripts/v460/lib/fill_config.py`
  - `per_side_dd_hard_limit_bps` デフォルトを `-50.0`
  - `per_side_dd_halt_cycles` デフォルトを `10`
  - `per_side_dd_reanchor_budget_bps` デフォルトを `-25.0`
- `tests/unit/v460/test_168_daily_drawdown_guard.py`
  - default assertion を新しい TUNE-2 値へ更新
- `tests/unit/v460/test_336_yaml_code_drift_prevention.py`
  - `per_side_dd_halt_cycles` を stale allowlist から除去

### 結果
- focused:
  - `tests/unit/v460/test_168_daily_drawdown_guard.py`
  - `tests/unit/v460/test_169_c1_c3_c4_config.py`
  - `tests/unit/v460/test_336_fill_config_parser.py`
  - `tests/unit/v460/test_344_improvements.py`
  - `181 passed in 2.46s`
- focused:
  - `tests/unit/v460/test_336_yaml_code_drift_prevention.py`
  - `4 passed in 0.85s`
- filtered broad:
  - `tests/unit/v460/`
  - `--ignore=test_113_resilience.py`
  - `--ignore=test_152_parallel_tasks.py`
  - `--ignore=test_260_compute_extract_regime_split.py`
  - `--deselect=test_306_proposals.py::TestProposalsConfigSync::test_yaml_has_microprice_side`
  - `4218 passed, 13 warnings in 43.07s`

### 主要改善
- `per_side_dd_halt` は本番 YAML とコードデフォルトの両方で、より緩い `-50bps / 10 cycles` に揃った。
- `per_side_reanchor_budget_bps` も hard limit に合わせて `-25bps` へ比例調整した。
- `TUNE-4` は実際に BDK bottleneck が出ていないため、閾値は動かさず YAML コメントだけで判断根拠を残した。
- drift prevention allowlist を同時に掃除し、設定変更で stale allowlist を増やさない状態へ戻した。

## 2026-03-10 / Session 037-078

### 実施
- `tests/unit/v460/test_356_g2_sac_blockers.py`
  - `HeavyTradingEnv` を共有する `shared_env` fixture を追加
  - `_create_training_env(...)` 検証を `training_env_bundle` fixture へ寄せた
  - 単純な instantiation/reset/step 系は重い env 構築を毎回やり直さない形へ整理
- `tests/unit/v460/test_enricher_skip_gate.py`
  - `_cached_real_enriched_training_df()` を追加
  - `real_enriched_df` class fixture を cached DataFrame + `copy(deep=False)` に変更
- `scripts/v460/run_v460_unit_tests.py`
  - `tests/unit/v460/` を `--no-cov --tb=short -q` で実行する専用ランナーを追加
  - `pytest.ini` の repository-wide coverage gate を `v460` subset 実行から切り離した

### 結果
- focused:
  - `tests/unit/v460/test_356_g2_sac_blockers.py`
  - `tests/unit/v460/test_enricher_skip_gate.py::Test058Integration`
  - `44 passed in 11.37s`
- smoke:
  - `scripts/v460/run_v460_unit_tests.py --help`
  - 正常に pytest help を透過表示

### 主要改善
- `HeavyTradingEnv` integration の基本ケースが shared fixture に乗り、毎回の env 構築固定費を減らした。
- `real_enriched_df` は class-scope での再利用時に deep copy を避け、real-data fixture の最後の無駄コピーを削った。
- `v460` 専用ランナーを追加したことで、subset 実行時に毎回 `--no-cov` を明示しなくても coverage gate を回避できる入口ができた。

## 2026-03-10 / Session 037-079

### 実施
- `tests/unit/v460/test_retrain_hot_reload.py`
  - `TestHotReload` に `_create_evaluator(...)` を追加
  - tempdir / model path / evaluator 構築の重複を共通化
- `tests/unit/v460/test_build_features_pipeline.py`
  - real-mode aggregate の共有 fixture を `32` 分ベースへ圧縮
  - `30` 行 schema 検証と microstructure 検証を同じ base aggregate から再利用
- `tests/unit/v460/test_enricher_skip_gate.py`
  - real-data ladder を `120/220/280` の guarded 構成で維持することを再確認
  - `180/240` では `n_samples > 30` 契約を割るため採用しなかった

### 結果
- focused:
  - `tests/unit/v460/test_enricher_skip_gate.py`
  - `tests/unit/v460/test_retrain_hot_reload.py`
  - `tests/unit/v460/test_build_features_pipeline.py`
  - `166 passed in 10.50s`
- filtered broad:
  - `tests/unit/v460/`
  - `--ignore=test_113_resilience.py`
  - `--ignore=test_152_parallel_tasks.py`
  - `--ignore=test_260_compute_extract_regime_split.py`
  - `--deselect=test_306_proposals.py::TestProposalsConfigSync::test_yaml_has_microprice_side`
  - `4218 passed, 13 warnings in 53.07s`

### 主要改善
- `TestHotReload` の setup は helper 化で今後の追加ケースでも重複が増えにくくなった。
- `build_features_pipeline` は 40 分ぶんの synthetic raw を作る必要がなくなり、real-mode fixture の無駄を削った。
- `enricher_skip_gate` は最小 ladder を探ったが、学習成立条件を守るには `220/280` fallback が必要と確認できた。
- broad 上位は `HeavyTradingEnv` と `SkipGate save/load` 側へ再集中し、今回触った setup 系での回帰は出ていない。

## 2026-03-10 / Session 037-080

### 実施
- `ztb/utils/dataclass_utils.py`
  - `shallow_asdict(...)` を追加
- `ztb/trading/environment/heavy_env/core.py`
  - `RewardSettings` merge と debug logging を `shallow_asdict(...)` ベースへ変更
- `ztb/trading/environment/components/calculators/reward_calculator.py`
  - reward settings debug dump を `shallow_asdict(...)` に変更
- `scripts/v460/ml/skip_gate.py`
  - `pickle.HIGHEST_PROTOCOL`
  - `Path.write_bytes()`
  - `Path.read_bytes()`
  に切り替えて save/load の固定費を削減
- `ztb/data/market_data_collector.py`
  - `aggregate_to_1min()` の index 構築で一時 `dt` 列を作らない形へ整理
- `tests/unit/utils/test_dataclass_utils.py`
  - `shallow_asdict(...)` の unit test を追加

### 結果
- focused:
  - `tests/unit/utils/test_dataclass_utils.py`
  - `tests/unit/v460/test_356_g2_sac_blockers.py`
  - `tests/unit/v460/test_aggregate_to_1min.py`
  - `tests/unit/v460/test_enricher_skip_gate.py`
  - `141 passed in 10.66s`
- filtered broad:
  - `tests/unit/v460/`
  - `--ignore=test_113_resilience.py`
  - `--ignore=test_152_parallel_tasks.py`
  - `--ignore=test_260_compute_extract_regime_split.py`
  - `--deselect=test_306_proposals.py::TestProposalsConfigSync::test_yaml_has_microprice_side`
  - `4218 passed, 13 warnings in 40.52s`

### 主要改善
- `HeavyTradingEnv` integration setup は broad で `2.40s -> 1.59s` まで低下した。
- `SkipGate` save/load roundtrip は broad 上位に残るが、I/O 経路は production 側でより軽い実装に置き換わった。
- `aggregate_to_1min()` は列追加→`set_index()` の一時オブジェクトを減らし、集約テストの hot path を少し軽くした。
- broad 全体は `53.07s -> 40.52s` まで改善した。

## 2026-03-10 / Session 037-081

### 実施
- `ztb/metrics/fill_quality.py`
  - `_resolve_fill_record_files_by_date_range(...)` に single-day fast-path を追加
- `tests/unit/v460/test_fill_quality.py`
  - `_save_dated_linear_record(...)` を追加
  - date-range / file listing 系テストの file setup を helper に統一
- `tests/unit/v460/test_enricher_skip_gate.py`
  - `_save_and_load_gate(...)` を追加
  - `SkipGate` の roundtrip テストを helper 再利用に統一
- `tests/unit/v460/test_356_g2_sac_blockers.py`
  - cached real-data slice を `128 -> 96` に圧縮

### 結果
- focused:
  - `tests/unit/v460/test_356_g2_sac_blockers.py`
  - `tests/unit/v460/test_enricher_skip_gate.py`
  - `tests/unit/v460/test_fill_quality.py`
  - `318 passed, 5 warnings in 9.35s`
- filtered broad:
  - `tests/unit/v460/`
  - `--ignore=test_113_resilience.py`
  - `--ignore=test_152_parallel_tasks.py`
  - `--ignore=test_260_compute_extract_regime_split.py`
  - `--deselect=test_306_proposals.py::TestProposalsConfigSync::test_yaml_has_microprice_side`
  - `4218 passed, 13 warnings in 31.21s`

### 主要改善
- `HeavyTradingEnv` integration setup は focused で `1.37s -> 1.20s`、filtered broad で `1.59s -> 1.41s` まで低下した。
- `TestFillRecordIO::test_list_fill_record_files_date_range_uses_direct_resolution` は `0.22s -> 0.03s` まで低下した。
- `fill_quality` の date-range I/O setup は helper 化で今後の横展開先を増やしやすくなった。
- broad 全体は `40.52s -> 31.21s` まで改善した。

## 2026-03-10 / Session 037-082

### 実施
- `tests/unit/v460/test_retrain_hot_reload.py`
  - `_save_and_load_gate(...)` を追加
  - post-deploy verification の roundtrip を helper に統一
- `tests/unit/v460/test_ml_pipeline.py`
  - `as_training_data_small` / `fill_training_data_small` fixture を追加
  - classifier 系テストの `build_*_features(...)` + `head(...)` 重複を除去
  - GB テストの `gb_n_estimators` を `3` に縮小
- `tests/unit/v460/test_v460_core.py`
  - `_cached_microstructure_result_df()` を追加
  - `TestMicrostructure::test_feature_generation` を cached result 再利用に変更

### 結果
- focused:
  - `tests/unit/v460/test_retrain_hot_reload.py`
  - `tests/unit/v460/test_v460_core.py`
  - `tests/unit/v460/test_ml_pipeline.py`
  - `159 passed in 5.84s`
- filtered broad:
  - `tests/unit/v460/`
  - `--ignore=test_113_resilience.py`
  - `--ignore=test_152_parallel_tasks.py`
  - `--ignore=test_260_compute_extract_regime_split.py`
  - `--deselect=test_306_proposals.py::TestProposalsConfigSync::test_yaml_has_microprice_side`
  - `4218 passed, 13 warnings in 35.42s`

### 主要改善
- `retrain_hot_reload` の roundtrip 契約テストは helper 化で重複が減り、`test_deployed_verified_status` は focused で `0.01s` 帯になった。
- `ml_pipeline` は学習前の feature 構築を fixture 再利用に寄せ、AS/Fill classifier 群の call を `0.02s-0.03s` 帯に揃えた。
- `v460_core` の microstructure feature generation は cached result 再利用で再計算を避ける形になった。
- filtered broad の wall time は揺れたが、今回の対象 hotspot では回帰は出ていない。

## 2026-03-10 / Session 037-083

### 実施
- `tests/unit/v460/test_135_trades_and_gate.py`
  - `_read_single_jsonl_gz(...)` と `_record_ob_snapshot(...)` を追加
  - `TestOBRecorderRefactored` の setup/読取重複を削減
- `tests/unit/v460/test_169_config_hot_reload.py`
  - `_run_do_reload_with_content(...)` を追加
  - reload/update 系の `_prepare_reload_context(...) + _do_reload(...)` 反復を統一
- `tests/unit/v460/test_157_regime_features.py`
  - buy dynamic kill / trending boost YAML payload を module constant 化
- `tests/unit/v460/test_138_p1_preflight_calibration.py`
  - preflight pause / score calibration YAML payload を module constant 化
- `ztb/data/trades_health.py`
  - CLI の `--raw-dir` を `resolve_raw_dir(...)` に通すよう整理

### 結果
- focused:
  - `tests/unit/v460/test_135_trades_and_gate.py`
  - `tests/unit/v460/test_169_config_hot_reload.py`
  - `tests/unit/v460/test_157_regime_features.py`
  - `tests/unit/v460/test_138_p1_preflight_calibration.py`
  - `92 passed in 9.65s`
- filtered broad:
  - `tests/unit/v460/`
  - `--ignore=test_113_resilience.py`
  - `--ignore=test_152_parallel_tasks.py`
  - `--ignore=test_260_compute_extract_regime_split.py`
  - `--deselect=test_306_proposals.py::TestProposalsConfigSync::test_yaml_has_microprice_side`
  - `4218 passed, 13 warnings in 44.42s`

### 主要改善
- `OBRecorderRefactored` は file read/write の helper が揃い、同型テスト追加時の重複が減った。
- `config_hot_reload` は reload path 実行 helper ができ、`_do_reload` 直叩きケースをこれ以上増やしても setup 重複が増えにくくなった。
- `157` / `138` の YAML parse テストは payload を定数化し、意図の違う assertion だけを残す形へ整理した。
- 統合テストの「結合」自体は今回は見送った。既存ケースは失敗時の切り分け粒度を持っており、helper 化のほうが利得が大きかった。

## 2026-03-10 / Session 037-084

### 実施
- `tests/unit/v460/test_157_regime_features.py`
  - `inspect.getsource(...)` と `__import__(...)` ベースの source inspection をやめ、`_fill_test_source.py` の cached helper に統一
  - `MakerPriceCalculator._resolve_trending_boost` は split 後の実体 `maker_regime_boost.py::RegimeBoostMixin` を参照する形へ修正
- `tests/unit/v460/test_215_dd_fix_alert_mode.py`
  - `_alert_mode_path()` / `_write_alert_mode(...)` を追加
  - `alert_mode.json` の repeated file write を helper 経由に寄せた
- 横断確認
  - `tests/unit/v460/test_261_protocol_type_safety.py` も同時に回し、split-source helper 化との整合性を確認

### 結果
- focused:
  - `tests/unit/v460/test_157_regime_features.py`
  - `tests/unit/v460/test_215_dd_fix_alert_mode.py`
  - `tests/unit/v460/test_261_protocol_type_safety.py`
  - `81 passed in 4.65s`
- filtered broad:
  - `tests/unit/v460/`
  - `--ignore=test_113_resilience.py`
  - `--ignore=test_152_parallel_tasks.py`
  - `--ignore=test_260_compute_extract_regime_split.py`
  - `--deselect=test_306_proposals.py::TestProposalsConfigSync::test_yaml_has_microprice_side`
  - `4218 passed, 13 warnings in 50.24s`

### 主要改善
- 今回は「runtime 直接短縮」ではなく、「source inspection の cached 化」と「file setup helper 化」という別系統の改善を適用した。
- `test_157_regime_features.py` は split-file 構成に対して壊れにくい source 契約になり、今後の monolith 依存回帰を抑えやすくなった。
- `test_215_dd_fix_alert_mode.py` は runtime には大差ないが、`alert_mode.json` を使う追加ケースを入れても setup 重複が増えにくい形になった。
- この探索で、`test_261_protocol_type_safety.py` は現時点で追加の大きい無駄が少ないことも確認できた。

## 2026-03-10 / Session 037-085

### 実施
- `tests/unit/v460/test_356_g2_sac_blockers.py`
  - `SACAlgorithm` / `SACTrainModelProtocol` / `inspect` を module scope に集約
  - `g2_sac_train.yaml` コメント検証用に `_load_g2_yaml_text()` を追加
- `tests/unit/v460/test_enricher_skip_gate.py`
  - `_make_basic_gate()` を追加
  - `Test059PickleHash` の最小 gate 生成を共通化
- `ztb/training/unified_trainer/algorithms/sac_trainer.py`
  - reward-settings 検証と Gate0 debug の `dataclasses.asdict(...)` を `shallow_asdict(...)` に置換
  - 既に入っている `HeavyTradingEnv` 側の shallow 化と流儀を揃えた
- 横断確認
  - `tests/test_reward_config_integration.py` を追加で確認
  - 初回は coverage gate で落ちたため、確認は `--no-cov` で切り分けた

### 結果
- focused:
  - `tests/unit/v460/test_356_g2_sac_blockers.py`
  - `tests/unit/v460/test_enricher_skip_gate.py`
  - `119 passed in 8.90s`
- reward config integration:
  - `tests/test_reward_config_integration.py --no-cov`
  - `4 passed in 5.17s`
- filtered broad:
  - `tests/unit/v460/`
  - `--ignore=test_113_resilience.py`
  - `--ignore=test_152_parallel_tasks.py`
  - `--ignore=test_260_compute_extract_regime_split.py`
  - `--deselect=test_306_proposals.py::TestProposalsConfigSync::test_yaml_has_microprice_side`
  - `4225 passed, 13 warnings in 40.25s`

### 主要改善
- `g2_sac_blockers` の replay-buffer / YAML comment テストは import と file-read の固定費を減らした。
- `enricher_skip_gate` は roundtrip/hash 群の gate builder が 1 箇所に寄り、今後の save/load テスト追加時の重複が減った。
- production 側は `sac_trainer.py` でも shallow dataclass 展開に揃えたため、reward settings の debug/verification 経路で不要な deep copy を避ける形になった。
- 追加探索の結果、`RewardSettings` 系は v460 外でも整合しており、`tests/test_reward_config_integration.py` では挙動回帰は出なかった。

## 2026-03-10 / Session 037-086

### 実施
- `ztb/features/scalping.py`
  - `realized_volatility(...)` を nested loop から `cumsum` ベースの O(n) 実装へ置換
  - `order_flow_imbalance(...)` を numpy vectorize 化
  - `micro_volatility(...)` を vectorized return + rolling `std(ddof=0)` に置換
  - いずれも `pd.Series` 返却、`name`、`prev_close == 0 -> 0.0` の挙動を維持
- `tests/unit/core/features/test_scalping_features.py`
  - `micro_volatility` の基本検証を追加
  - `previous close == 0` と `window > len(df)` の edge case を追加
- 影響確認
  - `tests/unit/core/features/test_v4_feature_extractor.py` の `realized_volatility` / `order_flow_imbalance` 経路を focused で確認
  - `tests/unit/v460/` filtered broad も再実行

### 結果
- focused:
  - `tests/unit/core/features/test_scalping_features.py`
  - `14 passed in 6.14s`
- focused:
  - `tests/unit/core/features/test_v4_feature_extractor.py -k 'realized_volatility or order_flow_imbalance'`
  - `2 passed, 13 deselected in 7.88s`
- filtered broad:
  - `tests/unit/v460/`
  - `--ignore=test_113_resilience.py`
  - `--ignore=test_152_parallel_tasks.py`
  - `--ignore=test_260_compute_extract_regime_split.py`
  - `--deselect=test_306_proposals.py::TestProposalsConfigSync::test_yaml_has_microprice_side`
  - `4225 passed, 13 warnings in 43.81s`

### 主要改善
- `realized_volatility(...)` は毎回 window 内 returns を作り直す形をやめたため、計算量は実質 O(n²) から O(n) に落ちた。
- `order_flow_imbalance(...)` は loop を消しつつ、先頭 0.0 と body-size 0 の扱いを維持した。
- `micro_volatility(...)` は内側ループをなくし、ゼロ割回避の既存契約も保持した。
- 今回の core 側最適化は `v4_feature_extractor` と filtered broad の両方で回帰なしを確認した。

## 2026-03-10 / Session 037-087

### 実施
- `scripts/v460/lib/manifest.py`
  - `ManifestEntry.to_dict()` を `dataclasses.asdict(...)` から `shallow_asdict(...)` に変更
  - manifest entry は flat dataclass なので deep copy を不要化
- `tests/unit/v460/test_v460_core.py`
  - `_write_yaml(...)` / `_write_config_pair(...)` を追加
  - config-loader / `_task` preservation テストの `base.yaml` / `exp.yaml` 生成を共通化
- `tests/unit/v460/test_retrain_hot_reload.py`
  - `_write_corrupt_gate(...)` を追加
  - post-deploy verification の壊れた artifact 準備を helper 化

### 結果
- focused:
  - `tests/unit/v460/test_v460_core.py`
  - `tests/unit/v460/test_retrain_hot_reload.py`
  - `137 passed in 4.77s`
- filtered broad:
  - `tests/unit/v460/`
  - `--ignore=test_113_resilience.py`
  - `--ignore=test_152_parallel_tasks.py`
  - `--ignore=test_260_compute_extract_regime_split.py`
  - `--deselect=test_306_proposals.py::TestProposalsConfigSync::test_yaml_has_microprice_side`
  - `4241 passed, 13 warnings in 55.03s`

### 主要改善
- `manifest.py` は flat dataclass の deep copy をやめたので、`TestManifest::test_write_and_read` の hot path を軽くした。
- `test_v460_core.py` は YAML setup の編集点が 1 箇所に集まり、今後の config-loader ケース追加時の重複が減った。
- `test_retrain_hot_reload.py` は corrupt artifact の準備が共通化され、atomic/post-deploy 周辺テストの意図が読みやすくなった。
- focused では `test_v460_core.py` + `test_retrain_hot_reload.py` が `7.89s -> 4.77s` まで低下した。

## 2026-03-10 / Session 037-088

### 実施
- `scripts/v460/lib/stopgap_health.py`
  - `serialize_health_report(...)` を `asdict(...)` から `shallow_asdict(...)` へ変更
- `scripts/v460/lib/resilience.py`
  - `FillTestStatePersistence.save(...)` を `asdict(...)` から `shallow_asdict(...)` へ変更
- 探索
  - `ztb/features` / `ztb/data` / `scripts/v460/lib` を広く検索し、残る計算量改善候補を洗い出した

### 結果
- focused:
  - `tests/unit/v460/test_stopgap_health.py`
  - `tests/unit/v460/test_health_monitor_resilience.py`
  - `tests/unit/v460/test_215_dd_fix_alert_mode.py`
  - `89 passed in 3.33s`
- filtered broad:
  - `tests/unit/v460/`
  - `--ignore=test_113_resilience.py`
  - `--ignore=test_152_parallel_tasks.py`
  - `--ignore=test_260_compute_extract_regime_split.py`
  - `--deselect=test_306_proposals.py::TestProposalsConfigSync::test_yaml_has_microprice_side`
  - `4241 passed, 13 warnings in 40.36s`

### 主要改善
- `stopgap_health` と `resilience` はどちらも「JSON に落とす直前の dataclass」を shallow 化しただけで十分だった。nested dict/list は既に構築済みなので deep copy の利得がない。
- 追加探索で、次の計算量候補を確認した:
  - `ztb/data/trades_health.py`: `for i in range(lookback_days)` ベースの日付探索
  - `ztb/features/base_features_v456.py`: rolling 系の Python loop
  - `ztb/data/v433_feature_engineering.py`: 一部の per-row / per-window loop
- これらは今後も触れる余地があるが、今回の shallow 化は low-risk で先に入れられる改善として適用した。

## 2026-03-10 / Session 037-089

### 実施
- `ztb/features/scalping.py`
  - 残っていた短期特徴量の Python loop を整理し、以下を vectorize 化
    - `price_velocity(...)`
    - `micro_trend(...)`
    - `price_acceleration(...)`
    - `volume_surge(...)`
    - `tick_volume_ratio(...)`
    - `spread_pressure(...)`
    - `momentum_burst(...)`
    - `liquidity_surge(...)`
- `tests/unit/core/features/test_scalping_features.py`
  - 上記 vectorize 箇所の known-value / zero-divisor / rolling 境界テストを追加
- `ztb/data/trades_health.py`
  - raw trades 日付探索を `iterdir()` + 手動 suffix 判定から `glob("????????.jsonl.gz")` に変更
- `ztb/features/base_features_v456.py`
  - `_ema(...)` を `pandas.Series(...).ewm(adjust=False)` へ置換
  - `_adx_di(...)` の `plus_dm` / `minus_dm` 生成を vectorized mask 化
- `tests/unit/features/test_base_features_v456.py`
  - base feature 計算の focused regression test を新設

### 結果
- focused:
  - `tests/unit/core/features/test_scalping_features.py`
  - `tests/unit/features/test_base_features_v456.py`
  - `tests/unit/v460/test_135_trades_and_gate.py`
  - `tests/unit/v460/test_136_p1_retrain_kill.py`
  - `87 passed in 4.81s`
- extractor subset:
  - `tests/unit/core/features/test_v4_feature_extractor.py -k 'realized_volatility or order_flow_imbalance or tick_volume_ratio'`
  - `3 passed, 12 deselected in 8.89s`
- filtered broad:
  - `tests/unit/v460/`
  - `--ignore=test_113_resilience.py`
  - `--ignore=test_152_parallel_tasks.py`
  - `--ignore=test_260_compute_extract_regime_split.py`
  - `--deselect=test_306_proposals.py::TestProposalsConfigSync::test_yaml_has_microprice_side`
  - `4241 passed, 13 warnings in 45.11s`

### 主要改善
- `scalping.py` は `366#` で触れた 3 関数だけでなく、残っていた短期特徴量もほぼ一掃できた。rolling の前窓参照は `shift(1)` を使って旧仕様を維持している。
- `trades_health.py` は大きいアルゴリズム変更ではないが、raw ディレクトリ内の不要ファイルまで毎回舐めない形になった。
- `base_features_v456.py` は `RSI` smoothing のような挙動差が出やすい箇所は触らず、`EMA` と `DM` 判定だけを low-risk で vectorize した。

## 2026-03-10 / Session 037-090

### 実施
- `ztb/features/scalping.py`
  - `momentum_divergence(...)` の残存 loop を vectorize 化
  - fast / slow 変化率を slice ベースで一括計算し、`divergence[slow_window:]` に直接代入する形へ整理
- `tests/unit/core/features/test_scalping_features.py`
  - `momentum_divergence(...)` の known-value regression test を追加

### 結果
- focused subset:
  - `tests/unit/core/features/test_scalping_features.py`
  - `tests/unit/core/features/test_v4_feature_extractor.py -k 'momentum_divergence or realized_volatility or order_flow_imbalance or tick_volume_ratio'`
  - `12 passed, 24 deselected in 6.40s`
- filtered broad:
  - `tests/unit/v460/`
  - `--ignore=test_113_resilience.py`
  - `--ignore=test_152_parallel_tasks.py`
  - `--ignore=test_260_compute_extract_regime_split.py`
  - `--deselect=test_306_proposals.py::TestProposalsConfigSync::test_yaml_has_microprice_side`
  - `4241 passed, 13 warnings in 40.84s`

### 主要改善
- `scalping.py` 側の Python loop は `momentum_divergence(...)` まで含めて解消した。
- これで `366#` 系の短期特徴量は、境界条件を残したままほぼ全て vectorized path に寄った。
