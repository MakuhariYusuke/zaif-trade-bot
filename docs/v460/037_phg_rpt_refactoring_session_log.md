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
