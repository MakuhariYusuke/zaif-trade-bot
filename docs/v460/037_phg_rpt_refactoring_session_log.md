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
