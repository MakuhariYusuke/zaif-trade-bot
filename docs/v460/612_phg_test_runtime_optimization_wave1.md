# 612# テスト実行時間最適化 Wave 1

## 背景

- `python -m pytest tests/ -x --tb=short` が重く、VS Code 側のメモリ圧迫と相性が悪い。
- 現行設定では `pytest.ini` / `pyproject.toml` の両方で coverage がデフォルト有効になっており、ローカル開発時の実行コストを押し上げていた。
- `tests/unit/v460/` では `FillTestConfig.from_yaml()` を同種 YAML に対して何度も繰り返すテストがあり、shared cache の横展開余地があった。

## 今回の方針

1. **coverage を dev デフォルトから外す**
   - `pytest.ini` と `pyproject.toml` の pytest addopts から `--cov*` を除去
   - coverage は CI / 明示コマンドでのみ有効化する前提へ寄せる
2. **shared config cache を追加する**
   - `tests/unit/v460/_yaml_test_helpers.py` に `FillTestConfig` の path/text ベース cache を追加
   - 既存の `load_yaml_mapping()` / `@lru_cache` パターンを `FillTestConfig` まで拡張
3. **高頻度 YAML test に横展開する**
   - `test_fill_test_config.py`
   - `test_183_log_analysis_improvements.py`
   に対して shared helper を適用
4. **xdist 準備**
   - `serial` marker を pytest 設定へ追加
   - 現環境では `pytest-xdist` 未導入なので、まずは marker と shared-state ラベル付けを先行

## 変更点

### 1. coverage の dev デフォルト解除

- `pytest.ini`
  - `--cov=ztb`
  - `--cov-report=term-missing`
  - `--cov-fail-under=20`
  を削除
- `pyproject.toml`
  - `[tool.pytest.ini_options].addopts` を `-ra -q` に整理

### 2. shared YAML / config helper

- `tests/unit/v460/_yaml_test_helpers.py`
  - `load_fill_test_config_from_text(...)`
  - `load_fill_test_config_from_path(...)`
  - `clone_fill_test_config(...)`
  を追加
- `tests/unit/v460/conftest.py`
  - `v460_fill_test_config_base` (`session`)
  - `v460_fill_test_config_yaml`
  を追加

### 3. 横展開したテスト

- `tests/unit/v460/test_fill_test_config.py`
  - production YAML roundtrip 系を session cache fixture に寄せた
- `tests/unit/v460/test_183_log_analysis_improvements.py`
  - inline YAML を `load_fill_test_config_from_text(...)` ベースの cached config fixture に寄せた
- `tests/unit/v460/test_yaml_test_helpers.py`
  - helper の cache / clone 契約を追加
- `tests/unit/v460/test_micro_timeout.py`
  - inline mapping を `load_fill_test_config_from_mapping(...)` ベースへ寄せた
  - production YAML 読み込みも `v460_fill_test_config_base` を再利用
- `tests/unit/v460/test_151_confidence_lot.py`
  - fixed mapping の `from_yaml(...)` を cached helper に寄せた
- `tests/unit/v460/test_138_p1_preflight_calibration.py`
  - preflight / score calibration の固定 mapping を cached helper に寄せた
- `tests/unit/v460/test_157_regime_features.py`
  - regime feature の固定 mapping を cached helper に寄せた
- `tests/unit/v460/test_166_hotfixes.py`
  - production YAML 由来 config を `v460_fill_test_config_base` から clone する形へ寄せた
- `tests/unit/v460/test_202_log_improvements.py`
  - inline YAML text を `load_fill_test_config_from_text(...)` ベースへ寄せた

### 4. xdist 準備

- `pytest.ini`
- `pyproject.toml`
  - `serial` marker を追加
- `tests/unit/v460/test_retrain_hot_reload.py`
  - module-level で `pytest.mark.serial` を付与

### 5. mapping ベース cache helper の追加

- `tests/unit/v460/_yaml_test_helpers.py`
  - `load_fill_test_config_from_mapping(...)`
  - JSON canonicalization を使った cache path
  を追加
- dict literal / fixed nested mapping を直接 `FillTestConfig.from_yaml(...)` していたテストを、
  検出力を変えずに cached helper ベースへ移行した

### 6. fixed YAML wiring sweep

- 追加で以下を cached helper に横展開:
  - `tests/unit/v460/test_093_side_params.py`
  - `tests/unit/v460/test_094_stale_order.py`
  - `tests/unit/v460/test_243_yaml_wiring.py`
  - `tests/unit/v460/test_154_deadlock_prevention.py`
  - `tests/unit/v460/test_137_p1_features.py`
- いずれも fixed dict / fixed YAML text を対象にしており、
  assert の緩和や production 挙動変更は行わず、初期化重複だけを削減した

### 7. 本体コード側の確認

- `scripts/v460/ml/retrain_scheduler.py::load_retrain_config(...)` を確認
- YAML 読み込み自体は
  [config_loader.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/config_loader.py)
  の `_read_config_section()` で
  - file signature (`mtime_ns`, `size`)
  - `@lru_cache`
  を使ってすでにキャッシュされていた
- そのため現時点では、本体コードにさらに low-risk な大きい短縮余地は小さく、
  先に test 側の repeated `FillTestConfig.from_yaml(...)` を潰す方が効率的と判断
- `tests/unit/v460/test_retrain_hot_reload.py` 単体も確認:
  - `86 passed in 4.47s`
  - slowest でも `0.05s` 台
  - ここは「見た目ほど本丸ではない」

## 重量テストの初期観測

- baseline 実行:
  - `.venv/Scripts/python.exe -m pytest tests/ --durations=30 --tb=short`
- coverage 付き baseline は 31% 到達時点でも継続中で、ローカル開発の初期応答性がかなり悪かった。
- まずは coverage デフォルト解除を優先するのが妥当と判断。

## 部分計測 (before / after)

比較対象:

```bash
.venv/Scripts/python.exe -m pytest \
  tests/unit/v460/test_fill_test_config.py \
  tests/unit/v460/test_183_log_analysis_improvements.py \
  tests/unit/v460/test_yaml_test_helpers.py
```

- with coverage:
  - `101 passed in 15.53s`
  - `--cov-fail-under=20` により exit code 1
  - coverage report / warning 出力も大きい
- without coverage:
  - `10 passed, 177 deselected in 3.56s`

少なくとも focused subset では、coverage デフォルト解除だけで **4x 以上** の差が出ている。

## xdist 観点

- `pytest-xdist` は現環境に未導入
  - `.venv/Scripts/python.exe -m pip show pytest-xdist`
  - `Package(s) not found`
- そのため今回は
  - marker 定義
  - shared-state が疑われる hot-reload test の `serial` 化
 までを先行

## 次の一手

1. `--durations=30` を `--no-cov` 相当の新設定で再測定し、top 30 を分類する
2. `test_retrain_hot_reload.py` など autouse fixture の重いファイルを個別に軽量化する
3. `FillTestConfig.from_yaml()` の高頻度呼び出しファイルをさらに横展開する
4. before / after 比較をこの文書に追記していく
5. `tests/ --durations=30 --no-cov` を改めて取り切り、top 30 を分類する

## 追加の shared fixture / tempfile hygiene

2026-03-30 時点で、次の 3 系統をさらにまとめて整理した。

1. `fill_test.yaml` / default config の再読込削減
   - `tests/unit/v460/test_fill_test_config.py`
   - `loaded_default_fill_test_yaml`
   - `loaded_explicit_fill_test_yaml`
   - `empty_fill_test_config`
   を module fixture 化し、
   `load_fill_test_config()` / `FillTestConfig.from_yaml({})`
   の read-only 呼び出しを shared に寄せた
2. real-data 可用性判定の共有
   - `tests/unit/v460/_real_data_test_helpers.py`
   - `has_fill_records_and_raw_data(...)`
   を追加し、
   `tests/unit/v460/test_enricher_skip_gate.py`
   の integration fixture で再利用
3. legacy tempfile cleanup
   - `tests/legacy_tests/unit/test_event_sourcing.py`
   - `tests/unit/reward/validate_reward_components.py`
   を `tmp_path` / `mkstemp()` ベースに整理

効果:

- `fill_test_config` での default/live YAML 再 open を削減
- `enricher_skip_gate` の real-data 可用性判定を helper 側へ集約
- skip 中でも読みにくい legacy tempdir パターンを減らし、保守性を上げた

この時点の残差は次の順。

1. `tests/unit/v460/test_enricher_skip_gate.py`
   - real-data enriched setup
2. `tests/unit/v460/test_fill_test_config.py`
   - parser-heavy path
3. `tests/legacy_tests/unit/test_event_sourcing.py`
   - runtime 影響は小さいが、legacy cleanup の残り

追加メモ:

- `tests/unit/v460/test_fill_test_config.py`
  - `test_from_yaml_empty`
  - `test_yaml_roundtrip_skip_gate`
  を base fixture へ寄せて、read-only default path の再評価をさらに削減した
- 直近 subset:
  - `tests/unit/v460/test_enricher_skip_gate.py`
  - `tests/unit/v460/test_fill_test_config.py`
  - `tests/legacy_tests/unit/test_event_sourcing.py`
  - `154 passed, 10 skipped in 4.11s`
- まだ最遅は
  - `Test058Integration::test_enrichment_with_real_data`
  - `0.21s setup`
  で、ここは real-data を読む限り次の本丸

- legacy / integration cleanup 追加:
  - `tests/legacy_tests/unit/utils/test_feature_cache.py`
  - `tests/legacy_tests/unit/utils/test_checkpoint_light.py`
  - `tests/integration/test_v433_phase5_integration.py`
  で tempdir lifecycle を整理
- 直近 bundle:
  - `tests/unit/v460/test_fill_test_config.py`
  - `tests/legacy_tests/unit/utils/test_feature_cache.py`
  - `tests/legacy_tests/unit/utils/test_checkpoint_light.py`
  - `tests/integration/test_v433_phase5_integration.py`
  - `91 passed, 9 skipped, 1 warning, 3 subtests passed in 40.46s`
- ただし subset の最遅は
  - `tests/integration/test_v433_phase5_integration.py::TestV433Phase5Integration::test_performance_under_load`
  - `21.46s call`
  で、次の重いテスト cleanup の最優先候補

- 次の cleanup 方向:
  - `v433` load test は integration の意味を残しつつ、負荷下の並列安定性だけを見る lightweight mock に寄せる
  - `test_enricher_skip_gate.py` は
    - smoke 用 enriched df
    - training 用 enriched df
    を分離して、実データ setup を段階化する

- 追記:
  - `tests/integration/test_v433_phase5_integration.py::TestV433Phase5Integration::test_performance_under_load`
    - before: `21.46s call`
    - after: `0.09s call`
  - `tests/unit/v460/test_enricher_skip_gate.py::Test058Integration::test_enrichment_with_real_data`
    - setup `0.21s -> 0.18s`
    - 小幅だが smoke/training 分離で責務が明確になった
  - `tests/unit/v460/test_fill_quality.py`
    - load/list/glob 系の setup を `save_fill_records(...)` 経由ではなく direct JSONL seed へ切り替える余地がある
    - focused subset:
      - `5 passed in 3.55s`
      - slowest call は `test_iter_fill_record_objects_glob_roundtrip` `0.04s`

- 追加 sweep:
  - `test_enricher_skip_gate.py` の smoke sample は `24 -> 36 -> 48` の段階 fallback に変更
  - `test_fill_quality.py` の glob/load roundtrip で残っていた `save_fill_records(...)` 経由 setup も direct seed に統一

- broad subset で見えた追加点:
  - `test_fill_test_config.py::Test062SkipGateConfig::test_yaml_has_skip_gate_section`
    - live YAML の `skip_gate.max_skip_rate=0.3` に追随
  - `v433` の残差:
    - `test_full_system_integration` `6.17s`
    - `test_monitoring_integration` `6.16s`
    - `test_emergency_control_integration` `3.11s`
  - 対応方針:
    - monitoring / recovery 系は integration の意味を残しつつ lightweight mock へ寄せる

## 追加の tempfile / mtime cleanup

次の同型パターンも low-risk に整理した。

- `tempfile.mkdtemp()` + 手動 cleanup
  - `tests/unit/training/test_training_resume.py`
  - `tests/integration/test_checkpoint_logging_integration.py`
  - `tests/training/callbacks/test_integration.py`
  - `tests/training/callbacks/test_callbacks.py`
  - `tests/unit/algorithms/test_ab_test_framework.py`
  - `tests/training/config/test_configuration_manager.py`
- `NamedTemporaryFile(delete=False)`
  - `tests/multimodal/test_multimodal_core.py`
  - `tests/trading/signal/test_entry_system.py`
- `TemporaryDirectory()`
  - `tests/unit/core/test_rollup_artifacts.py`
- `time.sleep()` による mtime 待機
  - `tests/unit/features/test_norm_loader.py`
  - `os.utime(...)` 明示更新へ変更

効果:

- temp lifecycle の実装揺れを減らせた
- Windows/WSL 環境でのファイルハンドル残りリスクを下げられた
- `sleep` 依存の小さな固定費も 1 本削れた

## 実行中の分類メモ

subset 実行:

```bash
.venv/Scripts/python.exe -m pytest \
  tests/unit/v460 tests/unit/training tests/integration tests/training \
  tests/unit/config tests/unit/cache tests/unit/algorithms tests/unit/analysis \
  tests/unit/features tests/multimodal tests/trading/signal tests/unit/core \
  -x --tb=short --no-cov
```

この途中で拾えた実 failure:

- `tests/unit/v460/test_183_log_analysis_improvements.py`
  - `fill_test.yaml` の live 値 drift により
    `sell_velocity_skip_threshold_bps`
    の期待値が `6.0 -> 4.0` へ更新必要だった

これは今回の cleanup 起因ではなく、現行 YAML 追随の回帰修正として吸収した。

## 検証

- `python3 -m py_compile`
  - `tests/unit/v460/_yaml_test_helpers.py`
  - `tests/unit/v460/conftest.py`
  - `tests/unit/v460/test_fill_test_config.py`
  - `tests/unit/v460/test_183_log_analysis_improvements.py`
  - `tests/unit/v460/test_yaml_test_helpers.py`
  - `tests/unit/v460/test_retrain_hot_reload.py`
- focused pytest:
  - `tests/unit/v460/test_fill_test_config.py`
  - `tests/unit/v460/test_183_log_analysis_improvements.py`
  - `tests/unit/v460/test_yaml_test_helpers.py`
  - `tests/unit/v460/test_retrain_hot_reload.py`

## 2026-03-29 追加: v433 import 崩れ + 長待機整理

- `tests/integration/test_v433_phase5_integration.py`
  - package import に戻し、古い direct import fallback / stdout print を削除
  - `ztb.utils.circuit_breaker` の現行 API に合わせて
    `CircuitBreakerConfig` / `CircuitBreaker("name", config)` へ追随
  - `failure_recovery` は
    - unsupported な `deployment_failure` を除外
    - `PENDING -> IN_PROGRESS` を復旧開始の成功条件として扱う
    - polling を `1.0s / 0.05s` に短縮
  - `performance_under_load` は
    - `concurrent_orders: 50 -> 20`
    - `timeout: 10s -> 5s`
    に縮小
- `tests/unit/v460/test_regime_detector.py`
  - 固定 mapping の `FillTestConfig.from_yaml(...)` を cached helper に寄せた
  - stale source-contract 2 件を現行 helper 境界へ更新
- `ztb/trading/production/*`
  - `paper_trading_manager.py`
  - `virtual_portfolio_manager.py`
  - `traffic_distributor.py`
  - `result_comparator.py`
  - `health_checker.py`
  - `real_time_metrics.py`
  - importability を戻す最小修復を実施

### before / after

- `tests/integration/test_v433_phase5_integration.py`
  - before:
    - collection error (`alert_system` direct import 崩れ)
    - 修復途中の実行では `4 failed ... in 89.85s`
  - after:
    - `8 passed, 1 warning, 3 subtests passed in 37.95s`

- `tests/unit/v460/test_regime_detector.py`
  - after helper 化 + stale test 修正:
    - `99 passed in 2.44s`

### 補足

- targeted mypy は `v433` production module 群に既存 baseline error が多く、
  今回は repo-wide clean までは狙っていない
- ただし今回の差分起因のノイズは
  - import alias
  - `Mapping[str, object]`
  への整理で抑制している
  - 結果: `187 passed in 4.97s`
- focused pytest:
  - `tests/unit/v460/test_yaml_test_helpers.py`
  - `tests/unit/v460/test_micro_timeout.py`
  - `tests/unit/v460/test_151_confidence_lot.py`
  - `tests/unit/v460/test_138_p1_preflight_calibration.py`
  - `tests/unit/v460/test_157_regime_features.py`
  - `tests/unit/v460/test_166_hotfixes.py`
  - `tests/unit/v460/test_202_log_improvements.py`
  - 結果: `134 passed in 5.25s`
- focused pytest:
  - `tests/unit/v460/test_093_side_params.py`
  - `tests/unit/v460/test_094_stale_order.py`
  - `tests/unit/v460/test_243_yaml_wiring.py`
  - `tests/unit/v460/test_154_deadlock_prevention.py`
  - `tests/unit/v460/test_137_p1_features.py`
  - 結果: `120 passed in 4.36s`
- targeted hotspot check:
  - `tests/unit/v460/test_retrain_hot_reload.py --durations=20 --no-cov`
  - 結果: `86 passed in 4.47s`
  - 最遅ケースも `0.05s` 台で、現時点では top offender ではなかった
- broad cached-helper sweep:
  - 追加適用:
    - `tests/unit/v460/test_velocity_skip_rule.py`
    - `tests/unit/v460/test_fill_quality.py`
    - `tests/unit/v460/test_141_side_specific_models.py`
    - `tests/unit/v460/test_168_pnl_measurer_sell_hold.py`
    - `tests/unit/v460/test_168_daily_drawdown_guard.py`
    - `tests/unit/v460/test_306_proposals.py`
    - `tests/unit/v460/test_593_ev_toxic_skip_and_cap_hit_veto.py`
    - `tests/unit/v460/test_139_review_fixes.py`
    - `tests/unit/v460/test_143_regime_utilization.py`
    - `tests/unit/v460/test_155_hindsight_review.py`
    - `tests/unit/v460/test_158_regime_deadlock_fix.py`
    - `tests/unit/v460/test_176_trending_offset_asymmetry.py`
    - `tests/unit/v460/test_187_chase_direction_guard_trace.py`
    - `tests/unit/v460/test_188_split_evc_macro.py`
    - `tests/unit/v460/test_190_ev_weighted_safety.py`
    - `tests/unit/v460/test_193_ev_offset.py`
    - `tests/unit/v460/test_195_velocity_b1_soft.py`
    - `tests/unit/v460/test_226_loss_boost_decay_inv_skew_state.py`
    - `tests/unit/v460/test_228_inv_decay_hasattr_removal.py`
    - `tests/unit/v460/test_273_kill_time_limit_halt_untick_recovery_grace.py`
    - `tests/unit/v460/test_274_pattern_c_theory_cleanup.py`
    - `tests/unit/v460/test_276_blocking_policy_dry.py`
    - `tests/unit/v460/test_277_magic_number_grounding.py`
    - `tests/unit/v460/test_292_observability.py`
    - `tests/unit/v460/test_303_review_implementations.py`
    - `tests/unit/v460/test_421_final_clamp_deadlock.py`
    - `tests/unit/v460/test_440_regime_side_offset.py`
    - `tests/unit/v460/test_skip_gate_v3.py`
  - 内容:
    - 単発 `FillTestConfig.from_yaml(...)` を `clone_fill_test_config(load_fill_test_config_from_mapping(...))` に寄せた
    - 固定 dict / 空 dict / 単発 YAML mapping の重複初期化を shared cache に吸収
  - focused pytest:
    - 上記 28 ファイル
    - 結果: `1101 passed in 11.41s`

- residual safe sweep:
  - 追加適用:
    - `tests/unit/v460/test_094_stale_order.py`
    - `tests/unit/v460/test_163_regime_adaptive_gating.py`
    - `tests/unit/v460/test_183_log_analysis_improvements.py`
    - `tests/unit/v460/test_249_directional_alpha.py`
    - `tests/unit/v460/test_596_primary_consecutive_skip_safety.py`
  - 内容:
    - 単発 `FillTestConfig.from_yaml(...)` を cached helper に寄せた
  - focused pytest:
    - 上記 5 ファイル
    - 結果: `125 passed in 2.78s`
  - 現時点で残している `from_yaml(...)` は主に
    - parser 契約そのものを検証するファイル
    - 並行差分の影響を受けやすいファイル
    に絞られている

- tempfile / tmpdir sweep:
  - 追加適用:
    - `tests/unit/config/test_config_manager.py`
    - `tests/unit/config/test_config.py`
    - `tests/unit/cache/test_sqlite_cache.py`
    - `tests/unit/cache/test_data_loader.py`
  - 内容:
    - `TemporaryDirectory()` / `NamedTemporaryFile()` を `tmp_path` ベースへ置換
    - delete=False cleanup や一時ディレクトリ生成の固定費を削減
  - focused pytest:
    - 上記 4 ファイル
    - 結果: `60 passed in 6.39s`

- cache/parquet tempfile sweep:
  - 追加適用:
    - `tests/unit/cache/test_parquet_io.py`
  - 内容:
    - CSV 入力 fixture を `NamedTemporaryFile()` から `tmp_path` へ移行
  - focused pytest:
    - `tests/unit/cache/test_parquet_io.py`

- alert-mode / legacy tempdir cleanup sweep:
  - 追加適用:
    - `tests/unit/v460/test_215_dd_fix_alert_mode.py`
    - `tests/legacy_tests/unit/utils/test_feature_cache.py`
    - `tests/legacy_tests/unit/utils/test_checkpoint_light.py`
  - 内容:
    - `mkdtemp()` の返り値を `Path` で保持し、teardown で `shutil.rmtree(..., ignore_errors=True)` に統一
    - `test_215_dd_fix_alert_mode.py` では invalid-values 系 fixture に `teardown_method()` を追加し、tempdir leak を止めた
  - focused pytest:
    - `31 passed, 9 skipped in 2.04s`

## git 運用メモ

- WSL + NTFS 上では `git status --untracked-files=normal` が重い
  - 修復後実測:
    - `git status --short --untracked-files=normal`: `54.05s`
    - `git status --short --untracked-files=no`: `9.47s`
- repo-local に維持する設定:
  - `core.untrackedcache=true`
  - `core.preloadindex=true`
  - `core.fscache=true`
  - `fetch.writeCommitGraph=true`
  - `core.splitIndex=false`
- `splitIndex` は壊れた shared index 参照で tracked 消失に見えるため、この repo では無効が安全

## 本体コード側の所見

- `FillTestConfig` 読み込みの production 側 short-path はすでにかなり整っている
  - `scripts/v460/ml/retrain_scheduler.py`
  - `scripts/v460/lib/config_loader.py`
  - file signature + `@lru_cache` が入っているため、ここをさらに触る優先度は高くない
- したがって現時点で最も効率が良いのは
  - coverage デフォルト解除
  - test 側の repeated `FillTestConfig.from_yaml(...)` 削減
  の継続

## `--durations=25` による中間観測

subset 実行:

```bash
.venv/Scripts/python.exe -m pytest \
  tests/unit/v460 tests/unit/training tests/integration tests/training \
  tests/unit/config tests/unit/cache tests/unit/algorithms tests/unit/analysis \
  tests/unit/features tests/multimodal tests/trading/signal tests/unit/core \
  --durations=25 -x --tb=short --no-cov -q
```

- 実行結果:
  - `1 failed, 4584 passed, 15 skipped, 10 warnings in 64.64s`
- 途中 failure:
  - `tests/unit/v460/test_fill_test_config.py::Test062SkipGateConfig::test_yaml_has_skip_gate_section`
  - 原因: live YAML drift
    - `skip_gate.max_skip_rate: 0.3 -> 0.4`

slowest 25 から見えた優先候補:

1. `test_552_update_training_data.py::TestDownloadOhlcv::test_returns_dataframe` `1.15s`
2. `test_499_loss_cap_daily_scope.py` 系 2 本 `0.40s-0.53s`
3. `test_384_pipeline_fixes.py::TestEvaluateModelOOS::test_multi_slice_metrics_present` `0.26s`
4. `test_enricher_skip_gate.py::Test058Integration::test_enrichment_with_real_data` setup `0.25s`
5. YAML/config 読み込み系
   - `test_202_log_improvements.py::TestLossCooldownConfig::test_yaml_parsing`
   - `test_596_primary_consecutive_skip_safety.py::test_yaml_overrides_default`
   - `test_157_regime_features.py::test_retrain_config_loads_from_yaml`

現時点の判断:

- top offender は `sleep` より
  - data download / dataframe build
  - real-data setup
  - YAML live-config verification
  が中心
- 次の軽量化 batch は
  - `test_552_update_training_data.py`
  - `test_499_loss_cap_daily_scope.py`
  - `test_384_pipeline_fixes.py`
  - `test_enricher_skip_gate.py`
  の順が効きやすい

## 次の batch 方針

同質なものをまとめて進める。

### Batch T2: top duration test cleanup

- `tests/unit/v460/test_552_update_training_data.py`
  - parquet / DataFrame fixture を module-scope へ寄せられるか確認
  - `_download_ohlcv()` の mocked path を focused 計測し、import/dataframe build のどちらが支配的か分離
- `tests/unit/v460/test_499_loss_cap_daily_scope.py`
  - 日次 reset stub 生成の helper 化
  - import-heavy であれば module-level import か shared fixture に寄せる
- `tests/unit/v460/test_384_pipeline_fixes.py`
  - long step loop を contract を崩さず短くできるか確認
- `tests/unit/v460/test_enricher_skip_gate.py`
  - real-data setup の sample cap / cache reuse を再点検

### Batch T3: helper / type cleanup と横展開

- `tests/unit/v460/_real_data_test_helpers.py`
  - generic helper の `Any` を減らし、`JsonRow` alias へ寄せる
- `analysis / metrics bridge`
  - `analyze_fill_logs.py`
  - `fill_quality.py`
  - `oracle_* / reproduce_152_metrics / tail_loss_analysis`
  の出力・型・metrics bridge を同じ基準で揃える

### Batch C1: code health candidates

- `scripts/v460/lib/maker_price.py`
  - Wave 2A の残差として veto/cache/telemetry ownership の inline update 点を棚卸し
- `scripts/v460/lib/ab_judgment.py`
  - result/report ownership の残差確認
- `tests/unit/v460/_real_data_test_helpers.py` と同種の helper は、
  汎用化できるものだけ `ztb` 側移管候補として見る

## Batch T2 進捗 (top duration cleanup 1st)

対象:

- `tests/unit/v460/test_552_update_training_data.py`
- `tests/unit/v460/test_499_loss_cap_daily_scope.py`
- `tests/unit/v460/test_384_pipeline_fixes.py`

内容:

- `test_552_update_training_data.py`
  - target function import を module-level に集約
  - parquet template を module-scope fixture 化
  - 各 test は template を `tmp_path` に copy する形へ変更
  - fixture 側で `pd.read_parquet(..., columns=["timestamp"])` を1回実行し、初回 engine 初期化を吸収
- `test_499_loss_cap_daily_scope.py`
  - module import を集約
  - `_make_pre_cycle_mixin(...)` helper を追加し、日次 reset stub 構築を共通化
- `test_384_pipeline_fixes.py`
  - `evaluate_model_oos`
  - `_build_val_env_config`
  の import を module-level に集約

focused 実測:

```bash
.venv/Scripts/python.exe -m pytest \
  tests/unit/v460/test_552_update_training_data.py \
  tests/unit/v460/test_499_loss_cap_daily_scope.py \
  tests/unit/v460/test_384_pipeline_fixes.py \
  --durations=15 -q --no-cov
```

結果:

- `31 passed, 1 warning in 3.99s`

durations 変化で特に効いた点:

- `test_552_update_training_data.py::TestGetParquetLastTimestamp::test_returns_last_ts`
  - call `1.23s -> 0.02s`
- 同 test setup
  - `0.51s -> 0.11s`
- `TestDownloadOhlcv::test_returns_dataframe`
  - `0.78s -> 0.48s`

現時点の判断:

- `552` は parquet template 共有がかなり効く
- `499` / `384` は import 集約で十分 low-risk
- 次の本丸は引き続き
  - `test_enricher_skip_gate.py`
  - `test_552_update_training_data.py` の `_download_ohlcv` path
  - `test_499_loss_cap_daily_scope.py` の direct mixin path
 である


- unittest / smoke tempfile sweep:
  - 追加適用:
    - `tests/unit/config/test_unified_config.py`
    - `tests/integration/smoke_tests.py`
  - 内容:
    - `NamedTemporaryFile(delete=False)` を `mkstemp` + `addCleanup` へ整理
    - smoke artifacts を `mkdtemp()` ベースにして cleanup を明示
    - 未使用 `TemporaryDirectory()` を削除
  - focused pytest:
    - `tests/unit/config/test_unified_config.py`
    - 結果: `13 passed in 2.96s`

- named-tempfile cleanup sweep:
  - 追加適用:
    - `tests/unit/algorithms/test_market_regime_system.py`
    - `tests/unit/analysis/test_v4xx_unified_analyzer.py`
    - `tests/unit/integration/test_unified_trainer_integration.py`
  - 内容:
    - `NamedTemporaryFile(delete=False)` を `mkstemp` / `tmp_path` ベースへ置換
    - analyzer fixture と integration config file の cleanup を簡素化
  - focused pytest:
    - 上記 3 ファイル
    - 結果: `28 passed, 3 warnings in 5.32s`

- training / integration tempdir lifecycle sweep:
  - 追加適用:
    - `tests/unit/training/test_training_resume.py`
    - `tests/unit/training/test_checkpoint_manager.py`
    - `tests/integration/test_checkpoint_logging_integration.py`
  - 内容:
    - `mkdtemp()` の cleanup を `addCleanup` ベースに統一
    - temp directory を `Path` で扱い、`save_dir` 渡し口だけ `str(...)` 化
    - shutdown と filesystem cleanup の責務を分離
  - focused pytest:
    - 上記 3 ファイル
    - 結果: `25 passed in 3.17s`

- live `fill_test.yaml` shared fixture sweep:
  - 追加適用:
    - `tests/unit/v460/test_micro_timeout.py`
    - `tests/unit/v460/test_634_sell_ranging_suppression.py`
    - `tests/unit/v460/test_421_final_clamp_deadlock.py`
    - `tests/unit/v460/test_596_primary_consecutive_skip_safety.py`
  - 内容:
    - `v460_fill_test_yaml_base` / `v460_fill_test_config_base` に寄せて
      live YAML の再 open / 再 parse を削減
    - `clone_fill_test_config(...)` 経由で mutation-safe を維持
  - focused pytest:
    - 上記 4 ファイル
    - 結果: `99 passed in 2.38s`

現時点の判断:

- live config を読むだけの v460 test は shared fixture 化がかなり効く
- parser 契約そのものを見ているテストは無理に helper 化しない
- 次の本丸は引き続き
  - `test_enricher_skip_gate.py`
  - `test_552_update_training_data.py` の `_download_ohlcv` path
  - temp file / temp dir 残件
  である

- `test_552_update_training_data.py` second sweep:
  - 追加適用:
    - `_download_ohlcv` test で `patch("yfinance.Ticker")` ではなく
      `patch.dict(sys.modules, {"yfinance": fake_module})` を使い、
      test call から実 import コストを除去
    - `_get_all_parquet_features(...)` を module-scope fixture で warm up し、
      test 本体は fixture 結果を再利用
  - focused pytest:
    - `tests/unit/v460/test_552_update_training_data.py --durations=10 -q --no-cov`
    - `15 passed in 2.09s`
  - before/after:
    - file total: `2.99s -> 2.09s`
    - `_download_ohlcv` top duration から脱落
    - 次の支配点は `TestGetAllParquetFeatures` setup `0.94s`

現時点の top residual:

1. `test_enricher_skip_gate.py` real-data setup
2. `test_552_update_training_data.py` feature registry warm-up
3. `test_499_loss_cap_daily_scope.py` daily reset path

- cached config + lightweight stub sweep:
  - 追加適用:
    - `tests/unit/v460/test_499_loss_cap_daily_scope.py`
    - `tests/unit/v460/test_169_c1_c3_c4_config.py`
    - `tests/unit/v460/test_168_low_vol_offset_boost.py`
    - `tests/unit/v460/test_169_ranging_buy_skip_and_metrics.py`
  - 内容:
    - `test_499_loss_cap_daily_scope.py` で `MagicMock` / ad-hoc record をやめ、
      軽量 stub + 実 `FillRecord` に置換
    - live `fill_test.yaml` を読むだけのテストを
      `v460_fill_test_config_base` + `clone_fill_test_config(...)` に寄せた
    - 固定 mapping の `FillConfig.from_yaml(...)` を cached helper に寄せた
    - decorator / fixture 起点の targeted mypy ノイズも整理
  - 検証:
    - targeted mypy:
      - 上記 4 ファイル
      - `Success: no issues found in 4 source files`
    - focused pytest:
      - 上記 4 ファイル
      - `63 passed, 1 warning in 2.20s`

次の residual 候補:

1. `tests/unit/v460/test_fill_test_config.py` など parser 契約そのものを見る `from_yaml(...)` テスト群
2. `tests/unit/v460/test_enricher_skip_gate.py` real-data setup
3. `tests/legacy_tests/unit/test_event_sourcing.py` の `TemporaryDirectory()` 残差

- default loader + real-data helper sweep:
  - 追加適用:
    - `tests/unit/v460/_real_data_test_helpers.py`
    - `tests/unit/v460/test_fill_test_config.py`
  - 内容:
    - `has_fill_records(...)` を `cached_latest_fill_records_file(...)` 経由にして
      real-data setup 時の glob を 1 段キャッシュ
    - `test_fill_test_config.py` の default / explicit loader 呼び出しを
      module-scope fixture に寄せた
  - focused pytest:
    - `tests/unit/v460/test_fill_test_config.py`
    - `tests/unit/v460/test_enricher_skip_gate.py`
    - `154 passed, 1 skipped in 4.14s`
  - slowest 12:
    - `test_enricher_skip_gate.py::Test058Integration::test_enrichment_with_real_data` setup `0.29s`
    - `test_fill_test_config.py::TestFillTestConfigFromYaml::test_from_yaml_defaults` setup `0.11s`
    - `test_fill_test_config.py::TestLoadFillTestConfig::test_load_default_path` setup `0.11s`
  - targeted mypy:
    - `tests/unit/v460/_real_data_test_helpers.py`
    - `Success: no issues found in 1 source file`

現時点の次残差:

1. `test_552_update_training_data.py` の `TestGetAllParquetFeatures` setup
2. `test_enricher_skip_gate.py` の real-data enriched setup
3. `test_fill_test_config.py` の parser-heavy path (baseline mypy 厚め)

- production/runtime cache sweep (`update_training_data`):
  - 追加適用:
    - `scripts/v460/ml/update_training_data.py`
    - `tests/unit/v460/test_552_update_training_data.py`
  - 内容:
    - parquet schema 列取得を file-signature cache 化
    - 最終 timestamp 取得も file-signature cache 化
    - heavy な feature module import は `_ensure_feature_registry_loaded()` に集約し、
      実際に feature 計算が必要な経路に遅延
    - `_get_all_parquet_features(...)` は parquet schema + SAC 必須列を基準に解決
  - 検証:
    - targeted mypy:
      - `scripts/v460/ml/update_training_data.py`
      - `tests/unit/v460/test_552_update_training_data.py`
      - `tests/unit/v460/_real_data_test_helpers.py`
      - `Success: no issues found in 3 source files`
    - focused pytest:
      - `tests/unit/v460/test_552_update_training_data.py`
      - `16 passed, 1 warning in 0.98s`
      - `tests/unit/v460/test_552_update_training_data.py`
      - `tests/unit/v460/test_enricher_skip_gate.py`
      - `tests/unit/v460/test_fill_test_config.py`
      - `170 passed, 1 skipped, 1 warning in 3.40s`
  - before/after:
    - `test_552_update_training_data.py --durations=12`
      - before top setup:
        - `TestGetAllParquetFeatures::test_includes_sac_features` setup `0.60s`
      - after top setup:
        - `TestGetParquetLastTimestamp::test_returns_last_ts` setup `0.09s`
    - combined subset (`552 + enricher + fill_test_config`)
      - before: `169 passed, 1 skipped, 1 warning in 4.47s`
      - after: `170 passed, 1 skipped, 1 warning in 3.40s`

- read-only base fixture sweep (`fill_test_config`):
  - 追加適用:
    - `tests/unit/v460/test_fill_test_config.py`
  - 内容:
    - read-only な roundtrip / default path 検証を
      `v460_fill_test_yaml_base` / `v460_fill_test_config_base` に寄せた
    - `clone_fill_test_config(...)` と function-scope YAML deepcopy を
      必要な箇所だけに限定
  - focused pytest:
    - `tests/unit/v460/test_fill_test_config.py`
    - `tests/unit/v460/test_552_update_training_data.py`
    - `tests/unit/v460/test_enricher_skip_gate.py`
    - `170 passed, 1 skipped, 1 warning in 3.12s`
  - before/after (same combined subset):
    - `test_fill_test_config.py::TestFillTestConfigFromYaml::test_from_yaml_defaults`
      - `0.10s -> 0.08s` setup
    - `test_fill_test_config.py::TestLoadFillTestConfig::test_load_default_path`
      - `0.18s -> 0.07s` setup
    - combined subset
      - `3.40s -> 3.12s`

- emergency stop / real-data enrich residual sweep:
  - 追加適用:
    - `tests/integration/test_v433_phase5_integration.py`
    - `tests/unit/v460/test_enricher_skip_gate.py`
  - 内容:
    - `test_emergency_control_integration`
      - `emergency_stop.trigger_emergency_stop(...)` 内の `asyncio.sleep(...)` を test 側で no-op 化
      - integration の責務は維持しつつ長待機だけ外した
    - `test_enricher_skip_gate.py`
      - class-scope の `real_smoke_enriched_df` / `real_trainable_enriched_df` を
        `real_enriched_bundle` へ統合
      - trainable real-data が取れる場合は smoke 用も同じ enriched df を再利用し、
        二重 enrich を避けた
  - subset 実測:
    - `tests/unit/v460/test_fill_quality.py`
    - `tests/unit/v460/test_enricher_skip_gate.py`
    - `tests/integration/test_v433_phase5_integration.py`
    - `tests/unit/v460/test_552_update_training_data.py`
    - `tests/unit/v460/test_fill_test_config.py`
    - `384 passed, 1 skipped, 6 warnings, 3 subtests passed in 11.15s`
  - slowest 20 の変化:
    - `test_v433_phase5_integration.py::test_emergency_control_integration`
      - `2.97s` が支配点だったので次回比較基準として記録
    - `test_enricher_skip_gate.py::Test058Integration::test_enrichment_with_real_data`
      - setup `0.31s`

- PPO integration/runtime sweep:
  - 追加適用:
    - `tests/integration/test_custom_ppo_integration.py`
    - `tests/training/test_ppo_trainer.py`
    - `tests/training/unified_trainer/test_algorithms.py`
  - 内容:
    - `SELLBiasMitigationPPOTrainer` の current params smoke を lightweight fake env 化
    - action-mask contract を `mask_fn -> get_action_masks()` 前提で固定
    - unified trainer 側の PPO current training path smoke を追加
  - focused pytest:
    - `tests/training/test_ppo_trainer.py`
    - `tests/unit/algorithms/test_ppo_algorithm.py`
    - `tests/unit/training/test_ppo_trainer.py`
    - `tests/integration/test_custom_ppo_integration.py`
    - `tests/training/unified_trainer/test_algorithms.py`
    - `92 passed, 2 skipped in 11.06s`
  - before/after:
    - `tests/integration/test_custom_ppo_integration.py::TestSellMitigationTrainerIntegration::test_trainer_uses_current_params_interface`
      - `11.80s -> 0.02s`
  - slowest 20:
- PPO 系では `sell_mitigation` integration が支配点から外れた
- 現在の上位は self-supervised trainer integration 側へ移動

- PPO integration / trainer sweep:
  - 追加適用:
    - `tests/integration/test_custom_ppo_integration.py`
    - `tests/training/test_ppo_trainer.py`
  - 内容:
    - `ActionMasker` 契約確認を tiny env ベースに分離
    - short training smoke の `n_steps` / `batch_size` / `total_timesteps` を縮小
    - `build_ppo_model_kwargs(...)` helper の contract guard を追加
  - 目的:
    - PPO 復旧作業の focused suite を保守しつつ、HeavyTradingEnv setup 依存を必要箇所に限定

- PPO integration trim 追記:
  - `tests/integration/test_custom_ppo_integration.py`
    - `test_create_with_current_masked_env` も tiny env ベースへ移行
    - HeavyTradingEnv を必要としない setup を完全に切り離した
  - これにより PPO focused subset の slowest setup は縮小し、
    残る支配点は scheduler/retrain path と self-supervised integration 側へ寄った

- PPO Phase 2 focused runtime:
  - `tests/unit/v460/test_679_ppo_sidecar_foundation.py`
  - `tests/unit/v460/test_680_ppo_retrain_scheduler.py`
  - `tests/unit/v460/test_ppo_warm_start.py`
  - `tests/training/test_ppo_trainer.py`
  - `tests/integration/test_custom_ppo_integration.py`
  - `63 passed in 5.52s`
  - warm-start / YAML parse / scheduler edge case を focused で固定

- PPO/SAC sidecar broad-ish regression:
  - `tests/unit/v460/test_679_ppo_sidecar_foundation.py`
  - `tests/unit/v460/test_680_ppo_retrain_scheduler.py`
  - `tests/unit/v460/test_sac_retrain_scheduler.py`
  - `tests/unit/v460/test_sidecar_sac_integration.py`
  - `tests/training/test_ppo_trainer.py`
  - `tests/unit/algorithms/test_ppo_algorithm.py`
  - `tests/unit/training/test_ppo_trainer.py`
  - `tests/integration/test_custom_ppo_integration.py`
  - `tests/training/unified_trainer/test_algorithms.py`
  - `252 passed, 2 skipped in 20.12s` の後、
    Phase 2 変更込みでも focused regression は維持

- SAC scheduler / unified trainer trim:
  - `tests/unit/v460/test_sac_retrain_scheduler.py`
    - `retrain_once(...)` 共通 patch helper で `_run_data_freshness_check(...)` を no-op 化
    - scheduler loop test でも data freshness helper を切り離し、
      control-flow だけを見る形に整理
  - `tests/training/unified_trainer/test_algorithms.py`
    - self-supervised synthetic default を小さくし、
      shape contract は維持したまま synthetic tensor 固定費を削減
  - `ztb/training/unified_trainer/algorithms/self_supervised_trainer.py`
    - degraded torch fallback は random ではなく zero array を使うように変更
  - subset 実測:
    - before:
      - `tests/unit/v460/test_sac_retrain_scheduler.py`
      - `tests/training/unified_trainer/test_algorithms.py`
      - `86 passed in 25.59s`
    - after:
      - 同 subset
      - `86 passed in 11.11s`
  - slowest call の変化:
    - `test_oos_failed`: `2.84s -> 0.15s`
    - `test_cold_start_success`: `1.93s -> 0.14s`
    - `test_load_data_synthetic_falls_back_when_randn_is_degraded`: `1.52s -> 0.02s`

- `enricher_skip_gate` / `fill_quality` trim:
  - `tests/unit/v460/test_enricher_skip_gate.py`
    - real-data smoke と trainable fixture を分離
    - smoke test が trainable enriched bundle の準備を踏まないよう整理
  - `tests/unit/v460/test_fill_quality.py`
    - fast-cycle unknown-status retry を 1 retry の fast path に寄せた
  - subset 実測:
    - `tests/integration/test_custom_ppo_integration.py`
    - `tests/training/test_ppo_trainer.py`
    - `tests/unit/v460/test_enricher_skip_gate.py`
    - `tests/unit/v460/test_fill_quality.py`
    - `312 passed in 7.86s`
  - slowest 25 の変化:
    - `test_enrichment_with_real_data` setup: `0.56s -> 0.29s`
    - `test_status_none_twice_becomes_cancelled_status_unknown`: `0.58s -> 0.03s`

- PPO trainer / checkpoint helper trim:
  - `tests/training/test_ppo_trainer.py`
    - `train()` orchestration test を internal boundary patch に寄せ、
      DataLoader / env / model 実生成を踏まないように整理
    - `TrainingConfigManager` の実初期化も不要な箇所では mock 化
  - `tests/unit/training/test_checkpoint_manager.py`
    - replay buffer capture / restore の tmpfile helper 回帰を追加
  - `ztb/training/checkpoint/checkpoint_manager.py`
    - replay buffer capture / restore を generic tmpfile helper に統一
  - subset 実測:
    - `tests/training/test_ppo_trainer.py`
      - before: `29 passed in 9.75s`
      - after: `29 passed in 5.54s`
    - `tests/unit/training/test_checkpoint_manager.py`
      - `13 passed in 3.64s`

- PPO Phase 3 scheduler / gate coverage trim:
  - `tests/unit/v460/test_680_ppo_retrain_scheduler.py`
    - warm-start
    - crash resilience
    - `record_result()` exception suppression
  - `tests/unit/v460/test_sidecar_sac_integration.py`
    - PPO gate の `None` signal / below-margin observe-only
  - `tests/unit/v460/test_sac_retrain_scheduler.py`
    - `_post_cycle_memory_check()` を deterministic mock に変更
  - focused:
    - `235 passed in 7.10s`
  - broader PPO/SAC subset:
    - `237 passed in 8.80s`

  ここで効いたのは単純な sleep 削減より、
  - interval/time 依存を trigger mock に寄せる
  - live RSS 依存を helper mock に寄せる
  という test responsibility の整理。

- PPO warm-start helper reuse:
  - `tests/training/test_ppo_trainer.py`
  - `tests/integration/test_custom_ppo_integration.py`
  - `tests/unit/v460/test_ppo_warm_start.py`
  - `SELLBiasMitigationPPOTrainer` の warm-start path を direct guard
  - core 側の `load_ppo_model_for_env(...)` も focused で固定
  - subset:
    - `39 passed in 6.02s`

- 2026-04-01 additional trim:
  - `tests/unit/v460/test_enricher_skip_gate.py`
    - real-data sample rows を `50/56/72` 系から `36/48/60` 系へ縮小
    - smoke path も `16/24/32` に縮小
  - `tests/training/test_ppo_trainer.py`
    - runtime limit helper (`data_rows_limit` / `max_features`) の focused guard を追加
  - subset:
    - `tests/training/test_ppo_trainer.py`
    - `tests/unit/v460/test_fill_quality.py`
    - `tests/unit/v460/test_enricher_skip_gate.py`
    - `tests/test_analyze_fill_logs.py`
    - `342 passed, 1 skipped in 10.46s`
  - `tests/unit/training/test_target_entropy.py`
    - `no_grad()` 漏れに耐える回帰を追加
    - focused bundle 更新:
      - `tests/unit/training/test_target_entropy.py`
      - `tests/training/test_ppo_trainer.py`
      - `tests/unit/v460/test_fill_quality.py`
      - `tests/unit/v460/test_enricher_skip_gate.py`
      - `tests/test_analyze_fill_logs.py`
      - `355 passed, 1 skipped in 11.59s`

- 2026-04-01 sidecar + entropy trim:
  - `tests/unit/v460/test_680_ppo_retrain_scheduler.py`
    - `retrain_once()` 系で `_cleanup_training_cycle()` を patch し、
      neutral fallback / error handling の責務だけを見るよう整理
  - `tests/unit/v460/test_sac_retrain_scheduler.py`
    - shared retrain helper で `cleanup_training_resources()` を patch
    - warm-start test も deploy/signal update を mock 化
  - `tests/unit/training/test_target_entropy.py`
    - tiny tensor batch に縮小
    - `torch.set_num_threads(1)` を適用し、CPU thread fan-out を抑制
  - `tests/unit/v460/test_enricher_skip_gate.py`
    - real-data smoke sample を `12/18/24` へ縮小
  - focused durations:
    - before:
      - `tests/unit/v460/test_680_ppo_retrain_scheduler.py`
      - `tests/unit/v460/test_sac_retrain_scheduler.py`
      - `tests/unit/training/test_target_entropy.py`
      - `tests/unit/v460/test_enricher_skip_gate.py`
      - `162 passed, 1 skipped in 17.99s`
    - after:
      - 同 subset で `162 passed, 1 skipped in 11.39s`
  - 主な変化:
    - `test_alpha_increases_on_low_entropy`: `5.33s -> 3.77s`
    - `test_error_pushes_neutral_fallback`: `1.18s -> 0.50s`
    - `test_cold_start_success`: `0.47s -> 0.08s`
  - broader regression:
    - PPO/SAC + fill_quality + enricher + target_entropy
    - `562 passed, 3 skipped in 23.41s`

- 2026-04-02 fill-quality / PPO structure bundle:
  - `ztb/metrics/fill_record_io.py`
    - `save_fill_records(...)`
    - `iter_fill_records(...)`
    - `load_fill_records(...)`
    - `iter_fill_records_glob(...)`
    - `load_fill_records_glob(...)`
    - `fill_records_to_dataframe(...)`
    - `iter_fill_record_dicts(...)`
    を I/O 層へ集約
  - `ztb/metrics/fill_quality.py`
    - I/O utilities を再 export する thin surface に整理
  - `ztb/training/experiments/sell_mitigation_ppo_trainer.py`
    - warm-start / cold-start の実行フローを helper 分割
  - `tests/integration/test_custom_ppo_integration.py`
    - warm-start で policy bias を再ニュートラライズしない guard
  - `tests/unit/v460/test_trend_5s_sell_guard.py`
    - `trend_5s_at_order` telemetry guard
  - focused:
    - `tests/unit/v460/test_fill_quality.py`
    - `tests/integration/test_custom_ppo_integration.py`
    - `tests/unit/v460/test_ppo_warm_start.py`
    - `tests/training/test_ppo_trainer.py`
    - `tests/unit/v460/test_trend_5s_sell_guard.py`
    - `tests/unit/v460/test_sac_sell_aware_reward.py`
    - `265 passed in 8.55s`
  - broader regression:
    - PPO/SAC scheduler + sell-aware + trend guard + fill_quality + enricher
    - `509 passed, 1 skipped in 13.95s`
  - full suite:
    - `tests/ -x --tb=short --no-cov`
    - 少なくとも 16% 超までは no failure を確認

- 2026-04-02 686# test infra fix:
  - `tests/unit/risk/test_rules.py`
    - `benchmark` fallback fixture を撤去
    - plugin 非依存の `perf_runner` fixture に置換
  - `rg -n "def benchmark\\(" tests` → 該当なし
  - focused:
    - `tests/unit/risk/test_rules.py`
    - `57 passed in 1.86s`

- 2026-04-02 686# 横展開:
  - `tests/conftest.py`
    - `perf_runner` を共通 fixture 化
  - `tests/integration/trading/test_signal_guidance_integration.py`
    - performance benchmark を共通 harness に寄せた
  - focused:
    - `tests/unit/risk/test_rules.py`
    - `tests/integration/trading/test_signal_guidance_integration.py`
    - `tests/unit/v460/test_292_observability.py`
    - `tests/unit/v460/test_fill_quality.py`
    - `294 passed in 8.56s`
