# 678# PPO runtime/type cleanup と unified smoke 追加

## 概要

676# / 677# で PPO の互換 shim と action-mask runtime は復旧したが、
まだ次の 3 点が残っていた。

1. `core` と `sell_mitigation` で action-mask の張り方が重複していた
2. `CustomPPO` / `PPOTrainer` 周辺に low-risk な mypy 残差が残っていた
3. unified trainer 側の PPO 実行経路に focused smoke が無かった

今回の batch では、runtime を変えすぎずにこの 3 点をまとめて整理した。

## 実施内容

### 1. action-mask 契約の共有化

- `ztb/training/core/ppo_trainer.py`
  - `_get_action_masks(...)`
  - `wrap_env_with_action_masker(...)`
  を追加
- `ztb/training/experiments/sell_mitigation_ppo_trainer.py`
  - 独自の `get_legal_actions().astype(bool)` 経路をやめ、
    `wrap_env_with_action_masker(...)` を再利用

これで PPO の active path はどちらも
`mask_fn=lambda env: env.get_action_masks()`
の current contract に揃った。

### 2. `SELLBiasMitigationPPOTrainer` の重複解消

- `sell_mitigation_ppo_trainer.py`
  - `PerActionAdvantageNormalizer`
  - `TargetEntropyController`
  - `StratifiedSampler`
  の trainer-side 重複初期化を削除

これらはすでに `CustomPPO` が正本として管理しており、
callback も `self.model.*` を見ていたため、trainer 側の別インスタンスは
保守ノイズになっていた。

### 3. low-risk mypy cleanup

対象:

- `ztb/training/models/custom_ppo.py`
- `ztb/training/core/ppo_trainer.py`
- `ztb/training/unified_trainer/algorithms/ppo_trainer.py`
- `ztb/training/experiments/sell_mitigation_ppo_trainer.py`

主な修正:

- `TypeAlias` で `Observation` / `PredictionResult` を明示
- `CustomPPO` の `action_space` 解決を helper 化
- 使われなくなった `type: ignore` を削除
- `explained_variance(...)` の戻り値を `float` に固定
- unified trainer 側の `self.model` 型注釈 drift を修正
- `train()` の bool 返却を明示化

### 4. テスト強化と高速化

- `tests/training/test_ppo_trainer.py`
  - core trainer が `ActionMasker(..., mask_fn=...)` を使うことを明示
- `tests/integration/test_custom_ppo_integration.py`
  - `SELLBiasMitigationPPOTrainer` の integration を lightweight fake env 化
  - fake env の `get_legal_actions()` は敢えて失敗させ、
    legacy mask path が再発しないことを guard
- `tests/training/unified_trainer/test_algorithms.py`
  - unified PPO trainer の current training path smoke を追加

## 検証

### targeted mypy

```bash
.venv/Scripts/python.exe scripts/quality/run_targeted_mypy.py \
  ztb/training/models/custom_ppo.py \
  ztb/training/experiments/sell_mitigation_ppo_trainer.py \
  ztb/training/unified_trainer/algorithms/ppo_trainer.py \
  ztb/training/core/ppo_trainer.py \
  tests/integration/test_custom_ppo_integration.py
```

- `Success: no issues found in 5 source files`

### focused pytest

```bash
.venv/Scripts/python.exe -m pytest \
  tests/training/test_ppo_trainer.py \
  tests/unit/algorithms/test_ppo_algorithm.py \
  tests/unit/training/test_ppo_trainer.py \
  tests/integration/test_custom_ppo_integration.py \
  tests/training/unified_trainer/test_algorithms.py \
  -x --tb=short --no-cov --durations=20
```

- `92 passed, 2 skipped in 11.06s`

主な改善:

- `tests/integration/test_custom_ppo_integration.py::TestSellMitigationTrainerIntegration::test_trainer_uses_current_params_interface`
  - `11.80s -> 0.02s`

## 追加で確認した archive 候補

今回の sweep では、次の PPO legacy script 群は引き続き archive 候補だった。

- `experiments/train_sac_v443_2_market_regime_adaptation.py`
- `experiments/train_v443_2_phase1.py`
- `experiments/train_v443_2_phase2.py`
- `experiments/sac_v446_algorithm_tuning.py`

いずれもヘッダ上は PPO 実験コードで、current v460/v461 active path ではない。
ただし今回の batch では runtime 復旧と current path の整理を優先し、移動は行っていない。

## 次の一手

1. `CustomPPO` / `SELLBiasMitigationPPOTrainer` の残る baseline mypy を小さく減らす
2. PPO sidecar scheduler の warm-start 設計を、current trainer 契約の上で整理する
3. legacy PPO 実験コードを archive batch として別 commit で切る

## ステータス追記

678# の後段として、current PPO foundation はさらに次の状態まで進んだ。

### 1. `PPOAlgorithm` wrapper の hidden gap を解消

- `ztb/training/algorithms/ppo/ppo_algorithm.py`
  - `create_model()` が current config から `CustomPPO` / `MaskablePPO` を選んで生成
  - `train()` は placeholder ではなく `model.learn(...)` に委譲

これで inventory 上 **ACTIVE** とされていた wrapper が、
実際に `BaseRLAlgorithm` 契約どおり前に進む状態になった。

### 2. PPO sidecar signal foundation を追加

- `scripts/v460/lib/sidecar_types.py`
  - `PPOSidecarSignal`
  - `normalize_ppo_action_probabilities(...)`
  - `resolve_ppo_sidecar_action(...)`
- `scripts/v460/lib/sidecar_signal_io.py`
  - `write_ppo_sidecar_signal(...)`
  - `read_ppo_sidecar_signal(...)`
  - `create_neutral_ppo_signal(...)`
- `scripts/v460/ml/ppo_sidecar_config.py`
  - discrete action / override threshold を固定する最小 config helper

SAC sidecar の atomic JSON I/O と TTL/stale 判定をそのまま再利用し、
PPO sidecar だけ別フォーマットを増やさない形に寄せた。

### 3. 設計上の固定

PPO sidecar は **side selection のみ** を扱い、
価格 aggressiveness は SAC / executor 側に残す。

この責務分離により:

- PPO:
  - BUY / SELL / SKIP の離散判断
  - top probability と probability gap による override 安定度
- SAC:
  - quote offset / aggressiveness の連続制御

という境界が保てる。

### 4. 追加 focused 確認

```bash
.venv/Scripts/python.exe scripts/quality/run_targeted_mypy.py \
  scripts/v460/lib/sidecar_types.py \
  scripts/v460/lib/sidecar_signal_io.py \
  scripts/v460/ml/ppo_sidecar_config.py \
  ztb/training/algorithms/ppo/ppo_algorithm.py \
  tests/unit/v460/test_679_ppo_sidecar_foundation.py
```

- `Success: no issues found in 5 source files`

```bash
.venv/Scripts/python.exe -m pytest \
  tests/unit/v460/test_679_ppo_sidecar_foundation.py \
  tests/unit/algorithms/test_ppo_algorithm.py \
  tests/unit/environment/test_heavy_env_initialization.py \
  tests/training/test_ppo_trainer.py \
  tests/unit/training/test_ppo_trainer.py \
  tests/integration/test_custom_ppo_integration.py \
  -x --tb=short --no-cov --durations=20
```

- `77 passed, 2 skipped in 17.71s`

### 5. PPO sidecar reader / safe veto wiring

current foundation の次段として、PPO signal を live pipeline に observe-safe に接続した。

- `scripts/v460/lib/cycle_gate_aggregator.py`
  - PPO signal を per-cycle gate に接続
  - confidence / action margin を通過したときだけ override を有効化
  - `skip` は veto
  - reverse-side signal も veto
  - same-side signal は telemetry のみ残して通す
- `scripts/v460/lib/orchestrator_mid_cycle.py`
  - `read_ppo_sidecar_signal_with_status()` を current cycle に注入
- `scripts/v460/lib/fill_record_builder.py`
- `scripts/v460/lib/fill_cycle_executor.py`
- `ztb/metrics/fill_quality.py`
  - PPO action / confidence / margin / model_version / signal_status / override_active を FillRecord に保存
- `scripts/v460/lib/orchestrator_lifecycle.py`
  - PPO sidecar signal cache cleanup を追加

この batch は live 挙動を急に大きく変えない。
`ppo_sidecar_signal.json` が存在しない限り従来動作のままで、
signal が存在しても **危険側 veto を先に有効化**する構造に留めている。

### 6. 追加 focused 確認

```bash
.venv/Scripts/python.exe -m pytest \
  tests/unit/v460/test_679_ppo_sidecar_foundation.py \
  tests/unit/v460/test_sidecar_sac_integration.py \
  tests/unit/environment/test_heavy_env_initialization.py \
  -x --tb=short --no-cov --durations=20
```

### 7. PPO sidecar scheduler foundation

current foundation の次段として、`scripts/v460/ml/ppo_retrain_scheduler.py` を追加した。

- `scripts/v460/ml/sidecar_scheduler_common.py`
  - scheduler の `result / history / file-mtime trigger` を SAC と最小共有化
- `scripts/v460/ml/ppo_sidecar_config.py`
  - `history_path`
  - `retrain_interval_max_sec`
  - trainer config の top-level flatten
  を追加し、current PPO trainer 契約に追随
- `scripts/v460/ml/ppo_retrain_scheduler.py`
  - `SELLBiasMitigationPPOTrainer` を用いた sidecar retrain foundation
  - atomic deploy
  - neutral fallback
  - action probability から `PPOSidecarSignal` を更新
- `scripts/v460/ml/sac_retrain_scheduler.py`
  - shared scheduler helper に history/result/trigger を寄せた

この batch でも live 安全性を優先している。

- model が無い:
  - cold-start の full timesteps
- model がある:
  - current trainer は warm-start API をまだ持たないため、
    **fresh fit の shorter budget** に留める

つまり、signal/deploy/history を先に正本化し、
weights の warm-start は別 batch に分離した。

### 8. 追加 focused 確認

```bash
.venv/Scripts/python.exe scripts/quality/run_targeted_mypy.py \
  scripts/v460/ml/sidecar_scheduler_common.py \
  scripts/v460/ml/ppo_sidecar_config.py \
  scripts/v460/ml/ppo_retrain_scheduler.py \
  tests/unit/v460/test_680_ppo_retrain_scheduler.py
```

### 21. warm-start helper の共通化

PPO Phase 3 の後段として、trainer ごとに重複していた warm-start / model-load 経路を整理した。

- `ztb/training/core/ppo_trainer.py`
  - `resolve_ppo_model_class(...)`
  - `load_ppo_model_for_env(...)`
- `ztb/training/experiments/sell_mitigation_ppo_trainer.py`
  - `_load_training_dataframe(...)`
  - `_build_env_config(...)`
  - `_create_training_env(...)`
  - `_create_custom_model(...)`
  - `_resolve_total_timesteps(...)`
  - `_ensure_lagrange_step_count(...)`
  - `load_and_continue(...)`

これにより、

- current PPO class の解決
- load 後の env bind
- mitigation trainer の cold / warm start 分岐

が 1 箇所ずつに寄った。

focused:

```bash
.venv/Scripts/python.exe scripts/quality/run_targeted_mypy.py \
  ztb/training/core/ppo_trainer.py \
  ztb/training/experiments/sell_mitigation_ppo_trainer.py \
  tests/training/test_ppo_trainer.py \
  tests/integration/test_custom_ppo_integration.py \
  tests/unit/v460/test_ppo_warm_start.py
```

- `Success: no issues found in 5 source files`

```bash
.venv/Scripts/python.exe -m pytest \
  tests/training/test_ppo_trainer.py \
  tests/integration/test_custom_ppo_integration.py \
  tests/unit/v460/test_ppo_warm_start.py \
  -x --tb=short --no-cov
```

- `39 passed in 6.02s`

- `Success: no issues found in 4 source files`

```bash
.venv/Scripts/python.exe -m pytest \
  tests/unit/v460/test_679_ppo_sidecar_foundation.py \
  tests/unit/v460/test_680_ppo_retrain_scheduler.py \
  tests/unit/v460/test_sac_retrain_scheduler.py \
  -x --tb=short --no-cov
```

- `69 passed in 14.18s`

### 9. scheduler staleness / timeout hardening

foundation の次段として、PPO sidecar scheduler の停止耐性とメモリ健全性を
さらに補強した。

- `scripts/v460/ml/sidecar_scheduler_common.py`
  - `DataFileRetrainTrigger.MAX_STALENESS_MULT = 3.0`
  - `mtime` 不変でも `effective_interval × 3` 経過後は
    `time_forced (...)` で再訓練を再開
- `scripts/v460/ml/ppo_retrain_scheduler.py`
  - `_TRAINING_TIMEOUT_SEC = 3600`
  - `_train_with_timeout(...)`
  - `_cleanup_training_cycle()`
  を追加
  - training timeout 時は `TimeoutError` で cycle を fail-fast
  - cycle 終了時に `clear_cuda_cache()` と `gc.collect()` を実行

この batch で追加した guard は次の性質を持つ。

1. data freshness 側がファイル更新を抑制しても scheduler が永久停止しない
2. 学習 thread が長時間ぶら下がるケースを live 側へ波及させない
3. retrain ごとに GPU/CPU の一時メモリを明示解放し、sidecar 常駐時のリークを抑える

### 10. focused test 追記

```bash
.venv/Scripts/python.exe -m pytest \
  tests/unit/v460/test_680_ppo_retrain_scheduler.py \
  tests/unit/v460/test_sac_retrain_scheduler.py \
  -x --tb=short --no-cov
```

- `time_forced` fallback
- training timeout
- neutral fallback / deploy path

を focused で確認した。

### 11. core / sell-mitigation の model kwargs 共有化

PPO current path で残っていた重複のうち、標準 PPO 引数の組み立てを
`core` 側 helper に寄せた。

- `ztb/training/core/ppo_trainer.py`
  - `build_ppo_model_kwargs(...)` を追加
- `ztb/training/experiments/sell_mitigation_ppo_trainer.py`
  - standard PPO kwargs を上記 helper から再利用
  - mitigation 固有引数だけを local で追加

この整理で、次の drift を防げる。

1. core trainer と sell-mitigation trainer の PPO ハイパーパラメータずれ
2. 新しい top-level PPO 引数追加時の片側更新漏れ
3. current trainer 契約の二重メンテナンス

### 12. PPO integration test の軽量化

PPO まわりの重い integration/setup も合わせて薄くした。

- `tests/integration/test_custom_ppo_integration.py`
  - `ActionMasker` 契約確認は `_TinyMaskedEnv` ベースへ移行
  - short training smoke も tiny env / shorter timestep に整理
- `tests/training/test_ppo_trainer.py`
  - `build_ppo_model_kwargs(...)` の focused guard を追加

HeavyTradingEnv を本当に見る必要がある test と、
mask contract / trainer wiring だけ見ればよい test を分離したことで、
検出力を維持したまま setup 固定費を落とした。

### 13. SAC/PPO scheduler helper の再共有化

foundation が揃ってきた段階で、scheduler 側に残っていた
「timeout 実行」と「best-effort cleanup」の重複も shared helper に戻した。

- `scripts/v460/ml/sidecar_scheduler_common.py`
  - `run_with_timeout(...)`
  - `best_effort_training_cleanup()`
  を追加
- `scripts/v460/ml/sac_retrain_scheduler.py`
  - local thread timeout 実装を shared helper に置換
  - post-cycle cleanup も shared helper を再利用
- `scripts/v460/ml/ppo_retrain_scheduler.py`
  - training timeout / cleanup を同 helper に統一

この整理で、SAC/PPO 間で次の drift を防げる。

1. timeout 例外の扱い差
2. cycle 後 cleanup 忘れ
3. WSL/Windows 環境での thread timeout 実装の二重保守

### 14. PPO Phase 2: warm-start / YAML / legacy config cleanup

Phase 2 では、foundation の上に運用面の不足を埋めた。

- `ztb/training/core/ppo_trainer.py`
  - `PPOTrainerAutoHalt.load_and_continue(...)` を追加
  - 既存 model を load し、fresh env を `set_env(...)` で再bindして
    incremental learn を継続できるようにした
  - load 失敗時は cold start に安全側 fallback
- `scripts/v460/ml/ppo_retrain_scheduler.py`
  - warm-start 時は `load_and_continue(...)` を優先
  - `trainer_mode` を `warm_start_resume` / `cold_start` で明示
- `scripts/v460/ml/ppo_sidecar_config.py`
  - `model_path`
  - `max_data_stale_hours`
  - `enable_action_masking`
  を current sidecar contract として保持
  - `ppo_sidecar:` 直下の hyperparameter flatten にも対応
- `configs/v460/experiments/g2_ppo_sidecar.yaml`
  - v460 現行運用用の PPO sidecar YAML を追加
- `tests/unit/v460/test_ppo_warm_start.py`
  - cold start -> save -> warm-start の focused roundtrip
  - model missing / load failure の fallback
- `tests/unit/v460/test_679_ppo_sidecar_foundation.py`
  - actual YAML parse guard を追加
- `tests/unit/v460/test_680_ppo_retrain_scheduler.py`
  - `_extract_action_probabilities()` edge cases
  - `_one_hot_ppo_probabilities()` clamp
  - trigger backoff reset
  - signal update failure 時 neutral fallback
  を追加

### 15. Legacy PPO config archive 整理

散在していた旧 PPO config は `archived/configs/ppo_legacy/` へ移動した。

- 対象:
  - `configs/v367` - `configs/v394` の PPO 実験 config
  - `configs/v428` の PPO config
  - `configs/training/ppo_*.json`
  - `configs/v1` / `v2` / `v3` の PPO config
- 維持:
  - `configs/ppo_test_config.yaml`
  - `configs/v460/**`

コード・docs 側の参照も archived path に追随させ、
prompt / archive 自体を除けば旧 path 参照は残さないところまで整理した。

### 16. SAC/PPO deploy helper 再共有 + scheduler test trim

sidecar scheduler の deploy 安全化で、SAC/PPO 間にまだ残っていた
tmp file + rename の重複を shared helper に寄せた。

- `scripts/v460/ml/sidecar_scheduler_common.py`
  - `atomic_replace_with_tmp(...)` を追加
- `scripts/v460/ml/ppo_retrain_scheduler.py`
  - PPO model deploy を shared helper に統一
- `scripts/v460/ml/sac_retrain_scheduler.py`
  - model deploy
  - buffer deploy
  - norm export
  を shared helper に統一

あわせて `sac_retrain_scheduler` の data freshness 呼び出しを
`_run_data_freshness_check(...)` に寄せ、test では non-fatal helper を
明示 patch できるようにした。これで scheduler test の責務が
「data updater 実行」から「scheduler control flow」へ戻った。

### 17. `CustomPPO` train ループの helper 分割

`ztb/training/models/custom_ppo.py` は train ループの責務が 1 箇所に寄りすぎていたため、
大きな継承追加ではなく helper 分割で整理した。

- 追加した helper:
  - `_normalize_advantages(...)`
  - `_compute_lagrange_penalty(...)`
  - `_resolve_entropy_coefficient(...)`
  - `_compute_value_loss(...)`
  - `_compute_entropy_loss(...)`
  - `_record_stats(...)`
- 目的:
  1. PAN / Lagrange / entropy controller の責務境界を明示
  2. train ループ本体を「PPO update の流れ」に寄せる
  3. `SELLBiasMitigationPPOTrainer` から見た component 境界を崩さない

これは god object を無理に大分割するのではなく、
hot path を読める長さに戻すための low-risk な整理。

### 18. Generic atomic helper の適用範囲確認 + PPO trainer test trim

今回の確認で、`atomic_replace_with_tmp(...)` は
`scripts/v460/ml/sidecar_scheduler_common.py` に閉じるより、
`ztb/utils` 配下の generic helper として持つ方が自然、という判断になった。

- 新設:
  - `ztb/utils/atomic_io.py`
    - `atomic_replace_with_tmp(...)`
    - `capture_bytes_via_tmpfile(...)`
    - `restore_bytes_via_tmpfile(...)`
- 横展開:
  - `scripts/v460/ml/sidecar_scheduler_common.py`
    - atomic deploy helper は generic util を再利用
  - `ztb/training/checkpoint/checkpoint_manager.py`
    - replay buffer の capture / restore を generic tmpfile helper に統一

あわせて、`tests/training/test_ppo_trainer.py` は
`train()` orchestration の確認に責務を寄せ、
`_create_environment()` / `_create_model()` / `_learn_model()` を patch する形へ整理した。

- before:
  - `tests/training/test_ppo_trainer.py`
  - `29 passed in 9.75s`
- after:
  - 同 focused suite
  - `29 passed in 5.54s`

支配点だった以下は大きく減っている。
- `test_create_sell_mitigation_ppo_trainer_uses_shim`
  - `0.55s -> 0.13s`
- `test_train_with_exception`
  - `0.53s -> 0.01s`
- `test_train_success_path`
  - `0.24s -> 0.01s`

`atomic_io` の他候補も見たが、今回は以下は据え置いた。
- `ztb/training/run_optimization.py`
  - temp JSON は subprocess に一時引き渡す用途で、atomic deploy helper の責務とは少し違う
- SAC/PPO scheduler の学習本体
  - timeout / cleanup / atomic deploy は共通化できるが、
    train/eval/signal semantics まで寄せると責務が崩れる

### 19. PPO Phase 3 coverage と loop crash-resilience

- `ppo_retrain_scheduler.py`
  - `run_scheduler()` を crash-resilient 化
  - `should_retrain()` 例外は loop 継続
  - `retrain_once()` 想定外例外は neutral fallback + error result 化
  - `record_result()` / history append は best-effort 化
- `tests/unit/v460/test_680_ppo_retrain_scheduler.py`
  - warm-start path
  - neutral fallback write failure suppression
  - single iteration / crash resilience / `record_result()` suppression
  を追加
- `tests/unit/v460/test_sidecar_sac_integration.py`
  - PPO gate の `None` signal / below-margin observe-only を追加
- `tests/unit/v460/test_enricher_skip_gate.py`
  - OB features の `SimpleImputer` 補完 path を追加
- `tests/unit/v460/test_sac_retrain_scheduler.py`
  - `_post_cycle_memory_check()` を deterministic mock に変更

focused:
- `235 passed in 7.10s`

broader PPO/SAC subset:
- `237 passed in 8.80s`

### 20. `scripts` 混雑緩和の最小移行

`sidecar_scheduler_common.py` は v460 専用 helper ではなくなっていたため、
実体を `ztb/training/sidecar/scheduler_common.py` に移し、
`scripts/v460/ml/sidecar_scheduler_common.py` は互換 shim にした。

これで
- current runtime entrypoint はそのまま
- generic sidecar helper は `ztb` 側に集約
- 後続の package 再整理がしやすい

という状態になった。

### 21. PPO runtime limit helper 共通化

`PPOTrainerAutoHalt` と `SELLBiasMitigationPPOTrainer` には、

- `data_rows_limit`
- `max_features`

の解決ロジックが重複していた。ここを
`ztb/training/core/runtime_limits.py`
へ寄せて、runtime helper として共通化した。

- 追加:
  - `resolve_data_rows_limit(...)`
  - `resolve_max_features(...)`
  - `load_training_dataframe_with_limit(...)`
- 適用:
  - `ztb/training/core/ppo_trainer.py`
  - `ztb/training/experiments/sell_mitigation_ppo_trainer.py`

新しい継承階層は増やしていない。ここは trainer の責務差を保ったまま、
shared helper で drift を止める方が安全と判断した。

### 22. Target entropy の grad-mode hardening

フル `tests/ -x --tb=short --no-cov` を回した際、
`tests/unit/training/test_target_entropy.py`
で、外側の grad-mode が崩れていると
`TargetEntropyController.update()` が `temp_loss.requires_grad=False`
で落ちる経路が見つかった。

今回は以下で hardening した。

- `ztb/training/entropy_temperature.py`
  - `log_alpha` を明示的に `device=self.device` で作成
  - `update()` を `with torch.enable_grad():` で保護
  - entropy 入力は `detach()` して、α 更新だけに責務を限定
- `tests/unit/training/test_target_entropy.py`
  - leaked `no_grad()` 下でも update が継続できる回帰を追加

これは速度改善というより、PPO を長時間回したときの
「他テスト/他コードが grad-mode を壊したまま残る」系の不安定性を潰す対応。

### 23. sidecar runtime helper の追加共有

`scripts/v460/ml` に残っていた scheduler bookkeeping の共通部分を、
さらに `ztb/training/sidecar/` へ寄せた。

- `ztb/training/sidecar/scheduler_common.py`
  - `append_history_best_effort(...)`
  - `record_trigger_result_best_effort(...)`
- `ztb/training/sidecar/ppo_policy.py`
  - `extract_action_probabilities(...)`
  - `one_hot_ppo_probabilities(...)`
  - `coerce_action_index(...)`

これにより
- `scripts/v460/ml/ppo_retrain_scheduler.py`
- `scripts/v460/ml/sac_retrain_scheduler.py`

は entrypoint / runtime wiring に寄せやすくなり、`scripts` 混雑を少し緩和できた。

無理に SAC/PPO の signal semantics を共通化するのではなく、
- timeout / bookkeeping / trigger bookkeeping
- PPO policy probability 抽出

のような純粋 helper だけを共有している。

### 24. PPO warm-start 周辺の hidden 残課題

684# とその周辺ドキュメントの意図から見える、未記載だが重要な残件は次のとおり。

1. `custom_ppo.py` と `sell_mitigation_ppo_trainer.py` の state/weight 継承境界
2. `ppo_retrain_scheduler.py` の warm-start deploy/fallback safety を、SAC 側と同粒度で維持
3. `scripts/` 配下に残る generic helper を今後も `ztb/training/sidecar/` へ戻すこと

今回の shared helper 追加は、この 3 点の前提整備として妥当だった。

### 25. SELL mitigation warm-start / cold-start の共有フロー整理

`ztb/training/experiments/sell_mitigation_ppo_trainer.py` は、

- cold start 時の model 準備
- warm-start 時の model load
- `learn()` 実行
- probe cleanup

が `train()` と `load_and_continue()` に分散しており、保守観点で drift しやすかった。

今回は以下へ分割した。

- `_prepare_cold_start_model()`
- `_prepare_warm_start_model(...)`
- `_run_current_model_training(...)`
- `_close_probe()`

重要な判断は、

- cold start では従来どおり `neutralize_policy_bias()` を維持
- warm-start では学習済み重みを再ニュートラライズしない

という点。これにより warm-start の state 継承を壊さず、trainer-owned な
probe / weighting の wiring だけ current contract に揃えられた。

合わせて `ztb/training/core/ppo_trainer.pyi` も runtime surface に追随させ、

- `build_ppo_model_kwargs(...)`
- `load_ppo_model_for_env(...)`
- `_create_callback()`
- `start_training()`
- `checkpoint_dir`

を明示した。PPO 周辺の `attr-defined` ノイズを増やさないための小さいが重要な整理。

### 26. 684# の hidden residual: trend_5s guard telemetry と regression

684# 系は大筋実装済みだったが、post-hoc 分析の前提として
`trend_5s_at_order` の guard 値保持を regression で固定した。

- `tests/unit/v460/test_trend_5s_sell_guard.py`
  - hard veto 時に `trend_5s_at_order == 4.0` を明示確認
- `tests/integration/test_custom_ppo_integration.py`
  - SELL mitigation warm-start で
    - `_setup_sell_bonus_weighting()` は current wiring として維持
    - `neutralize_policy_bias()` は warm-start で再実行しない
  ことを回帰化

これは 684# の本文に直接は書かれていないが、

- warm-start state continuity
- guard の事後分析可能性

の 2 点を考えると必要だった。
## 2026-04-02 追加

- PPO/SAC scheduler の neutral fallback 書き込みを
  [scheduler_common.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/ztb/training/sidecar/scheduler_common.py)
  の `push_neutral_signal_best_effort(...)` に寄せた
- 共有した責務は
  - neutral signal の生成/書き込み I/O 失敗 handling
  - success/failure logging
  に限定
- policy 判断や signal shape 自体は SAC/PPO で分けたままにしている
- overlap 確認:
  - 今回切り出した helper は既存 helper と実質重複していない
  - `atomic_replace_with_tmp(...)` と同じく `ztb.training.sidecar` 配下に置くのが妥当

## 2026-04-02 continuity 追記

- `ztb/training/core/ppo_trainer.py`
  - `continue_loaded_ppo_training(...)` を追加
  - `load_and_continue(...)` は `reset_num_timesteps=False` で継続学習するように修正
- `ztb/training/experiments/sell_mitigation_ppo_trainer.py`
  - warm-start でも同じ continuity 契約を使用
  - cold start のみ `neutralize_policy_bias()` を通す既存方針は維持
- `tests/unit/v460/test_ppo_warm_start.py`
  - cold start は `reset_num_timesteps=True`
  - warm start は `reset_num_timesteps=False`
  を明示的に guard
- `tests/integration/test_custom_ppo_integration.py`
  - SELL mitigation warm-start で continuity flag を確認
- `tests/training/test_ppo_trainer.py`
  - shared helper contract test を追加
- `scripts/v460/test_env_internal.py`
  - top-level `torch` import をやめ、`main()` 内 best-effort import に変更

回帰:

- targeted mypy:
  - `ztb/training/core/ppo_trainer.py`
  - `ztb/training/experiments/sell_mitigation_ppo_trainer.py`
  - `tests/unit/v460/test_ppo_warm_start.py`
  - `tests/integration/test_custom_ppo_integration.py`
  - `tests/training/test_ppo_trainer.py`
  - `scripts/v460/test_env_internal.py`
  - `Success: no issues found in 6 source files`
- focused pytest:
  - `tests/unit/v460/test_ppo_warm_start.py`
  - `tests/integration/test_custom_ppo_integration.py`
  - `tests/training/test_ppo_trainer.py`
  - `tests/unit/v460/test_enricher_skip_gate.py`
  - `tests/unit/v460/test_fill_quality.py`
  - `323 passed, 1 skipped, 5 warnings in 10.92s`
- broader regression:
  - `tests/unit/v460/test_679_ppo_sidecar_foundation.py`
  - `tests/unit/v460/test_680_ppo_retrain_scheduler.py`
  - `tests/unit/v460/test_sac_retrain_scheduler.py`
  - 上記 focused 群
  - `409 passed, 1 skipped, 5 warnings in 10.81s`
