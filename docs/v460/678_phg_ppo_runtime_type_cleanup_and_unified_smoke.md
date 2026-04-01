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
