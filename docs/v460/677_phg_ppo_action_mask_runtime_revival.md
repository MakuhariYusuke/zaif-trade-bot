# 677# PPO action-mask runtime revival と archive 候補確認

## 概要

676# の互換 shim 復旧の次段として、PPO の実行経路で止まっていた
`ActionMasker / TargetEntropyController` の runtime drift を修正した。

今回の狙いは次の 3 点:

1. pytest 上だけでなく、実 env でも `CustomPPO.learn()` が前に進む状態へ戻す
2. `tests/integration/test_custom_ppo_integration.py` を skip から active へ戻す
3. PPO archive 候補を軽く再確認し、今すぐ触らないものを明文化する

## 原因

### 1. local `ActionMasker` shim が env 契約を満たしていなかった

- `sb3_contrib/common/wrappers.py`
  - `ActionMasker` が `env` をぶら下げるだけの箱だった
  - `action_space` / `observation_space`
  - `action_masks()`
  - Gymnasium wrapper としての委譲
  が無かった

結果として、real runtime では `CustomPPO` 生成時に environment wrapping が破綻していた。

### 2. core trainer 側に古い keyword / method drift が残っていた

- `ztb/training/core/ppo_trainer.py`
  - `ActionMasker(..., action_mask_fn=lambda e: e.action_mask())`
  - という古い呼び出しが残っていた

今回、current contract の
`mask_fn=lambda e: e.get_action_masks()` に統一した。

### 3. `CustomPPO` が canonical ではなく experimental 実装を参照していた

- `ztb/training/models/custom_ppo.py`
- `ztb/training/experiments/sell_mitigation_ppo_trainer.py`

この 2 本が `ztb.training.experiments.entropy_temperature` を参照しており、
そちらには `compute_entropy` 周りの古い drift が残っていた。

今回 canonical `ztb.training.entropy_temperature` へ寄せた。

## 実施内容

### runtime / integration

- `sb3_contrib/common/wrappers.py`
  - `ActionMasker` を `Gymnasium Wrapper` 契約へ修正
  - `mask_fn` / `action_mask_fn` 両対応
  - `action_masks()` / `get_action_masks()` を追加
- `ztb/training/core/ppo_trainer.py`
  - `mask_fn=lambda e: e.get_action_masks()` に統一
- `ztb/training/models/custom_ppo.py`
  - canonical `TargetEntropyController` を参照
- `ztb/training/experiments/sell_mitigation_ppo_trainer.py`
  - 同上
- `tests/conftest.py`
  - test stub の `ActionMasker` も同じ契約へ追随
- `tests/integration/test_custom_ppo_integration.py`
  - module-level skip を廃止
  - current env / current trainer params に合わせた focused integration harness に再設計

## archive 候補の再確認

今回の sweep で確認できた PPO 関連 legacy ファイル:

| ファイル | 判定 | 理由 |
|---|---|---|
| `ztb/training/archive/ppo_trainer_old.py` | ARCHIVE 維持 | 旧 no-arg trainer 契約の保存先 |
| `ztb/training/experiments/sell_mitigation_ppo_trainer.py` | ACTIVE | current SELL mitigation trainer 本体 |

追加で `experiments/` / `scripts/training/` を確認したが、今回の sweep では
新たな PPO 実験コードは見当たらなかった。

## 検証

### focused pytest

```bash
.venv/Scripts/python.exe -m pytest tests/integration/test_custom_ppo_integration.py tests/training/test_ppo_trainer.py tests/unit/training/test_ppo_trainer.py tests/unit/environment/test_heavy_env_initialization.py -x --tb=short --no-cov
```

- `46 passed in 34.10s`

### runtime smoke

```bash
.venv/Scripts/python.exe - <<'PY'
...
model = CustomPPO(...)
learned = model.learn(total_timesteps=64, progress_bar=False)
...
PY
```

- `Discrete (3,)`
- `learned_is_self True`
- `entropy_updates 4`

### targeted mypy

```bash
.venv/Scripts/python.exe scripts/quality/run_targeted_mypy.py sb3_contrib/common/wrappers.py tests/integration/test_custom_ppo_integration.py
```

- `Success: no issues found in 2 source files`

補足:
- `ztb/training/models/custom_ppo.py`
- `ztb/training/experiments/sell_mitigation_ppo_trainer.py`

は runtime は復旧したが、mypy baseline はまだ厚い。ここは別 batch で減らす方が安全。

## 次の一手

1. `CustomPPO` / `SELLBiasMitigationPPOTrainer` の mypy baseline を小さい束で減らす
2. PPO sidecar signal/scheduler は、SAC sidecar の型/I/O を見ながら重複を避けて設計する
3. unified trainer 側の PPO smoke を追加して、current training path をもう一段固める
