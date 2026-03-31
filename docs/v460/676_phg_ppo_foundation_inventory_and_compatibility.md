# 676# PPO foundation 棚卸しと互換整理

## 概要

v461 の PPO sidecar 再開に向けて、現行 repo に残っている PPO 関連コードを棚卸しし、まず壊れやすい互換層を整理した。

今回の狙いは次の 3 点:

1. どの PPO ファイルが **ACTIVE / REFACTOR / ARCHIVE** なのかを固定する
2. 互換 shim の import 崩れを直して、focused test を pass / intentional skip に整理する
3. `HeavyTradingEnv` の離散 action mode を PPO 前提で明示的に守る

## 棚卸し結果

### コア実装

| ファイル | 判定 | メモ |
|---|---|---|
| `ztb/training/algorithms/ppo/ppo_algorithm.py` | ACTIVE | `BaseRLAlgorithm` 互換 wrapper。config validation と basic tests は通る |
| `ztb/training/core/ppo_trainer.py` | ACTIVE | いまの PPO 本体。`PPOTrainerAutoHalt` と `PPOTrainer` の正本 |
| `ztb/training/unified_trainer/algorithms/ppo_trainer.py` | REFACTOR | unified trainer 側。`HeavyTradingEnv(data=...)` の drift を今回修正 |
| `ztb/training/trainers/ppo_trainer.py` | ACTIVE | orchestration 層。trainer 生成パスに古い呼び出し形が残っていたので今回修正 |
| `ztb/training/custom_ppo.py` | ACTIVE | `ztb.training.models.custom_ppo.CustomPPO` への shim |
| `ztb/training/models/custom_ppo.py` | ACTIVE | current `CustomPPO` 本体。`TargetEntropyController` を canonical path に寄せた |
| `sb3_contrib/__init__.py` | ACTIVE | lightweight `MaskablePPO` compatibility shim |
| `sb3_contrib/common/wrappers.py` | REFACTOR → ACTIVE | `ActionMasker` を Gymnasium wrapper 契約へ修正 |

### 設定・互換層

| ファイル | 判定 | メモ |
|---|---|---|
| `ztb/training/config/ppo_config.py` | ACTIVE | canonical config 定義 |
| `ztb/training/ppo_config.py` | REFACTOR | 旧 stub が canonical config と乖離していたため、re-export shim に変更 |
| `ztb/training/ppo_trainer.py` | REFACTOR | `PPOTrainer` / `CustomPPO` / `MaskablePPO` を再 export する互換 shim に整理 |
| `ztb/training/core/ppo_trainer.pyi` | REFACTOR | 実装シグネチャに合わせて更新 |

### テスト

| ファイル | 判定 | メモ |
|---|---|---|
| `tests/training/test_ppo_trainer.py` | ACTIVE | core trainer / algorithm trainer の focused suite。pass |
| `tests/unit/algorithms/test_ppo_algorithm.py` | ACTIVE | `21 passed, 2 skipped`。skip は `get_default_config()` の仕様差を明示 |
| `tests/unit/training/test_ppo_trainer.py` | REFACTOR → ACTIVE | 古い no-arg trainer 前提をやめ、互換 shim contract test に刷新 |
| `tests/integration/test_custom_ppo_integration.py` | REFACTOR → ACTIVE | current env / current trainer params 前提の focused integration harness に更新 |

### archive / legacy

| ファイル | 判定 | メモ |
|---|---|---|
| `ztb/training/archive/ppo_trainer_old.py` | ARCHIVE | 旧 no-arg trainer contract の保存先。active path では使わない |
| `ztb/training/experiments/sell_mitigation_ppo_trainer.py` | ACTIVE | 専用 trainer。本件では archive しない |

## 今回の修正

### 1. 互換 shim の整理

- `ztb/training/ppo_trainer.py`
  - `PPOTrainer`
  - `PPOTrainingConfig`
  - `TrainingConfig`
  - `CustomPPO`
  - `MaskablePPO`
  を current implementation へ再 export
- `ztb/training/ppo_config.py`
  - canonical `ztb.training.config.ppo_config` の re-export shim に変更

### 2. trainer 生成 path の drift 修正

- `ztb/training/trainers/ppo_trainer.py`
  - standard path は `TrainerParams` を作り、`PPOTrainerAutoHalt(params)` で生成
  - SELL mitigation path も shim import に寄せ、`SELLBiasMitigationPPOTrainer(params)` を使う形に統一
- `ztb/training/unified_trainer/algorithms/ppo_trainer.py`
  - `HeavyTradingEnv(data=df, ...)` を `HeavyTradingEnv(df=df, ...)` に修正

### 2.5. ActionMasker / entropy controller の runtime drift 修正

- `sb3_contrib/common/wrappers.py`
  - `ActionMasker` を単なる箱から `Gymnasium Wrapper` 契約へ修正
  - `mask_fn` / `action_mask_fn` の両キーワードに対応
  - `action_masks()` / `get_action_masks()` を提供
- `ztb/training/core/ppo_trainer.py`
  - `ActionMasker(..., action_mask_fn=env.action_mask())` の古い呼び出しを
    `mask_fn=lambda e: e.get_action_masks()` に修正
- `ztb/training/models/custom_ppo.py`
  - `experiments.entropy_temperature` ではなく canonical
    `ztb.training.entropy_temperature` を使用
- `ztb/training/experiments/sell_mitigation_ppo_trainer.py`
  - 同じく canonical `TargetEntropyController` へ寄せた

### 3. HeavyTradingEnv 離散 action mode の guard 追加

- `tests/unit/environment/test_heavy_env_initialization.py`
  - `use_continuous_actions=False` で `Discrete(3)`
  - `action_space_type="discrete"` で `Discrete(3)`
  - `use_continuous_actions=True` で `Box`
  を明示的に確認
- discrete `step()` 自体は既存の `tests/unit/environment/test_forced_actions.py` が BUY/HOLD/SELL の流れを already cover している

## focused test 結果

```bash
.venv/Scripts/python.exe -m pytest tests/training/test_ppo_trainer.py -x --tb=short --no-cov
```

- `26 passed in 19.52s`

```bash
.venv/Scripts/python.exe -m pytest tests/unit/algorithms/test_ppo_algorithm.py -x --tb=short --no-cov
```

- `21 passed, 2 skipped in 20.69s`

```bash
.venv/Scripts/python.exe -m pytest tests/unit/training/test_ppo_trainer.py -x --tb=short --no-cov
```

- before: import 崩れで module skip
- after: compatibility shim test として active 化

```bash
.venv/Scripts/python.exe -m pytest tests/integration/test_custom_ppo_integration.py -x --tb=short --no-cov
```

- before: `1 skipped`
- after: `5 passed`

```bash
.venv/Scripts/python.exe -m pytest tests/integration/test_custom_ppo_integration.py tests/training/test_ppo_trainer.py tests/unit/training/test_ppo_trainer.py tests/unit/environment/test_heavy_env_initialization.py -x --tb=short --no-cov
```

- `46 passed in 34.10s`

## 残課題

1. `ztb/training/unified_trainer/algorithms/ppo_trainer.py`
   - real training path の smoke test を追加したい
2. `ztb/training/models/custom_ppo.py`
   - runtime は復旧したが、mypy baseline はまだ厚い
3. 過去バージョンの PPO 実験コード
   - いきなり archive 移動せず、参照実態を見てから別 batch で整理する
