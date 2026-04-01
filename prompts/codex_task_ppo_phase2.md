# Codex Task: PPO Sidecar Phase 2 — Warm-Start, Config, Legacy Cleanup

## 背景

Phase 1 (675#-680#) で以下が完了済:
- `scripts/v460/ml/ppo_retrain_scheduler.py` — scheduler 本体 (cold start のみ)
- `scripts/v460/ml/ppo_sidecar_config.py` — PPOSidecarConfig dataclass
- `scripts/v460/lib/sidecar_signal_io.py` — PPO signal read/write
- `scripts/v460/lib/orchestrator_mid_cycle.py` — PPO wiring (confidence gate 付き)
- `tests/unit/v460/test_679_ppo_sidecar_foundation.py` — signal/config テスト (8 tests)
- `tests/unit/v460/test_680_ppo_retrain_scheduler.py` — scheduler テスト (11 tests)

**未解決の課題:**
1. PPO warm-start がない (`SELLBiasMitigationPPOTrainer` に load API なし)
2. v460 用 PPO config YAML が未作成 (`configs/v460/experiments/` に存在しない)
3. 旧 PPO config (v367-v428) が 50+ ファイル散在
4. `test_custom_ppo_integration.py` の `test_trainer_uses_current_params_interface` が `total_timesteps` 分散問題を潜在的に抱えている

## タスク

### Task 1: PPO Warm-Start API の追加

`ztb/training/core/ppo_trainer.py` の `PPOTrainerAutoHalt` に warm-start 機能を追加:

```python
def load_and_continue(self, model_path: Path, total_timesteps: int, session_id: str) -> object:
    """既存モデルを読み込み、追加学習を実行する."""
```

実装要件:
- `MaskablePPO.load()` で既存モデルを読み込み
- 環境を wrap して `model.set_env()` で新環境をバインド
- `model.learn(total_timesteps=total_timesteps)` で追加学習
- 失敗時は cold start にフォールバック

その後 `ppo_retrain_scheduler.py` の `retrain_once()` で warm-start 分岐を実装:
```python
if is_warm_start and hasattr(trainer, 'load_and_continue'):
    model = trainer.load_and_continue(cfg.model_path, timesteps, session_id=model_version)
else:
    model = trainer.train(session_id=model_version)
```

テスト:
- `tests/unit/v460/test_ppo_warm_start.py` を新規作成
- cold start → save → warm start → save のラウンドトリップテスト
- model_path が存在しない場合の cold start フォールバックテスト

### Task 2: v460 PPO Config YAML の作成

`configs/v460/experiments/g2_ppo_sidecar.yaml` を作成。SAC の `g2_sac_train.yaml` に準拠した構造:

```yaml
ppo_sidecar:
  data_path: "data/btc_jpy_real_dataset.csv"
  model_path: "models/v461/ppo_sidecar.zip"
  signal_path: "cache/ppo_sidecar_signal.json"
  checkpoint_dir: "models/v461"
  total_timesteps: 200000
  incremental_timesteps: 50000
  retrain_interval_sec: 7200
  check_interval_sec: 300

  # 離散3行動 (SKIP=0, BUY=1, SELL=2)
  use_continuous_actions: false
  enable_action_masking: true

  # SAC 679# の教訓: simple reward + high gamma
  use_simple_reward: true
  gamma: 0.95
  reward_scaling: 100.0

  # PPO 固有
  n_steps: 256
  batch_size: 64
  n_epochs: 4
  learning_rate: 0.0003
  clip_range: 0.2
  ent_coef: 0.01

  # Bias mitigation (最小限のみ有効化)
  enable_pan: true
  enable_target_entropy: false
  enable_stratified_sampling: false
  allow_reverse: false

  # Deploy gate
  min_override_confidence: 0.55
  min_action_probability_gap: 0.15

  # Data freshness
  max_data_stale_hours: 1.5
```

`PPOSidecarConfig.from_yaml_dict()` がこの YAML を正しくパースできることをテストで検証。

### Task 3: レガシー PPO Config の整理

以下のディレクトリ配下の PPO config を `archived/configs/` に移動:
- `configs/v367/` ~ `configs/v394/` (PPO 実験 config, 30+ ファイル)
- `configs/v428/` (ensemble test)
- `configs/training/ppo_*.json` (初期実験)
- `configs/v1/`, `configs/v2/`, `configs/v3/` の PPO config

保持するもの:
- `configs/ppo_test_config.yaml` (テスト用、参照あり)
- `configs/v460/` 配下すべて

移動手順:
1. `archived/configs/ppo_legacy/` ディレクトリ作成
2. 対象ファイルを `git mv` で移動
3. import や参照がないことを `grep -r` で確認
4. テスト実行で壊れないことを確認

### Task 4: PPO Trainer テストの網羅性向上

既存テストの状態確認・修正:
- `tests/training/test_ppo_trainer.py` — 実行確認
- `tests/unit/algorithms/test_ppo_algorithm.py` — 実行確認
- `tests/unit/training/test_ppo_trainer.py` — 実行確認
- `tests/integration/test_custom_ppo_integration.py` — 全5テスト pass 確認済

追加テストが必要な箇所:
- `_extract_action_probabilities()` のエッジケース (policy=None, probs=None)
- `_one_hot_ppo_probabilities()` の境界値 (action_index < 0, > 2)
- `PPORetrainTrigger.record_result()` のバックオフ動作
- `retrain_once()` の例外時 neutral fallback パス

## 制約

- `git commit --no-verify -m "..."` でコミット
- `git add .` 禁止。変更ファイルを個別指定
- テスト: `python -m pytest tests/ -x --tb=short` — 既存 SAC テスト (2261 passing) を壊さない
- 型安全: Any 型は最小限。mypy 通過を目指す
- `_sb3_test_stub/` のスタブを活用してテスト環境で SB3 import を回避

## 成果物

1. PPO warm-start API + テスト
2. `configs/v460/experiments/g2_ppo_sidecar.yaml`
3. レガシー PPO config の `archived/configs/ppo_legacy/` への移動
4. PPO テスト網羅性向上 (追加テスト 5件以上)
5. ドキュメントは不要 (681# 以降で反映)

## 注意事項

- `scripts/v460/ml/sac_retrain_scheduler.py` は触らない (SAC 側は別管理)
- `scripts/v460/ml/sidecar_scheduler_common.py` は共有基盤。変更時は SAC テストも確認
- PPO の実行は `_sb3_test_stub` でモック可能。実際の SB3 学習は不要
- doc 番号 681# 以降は使用しない (こちらで採番する)
