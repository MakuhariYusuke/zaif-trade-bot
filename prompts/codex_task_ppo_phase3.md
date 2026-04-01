# Codex Task: PPO Phase 3 — テスト網羅性 + PPO Gate テスト + Skip Gate OB features

## 背景

Phase 1-2 で PPO sidecar の基盤は完了:
- `scripts/v460/ml/ppo_retrain_scheduler.py` — scheduler 本体 (warm-start + cold-start)
- `scripts/v460/ml/ppo_sidecar_config.py` — PPOSidecarConfig dataclass
- `scripts/v460/ml/sidecar_scheduler_common.py` — SAC/PPO 共通基盤 (trigger, timeout, atomic deploy)
- `scripts/v460/lib/cycle_gate_aggregator.py` — PPO gate wiring (`_apply_ppo_sidecar_gate()`)
- `configs/v460/experiments/g2_ppo_sidecar.yaml` — PPO sidecar 設定 YAML
- `tests/unit/v460/test_679_ppo_sidecar_foundation.py` — signal 契約テスト (8 tests)
- `tests/unit/v460/test_680_ppo_retrain_scheduler.py` — scheduler 基本テスト (11 tests)
- `tests/unit/v460/test_ppo_warm_start.py` — warm-start round-trip テスト

**テストギャップ (調査済み):**
1. PPO `retrain_once()` のモック統合テストなし (SAC 側には `TestRetrainOnce` が 7 test ある)
2. PPO `run_scheduler()` のループテストなし (SAC 側には `TestRunScheduler` がある)
3. `_apply_ppo_sidecar_gate()` の単体テストなし (cycle_gate_aggregator.py L475-530)
4. Skip Gate の OB features がデフォルト OFF のまま

## タスク

### Task 1: PPO Retrain Scheduler の包括テスト作成

**参照:** `tests/unit/v460/test_sac_retrain_scheduler.py` (54 tests, 包括的)

`tests/unit/v460/test_680_ppo_retrain_scheduler.py` に以下のテストクラスを追加:

#### 1a. `TestPPORetrainOnce` — retrain_once() のモック統合テスト
SAC の `TestRetrainOnce` (`test_sac_retrain_scheduler.py` L280-390) に準拠:

```python
class TestPPORetrainOnce:
    def test_cold_start_success(self):
        """model_path が存在しない → cold start で train() 呼び出し"""
    
    def test_warm_start(self):
        """model_path が存在する → load_and_continue() 呼び出し"""
    
    def test_oos_failed(self):
        """OOS eval が基準未達 → neutral signal push + 既存モデル保持"""
    
    def test_training_error_neutral_fallback(self):
        """trainer.train() が例外 → neutral signal が書き込まれる"""
    
    def test_deploy_gate_trade_count(self):
        """OOS trades < min_trade_count → deploy 拒否"""
    
    def test_deploy_gate_confidence(self):
        """action_probability_gap < min_action_probability_gap → deploy 拒否"""
```

モック対象:
- `PPOTrainerAutoHalt` → `MagicMock()` で `train()`, `load_and_continue()` をモック
- `load_training_data()` → 固定 DataFrame を返す
- `_extract_action_probabilities()` → テスト用の確率分布を返す
- `PPOSidecarConfig.from_yaml_dict()` → `tmp_path` ベースの config

#### 1b. `TestPPORunScheduler` — メインループテスト
```python
class TestPPORunScheduler:
    def test_single_iteration_then_shutdown(self):
        """1回 retrain → shutdown_event.set() → ループ終了"""
    
    def test_crash_resilience(self):
        """retrain_once() が例外 → ループは継続"""
```

#### 1c. `TestPPONeutralFallback`
```python
class TestPPONeutralFallback:
    def test_writes_neutral_signal(self):
        """push_neutral_fallback() が neutral PPO signal を書き込む"""
    
    def test_write_failure_suppressed(self):
        """write 失敗時も例外が伝搬しない"""
```

### Task 2: PPO Gate in CycleGateAggregator のテスト

`tests/unit/v460/test_cycle_gate_aggregator.py` に PPO gate テストを追加 (なければ新規作成):

```python
class TestPPOSidecarGate:
    """_apply_ppo_sidecar_gate() の単体テスト"""
    
    def test_skip_action_blocks(self):
        """PPO が SKIP を推奨 → gate_reason='ppo_sidecar_skip'"""
    
    def test_side_conflict_blocks(self):
        """PPO=buy, current=sell → gate_reason='ppo_sidecar_side_conflict'"""
    
    def test_matching_side_passes(self):
        """PPO=buy, current=buy → pass (telemetry のみ)"""
    
    def test_below_confidence_threshold_passes(self):
        """confidence < min_override_confidence → pass"""
    
    def test_below_margin_threshold_passes(self):
        """action_probability_gap < min_action_probability_gap → pass"""
    
    def test_none_signal_skips_gate(self):
        """ppo_sidecar_signal=None → gate 全体をスキップ"""
    
    def test_stale_signal_skips_gate(self):
        """signal status='stale' → gate スキップ"""
```

テスト実装ヒント:
- `CycleGateAggregator` のインスタンスを最小 config で作成
- `PPOSidecarSignal` を `scripts/v460/lib/sidecar_types.py` から import
- PPO signal の構造: `{action: int, action_probs: [skip, buy, sell], confidence: float, model_version: str, timestamp: str}`
- `should_activate_ppo_sidecar()` は `cycle_gate_aggregator.py` 内で定義

### Task 3: Skip Gate の OB Features 有効化テスト

`ztb/ml/skip_gate.py` の `use_ob_features` が `true` の場合の動作テストを追加。

`tests/unit/v460/test_enricher_skip_gate.py` に:

```python
class TestSkipGateOBFeatures:
    def test_ob_features_included_when_enabled(self):
        """use_ob_features=True → 特徴量に spread_bps_ob, depth_imbalance_ob, side_aligned_imbalance が含まれる"""
    
    def test_ob_features_excluded_when_disabled(self):
        """use_ob_features=False → OB 特徴量は含まれない (現行デフォルト)"""
    
    def test_ob_features_nan_handling(self):
        """OB 特徴量が NaN の場合 → SimpleImputer で処理される"""
```

### Task 4: 既知テスト問題の修正

以下の既知の問題を調査・修正:

1. `tests/training/unified_trainer/test_algorithms.py` — `test_execute_ppo_training_smoke_uses_current_env_path` が distributed training で `total_timesteps` が半減する件の修正 (assert を `>=` に変更するか、distributed detection を追加)

2. `tests/unit/v460/test_sac_retrain_scheduler.py::TestCrashResilience495::test_post_cycle_memory_check_runs` — 稀にタイミング依存で失敗するフレーキーテスト。sleep/timeout に余裕を持たせる

## 制約

- `git commit --no-verify -m "..."` でコミット
- `git add .` 禁止。変更ファイルを個別指定
- テスト: `python -m pytest tests/ -x --tb=short` — 既存テスト (2261+ passing) を壊さない
- 型安全: Any 型は最小限
- `_sb3_test_stub/` のスタブを活用して SB3 import を回避
- **doc 番号は使用しない** (本セッション側で採番する)
- `scripts/v460/ml/sac_retrain_scheduler.py` — 直接変更しない (SAC 側は別管理)

## 成果物

1. `tests/unit/v460/test_680_ppo_retrain_scheduler.py` に追加テスト 8+ 件
2. `tests/unit/v460/test_cycle_gate_aggregator.py` (新規) に PPO gate テスト 7+ 件
3. `tests/unit/v460/test_enricher_skip_gate.py` に OB features テスト 3 件
4. 既知フレーキーテスト 2 件の修正
5. コミットメッセージに修正内容を明記

## 参考: SAC テストの構造 (test_sac_retrain_scheduler.py)

```
TestSACRetrainConfig       — 9 tests  (defaults, from_yaml_dict, validation)
TestSACRetrainTrigger      — 8 tests  (first_run, interval, data_unchanged, backoff, time_forced)
TestRetrainResult          — 1 test   (to_dict)
TestRetrainOnce            — 7 tests  (cold/warm, oos_fail, data_error, debug)
TestAtomicDeploy           — 1 test
TestUpdateSidecarSignal    — 1 test
TestEvaluateModel          — 2 tests
TestAppendHistory          — 1 test
TestLoadConfig             — 2 tests
TestRunScheduler           — 1 test   (single_iteration_then_shutdown)
TestPushNeutralFallback    — 2 tests
TestReadSidecarCache       — 3 tests
TestCrashResilience495     — 7 tests
TestDataFreshnessDecoupling649 — 7 tests
```

PPO 側もこの構造に匹敵するカバレッジを目指す。
