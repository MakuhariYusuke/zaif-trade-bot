# 386# 報酬関数分析レポート

## 概要

385# baseline (gamma=0.80) で Seeds 123/456 が OOS 負になる根本原因を分析。
報酬のペナルティコンポーネントが PnL 報酬に対して過大であることを発見。

## 報酬コンポーネント構成 (全パイプライン)

```
calculate_reward():
  # 1. balance_penalty (BUY/SELL 前に計算)
  balance_penalty = -raw_excess × balance_penalty_value  # 外側

  # 2. base_reward (stage method、デフォルト = _calculate_default_reward)
  base_reward = pnl_reward            # PnL × reward_scaling(1.0) × pnl_reward_multiplier(1.0)
             + position_penalty       # -position_penalty_weight(0.01) × (pos/max_pos)²
             + hold_penalty           # HOLD時 -hold_penalty_weight(0.01)
             + consistency_penalty    # 反転時 -consistency_penalty(0.05)

  # 3. 外側のペナルティ・ボーナス
  total = base_reward
        + confidence_penalty          # 損失×高確信: -(loss_mag × (|action| - threshold) × factor)
        + action_bonus                # BUY/SELL/HOLD ボーナス (通常 0)
        + balance_penalty             # -raw_excess × balance_penalty_value(1.0)
        + skew_penalty                # BUY/SELL 偏り罰
        + balance_shaping             # バランス改善ボーナス
        + entropy_bonus               # 行動多様性

  # 4. 後処理
  → asymmetric_scaler(×1.0 no-op) → clip[-80, 80]
```

### キー設定のアクセスパス

| 設定 | キー | アクセスパス | デフォルト |
|------|------|------------|-----------|
| balance_penalty | `behavior_optimization.balance_penalty` | `config.behavior_optimization` dict | **1.0** |
| balance_penalty_tolerance | `behavior_optimization.balance_penalty_tolerance` | `config.behavior_optimization` dict | 0.05 |
| consistency_penalty | `behavior_optimization.consistency_penalty` | `reward_settings.consistency_penalty` | **0.05** |
| hold_penalty_weight | `hold_penalty_weight` | `reward_settings.custom_reward_params` | **0.01** |
| position_penalty_weight | `position_penalty_weight` | `reward_settings.custom_reward_params` | 0.01 |
| confidence_penalty_threshold | `confidence_penalty_threshold` | `reward_settings.custom_reward_params` | 0.05 |

### YAML 設定マッピング (386# 修正後)

```yaml
environment:
  behavior_optimization:       # → EnvironmentConfig.behavior_optimization dict
    balance_penalty: 0.1       #   → reward_calculator.balance_penalty
    consistency_penalty: 0.01  #   → reward_settings.consistency_penalty
reward_settings:               # → RewardSettings.custom_reward_params
  hold_penalty_weight: 0.001   #   → get_setting_float("hold_penalty_weight")
  confidence_penalty_threshold: 0.2  # → _get_setting("confidence_penalty_threshold")
```

## 問題分析

### 🔴 P0-3: `balance_penalty_value=1.0` が PnL に対して過大

| 条件 | PnL reward | balance_penalty | 比率 |
|------|-----------|-----------------|------|
| 微小 PnL step (5 JPY) | +5.0 | -0.5 (10%偏り) | 10% |
| 中程度 PnL step (20 JPY) | +20.0 | -0.5 | 2.5% |
| PnL ≈ 0 step | +0.0 | -0.5 | **支配** |

**影響**: PnL が小さい step でエージェントが「PnL 最大化」より「バランス維持」を学習。
市場状況に関わらず BUY/SELL を均等に打とうとし、トレンド追従が阻害される。

### 🔴 P0-4: `hold_penalty=-0.01` が過剰取引を誘発

- Seed 456: 59K trades on OOS (50K training steps 以上) = ティック毎に複数取引
- hold_penalty は HOLD するたびに -0.01 のコスト
- PnL が 0-5 JPY の低ボラ期間では hold_penalty の回避が優先される
- 結果: marginal な取引が増え、cumulative で負の PnL

### 🟡 P1-4: `consistency_penalty=-0.05` がトレンド転換対応を阻害

- BUY→SELL 反転を罰するため、トレンド転換時の方向転換が遅れる
- Seed 123 の保守性 (22K trades) の一因

### Seed 別影響

| Seed | OOS ROI | 主要影響因子 |
|------|---------|------------|
| 42 | +4.37% | PnL が十分大きく、ペナルティの影響が相対的に小さい |
| 123 | -0.39% | consistency_penalty → 方向転換遅延 + 取引機会損失 |
| 456 | -0.44% | hold_penalty → 過剰取引 + balance_penalty → 不適切なバランス取引 |
| 789 | +2.00% | 42 と類似、強い PnL シグナルがペナルティを上回る |

## 推奨調整案 (387# 予定)

gamma=0.95 実験結果を待ち、G2 PASS 不達の場合に適用。
YAML: `configs/v460/experiments/g2_sac_gamma095_reward_tuned.yaml`

### 数値効果検証 (386# 実測)

| 対象 | 現行値 | 推奨値 | 削減率 | YAML セクション |
|------|--------|--------|--------|----------------|
| `balance_penalty` | 1.0 | **0.1** | 90%↓ | `environment.behavior_optimization` |
| `hold_penalty_weight` | 0.01 | **0.001** | 90%↓ | `reward_settings` |
| `consistency_penalty` | 0.05 | **0.01** | 80%↓ | `environment.behavior_optimization` |
| `confidence_penalty_threshold` | 0.05 | **0.2** | — | `reward_settings` |

### 386# P0-5: YAML→env 伝播バグ修正

reward-tuned YAML を準備する過程で発見した伝播バグ:
1. `reward_settings` がYAMLトップレベルにある場合、`sac_trainer.py` が消失させていた → **修正済み**
2. `behavior_optimization` dict が `EnvironmentConfig.from_dict()` で `instance.behavior_optimization` に保存されていなかった → **修正済み**
3. 当初の reward-tuned YAML に無効キー (`balance_penalty_value`, `consistency_penalty_value`) があった → **正しいキー名に修正済み**

| 対象 | 現行値 | 推奨値 | 根拠 |
|------|--------|--------|------|
| `balance_penalty_value` | 1.0 | **0.1** | PnL SNR 改善、バランス強制の緩和 |
| `hold_penalty_weight` | 0.01 | **0.001** | 過剰取引の抑制 |
| `consistency_penalty` | -0.05 | **-0.01** | トレンド転換対応の改善 |
| `confidence_penalty_threshold` | 0.05 | **0.2** | 過剰なconfidence 検出の緩和 |

### 科学的アプローチ

1. **386# (現在)**: gamma=0.95 のみ変更 → 報酬ペナルティの影響を切り分け
2. **387# (予定)**: gamma=0.95 + reward tuning → 複合効果の検証
3. **388# (予定)**: 最適 gamma + 最適 reward → G2 PASS 達成

## PnL スケーリングについて

`reward_scaling=5.0-10.0` にすると PnL 報酬が 5-10 倍になり、
ペナルティとの SNR が改善される。ただし SAC の entropy term α との
バランスに注意が必要。

current: PnL(5 JPY) vs balance_penalty(-0.5) → SNR = 10
scaled:  PnL(25 JPY) vs balance_penalty(-0.5) → SNR = 50

→ penalty 縮小 + PnL スケーリングの組み合わせが最も効果的。
