# SAC Entropy Tuning - Critical Insight

## 🚨 Root Cause Analysis

### Problem
すべてのバージョン（v395a-d）で`ent_coef`が訓練中に上昇し続ける：
- v395a: 1.09 → 4.03
- v395b: 1.03 → 1.53  
- v395c: 1.03 → 1.53
- v395d: 1.09 → 3.58

### Why target_entropy=-0.5 Alone is Not Enough

`ent_coef_loss`を見ると：
- 常に負の値（-0.8 → -12.6）
- 絶対値が増加している

これは **エントロピーが目標値より常に大きい** ことを意味します。

### The Entropy Auto-Tuning Mechanism

SACのエントロピー自動調整：
```python
ent_coef_loss = -log(ent_coef) * (entropy - target_entropy).detach()
```

- `entropy > target_entropy` の場合
  → `ent_coef_loss < 0`
  → `ent_coef`を増やす（探索を抑制）

しかし、`ent_coef`を増やすと探索が減るはずが、なぜかエントロピーが下がらない！

## 💡 Real Problem: Excessive Initial Entropy

### Hypothesis
連続行動空間（Box[-1, 1]）では、初期の方策がガウス分布で：
- mean = 0.0
- std = 1.0 (デフォルト)

この場合、連続分布のエントロピーは：
```
H = 0.5 * log(2πe * σ²)
  = 0.5 * log(2πe * 1.0)
  ≈ 1.42
```

しかし、我々の`target_entropy = -0.5`は**負の値**！

### Why Negative Target Entropy?

SB3のSACでは、離散行動空間の場合：
```python
target_entropy = -dim(action_space)  # e.g., -3 for 3 actions
```

連続行動空間では：
```python
target_entropy = -dim(action_space)  # -1 for dim=1
```

しかし、連続分布のエントロピーは通常**正の値**（ガウス分布 ≈ 1.42）

## 🎯 Solution

### Option 1: Adjust target_entropy to Positive
```json
{
  "target_entropy": 0.5  // ガウス分布のエントロピーより小さい値
}
```

### Option 2: Reduce Initial Policy Std
```json
{
  "policy_kwargs": {
    "log_std_init": -2.0,  // std = exp(-2.0) ≈ 0.135
    "net_arch": [256, 256]
  }
}
```

### Option 3: Disable Auto-Tuning, Use Fixed ent_coef
```json
{
  "ent_coef": 0.1  // 固定値を使用
}
```

## 📊 Recommended Approach: v395e

**Hybrid Approach**: 小さい正の`target_entropy` + 低めの`log_std_init`

```json
{
  "sac_hyperparameters": {
    "learning_rate": 0.0003,
    "buffer_size": 20000,
    "learning_starts": 500,
    "batch_size": 128,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",              // 自動調整を継続
    "target_update_interval": 1,
    "target_entropy": 0.3,           // ✅ 正の値に変更！
    "policy_kwargs": {
      "log_std_init": -1.0,          // ✅ std ≈ 0.37に制限
      "net_arch": [256, 256]
    }
  }
}
```

### Expected Behavior
- 初期policy std = exp(-1.0) ≈ 0.37
- 初期entropy ≈ 0.5 * log(2πe * 0.37²) ≈ 0.2
- target_entropy = 0.3 → entropy を少し増やす方向
- ent_coef は小さい値（0.5-1.5）で安定

## 📋 Alternative: v395f (Conservative)

固定`ent_coef`アプローチ：

```json
{
  "sac_hyperparameters": {
    "learning_rate": 0.0003,
    "buffer_size": 20000,
    "learning_starts": 500,
    "batch_size": 128,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": 0.2,                 // ✅ 固定値（PPO v394fと同じ）
    "target_update_interval": 1
  }
}
```

## 🔬 Testing Plan

1. **v395e**: target_entropy=0.3 + log_std_init=-1.0
2. **v395f**: ent_coef=0.2 (fixed)
3. **v395g**: target_entropy=0.5 + log_std_init=-2.0 (very conservative)

各バージョンで5k timesteps訓練し、最良の設定を特定。
