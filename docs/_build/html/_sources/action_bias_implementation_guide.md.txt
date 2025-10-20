# Action Bias Fix - Detailed Implementation Guide

## Overview

This document provides specific file paths, function names, and code modifications needed to implement all fixes for the action bias issue.

## 1. Diagnostics Integration

### Files to Modify

#### `ztb/training/unified_trainer.py`

Add diagnostic logging to training loop:

```python
# At top of file, add import
from ztb.utils.diagnostics import ActionDiagnostics

# In UnifiedTrainer.__init__(), add:
self.diagnostics = ActionDiagnostics(save_dir="plots/diagnostics")

# In training loop (after policy update), add:
if self.diagnostics and step % 100 == 0:
    # Get logits and masks from last batch
    logits_raw = ...  # From policy network
    action_masks = ...  # From environment
    # Log diagnostics
    self.diagnostics.log_batch_diagnostics(
        step=step,
        logits_raw=logits_raw,
        logits_masked=masked_logits,
        action_masks=action_masks,
        probs_before_temp=probs_no_temp,
        probs_after_temp=probs_with_temp,
        temperature=1.0,  # or configured value
        actions_selected=actions,
        advantages=advantages,
        entropy=entropy,
        approx_kl=approx_kl,
        value_loss=value_loss,
        policy_loss=policy_loss,
        phase="train",
    )
```

## 2. Training-Time Action Mask Application

### Critical Fix: Apply Masks During Training

#### `ztb/training/environment/environment.py` (Already has get_legal_actions)

Current implementation:
```python
def get_legal_actions(self) -> NDArray[np.int_]:
    """Return binary mask of legal actions [HOLD, BUY, SELL]."""
    mask = np.ones(3, dtype=np.int_)
    
    # HOLD is always legal
    # BUY only legal if position <= 0 (no long position or flat)
    # SELL only legal if position >= 0 (have long position or flat)
    
    if self.position > 0:
        mask[1] = 0  # Can't BUY when holding long
    if self.position < 0:
        mask[2] = 0  # Can't SELL when holding short
    if self.position == 0:
        # When flat, both BUY and SELL are legal
        pass
    
    return mask
```

**This is CORRECT** - masks are state-dependent.

### Create Custom PPO Policy with Mask Application

#### New File: `ztb/training/policies/masked_policy.py`

```python
"""Custom PPO policy that applies action masks during forward pass."""

import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from typing import Tuple, Optional


class StrictMaskedPolicy(MaskableActorCriticPolicy):
    """Policy that strictly enforces action masks in forward and loss."""
    
    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with strict mask enforcement.
        
        Returns:
            actions, values, log_probs
        """
        # Get features
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Get action logits
        logits = self.action_net(latent_pi)
        
        # CRITICAL: Apply mask BEFORE softmax
        if action_masks is not None:
            # Set illegal actions to -inf
            logits = torch.where(
                action_masks.bool(),
                logits,
                torch.tensor(float('-inf'), device=logits.device)
            )
        
        # Create distribution and sample
        distribution = torch.distributions.Categorical(logits=logits)
        
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = distribution.sample()
        
        log_probs = distribution.log_prob(actions)
        
        # Get value
        values = self.value_net(latent_vf)
        
        return actions, values, log_probs
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions with mask consideration."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        logits = self.action_net(latent_pi)
        
        # Apply mask
        if action_masks is not None:
            logits = torch.where(
                action_masks.bool(),
                logits,
                torch.tensor(float('-inf'), device=logits.device)
            )
        
        distribution = torch.distributions.Categorical(logits=logits)
        
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(latent_vf)
        
        return values, log_probs, entropy
```

#### Modify `ztb/training/unified_trainer.py`

```python
# Import custom policy
from ztb.training.policies.masked_policy import StrictMaskedPolicy

# In _train_ppo() method:
model = MaskablePPO(
    policy=StrictMaskedPolicy,  # Use custom policy
    env=train_env,
    # ... other params
)
```

## 3. Deterministic Decoding Order Fix

### Modify Prediction in `paper_trade.py`

#### File: `ztb/training/paper_trade.py`

Current code:
```python
action, _ = cast(MaskablePPO, self.model).predict(
    predict_obs, action_masks=action_masks, deterministic=False
)
```

Add temperature-based evaluation:
```python
def _get_action_with_temperature(
    self,
    obs: np.ndarray,
    action_masks: np.ndarray,
    temperature: float = 0.7,
    deterministic: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    """Get action with temperature scaling."""
    # Get logits from policy
    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs).float()
        features = self.model.policy.extract_features(obs_tensor)
        latent_pi, _ = self.model.policy.mlp_extractor(features)
        logits = self.model.policy.action_net(latent_pi)
        
        # Apply mask
        mask_tensor = torch.from_numpy(action_masks).float()
        logits_masked = torch.where(
            mask_tensor.bool(),
            logits,
            torch.tensor(float('-inf'))
        )
        
        # Apply temperature
        logits_temp = logits_masked / temperature
        
        # Softmax
        probs = torch.softmax(logits_temp, dim=-1)
        
        # Select action
        if deterministic:
            action = torch.argmax(probs, dim=-1).item()
        else:
            action = torch.multinomial(probs, 1).item()
        
        # Return diagnostics
        diagnostics = {
            "probs": probs.numpy(),
            "logits_raw": logits.numpy(),
            "logits_masked": logits_masked.numpy(),
            "temperature": temperature,
        }
        
        return action, diagnostics
```

Use in simulation:
```python
# In _simulate_episode():
action, diag = self._get_action_with_temperature(
    predict_obs,
    action_masks,
    temperature=0.7,  # Soft-greedy
    deterministic=True,
)
```

## 4. Action Imbalance Correction

### Inverse-Frequency Loss Weighting

#### New File: `ztb/training/utils/action_weighting.py`

```python
"""Action frequency-based loss weighting."""

import torch
from collections import Counter
from typing import Dict


class ActionFrequencyWeighter:
    """Compute inverse-frequency weights for actions."""
    
    def __init__(self, n_actions: int = 3, min_weight: float = 0.5):
        self.n_actions = n_actions
        self.min_weight = min_weight
        self.action_counts = Counter()
    
    def update(self, actions: torch.Tensor) -> None:
        """Update action counts."""
        for action in actions.cpu().numpy():
            self.action_counts[int(action)] += 1
    
    def get_weights(self, actions: torch.Tensor) -> torch.Tensor:
        """Get weights for actions in batch."""
        total = sum(self.action_counts.values())
        if total == 0:
            return torch.ones_like(actions, dtype=torch.float32)
        
        weights = []
        for action in actions:
            action_int = int(action.item())
            freq = self.action_counts.get(action_int, 1) / total
            weight = max(self.min_weight, 1.0 / (freq + 1e-8))
            weights.append(weight)
        
        return torch.tensor(weights, device=actions.device)
```

#### Modify PPO Loss Calculation

In custom PPO implementation or callback:

```python
# In loss calculation:
action_weighter = ActionFrequencyWeighter()

# During training:
action_weighter.update(actions)
weights = action_weighter.get_weights(actions)

# Apply to policy loss
policy_loss = (weights * ratio_clipped).mean()
```

### Regime-Stratified Sampling

#### New File: `ztb/training/data/regime_sampler.py`

```python
"""Regime-balanced data sampling."""

import numpy as np
import pandas as pd
from typing import List, Dict


class RegimeSampler:
    """Sample data balanced across market regimes."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.regime_indices = self._identify_regimes()
    
    def _identify_regimes(self) -> Dict[str, List[int]]:
        """Identify different market regimes."""
        regimes = {}
        
        # Calculate returns
        returns = self.df['close'].pct_change()
        
        # Calculate volatility (rolling std)
        volatility = returns.rolling(20).std()
        
        # Trend: SMA crossover
        sma_short = self.df['close'].rolling(20).mean()
        sma_long = self.df['close'].rolling(50).mean()
        trend = sma_short > sma_long
        
        # High/Low volatility threshold
        vol_median = volatility.median()
        
        # Categorize
        regimes['uptrend_high_vol'] = self.df[(trend) & (volatility > vol_median)].index.tolist()
        regimes['uptrend_low_vol'] = self.df[(trend) & (volatility <= vol_median)].index.tolist()
        regimes['downtrend_high_vol'] = self.df[(~trend) & (volatility > vol_median)].index.tolist()
        regimes['downtrend_low_vol'] = self.df[(~trend) & (volatility <= vol_median)].index.tolist()
        
        return regimes
    
    def sample_balanced(self, n_samples: int) -> List[int]:
        """Sample indices balanced across regimes."""
        samples_per_regime = n_samples // len(self.regime_indices)
        
        all_indices = []
        for regime_name, indices in self.regime_indices.items():
            if len(indices) > 0:
                sampled = np.random.choice(
                    indices,
                    size=min(samples_per_regime, len(indices)),
                    replace=False
                )
                all_indices.extend(sampled)
        
        # Shuffle
        np.random.shuffle(all_indices)
        return all_indices[:n_samples]
```

## 5. Policy Head Bias Re-initialization

#### File: `ztb/training/unified_trainer.py`

After creating model, before training:

```python
# Re-initialize final layer bias to 0
if hasattr(model.policy, 'action_net'):
    with torch.no_grad():
        if hasattr(model.policy.action_net, 'bias') and model.policy.action_net.bias is not None:
            model.policy.action_net.bias.zero_()
            print("Re-initialized policy head bias to 0")
```

## 6. Hyperparameter Adjustments

### Add to Config

```json
{
  "target_kl": 0.02,
  "ent_coef": 0.6,
  "ent_coef_final": 0.2,
  "ent_coef_schedule": "cosine",
  "clip_range": 0.1,
  "gae_lambda": 0.9,
  "vf_coef": 0.3,
  "temperature_eval": 0.7
}
```

### Implement Entropy Coefficient Decay

```python
def cosine_decay(initial: float, final: float, step: int, total_steps: int) -> float:
    """Cosine annealing schedule."""
    progress = min(step / total_steps, 1.0)
    return final + (initial - final) * 0.5 * (1 + np.cos(np.pi * progress))

# In training loop:
current_ent_coef = cosine_decay(
    initial=0.6,
    final=0.2,
    step=current_step,
    total_steps=total_timesteps
)
model.ent_coef = current_ent_coef
```

## 7. Normalization Statistics Fix

### Save Scaler During Training

#### File: `ztb/trading/environment/environment.py`

```python
# After feature computation:
import joblib

if self.scaler is not None:
    scaler_path = Path("checkpoints") / "feature_scaler.pkl"
    joblib.dump(self.scaler, scaler_path)
```

### Load Scaler During Evaluation

#### File: `ztb/training/paper_trade.py`

```python
# In _create_env():
import joblib

scaler_path = Path("checkpoints") / "feature_scaler.pkl"
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
    config["scaler"] = scaler
    print(f"Loaded scaler from {scaler_path}")
```

## Summary Checklist

- [ ] Integrate ActionDiagnostics into training loop
- [ ] Create StrictMaskedPolicy for training-time masking
- [ ] Add temperature-based evaluation to paper_trade.py
- [ ] Implement ActionFrequencyWeighter
- [ ] Implement RegimeSampler
- [ ] Re-initialize policy head bias
- [ ] Add entropy coefficient cosine decay
- [ ] Adjust hyperparameters (target_kl, clip_range, gae_lambda, vf_coef)
- [ ] Save/load scaler for normalization
- [ ] Run forced action tests
- [ ] Run 50k × 3 seed validation
- [ ] Document results in fix_sell_bias.md

## Testing Commands

```bash
# Run forced action tests
python -m pytest tests/unit/environment/test_forced_actions.py -v

# Run short training with diagnostics
python -m ztb.training.unified_trainer --config ppo_50k_diagnostic_config.json

# Check diagnostic output
ls plots/diagnostics/

# Run paper trading with temperature
python -m ztb.training.paper_trade \
    --model-path models/ppo_50k_diagnostic.zip \
    --test-data btc_jpy_yahoo_real_dataset.csv \
    --episodes 3 \
    --config ppo_50k_diagnostic_config.json \
    --verbose
```
