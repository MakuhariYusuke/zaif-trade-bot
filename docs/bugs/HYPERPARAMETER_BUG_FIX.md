# ハイパーパラメータ適用バグ修正レポート

## 📅 実施日時
2025-10-11

## 🐛 発見されたバグ

### 問題の概要
v392訓練時、設定ファイルで指定したハイパーパラメータ（learning_rate等）が無視され、デフォルト値が使用されていた。

### 具体例
**設定ファイル（v392）**:
```json
{
  "ppo_hyperparameters": {
    "learning_rate": 0.007503,  // 二分探索最適化値（25倍）
    "batch_size": 256,          // 2倍
    "n_steps": 1024,            // 2倍
    "n_epochs": 16,             // 2.67倍
    "gamma": 0.8475,
    "max_grad_norm": 5.05       // 10倍
  },
  "lagrange_constraint": {
    "enabled": true,
    "r_target": 0.175,
    "tolerance": 0.042625,
    "eta": 0.062875
  },
  "environment": {
    "initial_balance": 200000,
    "max_position_size": 0.01,
    "transaction_cost": 0.0005
  }
}
```

**実際に使用された値**:
```
learning_rate: 0.0003  （デフォルト、設定の1/25）
batch_size: 32         （デフォルト、設定の1/8）
n_steps: 1024          （たまたま一致）
gamma: 0.99            （デフォルト）
max_grad_norm: 0.5     （デフォルト、設定の1/10）
```

### 影響
- ✅ 訓練自体は完了したが、最適化されたハイパーパラメータが適用されなかった
- ❌ 二分探索で得た+117pt改善が適用されなかった
- ❌ Learning rate が1/25なので学習速度が大幅に遅い

## 🔍 原因分析

### 根本原因
unified_trainer.pyの`get_ppo_core_config()`等のメソッドが、**トップレベルのキーのみ**を確認し、**階層化されたキー（ppo_hyperparameters、lagrange_constraint、environment）を見ていなかった**。

### 問題のコード（修正前）
```python
def get_ppo_core_config(self) -> PPOCoreConfig:
    return {
        "learning_rate": self.config.get("learning_rate", DEFAULT_PPO_CONFIG.get("learning_rate", 3e-4)),
        # ↑ トップレベルの "learning_rate" しか見ていない
        # v392設定ファイルでは "ppo_hyperparameters": {"learning_rate": 0.007503} となっている
    }
```

### 設定ファイルの2つのパターン

#### パターン1: トップレベル（古い形式）
```json
{
  "learning_rate": 0.0003,
  "batch_size": 32,
  ...
}
```

#### パターン2: 階層化（v392等の新形式）
```json
{
  "ppo_hyperparameters": {
    "learning_rate": 0.007503,
    "batch_size": 256,
    ...
  }
}
```

## ✅ 修正内容

### 1. get_ppo_core_config() 修正

**修正後**:
```python
def get_ppo_core_config(self) -> PPOCoreConfig:
    from ztb.training.config.ppo_config import DEFAULT_PPO_CONFIG
    
    # 🔧 FIX: ppo_hyperparametersキーもチェック（v392等で使用）
    ppo_hyperparams = self.config.get("ppo_hyperparameters", {})
    
    def get_param(key: str, default: Any = None) -> Any:
        """トップレベルとppo_hyperparametersの両方をチェック"""
        # トップレベルを優先、次にppo_hyperparameters、最後にデフォルト
        return self.config.get(key, ppo_hyperparams.get(key, DEFAULT_PPO_CONFIG.get(key, default)))
    
    return {
        "learning_rate": get_param("learning_rate", 3e-4),
        "n_steps": get_param("n_steps", 1024),
        "batch_size": get_param("batch_size", 32),
        "n_epochs": get_param("n_epochs", 10),
        "gamma": get_param("gamma", 0.99),
        "gae_lambda": get_param("gae_lambda", 0.95),
        "clip_range": get_param("clip_range", 0.2),
        "clip_range_vf": get_param("clip_range_vf"),
        "normalize_advantage": get_param("normalize_advantage", True),
        "ent_coef": get_param("ent_coef", 0.0),
        "vf_coef": get_param("vf_coef", 0.5),
        "max_grad_norm": get_param("max_grad_norm", 0.5),
        "use_sde": get_param("use_sde", False),
        "sde_sample_freq": get_param("sde_sample_freq", -1),
        "target_kl": get_param("target_kl"),
        "verbose": get_param("verbose", 1),
    }
```

**動作**:
1. トップレベルのキー（例: `learning_rate`）を確認
2. なければ `ppo_hyperparameters.learning_rate` を確認
3. なければデフォルト値を使用

### 2. get_environment_config() 修正

**修正後**:
```python
def get_environment_config(self) -> EnvironmentConfig:
    from ztb.training.config.ppo_config import DEFAULT_PPO_CONFIG
    
    # 🔧 FIX: environmentキーもチェック（v392等で使用）
    env_config = self.config.get("environment", {})
    
    def get_param(key: str, default: Any = None) -> Any:
        """トップレベルとenvironmentの両方をチェック"""
        return self.config.get(key, env_config.get(key, DEFAULT_PPO_CONFIG.get(key, default)))
    
    return {
        "max_position_size": get_param("max_position_size", 1.0),
        "initial_balance": get_param("initial_balance", 1000000),
        "transaction_cost": get_param("transaction_cost", 0.001),
        "reward_scaling": get_param("reward_scaling", 1.0),
    }
```

### 3. Lagrange制約パラメータ修正

**修正後**:
```python
# Build Lagrange parameters dict from config
# 🔧 FIX: lagrange_constraintキーもチェック（v392等で使用）
lagrange_config = self.config.get("lagrange_constraint", {})

def get_lagrange_param(key: str, default: Any = None) -> Any:
    """トップレベル（lagrange_プレフィックス）とlagrange_constraintの両方をチェック"""
    prefixed_key = f"lagrange_{key}"
    return self.config.get(prefixed_key, lagrange_config.get(key, LAGRANGE_DEFAULTS.get(key, default)))

lagrange_params = {}
# enable_lagrangeは特別扱い（プレフィックスなしとlagrange_constraint.enabledの両方をチェック）
enable_lagrange = self.config.get("enable_lagrange", lagrange_config.get("enabled", True))

if enable_lagrange:
    lagrange_params = {
        "r_target": get_lagrange_param("r_target"),
        "tolerance": get_lagrange_param("tolerance"),
        "eta": get_lagrange_param("eta"),
        "lambda_max": get_lagrange_param("lambda_max"),
        "warmup_steps": get_lagrange_param("warmup_steps"),
    }
```

## 📊 修正の影響範囲

### 修正したファイル
- `ztb/training/unified_trainer.py`:
  - `get_ppo_core_config()` (Lines 359-391)
  - `get_environment_config()` (Lines 336-358)
  - Lagrange parameter loading (Lines 645-668)

### 後方互換性
✅ **完全に後方互換**

**古い形式（トップレベル）**:
```json
{
  "learning_rate": 0.0003
}
```
→ トップレベルが優先されるので問題なし

**新しい形式（階層化）**:
```json
{
  "ppo_hyperparameters": {
    "learning_rate": 0.007503
  }
}
```
→ ppo_hyperparametersから読み込まれる

**混在**:
```json
{
  "learning_rate": 0.001,  // ← こちらが優先
  "ppo_hyperparameters": {
    "learning_rate": 0.007503
  }
}
```
→ トップレベルが優先（オーバーライド可能）

## 🎯 検証方法

### v392設定で再訓練
1. 修正後のunified_trainer.pyでv392を再訓練
2. ログで以下を確認：
   ```
   learning_rate: 0.007503  （✅ 設定値）
   batch_size: 256         （✅ 設定値）
   max_grad_norm: 5.05     （✅ 設定値）
   ```

### 簡易テスト
```python
from ztb.training.unified_trainer import UnifiedTrainer
import json

with open('configs/ppo_profitable_v392_bugfix.json') as f:
    config = json.load(f)

trainer = UnifiedTrainer(config)
ppo_config = trainer.get_ppo_core_config()

print(f"Learning rate: {ppo_config['learning_rate']}")  # 0.007503を期待
print(f"Batch size: {ppo_config['batch_size']}")        # 256を期待
print(f"Max grad norm: {ppo_config['max_grad_norm']}")  # 5.05を期待
```

## 📝 横展開調査結果

### ✅ 問題なし
- `checkpoint_interval`: トップレベルから直接読み取り
- `total_timesteps`: トップレベルから直接読み取り
- `data_path`: トップレベルから直接読み取り
- `model_name`: トップレベルから直接読み取り

### ⚠️ 将来的な懸念
以下のキーは現在トップレベルのみ対応だが、将来階層化される可能性がある：
- `enable_pan`
- `enable_target_entropy`
- `enable_stratified_sampling`
- `allow_reverse`

**推奨**: これらも同様のget_param()パターンで統一すべき

## 🎯 次のステップ

### 1. v393訓練（優先度：高）
- 修正後のunified_trainer.pyを使用
- v392設定をベースに再訓練
- ハイパーパラメータが正しく適用されることを確認

### 2. 設定ファイル標準化（優先度：中）
階層化形式を推奨形式として文書化：
```json
{
  "model_name": "...",
  "total_timesteps": 100000,
  "ppo_hyperparameters": { ... },
  "environment": { ... },
  "lagrange_constraint": { ... }
}
```

### 3. テストケース追加（優先度：中）
unified_trainer_test.pyに以下を追加：
- 階層化設定の読み込みテスト
- トップレベル設定の読み込みテスト
- 混在設定の優先順位テスト

## 📌 まとめ

### 修正前
```
v392設定: learning_rate=0.007503
実際の訓練: learning_rate=0.0003（デフォルト）
❌ 設定が無視された（unified_trainer.py + ppo_trainer.py両方のバグ）
```

### 修正後
```
v393設定: learning_rate=0.007503
実際の訓練: learning_rate=0.0075（✅ 設定値）
✅ 正しく適用される
```

### 修正箇所
1. **unified_trainer.py** (3箇所)
   - `get_ppo_core_config()`: ppo_hyperparametersキー対応
   - `get_environment_config()`: environmentキー対応
   - Lagrange制約設定: lagrange_constraintキー対応

2. **ppo_trainer.py** (1箇所) ← **重要な追加修正**
   - `TrainingConfig.from_dict()`: ppo_hyperparametersキー対応
   - unified_trainer経由で渡されたconfigをTrainingConfigが正しく解釈

### 影響
- ✅ v393訓練で二分探索最適化値が正しく適用される
- ✅ 学習効率が大幅に向上（learning rate 25倍）
- ✅ より強力な最適化（max_grad_norm 10倍等）
- ✅ 後方互換性を維持

### 検証結果
**v393訓練ログより**:
```
2025-10-11 00:52:34,692 - INFO - Learning rate: 0.007503 ✅
2025-10-11 00:52:34,692 - INFO - Batch size: 256 ✅
2025-10-11 00:52:34,692 - INFO - Epochs per update: 16 ✅
2025-10-11 00:52:34,692 - INFO - Gamma: 0.8475 ✅
--------------------------------------------------------------------
| train/                    |                                      |
|    learning_rate          | 0.0075                               |
--------------------------------------------------------------------
```

**🎉 修正成功！ハイパーパラメータが正しく適用されています。**
