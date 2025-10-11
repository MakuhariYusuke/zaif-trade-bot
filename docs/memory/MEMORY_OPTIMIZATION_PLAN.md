# メモリ最適化計画

## 現状の問題

### 1. 学習中のメモリ不足
- 30,000ステップ目付近で学習が停止
- メモリ関連のクラッシュと推測

### 2. モデル学習の問題  
- **学習時**: SELL率 26.9% (rollout統計)
- **評価時**: SELL率 98.0% (決定的推論)
- **根本原因**: Lagrange制約が学習中のみ適用され、モデル自体は学習できていない

## メモリ最適化戦略

### Phase 1: 即座に実装可能な最適化

#### 1.1 Gradient Checkpointing
```python
# ztb/training/custom_ppo.py に追加
policy_kwargs = dict(
    net_arch=[256, 256],
    activation_fn=nn.Tanh,
    ortho_init=False,
    # Gradient checkpointing for memory efficiency
    use_gradient_checkpointing=True,
)
```

**効果**: 約30-40% メモリ削減
**トレードオフ**: 約20% 学習速度低下

#### 1.2 Batch Size削減
```json
// ppo_balanced_test.json
{
  "n_steps": 1024,     // 現状
  "batch_size": 64,    // 現状
  
  // 最適化後
  "n_steps": 512,      // 半分に削減
  "batch_size": 32,    // 半分に削減
}
```

**効果**: 約50% メモリ削減
**トレードオフ**: 学習安定性が若干低下、収束までの時間増加

#### 1.3 データローディング最適化
```python
# environment.py
class HeavyTradingEnv:
    def __init__(self, df, config, ...):
        # メモリ効率の良いデータ型に変換
        self.df = df.copy()
        
        # float64 → float32 変換
        float_cols = self.df.select_dtypes(include=['float64']).columns
        self.df[float_cols] = self.df[float_cols].astype('float32')
        
        # 不要なカラムを削除
        essential_cols = ['close', 'open', 'high', 'low', 'volume', ...]
        self.df = self.df[essential_cols]
```

**効果**: 約20-30% データメモリ削減

#### 1.4 環境ベクトル化の最適化
```python
# n_envs を削減
"n_envs": 1,  # 現状: 複数環境でメモリ消費増
```

**効果**: 約40% メモリ削減（並列環境数による）
**トレードオフ**: サンプリング効率低下

### Phase 2: 構造的最適化

#### 2.1 特徴量の動的計算
```python
# 全ての特徴量を事前計算せず、必要時に計算
class LazyFeatureCalculator:
    def __init__(self, df_base):
        self.df_base = df_base  # 基本データのみ保持
        self._cache = {}
        
    def get_features(self, step):
        if step not in self._cache:
            self._cache[step] = self._calculate_features(step)
            
            # キャッシュサイズ制限
            if len(self._cache) > 1000:
                oldest = min(self._cache.keys())
                del self._cache[oldest]
        
        return self._cache[step]
```

**効果**: 約50-60% メモリ削減（大規模データセット）

#### 2.2 チェックポイント圧縮
```python
# ztb/training/base_trainer.py
def save_checkpoint(self, path):
    # モデルを圧縮保存
    import zipfile
    
    temp_path = path + ".tmp"
    self.model.save(temp_path)
    
    # 圧縮
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(temp_path)
    
    os.remove(temp_path)
```

**効果**: ディスク容量50-70%削減、メモリへの影響は限定的

### Phase 3: アルゴリズム改善

#### 3.1 Policy Regularization (Action Diversity)
```python
# CustomPPO に追加
class CustomPPO(PPO):
    def train(self):
        # Action entropy regularization
        action_probs = self.policy.get_distribution(obs).distribution.probs
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(-1).mean()
        
        # Entropy bonus (diversity encouragement)
        policy_loss = policy_loss - 0.05 * entropy  # ent_coef を強化
```

**効果**: モデル自体が多様な行動を学習

#### 3.2 Behavior Cloning from Balanced Policy
```python
# 事前にバランスの取れた行動をサンプリング
# それらから模倣学習

from stable_baselines3.common.evaluation import evaluate_policy

# ステップ1: ランダムバランスポリシーでデータ収集
balanced_policy = BalancedRandomPolicy()  # 33/33/33 の確率
dataset = collect_demonstrations(env, balanced_policy, n_steps=10000)

# ステップ2: Behavior Cloning
bc_model = BC(policy, dataset)
bc_model.train(n_epochs=10)

# ステップ3: Fine-tuning with PPO
ppo_model = PPO.load_from_bc(bc_model)
ppo_model.learn(total_timesteps=100000)
```

**効果**: 初期からバランスの取れた行動を学習

## 実装優先順位

### 🚨 緊急 (即座に実装)
1. ✅ **Batch Size削減**: `n_steps=512, batch_size=32`
2. ✅ **データ型最適化**: float64 → float32
3. ✅ **n_envs削減**: 複数環境 → 1環境

### 📊 高優先度 (1-2日以内)
4. **Gradient Checkpointing**: メモリ30-40%削減
5. **Policy Regularization**: モデルの学習品質向上
6. **動的特徴量計算**: 大幅メモリ削減

### 🔧 中優先度 (1週間以内)
7. **Behavior Cloning**: バランスの取れた初期ポリシー
8. **チェックポイント圧縮**: ディスク容量削減
9. **環境プーリング**: メモリ再利用

## メモリ使用量の目標

### 現状 (推定)
- **データ**: ~500MB (float64, 全特徴量)
- **モデル**: ~200MB (ポリシー + バリュー)
- **Rollout Buffer**: ~800MB (n_steps=1024, n_envs=複数)
- **Gradient計算**: ~600MB (バックプロパゲーション)
- **合計**: ~2.1GB

### 最適化後 (目標)
- **データ**: ~250MB (float32, 必要な特徴量のみ)
- **モデル**: ~200MB (変更なし)
- **Rollout Buffer**: ~200MB (n_steps=512, n_envs=1)
- **Gradient計算**: ~360MB (checkpointing使用)
- **合計**: ~1.0GB (-52%削減)

## 検証計画

### メモリ使用量測定
```python
import tracemalloc
import psutil
import os

def monitor_memory_usage():
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    
    # 学習前
    mem_before = process.memory_info().rss / 1024**2
    
    # 学習実行
    model.learn(total_timesteps=10000)
    
    # 学習後
    mem_after = process.memory_info().rss / 1024**2
    current, peak = tracemalloc.get_traced_memory()
    
    print(f"メモリ使用量: {mem_after - mem_before:.2f} MB")
    print(f"ピークメモリ: {peak / 1024**2:.2f} MB")
    
    tracemalloc.stop()
```

### パフォーマンス測定
```python
import time

start_time = time.time()
model.learn(total_timesteps=10000)
elapsed = time.time() - start_time

print(f"学習時間: {elapsed:.2f}秒")
print(f"FPS: {10000 / elapsed:.2f}")
```

## 次のステップ

1. **即座の対応**: メモリ最適化設定で再学習
2. **検証**: メモリ使用量とパフォーマンスを測定
3. **改善**: Behavior Cloning or Policy Regularization実装
4. **再検証**: バランスの取れたモデルが学習できるか確認

---

## 補足: SELL Bias問題の根本原因

Lagrange制約は**学習プロセス**に適用されるもので、**学習済みモデル**には影響しません:

```python
# 学習時
rollout_actions = model.collect_rollouts()  # ← Lagrange制約が適用
sell_rate = calculate_sell_rate(rollout_actions)  # 26.9%

# 評価時  
action = model.predict(obs, deterministic=True)  # ← 制約なし、純粋な推論
# → モデルが実際に学習した行動: 98% SELL
```

**解決策**:
1. **Action diversity penalty**をlossに直接追加
2. **Behavior Cloning**でバランスの取れた初期ポリシー
3. **Custom reward shaping**で多様な行動に報酬

これらは**モデル自体を変える**ので、評価時にも効果が持続します。
