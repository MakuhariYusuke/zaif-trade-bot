# 報酬関数の根本的問題点と再設計提案

## 🚨 現在の報酬関数の問題点

### 1. **報酬スケールの不整合**

#### 問題点
```python
# 利益ボーナス
base_profit_bonus = max(0.0, 1.5 * atr_normalised + 1.2 * portfolio_return)

# アクションペナルティ
action_penalty = 0.015  # BUY/SELL
action_penalty = 0.01 ~ 0.05  # HOLD
```

**分析**:
- `atr_normalised = pnl / atr` → ATRが小さいと爆発的に大きくなる
- `portfolio_return = pnl / initial_portfolio_value` → 初期値200000で割るため非常に小さい
- 結果: 報酬が-10 ~ +100の範囲で変動 → Q値が爆発

#### 証拠
- Critic Loss: 1e8 ~ 1e10 (異常に大きい)
- Actor Loss: 1e6 (報酬の分散が大きすぎる)

### 2. **報酬の方向性が矛盾**

```python
# 利益が出たら大きなボーナス
profit_bonus = base_profit_bonus * multipliers[action] * trend_multiplier
# 範囲: 0 ~ 100+

# 損失時は小さなペナルティ
loss_penalty = -0.2 * abs(atr_normalised)
# 範囲: 0 ~ -5

# 取引コストは無視
action_penalty = 0.015  # 固定値
```

**問題**:
- 利益に対する報酬 >> 損失に対するペナルティ
- エージェントが「大きな利益を夢見て無謀な取引」を学習
- 実際の取引コスト（0.0005 = 0.05%）が報酬に反映されていない

### 3. **HOLD行動への過剰なペナルティ**

```python
if action == ACTION_HOLD:
    position_size_factor = abs(position) / max_position_size
    volatility_factor = min(atr / (current_price * 0.01), 1.0)
    action_penalty = 0.01 + (0.04 * position_size_factor * volatility_factor)
    # 範囲: 0.01 ~ 0.05
```

**問題**:
- ポジションを持っている時のHOLDが常にペナルティ
- 「待つべき時に待つ」ことを学習できない
- 結果: 無意味な売買を繰り返す

### 4. **エントロピーが高すぎる原因**

現在の報酬関数では:
1. 報酬の分散が非常に大きい（-10 ~ +100）
2. 最適な行動が不明確（ノイズが多い）
3. SACが「不確実性が高い」と判断
4. エントロピーを高めて探索を続ける
5. ent_coef が上昇し続ける

## 💡 解決策: シンプルで安定した報酬関数

### 設計原則

1. **報酬を[-1, 1]の範囲に正規化**
2. **PnL（損益）を中心に設計**
3. **取引コストを明示的に考慮**
4. **HOLDを中立に扱う**
5. **不要な複雑性を排除**

### 新しい報酬関数 v2.0

```python
def calculate_reward_v2(
    self,
    action: int,
    pnl: float,
    transaction_cost: float,
    portfolio_value: float,
    position: float,
    old_position: float,
) -> float:
    """
    シンプルで安定した報酬関数 v2.0
    
    設計原則:
    1. PnL（損益）をベースとする
    2. 取引コストを明示的に考慮
    3. 報酬を[-1, 1]に正規化
    4. HOLDは中立（ペナルティなし）
    
    Args:
        action: 行動 (0=HOLD, 1=BUY, 2=SELL)
        pnl: 損益（円建て）
        transaction_cost: 取引コスト（割合）
        portfolio_value: ポートフォリオ価値
        position: 現在のポジション
        old_position: 前回のポジション
        
    Returns:
        正規化された報酬 [-1, 1]
    """
    
    # 1. PnLを正規化（ポートフォリオ価値の割合）
    # 例: pnl=1000, portfolio_value=200000 → pnl_ratio=0.005 (0.5%)
    pnl_ratio = pnl / max(portfolio_value, 1.0)
    
    # 2. 取引が発生したか確認
    position_changed = (position != old_position)
    
    # 3. 取引コストを計算（取引時のみ）
    if position_changed:
        # 取引額 = |新ポジション - 旧ポジション| * 現在価格
        # 簡略化: transaction_cost は既に考慮済みの pnl を使用
        # pnl には既に取引コストが引かれている
        cost_penalty = 0.0  # pnl に含まれているため追加ペナルティなし
    else:
        cost_penalty = 0.0
    
    # 4. ベース報酬 = PnL比率
    base_reward = pnl_ratio
    
    # 5. スケーリング: [-1, 1] の範囲に収める
    # 想定: 1ステップでの利益/損失は ±0.5% 程度
    # pnl_ratio を 10倍してクリッピング
    reward_scale = 10.0  # 0.1% の利益 → 報酬 1.0
    scaled_reward = base_reward * reward_scale
    
    # 6. クリッピング: [-1, 1]
    clipped_reward = max(-1.0, min(1.0, scaled_reward))
    
    return clipped_reward
```

### さらにシンプルな版（推奨）

```python
def calculate_reward_simple(
    self,
    pnl: float,
    portfolio_value: float,
) -> float:
    """
    最もシンプルな報酬関数
    
    報酬 = PnL / ポートフォリオ価値 * スケール
    
    これにより:
    - 利益 → 正の報酬
    - 損失 → 負の報酬
    - HOLD（ポジションなし） → 報酬 ≈ 0
    - HOLD（ポジションあり） → 価格変動に応じた報酬
    """
    # PnL比率を計算
    pnl_ratio = pnl / max(portfolio_value, 1.0)
    
    # スケーリング: 0.1%の利益で報酬1.0
    reward_scale = 1000.0
    reward = pnl_ratio * reward_scale
    
    # クリッピング: [-10, 10]
    # より大きな範囲で、大きな利益/損失も反映
    reward = max(-10.0, min(10.0, reward))
    
    return reward
```

## 📊 期待される効果

### 1. Critic Loss の安定化
- 報酬範囲: [-10, 10] (現在: -10 ~ +100)
- Q値の推定が容易になる
- Critic Loss: < 1e6 (現在: 1e8 ~ 1e10)

### 2. エントロピーの安定化
- 報酬のノイズが減少
- 最適な行動が明確になる
- ent_coef が 0.5 ~ 1.5 で安定

### 3. アクション分布の改善
- HOLDペナルティなし → 自然な行動選択
- 利益が出る時だけ取引 → 取引頻度が適切化
- HOLD比率: 60-70% (無駄な取引が減る)

### 4. 学習の高速化
- シンプルな報酬 → 学習が速い
- 5k timesteps で基本方策を獲得
- 50k ~ 100k で最適化

## 🔬 実装計画

### Phase 1: Simple Reward (v395f)
```json
{
  "reward_version": "simple_v2",
  "reward_scale": 1000.0,
  "reward_clip_min": -10.0,
  "reward_clip_max": 10.0
}
```

### Phase 2: 比較評価
- v395f (simple reward) vs v395a (complex reward)
- 5k timesteps で評価
- メトリクス:
  - Critic Loss
  - ent_coef
  - HOLD比率
  - 総収益

### Phase 3: 長期訓練
- 最良設定で 50k → 100k
- PPOとの比較

## 📝 実装例

### reward_calculator.py への追加

```python
def calculate_reward_simple(
    self,
    pnl: float,
    portfolio_value: float,
) -> float:
    """
    Simple PnL-based reward function.
    
    Reward = (PnL / Portfolio Value) * Scale
    Clipped to [-10, 10]
    
    This eliminates:
    - Complex penalty calculations
    - Action-specific bonuses
    - Position penalties
    - Diversity bonuses
    
    Focus purely on profit/loss.
    """
    # Normalize PnL by portfolio value
    pnl_ratio = pnl / max(portfolio_value, 1.0)
    
    # Scale: 0.1% profit = reward 1.0
    reward_scale = self.get_setting_float("reward_scale", 1000.0)
    reward = pnl_ratio * reward_scale
    
    # Clip to prevent extreme values
    clip_min = self.get_setting_float("reward_clip_min", -10.0)
    clip_max = self.get_setting_float("reward_clip_max", 10.0)
    reward = max(clip_min, min(clip_max, reward))
    
    # Optional: Small penalty for inactivity (very small)
    # This prevents pure HOLD strategy
    if self.get_setting_bool("enable_inactivity_penalty", False):
        inactivity_penalty = self.get_setting_float("inactivity_penalty", 0.001)
        reward -= inactivity_penalty
    
    return reward
```

## 🎯 次のステップ

1. ✅ `reward_calculator.py` に `calculate_reward_simple()` を追加
2. ✅ 新しい報酬設定を持つ `sac_v395f_simple_reward.json` を作成
3. ✅ 5k timesteps で訓練
4. ✅ ログ分析:
   - Critic Loss < 1e6 を確認
   - ent_coef < 2.0 を確認
   - HOLD比率 60-70% を確認
5. ✅ 成功したら 10k → 50k → 100k

---

**重要**: この変更は**根本的**です。複雑な報酬関数を完全に置き換えます。
まず5kで動作確認してから、長期訓練に進みましょう。
