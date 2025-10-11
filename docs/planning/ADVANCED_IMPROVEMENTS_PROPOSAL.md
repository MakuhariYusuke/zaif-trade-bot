# AIエージェント提案: 高度な改善案の検討

**日時**: 2025年10月6日  
**目的**: RecurrentPPO/LSTM化、連続ポジションサイズ化、自己教師タスク併用の技術的検討

---

## エグゼクティブサマリー

外部AIエージェントから提案された3つの高度な改善案について、技術的実現可能性、期待効果、実装コストを分析しました。

| 提案 | 実現可能性 | 期待効果 | 実装コスト | 優先度 |
|------|-----------|---------|-----------|--------|
| 1. RecurrentPPO/LSTM化 | ⭐⭐⭐⭐⭐ | 🔥🔥🔥🔥 | 中 (2-3週間) | **高** |
| 2. 連続ポジションサイズ化 | ⭐⭐⭐⭐ | 🔥🔥🔥 | 中 (1-2週間) | 中 |
| 3. 自己教師タスク併用 | ⭐⭐⭐ | 🔥🔥🔥🔥🔥 | 高 (3-4週間) | **高** |

**推奨順序**:
1. **RecurrentPPO/LSTM化** → 時系列依存性を捉える (最も影響大)
2. **自己教師タスク併用** → 表現学習を強化 (長期的価値)
3. **連続ポジションサイズ化** → アクション空間拡張 (追加の柔軟性)

---

## 提案1: RecurrentPPO/LSTM化

### 概要

現在のMLP (Multi-Layer Perceptron) ポリシーをLSTM (Long Short-Term Memory) ベースのRecurrentPPOに置き換え、時系列依存性を明示的にモデル化します。

### 技術的詳細

**現状のMLP**:
```python
# 現在: 各ステップ独立
observation_t → MLP → action_t
observation_t+1 → MLP → action_t+1
```

**提案のLSTM**:
```python
# 提案: 隠れ状態で時系列情報を保持
(observation_t, hidden_t-1) → LSTM → (action_t, hidden_t)
(observation_t+1, hidden_t) → LSTM → (action_t+1, hidden_t+1)
```

### 実装アプローチ

#### Option A: sb3-contrib RecurrentPPO (推奨)

**利用可能性**:
```bash
# sb3-contrib v2.0+
from sb3_contrib import RecurrentPPO
```

**実装例**:
```python
from sb3_contrib import RecurrentPPO

model = RecurrentPPO(
    policy="MlpLstmPolicy",  # LSTMポリシー
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    # LSTM specific
    n_lstm_layers=1,
    lstm_hidden_size=256,
    enable_critic_lstm=True,  # Criticにもchars使用
    # ... 既存のPPOパラメータ ...
)
```

**利点**:
- ✅ Stable-Baselines3公式サポート
- ✅ MaskablePPOと同様のAPI
- ✅ 既存コードベースとの互換性高い
- ✅ テスト済み実装

**課題**:
- ❌ CustomPPOの改修と統合が必要
- ❌ エピソード長の管理が複雑化

#### Option B: Custom LSTM Policy

**実装例**:
```python
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomLSTMPolicy(ActorCriticPolicy):
    def __init__(self, *args, lstm_hidden_size=256, n_lstm_layers=1, **kwargs):
        super().__init__(*args, **kwargs)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.features_dim,
            hidden_size=lstm_hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
        )
        
        # Policy head (actor)
        self.action_net = nn.Linear(lstm_hidden_size, self.action_space.n)
        
        # Value head (critic)
        self.value_net = nn.Linear(lstm_hidden_size, 1)
        
    def forward(self, obs, lstm_states, episode_starts, deterministic=False):
        # LSTM forward pass
        lstm_out, lstm_states = self.lstm(obs, lstm_states)
        
        # Policy and value
        action_logits = self.action_net(lstm_out)
        values = self.value_net(lstm_out)
        
        return action_logits, values, lstm_states
```

**利点**:
- ✅ 完全なカスタマイズ可能
- ✅ CustomPPOと統合しやすい

**課題**:
- ❌ 実装コスト高い
- ❌ デバッグが困難
- ❌ テストカバレッジ必要

### 期待効果

1. **時系列パターン認識**:
   - トレンド検出の改善
   - モメンタム変化の捕捉
   - 過去N時点の文脈を考慮

2. **SELL率改善**:
   - 「買った後に売る」という時系列関係を学習
   - ポジション保持期間の最適化
   - エントリー/エグジットタイミングの改善

3. **収益性向上**:
   - より洗練された取引戦略
   - 市場レジーム変化への適応
   - リスク管理の向上

### 実装計画

**Phase 1: 調査 (3-5日)**
1. sb3-contrib RecurrentPPOのAPI調査
2. LSTM policy設定の検証
3. エピソード管理の理解

**Phase 2: 統合 (5-7日)**
1. CustomRecurrentPPOクラス作成
2. RecurrentPPO + PAN統合
3. RecurrentPPO + Target Entropy統合
4. LSTMState管理実装

**Phase 3: テスト (3-5日)**
1. 10kスモークテスト
2. 50k収束テスト
3. バックテスト検証

**総所要時間**: 2-3週間

### リスクと制約

**技術的リスク**:
- エピソード境界でのLSTM状態リセット必要
- メモリ使用量増加 (LSTM状態保存)
- 学習速度低下の可能性

**軽減策**:
- Truncated BPTT (Backpropagation Through Time) 使用
- LSTM hidden size調整 (256→128)
- Gradient clipping強化

---

## 提案2: 連続ポジションサイズ化

### 概要

現在の離散的な3アクション (BUY/HOLD/SELL) を、連続的なポジションサイズ (-1.0 ~ +1.0) に拡張します。

### 技術的詳細

**現状の離散アクション**:
```python
action_space = Discrete(3)
# 0: BUY (1.0ポジション)
# 1: HOLD (ポジション変更なし)
# 2: SELL (0.0ポジション)
```

**提案の連続アクション**:
```python
action_space = Box(low=-1.0, high=1.0, shape=(1,))
# -1.0: 最大ショート
# 0.0: ポジションなし (HOLD)
# +1.0: 最大ロング
# 例: +0.5 = 50%ロングポジション
```

### 実装アプローチ

#### アクション空間変更

**environment.py修正**:
```python
from gym.spaces import Box

class HeavyTradingEnv(gym.Env):
    def __init__(self, *args, use_continuous_actions=False, **kwargs):
        self.use_continuous_actions = use_continuous_actions
        
        if use_continuous_actions:
            # 連続アクション: ポジションサイズ
            self.action_space = Box(
                low=np.array([-1.0]),
                high=np.array([1.0]),
                dtype=np.float32
            )
        else:
            # 従来の離散アクション
            self.action_space = Discrete(3)
    
    def step(self, action):
        if self.use_continuous_actions:
            # actionは-1.0~1.0の連続値
            target_position = float(action[0])
            
            # ポジション調整
            current_position = self.position
            position_change = target_position - current_position
            
            # トレード実行
            if abs(position_change) > 0.01:  # 閾値
                self._execute_trade(position_change)
        else:
            # 従来の離散アクション処理
            # ...
```

#### PPOからPPO (Continuous)への移行

**PolicyType変更**:
```python
# 離散アクション用
from sb3_contrib import MaskablePPO

# 連続アクション用
from stable_baselines3 import PPO  # 標準PPO (連続対応)

model = PPO(
    policy="MlpPolicy",
    env=env,
    # ... 既存のPPOパラメータ ...
)
```

**Action Maskingの代替**:
連続アクション空間ではAction Maskingが使えないため、以下で対応:

```python
def step(self, action):
    target_position = np.clip(action[0], -1.0, 1.0)
    
    # 制約チェック
    if self._is_invalid_trade(target_position):
        # 無効なアクションには大きなペナルティ
        reward = -10.0
        # ポジションは変更しない
        target_position = self.position
    
    # ...
```

### 期待効果

1. **ポジションサイズ最適化**:
   - 確信度に応じたポジションサイジング
   - リスク管理の向上
   - 部分利確/損切りが可能

2. **アクション柔軟性**:
   - "50%売却"などの中間アクション
   - グラデーショナルなポジション調整
   - 市場状況に応じた微調整

3. **収益性向上**:
   - ドローダウン削減
   - リスク調整後リターン改善
   - 資金効率の最適化

### 実装計画

**Phase 1: 環境拡張 (3-5日)**
1. `use_continuous_actions`フラグ追加
2. 連続アクション処理実装
3. 報酬関数調整

**Phase 2: ポリシー移行 (3-5日)**
1. PPO (continuous) 統合
2. Action constraint実装
3. CustomPPO適応

**Phase 3: テスト (3-5日)**
1. 単体テスト
2. 統合テスト
3. バックテスト比較

**総所要時間**: 1-2週間

### リスクと制約

**技術的リスク**:
- Action Maskingが使えない
- 無効アクション処理が複雑化
- 学習が不安定になる可能性

**軽減策**:
- Reward shapingで制約を表現
- Action normalizationを強化
- Entropy regularization調整

**互換性**:
- MaskablePPOが使えない
- CustomPPO全面改修が必要

---

## 提案3: 自己教師タスク併用

### 概要

取引タスク (メインタスク) に加えて、価格予測などの補助タスク (自己教師あり学習) を併用し、より良い特徴表現を学習します。

### 技術的詳細

**Multi-task Learning アーキテクチャ**:

```
                     ┌─────────────────┐
                     │  Shared Encoder │
                     │   (LSTM/MLP)    │
                     └────────┬────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
         ┌───────▼────────┐       ┌───────▼────────┐
         │  Policy Head   │       │ Auxiliary Head │
         │   (Actor)      │       │  (Predictor)   │
         └───────┬────────┘       └───────┬────────┘
                 │                         │
         ┌───────▼────────┐       ┌───────▼────────┐
         │Trading Actions │       │ Price Prediction│
         │ (BUY/HOLD/SELL)│       │  (next_price)   │
         └────────────────┘       └────────────────┘
```

### 実装アプローチ

#### Auxiliary Task候補

1. **価格予測** (最有力):
```python
# 次のN時点の価格変化を予測
next_price_change = (price_t+N - price_t) / price_t
```

2. **トレンド分類**:
```python
# 上昇/下降/横ばいのトレンド分類
trend = classify_trend(price_history)  # 0/1/2
```

3. **ボラティリティ予測**:
```python
# 次のN時点のボラティリティ予測
volatility = std(price_t+1:t+N)
```

#### カスタムポリシー実装

```python
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

class MultiTaskPolicy(ActorCriticPolicy):
    def __init__(self, *args, aux_task_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Shared feature extractor (既存)
        # self.features_extractor = ...
        
        # Policy head (Actor) - 既存
        self.action_net = nn.Linear(self.features_dim, self.action_space.n)
        
        # Value head (Critic) - 既存
        self.value_net = nn.Linear(self.features_dim, 1)
        
        # Auxiliary task head - 新規
        self.aux_net = nn.Sequential(
            nn.Linear(self.features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # 価格変化予測
        )
        
        self.aux_task_weight = aux_task_weight
        
    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        
        # Main task: Policy and Value
        action_logits = self.action_net(features)
        value = self.value_net(features)
        
        # Auxiliary task: Price prediction
        price_pred = self.aux_net(features)
        
        return action_logits, value, price_pred
    
    def evaluate_actions(self, obs, actions):
        # Standard PPO loss
        action_log_probs, value, entropy = super().evaluate_actions(obs, actions)
        
        # Auxiliary task loss
        _, _, price_pred = self.forward(obs)
        
        # 価格変化のground truth (環境から取得)
        next_price_change = self._get_next_price_change(obs)
        aux_loss = nn.MSELoss()(price_pred, next_price_change)
        
        # Total loss = PPO loss + aux_weight * aux_loss
        # (CustomPPO内で統合)
        
        return action_log_probs, value, entropy, aux_loss
```

#### CustomPPOへの統合

```python
class CustomMultiTaskPPO(CustomPPO):
    def __init__(self, *args, aux_task_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_task_weight = aux_task_weight
        
    def train(self) -> None:
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            # ... 既存のPPO train処理 ...
            
            # Auxiliary task loss追加
            _, _, _, aux_loss = self.policy.evaluate_actions(
                rollout_data.observations, rollout_data.actions
            )
            
            # Total loss
            total_loss = (
                policy_loss
                + self.vf_coef * value_loss
                - self.ent_coef * entropy_loss
                + self.aux_task_weight * aux_loss  # ← 追加
            )
            
            # Backward and optimize
            self.policy.optimizer.zero_grad()
            total_loss.backward()
            self.policy.optimizer.step()
            
            # Logging
            self.logger.record("train/aux_loss", aux_loss.item())
```

### 期待効果

1. **表現学習の改善**:
   - より豊かな特徴表現
   - 一般化性能向上
   - Transfer learning効果

2. **収益性向上**:
   - 価格変動の理解向上
   - トレンド検出精度向上
   - リスク管理の改善

3. **学習効率向上**:
   - 少ないデータで学習可能
   - オーバーフィッティング抑制
   - 安定した学習

### 実装計画

**Phase 1: Auxiliary Task設計 (5-7日)**
1. 価格予測タスク実装
2. Ground truthラベル生成
3. Loss function設計

**Phase 2: MultiTaskPolicy実装 (7-10日)**
1. カスタムポリシークラス作成
2. Auxiliary head追加
3. Multi-task loss実装

**Phase 3: CustomPPO統合 (5-7日)**
1. CustomMultiTaskPPO作成
2. Loss weightingバランス調整
3. Logging追加

**Phase 4: テスト (5-7日)**
1. 単体テスト
2. 10kスモークテスト
3. Ablation study (aux taskあり/なし比較)

**総所要時間**: 3-4週間

### リスクと制約

**技術的リスク**:
- Multi-task学習のバランス調整が困難
- Auxiliary taskがメインタスクを阻害する可能性
- 計算コスト増加

**軽減策**:
- Loss weightingを慎重に調整 (0.01~0.1)
- Auxiliary task勾配のクリッピング
- 段階的統合 (最初は低weight)

---

## 統合実装ロードマップ

### Short-term (1-2ヶ月)

**Week 1-3: RecurrentPPO/LSTM化**
- 優先度: **最高**
- 理由: 時系列特性を最も活かせる
- 期待効果: SELL率改善、収益性向上

**Week 4-5: 10k→50k検証**
- RecurrentPPO動作確認
- バックテスト実施
- パフォーマンス評価

### Mid-term (2-4ヶ月)

**Week 6-9: 自己教師タスク併用**
- 優先度: **高**
- 理由: 長期的な表現学習改善
- 期待効果: 一般化性能向上

**Week 10-11: 統合テスト**
- RecurrentPPO + Multi-task learning
- Ablation study
- ハイパーパラメータ最適化

### Long-term (4-6ヶ月)

**Week 12-14: 連続ポジションサイズ化** (オプション)
- 優先度: **中**
- 理由: 追加の柔軟性
- 期待効果: ポジション最適化

**Week 15-16: 最終統合**
- 全機能統合
- 本番環境テスト
- ドキュメント作成

---

## 技術スタック

### 必要なライブラリ

```python
# Stable-Baselines3 ecosystem
stable-baselines3>=2.0.0
sb3-contrib>=2.0.0  # RecurrentPPO, MaskablePPO

# Deep Learning
torch>=2.0.0
torch-tb-profiler  # TensorBoard profiling

# Auxiliary task
scikit-learn  # Preprocessing, metrics
```

### 開発環境

```bash
# RecurrentPPO検証環境
python -m venv venv_recurrent
pip install stable-baselines3[extra] sb3-contrib torch

# Multi-task learning検証環境
python -m venv venv_multitask
pip install stable-baselines3 torch tensorboard
```

---

## 成功基準

### RecurrentPPO/LSTM化

| 指標 | 現状 (MLP) | 目標 (LSTM) |
|------|-----------|------------|
| SELL率 | 0-5% | **≥15%** |
| Sharpe Ratio | 0.5 | **≥1.0** |
| Max Drawdown | -20% | **≤-15%** |
| 学習時間 | 100% | ≤150% |

### 自己教師タスク併用

| 指標 | Single-task | Multi-task目標 |
|------|------------|---------------|
| 価格予測MSE | N/A | **≤0.01** |
| 収益性 | 100% | **≥120%** |
| 過学習度 | train/val=1.5 | **≤1.2** |

### 連続ポジションサイズ化

| 指標 | 離散 | 連続目標 |
|------|------|---------|
| ポジション利用率 | 0/50/100% | **0-100%連続** |
| リスク調整後リターン | 100% | **≥110%** |
| 取引頻度 | 同じ | 同じ |

---

## 推奨事項

### 即座に実施

1. ✅ **RecurrentPPO/LSTM化の調査開始**
   - sb3-contrib RecurrentPPO API確認
   - 簡単なLSTMポリシー実験
   - エピソード管理の理解

### 短期的 (1-2週間)

2. ✅ **RecurrentPPO Prototypeング**
   - 10kスモークテスト実施
   - PAN/Target Entropy統合確認
   - パフォーマンス初期評価

### 中期的 (1-2ヶ月)

3. ✅ **自己教師タスク設計**
   - 価格予測タスク仕様策定
   - Multi-taskポリシー設計
   - Loss weightingプロトタイピング

### 長期的 (2-3ヶ月)

4. 🔄 **連続ポジションサイズ検討**
   - RecurrentPPO安定後に評価
   - プロトタイプ作成
   - 効果測定

---

## 結論

3つの提案はいずれも技術的に実現可能であり、大きな改善効果が期待できます。

**推奨実装順序**:
1. **RecurrentPPO/LSTM化** (最優先) → 時系列モデリング
2. **自己教師タスク併用** (高優先度) → 表現学習強化
3. **連続ポジションサイズ化** (中優先度) → 追加の柔軟性

**次のステップ**:
- RecurrentPPO調査開始
- sb3-contrib RecurrentPPO APIドキュメント確認
- 簡単なLSTMポリシー実験

これらの改善は、CustomPPOと相補的であり、統合することでさらに強力なトレーディングシステムを構築できます。

---

**レポート作成日**: 2025年10月6日  
**作成者**: GitHub Copilot  
**ステータス**: 検討資料  
**次回更新**: RecurrentPPO調査完了後
