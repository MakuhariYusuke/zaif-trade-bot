# v394シリーズ最終レポート: エントロピー係数の限界

## 📊 実験結果サマリー

### 完了した訓練

| Version | ent_coef | HOLD罰則 | 取引報酬 | Timesteps | 初期HOLD | 最終HOLD | 最終エントロピー |
|---------|----------|----------|----------|-----------|----------|----------|-----------------|
| v394d | 0.01 | 0.1 (5x) | 5.0 (5x) | 100,352 ✅ | **50.0%** | 89.1% | 0.610 |
| v394f | **0.2** | 0.1 (5x) | 5.0 (5x) | 100,352 ✅ | ? | 89.1% | 0.597 |

### 🚨 衝撃的な発見

**ent_coef を20倍（0.01 → 0.2）にしても結果は同じ！**

```
v394d (ent_coef=0.01):  HOLD 89.1%, entropy 0.610
v394f (ent_coef=0.2):   HOLD 89.1%, entropy 0.597
差分:                    0.0%,       -0.013
```

## 🔍 詳細分析

### 1. Action分布の推移

#### v394d (ent_coef=0.01)
```
Timesteps | HOLD% | BUY+SELL% | Entropy
----------|-------|-----------|--------
2,048     | 50.0% |   50.0%   | ~1.00
10,240    | 90.6% |    9.4%   | ~0.65
100,352   | 89.1% |   10.9%   | 0.610
```

#### v394f (ent_coef=0.2)
```
Timesteps | HOLD% | BUY+SELL% | Entropy
----------|-------|-----------|--------
94,208    | 91.4% |    8.6%   | 0.596
95,232    | 87.1% |   12.9%   | 0.596
100,352   | 89.1% |   10.9%   | 0.597
```

### 2. エントロピー係数の効果

**期待**:
- ent_coef=0.2でエントロピー>1.0を維持
- HOLD比率<70%を達成

**現実**:
- エントロピー: 0.597（v394dとほぼ同じ）
- HOLD比率: 89.1%（v394dと完全同一）

**結論**: **エントロピー係数は根本的な解決策ではない**

### 3. 根本原因の特定

#### 仮説1: 報酬シグナルの弱さ
```python
# 現在の報酬設定
HOLD penalty:     -0.1 per step
Trading cost:     -0.0005 * price  # ~数百円
Profit reward:    +5.0 * (利益率)

# 問題
- 取引コスト > HOLD罰則の可能性
- 小さな損失でも大きなマイナス報酬
- HOLDが「安全な選択」として学習される
```

#### 仮説2: PPOの保守的性質
```
PPO (Proximal Policy Optimization):
- 保守的なPolicy更新（clip_range制約）
- 安定性優先の設計
- リスク回避傾向

→ HOLDが最適解として収束しやすい
```

#### 仮説3: エントロピー正則化の限界
```
ent_coef × entropy_loss:
- 0.01 × -0.61 = -0.0061  (v394d)
- 0.2  × -0.60 = -0.12    (v394f)

しかし、value_lossやpolicy_lossに比べて小さい
→ エントロピーボーナスが報酬シグナルに負ける
```

## 💡 次のアプローチ

### Strategy A: Stochastic推論評価 ⭐⭐⭐⭐⭐
**最優先で実施**

```python
# deterministic=False でバックテスト
# 訓練時のAction分布（HOLD 50-89%）を活用
# 期待: 確率的サンプリングで多様な行動
```

**メリット**:
- 既存モデルを活用
- 追加訓練不要
- 即座に評価可能

**実施**:
```bash
.venv311\Scripts\python.exe stochastic_backtest.py \
  --model checkpoints/ppo_session_11 \
  --data btc_jpy_real_dataset.csv \
  --episodes 10
```

### Strategy B: 報酬関数の抜本的見直し ⭐⭐⭐⭐

**変更案**:
```json
{
  "hold_penalty_weight": 0.5,           // 0.1 → 0.5 (5倍)
  "consecutive_hold_penalty": 0.2,      // 0.05 → 0.2 (4倍)
  "successful_trade_bonus": 20.0,       // 5.0 → 20.0 (4倍)
  "profit_reward_multiplier": 50.0,     // 10.0 → 50.0 (5倍)
  "trading_frequency_bonus": 1.0,       // 0.3 → 1.0 (3倍)
  "transaction_cost_in_reward": false   // 🔥 NEW: コストを報酬から除外
}
```

**理論**:
- HOLD罰則 >> 取引コスト
- 成功取引報酬 >> HOLD罰則
- 明確な報酬勾配

### Strategy C: Early Stopping ⭐⭐⭐⭐

**発見**: v394dは2,048 stepsでHOLD 50%達成

```python
# チェックポイントから2k steps時点を使用
# それ以降は悪化するだけ
# 「過学習」の一種
```

**実装**:
```python
# checkpoint_interval: 2000
# early_stopping_patience: 3
# early_stopping_metric: "action_diversity"
```

### Strategy D: 異なるアルゴリズム ⭐⭐⭐

#### Soft Actor-Critic (SAC)
```
特徴:
- 最大エントロピーRLの設計
- 探索がアルゴリズムに組み込まれている
- 連続・離散両方対応

期待:
- HOLDへの収束が遅い
- より多様な行動パターン
```

#### TD3 (Twin Delayed DDPG)
```
特徴:
- 連続行動空間
- より積極的な探索

期待:
- 売買タイミングの最適化
```

### Strategy E: カリキュラム学習 ⭐⭐

```python
# 段階的にHOLD罰則を強化
stage_1: HOLD penalty = 0.01, ent_coef = 0.2  (探索)
stage_2: HOLD penalty = 0.1,  ent_coef = 0.1  (学習)
stage_3: HOLD penalty = 0.5,  ent_coef = 0.05 (活用)
```

## 🎯 実行計画

### Phase 1: 即座に実行（今日）

1. **Stochasticバックテスト** ✅ 最優先
   - v394d, v394f両方でテスト
   - deterministic=False
   - Return > 0%が目標

2. **初期チェックポイント評価**
   - 2k steps時点のモデル（HOLD 50%）
   - 「Early Stopping」の検証

### Phase 2: 次の訓練（明日以降）

3. **報酬関数見直し版（v395）**
   - HOLD罰則5倍（0.5）
   - 取引報酬20倍（20.0）
   - 取引コスト除外

4. **SAC実装**
   - 最大エントロピーRL
   - 別アルゴリズムでの検証

### Phase 3: 最終判断

5. **比較評価**
   - Stochastic vs Deterministic
   - PPO vs SAC
   - Early Stopping vs Full Training

6. **実運用決定**
   - 最も収益性の高いモデル
   - リスク管理設定
   - ライブトレード開始

## 📋 重要な学び

### ✅ 成功した点
1. **初期Action分布**: v394dの2k stepsでHOLD 50%達成
2. **報酬設定の方向性**: HOLD罰則+取引報酬の組み合わせが重要
3. **完全な訓練実施**: 100k timestepsで完走×2回

### 🚨 失敗した点
1. **エントロピー係数の過信**: 20倍でも効果なし
2. **長期訓練の弊害**: 訓練が進むほど悪化
3. **モデル保存の問題**: best_model.zipが保存されず

### 💡 新しい洞察
1. **Early Stoppingの重要性**: 2k steps時点が最良
2. **報酬設計の再考**: より大きな差をつける必要
3. **アルゴリズムの選択**: PPOの限界、SACへの移行検討

## 🚀 Next Step

```bash
# 1. Stochasticバックテスト実行（最優先）
.venv311\Scripts\python.exe stochastic_backtest.py \
  --model checkpoints/ppo_session_11 \
  --data btc_jpy_real_dataset.csv \
  --episodes 10

# 2. 結果に応じて方針決定
# Return > 0% → 実運用準備
# Return ≤ 0% → Strategy B-E実施
```

---

**結論**: エントロピー係数は根本的な解決策ではなかった。Stochastic推論評価と報酬関数の抜本的見直しが必要。
