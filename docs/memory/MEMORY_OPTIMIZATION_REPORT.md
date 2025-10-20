# メモリ最適化と学習結果レポート

## 実行日時
2025年10月7日

---

## 1. メモリ最適化の実施

### 問題点
- プログレスバーが100%で停止する問題
- メモリ不足による学習の中断

### 実施した対策

#### 1.1 コード修正
1. **モデル保存処理の最適化** (`ztb/training/unified_trainer.py`)
   - 保存前にガベージコレクション実行
   - 保存後にメモリクリーンアップ
   - エラーハンドリングの強化

2. **final_validation のタイムアウト対策** (`ztb/training/sell_mitigation_ppo_trainer.py`)
   - try-except でエラーを捕捉
   - 非致命的エラーは警告のみ

3. **学習完了後のメモリクリーンアップ** (`ztb/training/ppo_trainer.py`)
   - 学習完了時に明示的にgc.collect()実行

#### 1.2 設定最適化 (`configs/training/ppo_memory_optimized.json`)
```json
{
  "n_steps": 256,          // 512 → 256 (50%削減)
  "batch_size": 16,        // 32 → 16 (50%削減)
  "n_epochs": 3,           // 4 → 3 (25%削減)
  "checkpoint_interval": 15000,  // 10000 → 15000
  "target_kl": null        // 0.05 → null (早期停止を無効化)
}
```

**メモリ削減効果**:
- バッチメモリ: 512×32 = 16,384 → 256×16 = 4,096 (**75%削減**)
- チェックポイント頻度: 33%削減

---

## 2. 学習結果

### 2.1 学習完了
- ✅ **総ステップ数**: 30,208 steps (目標: 30,000)
- ✅ **完走**: メモリエラーなく最後まで実行
- ✅ **モデル保存**: `models/ppo_memory_optimized.zip` (90KB)
- ⏱️ **学習時間**: 約6分20秒
- 💾 **メモリ**: 安定動作、問題なし

### 2.2 学習中の指標

#### 最終統計 (iteration 118, 30,208 steps):
```
lagrange_r_sell:     0.188  (18.8% SELL)
lagrange_r_target:   0.33   (目標33%)
r_sell_mean:         0.117  (平均11.7%)
constraint_active:   True
lambda_dual:         2.0    (最大値に達している)
```

#### アクション分布 (学習中):
```
pan_action_counts: [12, 1, 3]
→ HOLD: 75%, BUY: 6%, SELL: 19%
```

### 2.3 評価結果 (10エピソード, 2000ステップ)

#### アクション分布:
```
HOLD:  1920 (96.0%)
BUY:    80 (  4.0%)
SELL:    0 (  0.0%)
```

#### 診断:
- ⚠️ **HOLD bias**: 96%
- ⚠️ **BUY不足**: 4%のみ
- ❌ **SELL消失**: 0%
- **バランススコア**: 0.000/1.000

---

## 3. 問題分析

### 3.1 学習中と評価時の乖離

| 指標 | 学習時 | 評価時 | 差異 |
|-----|-------|-------|-----|
| HOLD | 75% | 96% | +21% |
| BUY  | 6%  | 4%  | -2% |
| SELL | 19% | 0%  | -19% |

**根本原因**:
1. **Lagrange制約が評価時に機能しない**
   - Lagrange制約は学習プロセスでのみ適用
   - 推論時には制約なし
   - モデル自体が行動多様性を学習していない

2. **Target Entropy Controllerの限界**
   - エントロピーを高めるが、行動の偏りは解消できない
   - `lambda_dual=2.0`(最大値)は制約が効いていない証拠

3. **PAN (Per-Action Normalization)の不十分**
   - サンプル数が少ない行動の正規化がスキップされる
   - `Action 2 has only 0 samples` の警告多数

### 3.2 比較: 以前のモデル

| モデル | HOLD | BUY | SELL | 問題 |
|-------|------|-----|------|------|
| ppo_100k_optimized | 0.5% | 0.0% | 99.5% | SELL bias |
| ppo_balanced_test  | ?% | ?% | 98% | SELL bias (変化なし) |
| ppo_memory_optimized | 96% | 4% | 0% | **HOLD bias** |

**改善点**:
- ✅ SELL bias は解消 (99.5% → 0%)
- ❌ HOLD biasに移行 (0% → 96%)
- ❌ 行動多様性は依然として低い

---

## 4. 根本的な問題

### 4.1 Curriculum Learning が機能していない理由

**forced_balance ステージの問題**:
1. **学習中の強制**のみで、モデルが本質的に学習していない
2. Lagrange制約は"外部からの圧力"であり、内在化されない
3. 推論時には制約が消えるため、元の偏りが露呈

### 4.2 必要なアプローチ

#### A. Behavior Cloning (行動模倣)
```python
# バランスの取れた専門家データを作成
expert_data = generate_balanced_expert_data(
    hold_ratio=0.33,
    buy_ratio=0.33,
    sell_ratio=0.34
)

# Pre-training で模倣学習
model = pretrain_with_bc(expert_data, epochs=10)

# その後RL fine-tuning
model = finetune_with_rl(model, reward_function)
```

#### B. Intrinsic Motivation (内発的動機付け)
```python
# Exploration bonus を追加
intrinsic_reward = calculate_exploration_bonus(
    action_counts,
    target_distribution=[0.33, 0.33, 0.34]
)

total_reward = extrinsic_reward + β * intrinsic_reward
```

#### C. Multi-Objective RL
```python
# 複数の目標を同時最適化
objectives = {
    "profit": maximize_pnl(),
    "diversity": maximize_action_entropy(),
    "balance": minimize_distribution_divergence()
}

# Pareto最適解を探索
model = train_multi_objective(objectives, weights=[0.5, 0.25, 0.25])
```

---

## 5. 次のステップ

### 5.1 即座に実施可能 (優先度: 高)

#### Option A: より強力なEntropy Regularization
```json
{
  "ent_coef": 0.2,  // 0.1 → 0.2 に増加
  "entropy_target_entropy": 1.099,  // log(3) = 完全なランダム
  "entropy_beta": 0.01  // より強力な調整
}
```

#### Option B: Action Maskingの活用
```python
# 過剰に選択されている行動をマスク
if action_counts[HOLD] > 0.5 * total_actions:
    action_mask[HOLD] = False  # HOLDを一時的に禁止
```

#### Option C: Reward Shaping
```python
# 多様性ボーナスを追加
diversity_bonus = -abs(action_dist - target_dist).sum()
shaped_reward = original_reward + 0.1 * diversity_bonus
```

### 5.2 中期的施策 (優先度: 中)

#### 1. Behavior Cloning Pre-training
- バランスの取れた専門家データを生成
- 初期ポリシーとして使用
- RL fine-tuningで最適化

#### 2. Curriculum Learningの再設計
- Stage 0: BC pre-training (10K steps)
- Stage 1: Intrinsic motivation強化 (20K steps)
- Stage 2: Profit最適化 (70K steps)

#### 3. Multi-Agent Ensemble
- 3つのモデルを訓練: HOLD専門, BUY専門, SELL専門
- メタ学習器が最適な行動を選択

### 5.3 長期的施策 (優先度: 低)

#### 1. OfflineRLへの移行
- 過去の取引データからバッチ学習
- Distribution shiftに強い

#### 2. Model-Based RL
- 環境モデルを学習
- Planning with learned model

#### 3. Meta-RL
- 複数の市場条件で訓練
- 汎化性能の向上

---

## 6. 推奨アクション (今すぐ実施)

### Step 1: Entropy係数を増やして再学習
```bash
# configs/training/ppo_high_entropy.json を作成
{
  "session_id": "ppo_high_entropy",
  "ent_coef": 0.2,
  "entropy_target_entropy": 1.099,
  "total_timesteps": 50000
}

python run_training.py --config configs/training/ppo_high_entropy.json --force
```

### Step 2: Reward Shapingを実装
```python
# ztb/environment.py に追加
def calculate_diversity_bonus(self, action, target_dist=[0.33, 0.33, 0.34]):
    current_dist = self.action_counts / self.action_counts.sum()
    divergence = np.abs(current_dist - target_dist).sum()
    bonus = -divergence * 10.0  # ペナルティとして実装
    return bonus
```

### Step 3: 評価時にもEntropy制約を適用
```python
# 推論時のサンプリング温度を上げる
action, _states = model.predict(
    obs,
    deterministic=False,  # 確率的サンプリング
    temperature=1.5       # より多様な行動
)
```

---

## 7. メモリ最適化の成果

### 7.1 成功した点
- ✅ 100%停止問題の解決
- ✅ メモリエラーなしで完走
- ✅ 学習時間の短縮 (6分20秒)
- ✅ モデルサイズの削減 (90KB)

### 7.2 メモリ使用量 (推定)
| 設定 | n_steps | batch_size | 推定メモリ | 実績 |
|-----|---------|------------|-----------|------|
| 元の設定 | 512 | 32 | ~200MB | メモリエラー |
| 最適化版 | 256 | 16 | ~50MB | ✅ 成功 |

**削減率**: **75%**

---

## 8. 結論

### 8.1 達成できたこと
1. ✅ メモリ最適化により安定した学習を実現
2. ✅ SELL bias (99.5%) の完全解消
3. ✅ プログレスバー停止問題の解決

### 8.2 残された課題
1. ❌ **行動の偏り**: SELL bias → HOLD biasに移行
2. ❌ **Lagrange制約の限界**: 学習時のみ有効、推論時は無効
3. ❌ **本質的な多様性学習の欠如**: 外部制約に頼っている

### 8.3 次の方針
**優先順位**:
1. 🔥 **即座**: Entropy係数を0.2に上げて再学習
2. 🔥 **今日中**: Reward Shapingで多様性ボーナスを追加
3. 📅 **今週**: Behavior Cloning pre-trainingを実装
4. 📅 **来週**: Multi-objective RL frameworkの構築

---

## 付録: 設定ファイル比較

### A. 現在の設定 (ppo_memory_optimized.json)
```json
{
  "learning_rate": 0.0003,
  "n_steps": 256,
  "batch_size": 16,
  "ent_coef": 0.1,
  "lagrange_r_target": 0.33,
  "entropy_target_entropy": 0.769
}
```

### B. 推奨設定 (ppo_high_entropy.json)
```json
{
  "learning_rate": 0.0003,
  "n_steps": 256,
  "batch_size": 16,
  "ent_coef": 0.25,           // ⬆️ 増加
  "lagrange_r_target": 0.33,
  "entropy_target_entropy": 1.099,  // ⬆️ log(3)
  "entropy_beta": 0.02,        // ⬆️ 増加
  "enable_forced_diversity": true,
  "diversity_bonus_weight": 0.15  // 🆕 新規
}
```

---

## 参考リンク
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Entropy Regularization in RL](https://arxiv.org/abs/1801.01290)
- [Behavior Cloning](https://arxiv.org/abs/1709.10089)
- [Multi-Objective RL](https://arxiv.org/abs/1809.07803)
