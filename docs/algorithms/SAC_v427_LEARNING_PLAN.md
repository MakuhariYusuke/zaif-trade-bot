# SAC v427 学習計画と実行ガイド

## 概要
報酬関数とパラメータを最適化したSAC v427モデルに対して、包括的な学習計画を策定します。150次元特徴拡張、多様な学習方法、堅牢なバックテストシステムを活用した学習戦略です。

## 学習フェーズ設計

### Phase 1: 初期学習 (Foundation Training)
**目的**: 基本的な取引パターンの学習と安定性の確保
- **ステップ数**: 100,000 steps
- **エピソード数**: 50-100 episodes
- **学習率**: 3e-4 (初期設定)
- **評価頻度**: 毎10,000 steps
- **評価エピソード数**: 10 episodes
- **目標**: 基本的な収益性と安定性の達成

### Phase 2: 最適化フェーズ (Optimization Phase)
**目的**: 多様な最適化手法によるハイパーパラメータチューニング
- **Ray Tune最適化**: 50 trials × 50,000 steps each
- **Hyperopt最適化**: 100 trials × 25,000 steps each
- **BOHB最適化**: 30 trials × 75,000 steps each
- **並列実行**: 最大4並列プロセス
- **目標**: 最適ハイパーパラメータの特定

### Phase 3: 微調整学習 (Fine-tuning)
**目的**: 最適化されたパラメータでの本格的な学習
- **ステップ数**: 500,000 steps
- **エピソード数**: 200-300 episodes
- **学習率**: 最適化された値 (1e-4 - 1e-5)
- **評価頻度**: 毎25,000 steps
- **評価エピソード数**: 20 episodes
- **目標**: 最高性能モデルの生成

### Phase 4: 最終検証 (Final Validation)
**目的**: 学習成果の包括的検証
- **バックテスト**: 複数市場データでの検証
- **ウォークフォワード分析**: 時系列安定性の確認
- **リスク指標評価**: Sharpe ratio, Max Drawdown, Win Rate
- **目標**: 本番環境での使用準備完了

## 学習実行手順

### 1. 環境準備
```bash
# 必要なパッケージのインストール
pip install stable-baselines3[extra] ray[tune] hyperopt hpbandster

# 環境変数の設定
export CUDA_VISIBLE_DEVICES=""  # CPU学習の場合
export OMP_NUM_THREADS=4
```

### 2. 初期学習実行
```bash
# Phase 1: 初期学習
python -m ztb.training.unified_trainer.main \
  --config configs/sac_v427_market_adaptive_ensemble.json \
  --total-timesteps 100000 \
  --eval-freq 10000 \
  --n-eval-episodes 10 \
  --save-freq 25000
```

### 3. ハイパーパラメータ最適化
```python
# 多様な学習方法での最適化
from ztb.optimization.diverse_learning_methods import DiverseLearningMethods

optimizer = DiverseLearningMethods()

# Ray Tune最適化
ray_results = optimizer.optimize_hyperparameters(
    objective_function=training_objective,
    search_space=ray_search_space,
    framework='ray_tune',
    max_evals=50
)

# Hyperopt最適化
hyperopt_results = optimizer.optimize_hyperparameters(
    objective_function=training_objective,
    search_space=hyperopt_search_space,
    framework='hyperopt',
    max_evals=100
)

# BOHB最適化
bohb_results = optimizer.optimize_hyperparameters(
    objective_function=training_objective,
    search_space=bohb_search_space,
    framework='bohb',
    max_evals=30
)
```

### 4. 微調整学習実行
```bash
# Phase 3: 最適化パラメータでの微調整
python -m ztb.training.unified_trainer.main \
  --config configs/sac_v427_optimized.json \
  --total-timesteps 500000 \
  --eval-freq 25000 \
  --n-eval-episodes 20 \
  --save-freq 100000 \
  --load-best  # 最適化フェーズのベストモデルから再開
```

### 5. 最終検証
```bash
# バックテスト検証
python scripts/run_backtest.py \
  --model-path models/sac_v427_final.zip \
  --data-path data/btc_jpy_real_dataset.csv \
  --config configs/sac_v427_market_adaptive_ensemble.json

# ウォークフォワード分析
python scripts/walk_forward_analysis.py \
  --model-path models/sac_v427_final.zip \
  --window-size 1000 \
  --step-size 200
```

## 学習パラメータ設定

### SACアルゴリズム設定
```json
{
  "learning_rate": 0.0003,
  "buffer_size": 50000,
  "learning_starts": 1000,
  "batch_size": 256,
  "tau": 0.005,
  "gamma": 0.99,
  "ent_coef": 0.01,
  "target_update_interval": 1,
  "target_entropy": -2.0
}
```

### 環境設定
```json
{
  "initial_balance": 200000.0,
  "transaction_cost": 0.00001,
  "max_position_size": 1.0,
  "feature_set": "v427_adaptive",
  "reward_scale": 500.0,
  "curriculum_stage": "strong_penalty_trading"
}
```

### 評価指標
- **主要指標**: Sharpe Ratio, Total Return, Win Rate
- **リスク指標**: Maximum Drawdown, Volatility, Calmar Ratio
- **安定性指標**: Consistency Score, Recovery Factor

## 学習監視と管理

### TensorBoard監視
```bash
# TensorBoard起動
tensorboard --logdir ./tensorboard/sac_v427

# 監視項目:
# - episode_reward: エピソード報酬
# - eval/mean_reward: 評価平均報酬
# - train/learning_rate: 学習率
# - train/ent_coef: エントロピー係数
```

### 学習進捗確認
```python
# 学習状態確認
from ztb.training.unified_trainer import UnifiedTrainer

trainer = UnifiedTrainer.load("models/sac_v427_checkpoint.zip")
stats = trainer.get_training_stats()
print(f"学習ステップ: {stats['total_timesteps']}")
print(f"ベスト報酬: {stats['best_mean_reward']}")
print(f"評価回数: {stats['n_evaluations']}")
```

## トラブルシューティング

### 学習が不安定な場合
1. **学習率調整**: 学習率を1/10に減らす (3e-4 → 3e-5)
2. **バッファサイズ増大**: buffer_sizeを100,000に増加
3. **バッチサイズ調整**: batch_sizeを128に減少
4. **報酬スケーリング**: reward_scaleを調整

### メモリ不足の場合
1. **バッファサイズ削減**: buffer_sizeを25,000に減少
2. **特徴数制限**: max_featuresを50に設定
3. **GC強制実行**: 定期的なガベージコレクション

### 収束が遅い場合
1. **学習率増加**: 学習率を1/3に増加 (3e-4 → 1e-3)
2. **エントロピー係数調整**: ent_coefを0.1に増加
3. **ターゲットエントロピー調整**: target_entropyを-1.0に変更

## 学習成果評価基準

### 成功基準
- **Sharpe Ratio**: > 1.5 (バックテスト)
- **Total Return**: > 50% (1年シミュレーション)
- **Win Rate**: > 55%
- **Maximum Drawdown**: < 20%
- **Calmar Ratio**: > 0.8

### 品質ゲート
- [ ] 学習安定性: 100エピソード連続でクラッシュなし
- [ ] バックテスト性能: 複数市場で安定した成績
- [ ] ウォークフォワード安定性: 時系列での一貫性
- [ ] リスク管理: 適切なリスク指標

## 今後の改善方針

### 短期改善 (1-2ヶ月)
1. **特徴拡張**: 追加の市場指標統合
2. **報酬関数改良**: アンサンブル報酬設計
3. **学習安定化**: 高度な正則化手法

### 中期改善 (3-6ヶ月)
1. **マルチエージェント学習**: 複数戦略の協調学習
2. **転移学習**: 異なる市場間での知識転移
3. **オンライン学習**: リアルタイム適応能力

### 長期改善 (6ヶ月以上)
1. **メタ学習**: タスク適応能力の向上
2. **継続学習**: 長期的な知識蓄積
3. **連合学習**: 分散型学習インフラ

---

## 学習実行結果

### Phase 1: 初期学習 ✅ 完了
**実行日時**: 2025年10月19日
**学習ステップ数**: 100,000 steps
**学習時間**: 3,442秒 (約57分)
**ステップ/秒**: 29.05 SPS
**アクション分布**:
- HOLD: 2.8%
- BUY: 5.3%
- SELL: 91.9% (支配的)

**結果**: 学習成功、モデル保存完了 (`models/sac_v427_market_adaptive_ensemble.zip`)

### 次のステップ
1. **Phase 2**: ハイパーパラメータ最適化 (Ray Tune, Hyperopt, BOHB)
2. **Phase 3**: 最適化パラメータでの微調整学習 (500kステップ)
3. **Phase 4**: 最終検証とバックテスト評価

---

**最終更新**: 2025年10月19日
**バージョン**: SAC v427 Final
**責任者**: 開発チーム</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\docs\SAC_v427_LEARNING_PLAN.md
