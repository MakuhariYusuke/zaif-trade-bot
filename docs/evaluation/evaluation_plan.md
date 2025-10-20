# 評価計画ドキュメント

## 概要

このドキュメントは、Zaif Trade Botのアンサンブル学習改善のための包括的な評価計画を定義します。ベンチマークスイート、クロスバリデーション、アブレーションスタディ、トレード分析を通じて、モデルの性能を体系的に評価します。

## 評価タスク

### 1. ベンチマークスイートの設計

#### 設定
- **目的**: シングルモデル vs アンサンブルモデルの性能比較
- **データ**: `ml-dataset-enhanced.csv` (トレーニング済みデータ)
- **評価期間**: 過去30日分のデータを使用
- **評価エピソード数**: 10エピソード/モデル

#### 実装手順
```bash
# ベンチマーク実行
python comprehensive_benchmark.py \
  --data ml-dataset-enhanced.csv \
  --single-model models/single_model.zip \
  --ensemble-models models/ensemble_model_1.zip models/ensemble_model_2.zip \
  --episodes 10 \
  --output-dir benchmark_results
```

#### 出力例
```
Model      | Total Return | Sharpe | Sortino | Win Rate | Max DD | Total Trades
-----------|--------------|--------|---------|----------|--------|-------------
Single     | 1.234        | 1.45   | 1.67    | 62.5%    | -0.123 | 45
Ensemble   | 1.567        | 1.78   | 2.01    | 68.2%    | -0.089 | 52
```

#### 評価指標
- **Sharpe Ratio**: (平均超過リターン) / ボラティリティ
- **Sortino Ratio**: (平均超過リターン) / 下方偏差
- **Calmar Ratio**: (総リターン) / 最大ドローダウン
- **Win Rate**: ポジティブリターンの割合
- **Total Trades**: 総トレード数

### 2. クロスバリデーション

#### 設定
- **目的**: モデルの安定性と汎化性能の評価
- **分割数**: K=5 (時系列を考慮した分割)
- **評価指標**: 各foldのSharpe ratioの平均と標準偏差

#### 実装手順
```bash
# CV評価実行
python comprehensive_benchmark.py \
  --data ml-dataset-enhanced.csv \
  --single-model models/single_model.zip \
  --cv-folds 5 \
  --episodes 5
```

#### 出力例
```
Cross-Validation Results (Single Model):
Fold 1: Sharpe = 1.23
Fold 2: Sharpe = 1.45
Fold 3: Sharpe = 1.12
Fold 4: Sharpe = 1.67
Fold 5: Sharpe = 1.34
Mean: 1.36 ± 0.19
```

### 3. アブレーションスタディ

#### 設定
- **目的**: 各改善要素の寄与度評価
- **比較対象**:
  - 動的HOLDペナルティ
  - Sortino/Calmar比率導入
  - Cosine Annealing LR
  - Early Stopping
  - 信頼度加重アンサンブル

#### 実装手順
```bash
# アブレーションスタディ実行
python ablation_study.py \
  --data ml-dataset-enhanced.csv \
  --model models/baseline_model.zip \
  --episodes 5 \
  --output-dir ablation_results
```

#### 出力例
```
Configuration          | Sharpe | Improvement
-----------------------|--------|------------
baseline              | 1.45   | 0.00
no_dynamic_hold       | 1.23   | -0.22
no_sortino_calmar     | 1.34   | -0.11
no_cosine_lr          | 1.28   | -0.17
no_early_stopping     | 1.31   | -0.14
no_improvements       | 0.89   | -0.56
```

### 4. トレード分析

#### 設定
- **目的**: 行動パターンの分析とHOLD偏重の改善確認
- **分析対象**: BUY/SELL/HOLDの分布と遷移

#### 実装手順
```bash
# トレード分析実行
python trade_analysis.py \
  --data ml-dataset-enhanced.csv \
  --models baseline:models/baseline_model.zip improved:models/improved_model.zip \
  --episodes 5 \
  --output-dir trade_analysis
```

#### 出力例
```
Model     | HOLD % | BUY % | SELL % | Avg HOLD Streak
----------|--------|-------|--------|----------------
Baseline  | 45.2%  | 32.1% | 22.7%  | 3.2
Improved  | 52.8%  | 28.4% | 18.8%  | 4.7
```

## 追加検討質問への回答

### 1. 評価用データ期間の分割

**推奨分割**:
- **トレーニング期間**: 過去180日 (6ヶ月)
- **検証期間**: 過去30日 (直近1ヶ月)
- **テスト期間**: 最新7日 (直近1週間)

**理由**:
- 時系列データの性質上、未来データを予測するため直近データをテストに使用
- 検証期間でハイパーパラメータチューニング
- テスト期間は最終評価のみ使用（データ汚染防止）

**実装例**:
```python
# データ分割関数
def split_data_for_evaluation(df, train_days=180, val_days=30, test_days=7):
    end_date = df['timestamp'].max()
    test_start = end_date - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)
    train_start = val_start - pd.Timedelta(days=train_days)

    train_data = df[df['timestamp'] >= train_start][df['timestamp'] < val_start]
    val_data = df[df['timestamp'] >= val_start][df['timestamp'] < test_start]
    test_data = df[df['timestamp'] >= test_start]

    return train_data, val_data, test_data
```

### 2. アンサンブル重み付け方式の検証

**検証アプローチ**:
- **固定重み**: 各モデルの貢献度を事前定義
- **動的重み**: 予測信頼度に基づく動的調整
- **パフォーマンスベース**: 過去の実績に基づく重み調整

**実装例**:
```python
# 信頼度ベースの動的重み付け
def confidence_weighted_ensemble(predictions, confidences):
    weights = np.array(confidences) / np.sum(confidences)
    ensemble_pred = np.average(predictions, weights=weights, axis=0)
    return ensemble_pred

# パフォーマンスベース重み付け
def performance_based_weights(model_histories, window=30):
    recent_performance = []
    for history in model_histories:
        recent_sharpe = calculate_sharpe_ratio(history[-window:])
        recent_performance.append(recent_sharpe)

    weights = np.array(recent_performance)
    weights = np.maximum(weights, 0)  # 負の性能は0に
    weights = weights / np.sum(weights)
    return weights
```

### 3. 再現性保証のための仕組み

**シード管理**:
```python
# グローバルシード設定
def set_experiment_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Stable Baselines3 の決定論的動作
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# 実験設定の保存
experiment_config = {
    'seed': 42,
    'model_configs': [...],
    'training_params': {...},
    'environment_params': {...},
    'timestamp': datetime.now().isoformat()
}
```

**ログ保存**:
```python
# 実験追跡
def log_experiment_results(config, results, output_dir):
    experiment_id = f"{config['timestamp']}_{config['seed']}"

    # 設定保存
    with open(output_dir / f"{experiment_id}_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # 結果保存
    with open(output_dir / f"{experiment_id}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # モデル保存
    model.save(output_dir / f"{experiment_id}_model.zip")
```

## PR タイトル案

### 主要PR
```
feat: Add comprehensive evaluation suite for ensemble trading models

- Add benchmark suite for single vs ensemble model comparison
- Implement cross-validation with time-series aware splitting
- Create ablation study framework for improvement analysis
- Add trade action distribution analysis and visualization
- Include data splitting utilities for train/val/test periods
- Add reproducibility controls with seed management
```

### ドキュメントPR
```
docs: Add evaluation plan and methodology documentation

- Document comprehensive evaluation framework
- Include benchmark metrics and interpretation guidelines
- Add ablation study methodology and result interpretation
- Document data splitting strategies and validation approaches
- Include reproducibility best practices
```

## 次のステップ

1. **実装検証**: 各スクリプトの動作確認
2. **統合テスト**: エンドツーエンドの評価パイプライン実行
3. **結果分析**: 改善効果の定量評価
4. **ドキュメント更新**: 実際の結果に基づく更新
5. **継続的評価**: CI/CDへの評価パイプライン統合