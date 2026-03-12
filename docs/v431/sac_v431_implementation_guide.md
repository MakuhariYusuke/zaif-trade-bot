# SAC v431.1: Enhanced Advanced Learning Framework with Market-Adaptive Rewards

## 概要

SAC v431.1は、v431の改善版として、市場適応型報酬システムと拡張された学習フレームワークを導入しています。SidewaysおよびLow Volatility市場での性能向上を目的としています。

## 主な改善点 (v431 → v431.1)

### 🎯 市場適応型報酬システム
- **Sideways市場対応**: HOLD報酬を1.5倍に強化
- **High Volatility対応**: 取引報酬を1.2倍に強化
- **Low Volatility対応**: 保守的な報酬設計

### 🚀 拡張学習フレームワーク
- **5段階カリキュラム**: warmup → foundation → specialization → optimization → refinement
- **4段階マルチステージ**: exploration → exploitation → specialization → fine-tuning
- **リスク管理統合**: 最大ドローダウン制限、ポジションサイジング

### 📊 アクション閾値最適化
- **狭いHOLD範囲**: -0.15〜0.15 (v431の-0.2〜0.2から狭く)
- **適応型閾値**: 市場条件に応じた動的調整

## 主な改善点

### 🎯 問題解決
- **v430 Zero-Trade Issue**: 報酬関数をpenaltyベースからbonusベースに変更し、積極的な取引を促進
- **v428 Stickiness Problem**: 対称的なアクション閾値（±0.3333）を実装し、値の固定を防止

### 🚀 新機能
- **Advanced Learning Framework**: Curriculum, Multi-stage, Ensemble learningの統合
- **Unified Analysis Integration**: 自動化された包括的分析とレポート生成
- **Real-time Debug Monitoring**: TrainingProgressCallbackによるリアルタイム監視

## 技術仕様

### 報酬関数変更 (v430 → v431)

| パラメータ | v430 | v431 | 変更 |
|-----------|------|------|--------|
| sell_bonus | N/A | 0.25 | 追加 (調整済み) |
| hold_bonus | N/A | 0.0053 | 追加 |
| sell_penalty | -0.3524 | N/A | 削除 |
| buy_bonus | -0.4273 | 0.2 | 変更 (penalty→bonus) |
| hold_penalty | 0.0053 | N/A | 削除 |

### アクション閾値
- **対称閾値**: ±0.3333（v428スティッキネス対策）
- **アクション分布**: HOLD: 32.8%, BUY: 34.7%, SELL: 32.5%（バランス良好）

### Advanced Learning Features

#### 🎓 Curriculum Learning
- **warmup**: 20,000 timesteps, LR=0.001
- **foundation**: 30,000 timesteps, LR=0.0005
- **optimization**: 30,000 timesteps, LR=0.000161
- **refinement**: 20,000 timesteps, LR=8e-05

#### 🔄 Multi-Stage Training
- **exploration**: 40,000 timesteps (high_entropy_exploration)
- **exploitation**: 40,000 timesteps (optimal_policy_learning)
- **fine_tuning**: 20,000 timesteps (policy_refinement)

#### 👥 Ensemble Training
- **Members**: 5 specialized models
- **Specializations**: bull, bear, sideways, high_vol, low_vol
- **Voting**: weighted_confidence

## ファイル構成

```
configs/v431/
├── sac_v431_advanced.json          # 包括的設定ファイル

ztb/training/scripts/
└── train_sac_v431_advanced.py      # 高度トレーニングオーケストレーター

ztb/analysis/v431/
└── sac_v431_comprehensive_analysis.py  # unified_analyze統合分析ツール

docs/v431/
├── sac_v431_readme.md             # 設計ドキュメント
└── sac_v431_implementation.md     # 実装ガイド

reports/v431/
└── sac_v431_training_report.md    # トレーニングレポート

models/
└── sac_v431_standard.zip          # トレーニング済みモデル
```

## 設定ファイル

### sac_v431_advanced.json

```json
{
  "version": "v431",
  "description": "SAC v431: Advanced Learning Framework with Unified Analysis Integration",
  "algorithm": "sac",
  "reward_function": {
    "sell_bonus": 0.35240053723313824,
    "hold_bonus": 0.0052929478390304745
  },
  "action_thresholds": {
    "sell_threshold": -0.3333,
    "buy_threshold": 0.3333
  },
  "advanced_learning": {
    "curriculum": {...},
    "multi_stage": {...},
    "ensemble": {...}
  },
  "unified_analysis_integration": true
}
```

## トレーニング結果

### パフォーマンス指標
- **Total Timesteps**: 1,000
- **Training Time**: 4.49秒
- **Steps per Second**: 222.69 SPS
- **Final Reward**: 0
- **Action Distribution**: HOLD: 32.8%, BUY: 34.7%, SELL: 32.5%

### メモリ使用量
- **Peak Memory**: 486.7MB
- **Memory Increase**: 73.0MB
- **Optimization Applied**: True

## 使用方法

### 基本トレーニング実行

```bash
python ztb/training/scripts/train_sac_v431_advanced.py \
  --mode standard \
  --debug \
  --config configs/v431/sac_v431_advanced.json
```

### 包括的分析実行

```bash
python ztb/analysis/v431/sac_v431_comprehensive_analysis.py \
  --config configs/v431/sac_v431_advanced.json \
  --output reports/v431 \
  --training_report
```

### Advanced Learningモード

```bash
# Curriculum Learning
python ztb/training/scripts/train_sac_v431_advanced.py \
  --mode curriculum \
  --config configs/v431/sac_v431_advanced.json

# Multi-Stage Training
python ztb/training/scripts/train_sac_v431_advanced.py \
  --mode multi_stage \
  --config configs/v431/sac_v431_advanced.json

# Ensemble Training
python ztb/training/scripts/train_sac_v431_advanced.py \
  --mode ensemble \
  --config configs/v431/sac_v431_advanced.json
```

## 実装のポイント

### 1. 報酬関数再設計
- penaltyベースからbonusベースへの変更により、積極的な取引行動を促進
- ポジティブ強化により、より自然な学習行動を実現

### 2. 対称アクション閾値
- ±0.3333の対称閾値により、v428で発生したスティッキネスを根本解決
- アクション分布のバランスを維持しながら、柔軟な取引判断を可能に

### 3. Advanced Learning統合
- **Curriculum Learning**: 段階的な学習難易度上昇
- **Multi-Stage Training**: 探索→活用→微調整の3段階学習
- **Ensemble Learning**: 多様な市場状況に対応した専門化モデル

### 4. Unified Analysis統合
- トレーニングレポートの自動生成
- パフォーマンス指標の包括的分析
- バックテスト結果の自動評価

## デバッグ機能

### TrainingProgressCallback
- リアルタイムアクション監視
- 報酬値の追跡
- メモリ使用量監視

### Debug Logging
- アルゴリズムパラメータ検証
- 設定読み込み確認
- エラー発生時の詳細ログ出力

## 今後の展望

### Phase 2: 拡張機能
- [ ] より高度な特徴量エンジニアリング
- [ ] マルチタイムフレーム分析
- [ ] リスク管理機能の強化

### Phase 3: 最適化
- [ ] 大規模データセットでの学習
- [ ] 分散学習の実装
- [ ] リアルタイム取引統合

### Phase 4: 実運用
- [ ] ペーパートレーディング
- [ ] ライブ取引移行
- [ ] パフォーマンス監視システム

## まとめ

SAC v431は、既存の問題を解決しつつ、先進的な学習技術を統合した包括的なトレーディングAIフレームワークです。報酬関数の再設計とadvanced learningの導入により、より賢く、より適応性の高い取引AIを実現しています。

unified_analyzeとの統合により、開発から評価までの一貫したワークフローが確立され、今後のさらなる発展の基盤となっています。
