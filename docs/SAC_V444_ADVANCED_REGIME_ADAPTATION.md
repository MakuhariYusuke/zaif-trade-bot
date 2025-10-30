# SAC v444: Advanced Market Regime Adaptation System

## 概要

SAC v444は、市場状態を12種類に細分化した高度な適応型トレーディングシステムです。従来の4分類（Bull/Bear/Sideways/Volatile）を大幅に拡張し、より精密な市場適応を実現します。

## 主な特徴

### 1. 12レジーム分類システム

市場状態を以下の12種類に分類：

#### 強気トレンド系
- **strong_bull_trend**: 明確な上昇相場（トレンド強度 > 0.02）
- **moderate_bull_trend**: 中程度の上昇相場（トレンド強度 0.01-0.02）
- **weak_bull_trend**: 弱い上昇相場（トレンド強度 0.005-0.01）

#### 弱気トレンド系
- **strong_bear_trend**: 明確な下降相場（トレンド強度 < -0.02）
- **moderate_bear_trend**: 中程度の下降相場（トレンド強度 -0.02～-0.01）
- **weak_bear_trend**: 弱い下降相場（トレンド強度 -0.01～-0.005）

#### レンジ系
- **high_volatility_ranging**: 高ボラティリティレンジ（ボラティリティ > 0.02）
- **moderate_volatility_ranging**: 中程度ボラティリティレンジ（ボラティリティ 0.01-0.02）
- **low_volatility_ranging**: 低ボラティリティレンジ（ボラティリティ < 0.01）

#### 特殊状態
- **extreme_volatility**: 極端な高ボラティリティ相場（ボラティリティ > 0.03）
- **consolidation**: 統合相場（低ボラティリティ・低トレンド・高出来高）
- **breakout_setup**: ブレイクアウト準備相場
- **breakdown_setup**: ブレークダウン準備相場

### 2. レジーム別最適化パラメータ

各レジームに対して最適化された行動パラメータ：

| レジーム | Action Balance | Entropy Reg. | Consistency Penalty | Position Size |
|----------|----------------|--------------|-------------------|---------------|
| strong_bull_trend | 0.75 | 0.005 | 0.02 | 0.8x |
| high_vol_ranging | 0.7 | 0.02 | 0.01 | 0.4x |
| consolidation | 0.95 | 0.005 | 0.08 | 0.8x |
| extreme_volatility | 0.6 | 0.025 | 0.005 | 0.2x |

### 3. 動的特徴量選択エンジン

レジームに応じて特徴量の重みを動的に調整：

```json
"feature_weights": {
  "momentum_indicators": 0.9,    // トレンド系で重視
  "trend_indicators": 0.95,     // トレンド系で最重視
  "volatility_indicators": 0.6, // レンジ系で重視
  "volume_indicators": 0.7      // 特殊状態で重視
}
```

### 4. マルチタイムフレーム統合

5つの時間軸を統合した階層的分析：
- **短期 (5-15分)**: エントリー/エグジットタイミング
- **中期 (1-4時間)**: トレンド方向性とレジーム判定
- **長期 (日次)**: 全体的な市場環境把握

### 5. 高度なリスク管理

- **VaR統合**: リアルタイムValue at Risk計算
- **多層ストップシステム**: 固定/トレーリング/時間ベース複合
- **レジーム調整ポジションサイジング**: 12レジームそれぞれに最適化

## 技術仕様

### 設定ファイル
```json
{
  "version": "1.0",
  "algorithm": "ppo",
  "model_name": "ppo_v444_advanced_regime_adaptation",
  "advanced_market_regime": {
    "regime_classifications": {
      // 12レジームそれぞれの定義
    },
    "multi_timeframe_integration": {
      // マルチタイムフレーム設定
    }
  }
}
```

### Analyzer API

```python
from ztb.analysis.v444_regime_analyzer import V444RegimeAnalyzer

analyzer = V444RegimeAnalyzer()

# レジームパフォーマンス分析
performance_matrix = analyzer.analyze_regime_performance_matrix(
    backtest_results, regime_data
)

# レジーム遷移分析
transitions = analyzer.analyze_regime_transitions(
    historical_data, regime_labels
)

# アダプティブ戦略検証
validation = analyzer.validate_adaptive_strategy(
    predictions, performance, context
)
```

## 性能目標

### 改善目標
- **総合リターン**: v443.2比 +25%
- **リスク調整リターン**: +30%
- **ドローダウン**: -20%
- **Sharpe Ratio**: +0.2
- **レジーム適応スコア**: 1.2（従来比+20%）

### 成功基準
- 12レジーム全てでSharpe Ratio > 0.1
- レジーム適応精度 > 80%
- 安定性スコア > 0.7

## 実装ロードマップ

### Phase 1: 基盤構築 (2週間)
- [ ] 12レジーム分類システムの実装
- [ ] 基本的なレジーム別パラメータ設定
- [ ] 単体テストと検証

### Phase 2: 高度機能統合 (3週間)
- [ ] マルチタイムフレーム統合
- [ ] 動的特徴量選択エンジン
- [ ] 高度リスク管理の実装

### Phase 3: 最適化と検証 (2週間)
- [ ] パフォーマンス最適化
- [ ] 包括的バックテスト
- [ ] 実践環境デプロイ

### Phase 4: モニタリング (1週間)
- [ ] ライブトレーディング監視
- [ ] パフォーマンス分析
- [ ] 継続的改善

## 使用方法

### 1. 設定ファイルの準備
```bash
cp config/sac_v444_advanced_regime_adaptation_config.json config/my_v444_config.json
# 必要に応じてパラメータを調整
```

### 2. トレーニング実行
```bash
python scripts/training/train_sac_v444.py --config config/my_v444_config.json
```

### 3. 分析実行
```python
from ztb.analysis.v444_regime_analyzer import create_v444_regime_analysis_report

report = create_v444_regime_analysis_report(
    analyzer, backtest_results, historical_data, regime_labels
)
print(report)
```

## 主要な洞察と教訓

### 深掘り分析の結果

1. **レジーム分類の重要性**
   - 従来の4分類では不十分な市場状態が多く存在
   - 12分類により、より精密な適応が可能に

2. **マルチタイムフレームの効果**
   - 単一時間軸では捉えきれない市場構造を把握
   - 短期/中期/長期の統合が安定性を向上

3. **動的適応の必要性**
   - 静的なパラメータでは最適化が不十分
   - 市場状態に応じた動的調整が鍵

4. **リスク管理の多層化**
   - 単一のリスク管理手法では不十分
   - VaR、多層ストップ、ポジション調整の統合が必要

### 追加の気づき

1. **特徴量の文脈依存性**
   - 同じ特徴量でもレジームによって重要度が異なる
   - レジーム別特徴量重みの最適化が有効

2. **遷移確率の活用**
   - レジーム間の遷移確率を予測することで
   - より積極的なポジション取りが可能

3. **安定性のトレードオフ**
   - 高リターンを目指すと安定性が低下
   - レジーム別リスク調整が重要

## 次のステップ

v444の実装完了後、以下のさらなる改善を検討：

1. **リアルタイム適応の強化**
   - 市場変化に対するより迅速な適応
   - オンライン学習の実装

2. **Ensemble Learningの導入**
   - 複数モデルの統合
   - 多様な戦略の組み合わせ

3. **高度な特徴量エンジニアリング**
   - 深層学習ベースの特徴量生成
   - 市場構造の自動発見

---

*このドキュメントはv444開発中の暫定版です。実装が進むにつれて更新されます。*