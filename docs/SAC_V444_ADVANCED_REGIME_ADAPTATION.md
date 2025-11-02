# SAC v444: Advanced Market Regime Adaptation System

## 概要

SAC v444は、市場状態を16種類に細分化した高度な適応型トレーディングシステムです。従来の4分類（Bull/Bear/Sideways/Volatile）を大幅に拡張し、より精密な市場適応を実現します。特にSELL bias対策として、4つのSELL特化レジームを追加しました。

## 主な特徴

### 1. 16レジーム分類システム

市場状態を以下の16種類に分類：

#### 強気トレンド系
- **strong_bull_trend**: 明確な上昇相場（トレンド強度 > 0.02）
- **moderate_bull_trend**: 中程度の上昇相場（トレンド強度 0.01-0.02）
- **weak_bull_trend**: 弱い上昇相場（トレンド強度 0.005-0.01）

#### 弱気トレンド系
- **strong_bear_trend**: 明確な下降相場（トレンド強度 < -0.02）
- **moderate_bear_trend**: 中程度の下降相場（トレンド強度 -0.02～-0.01）
- **weak_bear_trend**: 弱い下降相場（トレンド強度 -0.01～-0.005）

#### SELL特化レジーム（SELL bias対策）
- **sell_breakdown**: 強いbreakdownパターン（優先度16）
- **sell_divergence**: 弱気ダイバージェンス検出（優先度15）
- **sell_momentum_weak**: 弱いモメンタムでの下降（優先度14）
- **sell_volume_surge**: 出来高急増時の下降（優先度13）

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

| レジーム | Action Balance | Entropy Reg. | Consistency Penalty | Position Size | SELL Bias |
|----------|----------------|--------------|-------------------|---------------|-----------|
| strong_bull_trend | 0.75 | 0.005 | 0.02 | 0.8x | 0.3 |
| sell_breakdown | 0.4 | 0.01 | 0.05 | 0.6x | 0.8 |
| sell_divergence | 0.45 | 0.008 | 0.04 | 0.5x | 0.75 |
| sell_momentum_weak | 0.5 | 0.007 | 0.03 | 0.4x | 0.7 |
| sell_volume_surge | 0.48 | 0.009 | 0.035 | 0.45x | 0.72 |
| strong_bear_trend | 0.35 | 0.012 | 0.06 | 0.7x | 0.85 |
| high_vol_ranging | 0.7 | 0.02 | 0.01 | 0.4x | 0.4 |
| consolidation | 0.95 | 0.005 | 0.08 | 0.8x | 0.35 |
| extreme_volatility | 0.6 | 0.025 | 0.005 | 0.2x | 0.5 |

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
- **レジーム調整ポジションサイジング**: 16レジームそれぞれに最適化

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

## バックテスト修正と正規化改善 (2025-11-02)

### アクション分布バランスの問題解決

#### 問題の特定
- **アクション固定問題**: バックテストで常に同じアクション（例: 100% BUY）が出力される
- **正規化統計不一致**: トレーニング時の68特徴量 vs バックテスト時の212特徴量
- **確定的予測の影響**: `deterministic=True`によるエントロピー低下

#### 実装された解決策

##### 1. 正規化統計の再生成
```python
# 環境ウォームアップによる統計収集
for _ in range(5000):
    obs = vec_env.reset()
    action = model.predict(obs, deterministic=False)[0]
    obs, _, _, _ = vec_env.step(action)

# VecNormalizeから統計を抽出
norm_stats = NormalizationStats.from_vec_normalize(vec_norm, feature_names)
norm_stats.save("models/scaler_v444_regenerated.npz")
```

##### 2. 確率的予測の実装
```python
# 確率的サンプリングによる多様性確保
action, _ = model.predict(obs_for_predict, deterministic=False)
```

##### 3. 環境設定の統一
- `curriculum_stage="forced_balance"` の強制適用
- 連続アクション空間の維持
- 報酬クリッピングの拡張 (-10000 to 10000)

#### 結果の改善
- **アクション分布**: HOLD 28.3%, BUY 36.6%, SELL 35.1% (1000ステップ)
- **特徴量整合性**: 212特徴量の完全一致
- **強制バランスペナルティ**: 計算とログ出力の確認

### コード品質の向上
- **型安全性**: バックテストスクリプトの型アノテーション改善
- **エラーハンドリング**: 環境初期化とモデル読み込みの強化
- **デバッグ機能**: 最初の5ステップの詳細ログ出力

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

## 過学習対策とロバストネス向上

### 過学習リスクの特定

現在のシステムは122個の特徴量を使用しており、以下の過学習リスクが懸念されます：

1. **特徴量過多**: 122個の特徴量はトレーニングデータに過剰適合する可能性
2. **レジーム分類の複雑化**: 16レジーム分類による過学習
3. **弱気シグナル特徴量の追加**: 13個のSELL bias対策特徴量による局所最適解

### 実装済みの対策

#### 1. 特徴量次元削減
- **相関分析**: 高度に相関する特徴量の除去
- **重要度ランキング**: 特徴量重要度に基づく選択的上位50-70個の使用
- **PCA適用**: 主成分分析による次元圧縮（検討中）

#### 2. 正則化の強化
- **L2正則化**: ネットワーク重みのペナルティ増加
- **ドロップアウト**: 隠れ層に0.1-0.2のドロップアウト適用
- **バッチ正規化**: 各層での安定した学習

#### 3. トレーニング戦略の改善
- **早期停止**: 検証損失が改善しなくなった時点で停止
- **学習率減衰**: 学習が進むにつれて学習率を段階的に減少
- **データ拡張**: 市場データの時間軸シフトによる多様性確保

### 追加検討中の対策

#### 1. アンサンブル学習
```python
# 複数モデルの統合によるロバストネス向上
ensemble_models = [
    SACModel(config='conservative'),
    SACModel(config='aggressive'),
    SACModel(config='balanced')
]
```

#### 2. 交差検証の実装
- **時系列交差検証**: 過去データでの複数分割検証
- **ウォークフォワード最適化**: 順次的なトレーニング期間の拡張

#### 3. 特徴量選択の自動化
- **ランダムフォレスト重要度**: 特徴量重要度の自動計算
- **再帰的特徴量除去**: 重要度の低い特徴量を順次除去
- **LASSO回帰**: L1正則化による特徴量選択

#### 4. アーキテクチャの簡素化
- **ネットワークサイズ削減**: 隠れ層のニューロン数を50%削減
- **特徴量グループ化**: 相関の高い特徴量を統合
- **Attention機構**: 重要な特徴量への集中学習

### 評価指標の拡張

過学習検知のための追加指標：
- **トレーニング/検証スコアのギャップ**: 0.1以上のギャップで過学習警告
- **特徴量重要度の偏り**: 特定特徴量への過度な依存を検知
- **予測安定性**: 異なる市場条件での予測一貫性評価

### 実装計画

1. **Phase 1**: 特徴量次元削減（次回トレーニングから適用）
2. **Phase 2**: 正則化強化と早期停止の実装
3. **Phase 3**: アンサンブル学習の導入
4. **Phase 4**: 自動特徴量選択システムの実装

---

*このドキュメントはv444開発中の暫定版です。実装が進むにつれて更新されます。*
