# Action Signal Guide 統合ドキュメント

## システム概要

Action Signal Guideは、SAC（Soft Actor-Critic）強化学習システムの補助機能として位置づけられる高度なテクニカル分析パターン認識システムです。古典的な日本式ローソク足パターンと西洋テクニカル指標を統合し、RLエージェントの学習効率を向上させます。

### 主要機能

#### 1. パターン認識 (27種類以上)
- **調和パターン**: Gartley, Butterfly, Bat, Crab
- **ダウ理論**: 主要トレンドと修正波
- **フィボナッチ分析**:  retracement, extension, projection
- **ローソク足パターン**: 伝統的な日本式11パターン
- **オシレーター**: RSI, Stochastic, MACD (適応型閾値)
- **ADXパターン**: トレンド強度分析
- **ボリンジャーバンド**: ボラティリティベースシグナル

#### 2. 適応型アルゴリズム
- **レジーム連動型特徴量調整**: SAC v444の16レジーム分類に基づく動的スケーリング
- **並列処理**: ThreadPoolExecutorによる2.2倍速度向上
- **LRUキャッシュ**: メモリ効率の高いキャッシュシステム

#### 3. 信号集約戦略
- **重み付き平均集約**: 強度×信頼度ベースの統合
- **パターン別重み付け**: 調和(1.2), ダウ理論(1.1), フィボナッチ(1.0), ローソク足(0.9)
- **階層的フィルタリング**: 信頼度→整合性→環境適合性の3段階検証

#### 4. 多時間足分析
- **5段階階層**: 1分→5分→15分→1時間→4時間
- **アライメントスコアリング**: 時間軸間一貫性評価
- **適応型時間軸選択**: 市場ボラティリティに応じた重み調整

## 技術仕様

### アーキテクチャ

```
ActionSignalGuide (メインクラス)
├── SignalGenerator (信号生成コア)
│   ├── PatternRecognizer (個別パターン認識)
│   ├── PatternStatistics (パフォーマンス追跡)
│   └── CacheManager (LRUキャッシュ)
├── PerformanceTracker (学習効果分析)
├── MarketRegimeAdapter (市場レジーム適応)
└── Validation (シグナル品質検証)
```

### パフォーマンス特性

#### 処理性能
- **並列処理**: 4スレッド自動調整
- **キャッシュ**: LRU 1000エントリ、TTL 300秒
- **バッチ処理**: 5000データポイントを126秒で処理

#### シグナル品質
- **生成数**: 5000データポイント → 5000シグナル
- **分布**: 買い131, 売り203, ホールド4666
- **パターン別性能**: ADX(0.54), Wave(0.63), Oscillator(0.72)

### 設定パラメータ

#### 基本設定
```python
ActionSignalGuideConfig = {
    'guidance_level': 'weak',  # none, weak, strong
    'enabled_patterns': ['fibonacci', 'harmonic', 'candlestick'],
    'signal_weights': {'bullish': 0.7, 'bearish': 0.8},
    'cache_size': 1000,
    'max_signals_per_bar': 5
}
```

#### パターン別設定
```python
pattern_config = {
    'fibonacci': {
        'retracement_levels': [0.236, 0.382, 0.5, 0.618, 0.786],
        'extension_levels': [1.272, 1.414, 1.618, 2.618],
        'deviation_tolerance': 0.05
    },
    'harmonic': {
        'ratio_tolerance': 0.05,
        'min_amplitude': 0.01,
        'max_amplitude': 0.10
    }
}
```

## 使用方法

### 基本的な使用例

#### 1. 初期化
```python
from ztb.trading.strategies.action_signal_guide import ActionSignalGuide
from ztb.tests.unit.trading.strategies.action_signal_guide import get_optimized_config

# 最適化設定取得
config = get_optimized_config()

# Action Signal Guide初期化
guide = ActionSignalGuide(config)
```

#### 2. シグナル生成
```python
# 市場データでのシグナル生成
signals = guide.generate_signals(market_data)

# シグナル処理
for signal in signals:
    if signal.strength > 0.7:
        print(f"強力な{signal.type}シグナル: {signal.description}")
        # 取引実行ロジック
```

#### 3. バッチ処理
```python
# 大量データの一括処理
all_signals = guide.generate_signals_batch(large_dataset)
print(f"生成されたシグナル数: {len(all_signals)}")
```

### 高度な使用例

#### SAC学習との統合
```python
# トレーニング中のシグナル活用
state = env.get_state()
action_signals = guide.get_action_signals(state)

# 報酬へのシグナル反映
if action_signals.preferred_action == agent_action:
    reward += signal_alignment_bonus
```

#### リアルタイム適応
```python
# 市場レジームに応じた動的調整
regime = guide.detect_market_regime(current_data)
guide.adjust_for_regime(regime)

# パフォーマンス監視
stats = guide.get_signal_statistics()
guide.optimize_weights_based_on_performance(stats)
```

## 実装状況とロードマップ

### 完了済み機能 (Phase 1-2)
- ✅ 基本パターン認識システム
- ✅ 適応型特徴量前処理
- ✅ 並列処理最適化
- ✅ LRUキャッシュシステム
- ✅ 信号集約アルゴリズム
- ✅ 多時間足サポート
- ✅ パフォーマンス統計追跡

### 開発中機能 (Phase 3)
- 🔄 動的閾値調整システム
- 🔄 リスク管理機能 (ストップロス/テイクプロフィット)
- 🔄 パターン認識器パフォーマンス分析
- 🔄 高度なシグナル集約アルゴリズム
- 🔄 パターンベース動的重み付け

### 計画機能 (Phase 4-5)
- 📋 機械学習ベース信号融合
- 📋 オンライン学習システム
- 📋 リアルタイム適応アルゴリズム
- 📋 ポートフォリオレベル最適化

## パフォーマンス分析

### バックテスト結果
```
最終資本: $1,774,568,573 (17,744倍リターン)
年間リターン: 63.74%
最大ドローダウン: -795.78%
Sharpe Ratio: -0.25
勝率: 59.76%
総取引数: 169
プロフィットファクター: 1.82
```

### パターン別貢献度
- **ADXパターン**: 相関0.106 (最高貢献)
- **Waveパターン**: 強度0.63
- **Oscillatorパターン**: 強度0.72
- **Fibonacciパターン**: 安定したサポート/レジスタンス

### 課題と改善点
1. **過剰最適化の可能性**: 17744倍リターンは現実的でない
2. **リスク調整不足**: 負のSharpe Ratio
3. **適応性の限界**: 市場変化への対応が不十分

## 拡張ガイド

### 新規パターン追加
```python
class CustomPattern(PatternRecognizer):
    def recognize(self, data: pd.DataFrame) -> List[PatternResult]:
        # カスタムパターン認識ロジック
        patterns = self._detect_custom_patterns(data)
        return self._create_pattern_results(patterns)
```

### 設定のカスタマイズ
```python
# カスタム設定の作成
custom_config = ActionSignalGuideConfig(
    guidance_level='strong',
    enabled_patterns=['custom_pattern'],
    custom_weights={'custom': 1.5}
)

guide = ActionSignalGuide(custom_config)
```

## テストと検証

### 単体テスト
```bash
# Action Signal Guideテスト実行
pytest tests/unit/trading/strategies/action_signal_guide/ -v
```

### 統合テスト
```bash
# SACとの統合テスト
pytest tests/integration/test_sac_action_signal_guide.py -v
```

### バックテスト検証
```bash
# 独立バックテスト
python backtest/backtest_action_signal_guide.py
```

## 参照ドキュメント

1. **メイン仕様**: `docs/features/action_signal_guide.md`
2. **技術詳細**: `docs/CURRENT_STATUS_IMPROVEMENTS_FUTURE_OUTLOOK.md`
3. **使用例**: `README.md` (Action Signal Guideセクション)
4. **実装例**: `ztb/trading/strategies/action_signal_guide/`

## 更新履歴

- **2025-11-09**: 統合ドキュメント作成、収益化施策整理
- **2025-11-05**: SAC v444統合完了、適応型アルゴリズム実装
- **2025-10-15**: 基本パターン認識システム完成
- **2025-09-01**: プロジェクト開始、パターン認識基盤構築
