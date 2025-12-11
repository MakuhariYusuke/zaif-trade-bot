# SIGNAL_GUIDANCEシステム 包括的改善計画

## 概要

SAC v445 SIGNAL_GUIDANCEシステムの包括的改善計画。現在の勝率34.38%、ドローダウン-0.46%を、勝率50%+、ドローダウン-0.2%以下まで向上させることを目標とする。本ドキュメントは、テクニカル指標再設計から市場レジーム適応、アンサンブル手法、分足対応までをカバーした詳細な実装計画。

### 品質保証とテスト基盤強化 ✅
- **テストコード分離**: SignalQualityScorerのテストスイートを本番コードから分離し、pytest標準に準拠
- **テスト構造最適化**: 単体テスト8件 + 統合テスト1件の包括的テストスイート構築
- **コード品質向上**: 階層化されたテスト構造（tests/unit/trading/signal/quality_scorer/）の実現
- **CI/CD対応**: 自動テストスイートにより継続的品質保証を実現

## 開発フェーズ完了状況

### ✅ Phase 1: テクニカル指標再設計 (COMPLETED)
- RSIスコアリングの改善（オーバーソールド/オーバーバウトの感度調整）
- ウェイトバランスの最適化（RSI 0.22, MACD 0.22, Bollinger 0.18, ATR 0.13, Trend 0.13, Momentum 0.07, Stochastic 0.05）
- ATRの文脈化（市場ボラティリティに応じた解釈）
- 新規指標追加（モメンタム, ストキャスティクス）
- SignalQualityScorerのBUY/SELLバランス改善

### ✅ Phase 4: 分足対応アーキテクチャ (COMPLETED)
- **AdaptiveTimeframeManager**: 市場条件に応じたタイムフレーム適応
- **MultiTimeframeSignalValidator**: 複数タイムフレーム間の一貫性検証
- **MinuteDataPipeline**: 非同期データ取得と品質管理
- **Phase4MinuteTradingManager**: 分足取引統合マネージャー
- **高頻度取引対応**: 1分足/5分足/15分足/1時間足のマルチタイムフレーム処理
- **MarketRegimeClassifier統合**: 既存の市場レジーム分類器との連携
- **動的パラメータ調整**: レジームに応じたウェイトと閾値の適応
- **レジーム別最適化**: 低ボラティリティ/高ボラティリティ/トレンド相場別パラメータ
- **適応性検証**: 異なる市場条件でのパフォーマンス評価

### ⚠️ SIGNAL_GUIDANCE バックテスト結果分析 (CRITICAL ISSUE IDENTIFIED)

#### バックテスト性能調査結果
- **SIGNAL_GUIDANCE実装状況**: Phase 1-4の機能統合完了、V4FeatureExtractorとの互換性確保
- **スコアリング機能**: SIGNAL_GUIDANCEスコアリング正常動作、V4特徴量（Supertrend, Supertrend_Direction, OBV）の適切な抽出
- **性能劣化問題**: SIGNAL_GUIDANCE導入により深刻な性能劣化（平均リターン -81.93% vs ベースライン -6.56%）
- **スコア分布**: SIGNAL_GUIDANCEスコア範囲38-65（平均47.86）、55%が50-54の範囲だが、性能との正の相関なし
- **比較分析**: SIGNAL_GUIDANCEはベースライン比75.38%の性能劣化、根本的なスコアリングロジックの逆転を示唆

#### 技術的問題点
- **スコア解釈問題**: 高いSIGNAL_GUIDANCEスコアが悪い取引判断と相関する可能性
- **V4特徴量マッピング**: V4FeatureExtractor特徴量（Supertrend, Supertrend_Direction, OBV）の正常マッピング、BB_Position近似
- **スコアリングロジック逆転**: 現在の実装でスコア-アクション関係が逆転している可能性、完全な再設計が必要
- **デバッグ分析必要**: SIGNAL_GUIDANCEスコアと実際の取引結果の相関関係の詳細分析

#### 次のステップ
- **スコアリングロジック再設計**: SIGNAL_GUIDANCEスコア解釈とアクションガイダンスの完全な見直し
- **相関分析**: スコア-アクション関係の詳細分析で逆転パターンの特定
- **簡素化実装**: 複雑な重み付け前に基本的なSupertrend_Directionシグナルから開始
- **閾値ベースアプローチ**: SIGNAL_GUIDANCEを直接アクションガイダンスではなくゲーティング機構として検討

### 📋 Phase 3-6: アンサンブル手法以降 (NOT STARTED)
- Phase 3: アンサンブル手法の導入
- Phase 4: 分足対応アーキテクチャ
- Phase 5: 高度なリスク管理統合
- Phase 6: バックテスト環境の拡張

## Phase 2: 市場レジーム適応の詳細設計

### 2.1 16種類の市場レジーム分類システム

現在のMarketRegimeClassifierは、SELLバイアス是正のために設計された16種類のレジームをサポートしています。各レジームは優先度ベースの分類システムで評価され、SIGNAL_GUIDANCEシステムに動的適応を提供します。

#### SELL特化レジーム（最高優先度）
1. **SELL_BREAKDOWN** (優先度: 16)
   - **条件**: trend_strength ≤ -2.5, bear_strength ≥ 1.8, volatility ≥ 0.08
   - **特徴**: 強いブレイクダウンパターン、SELLシグナル優先
   - **SIGNAL_GUIDANCE適応**: RSIウェイト20%減、トレンドウェイト30%増、SELL閾値15

2. **SELL_DIVERGENCE** (優先度: 15)
   - **条件**: -1.5 ≤ trend_strength ≤ 1.5, bear_strength ≥ 1.2, MACD ≤ -0.5, RSI ≤ 65
   - **特徴**: 弱気ダイバージェンス検出、SELL機会強化
   - **SIGNAL_GUIDANCE適応**: MACDウェイト30%、SELL閾値20、モメンタム考慮強化

3. **SELL_MOMENTUM_WEAK** (優先度: 14)
   - **条件**: trend_strength ≤ -1.0, momentum ≤ -0.3, volatility ≥ 0.05, ADX ≥ 20
   - **特徴**: 弱気トレンドでのモメンタム減衰
   - **SIGNAL_GUIDANCE適応**: モメンタムウェイト25%、ATRウェイト20%増

4. **SELL_VOLUME_SURGE** (優先度: 13)
   - **条件**: trend_strength ≤ -1.2, volume_trend ≥ 0.15, Bollinger_position ≤ 0.3
   - **特徴**: 弱気トレンドでの出来高急増
   - **SIGNAL_GUIDANCE適応**: 出来高分析ウェイト20%、SELL閾値15

#### Bullトレンドレジーム（優先度: 12-10）
5. **STRONG_BULL_TREND** (優先度: 12)
   - **条件**: trend_strength ≥ 3.0, bull_strength ≥ 2.5, volatility ≤ 0.15
   - **特徴**: 強力な上昇トレンド、高確信度
   - **SIGNAL_GUIDANCE適応**: トレンドウェイト35%、BUY閾値70、RSIウェイト25%

6. **MODERATE_BULL_TREND** (優先度: 11)
   - **条件**: 2.0 ≤ trend_strength ≤ 3.0, 1.5 ≤ bull_strength ≤ 2.5, volatility ≤ 0.20
   - **特徴**: 安定した上昇トレンド、持続的利益
   - **SIGNAL_GUIDANCE適応**: トレンドウェイト30%、MACDウェイト25%、BUY閾値75

7. **WEAK_BULL_TREND** (優先度: 10)
   - **条件**: 1.0 ≤ trend_strength ≤ 2.0, 0.5 ≤ bull_strength ≤ 1.5, volatility ≤ 0.25
   - **特徴**: 弱い上昇傾向、低モメンタム
   - **SIGNAL_GUIDANCE適応**: RSIウェイト30%、モメンタムウェイト20%、BUY閾値80

#### Bearトレンドレジーム（優先度: 9-7、SELLバイアス是正）
8. **STRONG_BEAR_TREND** (優先度: 9)
   - **条件**: trend_strength ≤ -2.8, bear_strength ≥ 2.2, volatility ≤ 0.18
   - **特徴**: 強力な下降トレンド、高確信度
   - **SIGNAL_GUIDANCE適応**: トレンドウェイト35%、SELL閾値20、RSIウェイト20%

9. **MODERATE_BEAR_TREND** (優先度: 8)
   - **条件**: -1.8 ≥ trend_strength ≥ -2.8, 1.3 ≤ bear_strength ≤ 2.2, volatility ≤ 0.22
   - **特徴**: 安定した下降トレンド、持続的損失
   - **SIGNAL_GUIDANCE適応**: トレンドウェイト30%、MACDウェイト25%、SELL閾値25

10. **WEAK_BEAR_TREND** (優先度: 7)
    - **条件**: -0.8 ≥ trend_strength ≥ -1.8, 0.3 ≤ bear_strength ≤ 1.3, volatility ≤ 0.28
    - **特徴**: 弱い下降傾向、低モメンタム
    - **SIGNAL_GUIDANCE適応**: RSIウェイト30%、モメンタムウェイト20%、SELL閾値30

#### レンジ相場レジーム（優先度: 6-4）
11. **HIGH_VOLATILITY_RANGE** (優先度: 6)
    - **条件**: volatility ≥ 0.15, -2.0 ≤ trend_strength ≤ 2.0
    - **特徴**: 高ボラティリティな横ばい相場
    - **SIGNAL_GUIDANCE適応**: ATRウェイト30%、ボリンジャーウェイト25%、厳格閾値

12. **MODERATE_VOLATILITY_RANGE** (優先度: 5)
    - **条件**: 0.10 ≤ volatility ≤ 0.15, -1.5 ≤ trend_strength ≤ 1.5
    - **特徴**: 中程度ボラティリティのコンソリデーション
    - **SIGNAL_GUIDANCE適応**: ストキャスティクスウェイト25%、レンジ分析強化

13. **LOW_VOLATILITY_RANGE** (優先度: 4)
    - **条件**: volatility ≤ 0.10, -1.0 ≤ trend_strength ≤ 1.0
    - **特徴**: 低ボラティリティのタイトレンジ
    - **SIGNAL_GUIDANCE適応**: サポート/レジスタンスウェイト25%、高頻度適応

#### 特殊条件レジーム（優先度: 3-1）
14. **EXTREME_VOLATILITY** (優先度: 3)
    - **条件**: volatility ≥ 0.20
    - **特徴**: 極端な市場ボラティリティ
    - **SIGNAL_GUIDANCE適応**: ATRウェイト40%、全閾値厳格化、リスク管理優先

15. **CONSOLIDATION** (優先度: 2)
    - **条件**: volatility ≤ 0.08, -0.8 ≤ trend_strength ≤ 0.8
    - **特徴**: 最小変動のコンソリデーション
    - **SIGNAL_GUIDANCE適応**: 低ウェイト全指標、HOLDシグナル優位

16. **BREAKOUT_SETUP** (優先度: 1)
    - **条件**: volatility ≤ 0.12, -1.2 ≤ trend_strength ≤ 1.2, support_resistance ≥ 0.7
    - **特徴**: コンソリデーションからのブレイク準備
    - **SIGNAL_GUIDANCE適応**: ブレイクアウト検出ウェイト30%、ダイナミック閾値

17. **BREAKDOWN_SETUP** (優先度: 1)
    - **条件**: volatility ≤ 0.12, -1.2 ≤ trend_strength ≤ 1.2, support_resistance ≥ 0.7
    - **特徴**: コンソリデーションからのブレークダウン準備
    - **SIGNAL_GUIDANCE適応**: ブレークダウン検出ウェイト30%、SELL優先適応

### 2.2 レジーム分類の技術的詳細

#### メトリクス計算仕様
```python
@dataclass
class RegimeMetrics:
    """レジーム分類用の計算メトリクス"""
    trend_strength: float          # トレンド強度（-5.0 to +5.0）
    bull_strength: float           # 強気強度（0.0 to 5.0）
    bear_strength: float           # 弱気強度（0.0 to 5.0）
    volatility: float              # 正規化ボラティリティ（0.0 to 1.0）
    momentum: float                # モメンタム指標（-1.0 to +1.0）
    volume_trend: float            # 出来高トレンド（-1.0 to +1.0）
    price_range_ratio: float       # 価格レンジ比率（0.0 to 1.0）
    adx: float                     # ADX指標（0 to 100）
    rsi: float                     # RSI（0 to 100）
    macd_signal: float             # MACDシグナル（-5.0 to +5.0）
    bollinger_position: float      # ボリンジャーバンド位置（0.0 to 1.0）
    support_resistance_strength: float  # サポート/レジスタンス強度（0.0 to 1.0）
```

#### 分類アルゴリズム
```python
def _classify_regime(self, metrics: RegimeMetrics) -> Tuple[RegimeType, float]:
    """優先度ベースのレジーム分類"""
    # 全レジーム定義を優先度順に評価
    for regime_def in sorted(self.regime_definitions, key=lambda x: x.priority, reverse=True):
        score, confidence = self._evaluate_regime_conditions(metrics, regime_def)

        if confidence >= self.config.get('confidence_threshold', 0.6):
            return regime_def.regime_type, confidence

    # デフォルト: コンソリデーション
    return RegimeType.CONSOLIDATION, 0.5
```

#### 条件評価ロジック
```python
def _evaluate_regime_conditions(self, metrics: Dict, regime_def: RegimeDefinition) -> Tuple[float, float]:
    """レジーム条件の評価"""
    matched_conditions = 0
    total_conditions = len(regime_def.conditions)

    for metric_name, conditions in regime_def.conditions.items():
        metric_value = getattr(metrics, metric_name)
        condition_met = True

        # min/max条件チェック
        if 'min' in conditions and metric_value < conditions['min']:
            condition_met = False
        if 'max' in conditions and metric_value > conditions['max']:
            condition_met = False

        if condition_met:
            matched_conditions += 1

    confidence = matched_conditions / total_conditions if total_conditions > 0 else 0.0
    return matched_conditions, confidence
```

#### 統合クラス設計
```python
class EnhancedSignalGuidanceSystem(SignalGuidanceSystem):
    """市場レジーム適応型SIGNAL_GUIDANCE"""

    def __init__(self, config: GuidanceConfig):
        super().__init__(config)
        # 市場レジーム分類器統合
        self.regime_classifier = MarketRegimeClassifier()
        self.regime_adaptation = {
            'low_volatility': {
                'rsi_weight': 0.3,
                'trend_weight': 0.25,
                'buy_threshold': 80,
                'sell_threshold': 20,
                'atr_multiplier': 1.0
            },
            'high_volatility': {
                'rsi_weight': 0.2,
                'atr_weight': 0.25,
                'buy_threshold': 85,
                'sell_threshold': 15,
                'atr_multiplier': 2.0
            },
            'trending': {
                'trend_weight': 0.3,
                'macd_weight': 0.25,
                'buy_threshold': 75,
                'sell_threshold': 25,
                'atr_multiplier': 2.5
            },
            'sideways': {
                'bollinger_weight': 0.3,
                'stochastic_weight': 0.25,
                'buy_threshold': 70,
                'sell_threshold': 30,
                'atr_multiplier': 1.2
            }
        }
```

#### レジーム分類連携
```python
def classify_market_regime(self, market_data: pd.DataFrame) -> str:
    """市場レジームの分類"""
    # MarketRegimeClassifierの既存機能を利用
    volatility = self._calculate_volatility(market_data)
    trend_strength = self._calculate_trend_strength(market_data)
    volume_profile = self._analyze_volume_profile(market_data)

    # 分類ロジック
    if volatility < 0.02 and trend_strength < 0.3:
        return 'low_volatility'
    elif volatility > 0.05 and trend_strength < 0.4:
        return 'high_volatility'
    elif trend_strength > 0.6:
        return 'trending'
    else:
        return 'sideways'
```

### 2.3 SIGNAL_GUIDANCEとの統合実装

#### EnhancedSignalGuidanceSystemクラス設計
```python
class EnhancedSignalGuidanceSystem(SignalGuidanceSystem):
    """16レジーム適応型SIGNAL_GUIDANCEシステム"""

    def __init__(self, config: GuidanceConfig):
        super().__init__(config)
        self.regime_classifier = MarketRegimeClassifier()
        self.current_regime = None
        self.regime_history = []

        # レジーム別適応パラメータ
        self.regime_adaptations = self._initialize_regime_adaptations()

    def _initialize_regime_adaptations(self) -> Dict[RegimeType, Dict[str, Any]]:
        """各レジームに対する適応パラメータ初期化"""
        return {
            # SELL特化レジーム - 積極的なSELLシグナル生成
            RegimeType.SELL_BREAKDOWN: {
                'rsi_weight': 0.15, 'macd_weight': 0.30, 'trend_weight': 0.25,
                'buy_threshold': 85, 'sell_threshold': 15, 'hold_threshold': 40,
                'description': '強いSELLシグナル優先'
            },
            RegimeType.SELL_DIVERGENCE: {
                'rsi_weight': 0.20, 'macd_weight': 0.35, 'momentum_weight': 0.20,
                'buy_threshold': 80, 'sell_threshold': 20, 'hold_threshold': 45,
                'description': 'ダイバージェンスベースSELL'
            },
            RegimeType.SELL_MOMENTUM_WEAK: {
                'momentum_weight': 0.30, 'trend_weight': 0.25, 'rsi_weight': 0.20,
                'buy_threshold': 75, 'sell_threshold': 25, 'hold_threshold': 45,
                'description': 'モメンタム減衰SELL'
            },
            RegimeType.SELL_VOLUME_SURGE: {
                'volume_weight': 0.25, 'trend_weight': 0.25, 'bollinger_weight': 0.20,
                'buy_threshold': 80, 'sell_threshold': 20, 'hold_threshold': 40,
                'description': '出来高急増SELL'
            },

            # Bullトレンドレジーム - BUYシグナル優位
            RegimeType.STRONG_BULL_TREND: {
                'trend_weight': 0.35, 'rsi_weight': 0.25, 'macd_weight': 0.20,
                'buy_threshold': 70, 'sell_threshold': 30, 'hold_threshold': 50,
                'description': '強気トレンド追従'
            },
            RegimeType.MODERATE_BULL_TREND: {
                'trend_weight': 0.30, 'macd_weight': 0.25, 'rsi_weight': 0.20,
                'buy_threshold': 75, 'sell_threshold': 25, 'hold_threshold': 45,
                'description': '安定上昇適応'
            },
            RegimeType.WEAK_BULL_TREND: {
                'rsi_weight': 0.30, 'momentum_weight': 0.20, 'trend_weight': 0.20,
                'buy_threshold': 80, 'sell_threshold': 20, 'hold_threshold': 40,
                'description': '弱気上昇支援'
            },

            # Bearトレンドレジーム - SELLシグナル強化
            RegimeType.STRONG_BEAR_TREND: {
                'trend_weight': 0.35, 'rsi_weight': 0.20, 'macd_weight': 0.25,
                'buy_threshold': 80, 'sell_threshold': 20, 'hold_threshold': 40,
                'description': '強気下降追従'
            },
            RegimeType.MODERATE_BEAR_TREND: {
                'trend_weight': 0.30, 'macd_weight': 0.25, 'rsi_weight': 0.20,
                'buy_threshold': 75, 'sell_threshold': 25, 'hold_threshold': 45,
                'description': '安定下降適応'
            },
            RegimeType.WEAK_BEAR_TREND: {
                'rsi_weight': 0.30, 'momentum_weight': 0.20, 'trend_weight': 0.20,
                'buy_threshold': 70, 'sell_threshold': 30, 'hold_threshold': 50,
                'description': '弱気下降支援'
            },

            # レンジ相場レジーム - 厳格なシグナルフィルタリング
            RegimeType.HIGH_VOLATILITY_RANGE: {
                'atr_weight': 0.30, 'bollinger_weight': 0.25, 'stochastic_weight': 0.20,
                'buy_threshold': 85, 'sell_threshold': 15, 'hold_threshold': 35,
                'description': '高ボラティリティレンジ'
            },
            RegimeType.MODERATE_VOLATILITY_RANGE: {
                'bollinger_weight': 0.30, 'stochastic_weight': 0.25, 'rsi_weight': 0.20,
                'buy_threshold': 80, 'sell_threshold': 20, 'hold_threshold': 40,
                'description': '中ボラティリティレンジ'
            },
            RegimeType.LOW_VOLATILITY_RANGE: {
                'support_resistance_weight': 0.25, 'bollinger_weight': 0.25, 'rsi_weight': 0.20,
                'buy_threshold': 75, 'sell_threshold': 25, 'hold_threshold': 45,
                'description': '低ボラティリティレンジ'
            },

            # 特殊条件レジーム
            RegimeType.EXTREME_VOLATILITY: {
                'atr_weight': 0.40, 'trend_weight': 0.20, 'volatility_weight': 0.15,
                'buy_threshold': 90, 'sell_threshold': 10, 'hold_threshold': 30,
                'description': '極端ボラティリティ対応'
            },
            RegimeType.CONSOLIDATION: {
                'bollinger_weight': 0.20, 'rsi_weight': 0.15, 'momentum_weight': 0.10,
                'buy_threshold': 85, 'sell_threshold': 15, 'hold_threshold': 55,
                'description': 'コンソリデーションHOLD優位'
            },
            RegimeType.BREAKOUT_SETUP: {
                'support_resistance_weight': 0.30, 'bollinger_weight': 0.25, 'volume_weight': 0.20,
                'buy_threshold': 75, 'sell_threshold': 25, 'hold_threshold': 45,
                'description': 'ブレイクアウト検出'
            },
            RegimeType.BREAKDOWN_SETUP: {
                'support_resistance_weight': 0.30, 'bollinger_weight': 0.25, 'volume_weight': 0.20,
                'buy_threshold': 80, 'sell_threshold': 20, 'hold_threshold': 40,
                'description': 'ブレークダウン検出'
            }
        }
```

#### 動的適応実行メソッド
```python
def calculate_adaptive_signal_quality(self, market_data: pd.DataFrame,
                                    continuous_action: float,
                                    portfolio: Dict) -> Tuple[int, float]:
    """レジーム適応型シグナル品質計算"""

    # 1. 現在の市場レジーム検出
    regime_result = self.regime_classifier.detect_regime(market_data)
    self.current_regime = regime_result.primary_regime
    self.regime_history.append(regime_result)

    # 2. レジーム別適応パラメータ取得
    regime_params = self.regime_adaptations.get(
        self.current_regime,
        self._get_default_adaptation()  # 未知レジーム用デフォルト
    )

    # 3. 適応ウェイトの適用
    adapted_weights = self._apply_regime_weights(regime_params)

    # 4. 適応されたシグナル品質計算
    quality_score = self._calculate_adaptive_score(
        market_data, adapted_weights, continuous_action, portfolio
    )

    # 5. 適応閾値での離散アクション決定
    discrete_action = self._determine_adaptive_action(
        quality_score, regime_params
    )

    # 6. 適応結果のログ記録
    self._log_regime_adaptation(regime_result, regime_params, quality_score)

    return discrete_action, quality_score

def _apply_regime_weights(self, regime_params: Dict[str, Any]) -> Dict[str, float]:
    """レジーム別ウェイト適用"""
    adapted_weights = self.weights.copy()

    # パラメータで指定されたウェイトを適用
    for param_name, value in regime_params.items():
        if param_name.endswith('_weight'):
            indicator_name = param_name.replace('_weight', '')
            if indicator_name in adapted_weights:
                adapted_weights[indicator_name] = value

    # ウェイト正規化
    total_weight = sum(adapted_weights.values())
    if total_weight > 0:
        adapted_weights = {k: v/total_weight for k, v in adapted_weights.items()}

    return adapted_weights

def _determine_adaptive_action(self, quality_score: float,
                              regime_params: Dict[str, Any]) -> int:
    """適応閾値によるアクション決定"""
    buy_threshold = regime_params.get('buy_threshold', self.buy_threshold)
    sell_threshold = regime_params.get('sell_threshold', self.sell_threshold)
    hold_threshold = regime_params.get('hold_threshold', self.hold_threshold)

    if quality_score >= buy_threshold:
        return 1  # BUY
    elif quality_score <= sell_threshold:
        return -1  # SELL
    else:
        return 0  # HOLD
```
```python
def adapt_to_market_regime(self, market_data: pd.DataFrame) -> Dict[str, Any]:
    """市場レジームに応じた動的パラメータ調整"""
    regime = self.classify_market_regime(market_data)
    regime_params = self.regime_adaptation[regime]

    # ウェイトの動的調整
    adapted_weights = self.weights.copy()
    for indicator, weight in regime_params.items():
        if indicator.endswith('_weight'):
            indicator_name = indicator.replace('_weight', '')
            if indicator_name in adapted_weights:
                adapted_weights[indicator_name] = weight

    # 閾値の適応
    adapted_thresholds = {
        'buy': regime_params.get('buy_threshold', self.buy_threshold),
        'sell': regime_params.get('sell_threshold', self.sell_threshold),
        'hold': self.hold_threshold  # HOLD閾値は固定
    }

    # ATR乗数の適用
    atr_multiplier = regime_params.get('atr_multiplier', 1.0)
    adapted_weights['atr'] *= atr_multiplier

    return {
        'regime': regime,
        'weights': adapted_weights,
        'thresholds': adapted_thresholds,
        'atr_multiplier': atr_multiplier
    }
```

#### 適応結果の適用
```python
def calculate_adaptive_signal_quality(self, market_data: pd.DataFrame,
                                    continuous_action: float,
                                    portfolio: Dict) -> Tuple[int, float]:
    """適応型シグナル品質計算"""
    # レジーム分析とパラメータ適応
    adaptation_params = self.adapt_to_market_regime(market_data)

    # 適応されたウェイトでスコア計算
    quality_score = self._calculate_weighted_score(
        market_data, adaptation_params['weights']
    )

    # 適応された閾値で離散アクション決定
    thresholds = adaptation_params['thresholds']
    if quality_score >= thresholds['buy']:
        discrete_action = 1  # BUY
    elif quality_score <= thresholds['sell']:
        discrete_action = -1  # SELL
    else:
        discrete_action = 0  # HOLD

    return discrete_action, quality_score
```

### 2.4 レジーム別最適化戦略の詳細

#### SELL特化レジーム群の適応戦略

**SELL_BREAKDOWN (強いブレイクダウン)**
- **市場状況**: 明確な下降トレンド形成、強い弱気 momentum
- **適応戦略**:
  - MACDウェイト35% - トレンド転換シグナルの優先
  - トレンドウェイト25% - 下降トレンドの確認
  - RSIウェイト15% - オーバーバウト検出の抑制
  - SELL閾値15 - 積極的なSELLシグナル生成
- **期待効果**: 下降トレンドでの高勝率（70%+）、早期SELL機会の確保

**SELL_DIVERGENCE (弱気ダイバージェンス)**
- **市場状況**: 価格とモメンタムの乖離、潜在的な反転シグナル
- **適応戦略**:
  - MACDウェイト35% - ダイバージェンス検出の強化
  - モメンタムウェイト20% - モメンタム減衰の把握
  - RSIウェイト20% - オーバーバウト領域の監視
  - SELL閾値20 - ダイバージェンス時のSELL優先
- **期待効果**: 反転ポイントでの高精度SELLシグナル

**SELL_MOMENTUM_WEAK (弱いモメンタム)**
- **市場状況**: 下降トレンド継続だが、勢いが衰え始め
- **適応戦略**:
  - モメンタムウェイト30% - 勢い減衰の早期検出
  - トレンドウェイト25% - トレンド継続性の評価
  - ATRウェイト20% - ボラティリティ適応
  - SELL閾値25 - モメンタム減衰時のSELL
- **期待効果**: トレンド終焉時の適切な手仕舞い

**SELL_VOLUME_SURGE (出来高急増)**
- **市場状況**: 下降トレンド中の出来高急増、機関投資家の参入
- **適応戦略**:
  - 出来高ウェイト25% - 出来高急増の検出
  - トレンドウェイト25% - 下降トレンドの確認
  - ボリンジャーウェイト20% - 価格位置の評価
  - SELL閾値20 - 出来高急増時のSELL優先
- **期待効果**: 大口SELLシグナルの高精度検出

#### Bullトレンドレジーム群の適応戦略

**STRONG_BULL_TREND (強い上昇トレンド)**
- **市場状況**: 強力な上昇 momentum、高確信度のトレンド
- **適応戦略**:
  - トレンドウェイト35% - 強トレンドの追従
  - RSIウェイト25% - 押し目買いの最適化
  - MACDウェイト20% - トレンド継続の確認
  - BUY閾値70 - 積極的なBUYシグナル
- **期待効果**: 強気相場での高リターン（勝率65%+）

**MODERATE_BULL_TREND (中程度上昇トレンド)**
- **市場状況**: 安定した上昇トレンド、持続的な利益機会
- **適応戦略**:
  - トレンドウェイト30% - 安定トレンドの追従
  - MACDウェイト25% - トレンド転換の監視
  - RSIウェイト20% - 中間的シグナル生成
  - BUY/SELL閾値75/25 - バランスの取れた適応
- **期待効果**: 安定した上昇相場での堅実なパフォーマンス

**WEAK_BULL_TREND (弱い上昇トレンド)**
- **市場状況**: 弱い上昇傾向、低 momentum環境
- **適応戦略**:
  - RSIウェイト30% - オーバーソールド検出の強化
  - モメンタムウェイト20% - 弱気上昇の把握
  - トレンドウェイト20% - トレンド方向性の確認
  - BUY閾値80 - 弱気相場でのBUY機会確保
- **期待効果**: 弱気相場での機会損失低減

#### Bearトレンドレジーム群の適応戦略

**STRONG_BEAR_TREND (強い下降トレンド)**
- **市場状況**: 強力な下降 momentum、高確信度の弱気トレンド
- **適応戦略**:
  - トレンドウェイト35% - 強トレンドの追従
  - MACDウェイト25% - トレンド継続の確認
  - RSIウェイト20% - 戻り売りの最適化
  - SELL閾値20 - 積極的なSELLシグナル
- **期待効果**: 弱気相場での高リターン（勝率65%+）

**MODERATE_BEAR_TREND (中程度下降トレンド)**
- **市場状況**: 安定した下降トレンド、持続的なSELL機会
- **適応戦略**:
  - トレンドウェイト30% - 安定トレンドの追従
  - MACDウェイト25% - トレンド転換の監視
  - RSIウェイト20% - 中間的シグナル生成
  - BUY/SELL閾値75/25 - バランスの取れた適応
- **期待効果**: 安定した下降相場での堅実なパフォーマンス

**WEAK_BEAR_TREND (弱い下降トレンド)**
- **市場状況**: 弱い下降傾向、低 momentum環境
- **適応戦略**:
  - RSIウェイト30% - オーバーバウト検出の強化
  - モメンタムウェイト20% - 弱気下降の把握
  - トレンドウェイト20% - トレンド方向性の確認
  - SELL閾値30 - 弱気相場でのSELL機会確保
- **期待効果**: 弱気相場での機会損失低減

#### レンジ相場レジーム群の適応戦略

**HIGH_VOLATILITY_RANGE (高ボラティリティレンジ)**
- **市場状況**: 高ボラティリティな横ばい相場、ノイズの多い環境
- **適応戦略**:
  - ATRウェイト30% - ボラティリティ適応の優先
  - ボリンジャーウェイト25% - レンジ境界の検出
  - ストキャスティクスウェイト20% - オーバーバウト/オーバーソールド
  - 厳格閾値85/15 - 高品質シグナルのみ取引
- **期待効果**: 高ボラティリティ環境でのリスク低減

**MODERATE_VOLATILITY_RANGE (中ボラティリティレンジ)**
- **市場状況**: 中程度のボラティリティ、比較的安定したレンジ
- **適応戦略**:
  - ボリンジャーウェイト30% - レンジ取引の最適化
  - ストキャスティクスウェイト25% - 反転シグナルの検出
  - RSIウェイト20% - 中間的シグナル生成
  - 閾値80/20 - 適度な取引頻度
- **期待効果**: 中ボラティリティ環境での安定収益

**LOW_VOLATILITY_RANGE (低ボラティリティレンジ)**
- **市場状況**: 低ボラティリティのタイトレンジ、微小変動相場
- **適応戦略**:
  - サポート/レジスタンスウェイト25% - キーレベル検出
  - ボリンジャーウェイト25% - 狭いレンジ適応
  - RSIウェイト20% - 高頻度シグナル生成
  - 閾値75/25 - 低ボラティリティ対応
- **期待効果**: 低ボラティリティ環境での高頻度取引機会

#### 特殊条件レジーム群の適応戦略

**EXTREME_VOLATILITY (極端ボラティリティ)**
- **市場状況**: 異常な市場ボラティリティ、フラッシュクラッシュの可能性
- **適応戦略**:
  - ATRウェイト40% - ボラティリティ適応の最大化
  - トレンドウェイト20% - トレンド方向性の把握
  - ボラティリティウェイト15% - 異常検知
  - 超厳格閾値90/10 - リスク回避優先
- **期待効果**: 極端相場での損失最小化

**CONSOLIDATION (コンソリデーション)**
- **市場状況**: 最小変動の停滞相場、方向性不明
- **適応戦略**:
  - 全指標ウェイト低減 - ノイズシグナル回避
  - HOLD閾値55 - HOLDシグナルの優位
  - 厳格閾値85/15 - 高品質シグナルのみ取引
  - 低頻度取引適応
- **期待効果**: 停滞相場での不必要な取引回避

**BREAKOUT_SETUP/BREAKDOWN_SETUP (ブレイク準備)**
- **市場状況**: コンソリデーションからのブレイク準備段階
- **適応戦略**:
  - サポート/レジスタンスウェイト30% - キーレベル監視
  - ボリンジャーウェイト25% - ブレイクシグナル検出
  - 出来高ウェイト20% - ブレイク確認
  - 適度な閾値設定 - ブレイク機会の確保
- **期待効果**: 主要ブレイクムーブでの高精度エントリー

### 2.5 16レジーム適応の実装ロードマップと検証計画

#### Week 1-2: 基本統合とSELL特化レジーム実装
- [ ] MarketRegimeClassifierのSIGNAL_GUIDANCE統合完了
- [ ] SELL_BREAKDOWN, SELL_DIVERGENCE, SELL_MOMENTUM_WEAK, SELL_VOLUME_SURGEの実装
- [ ] SELLバイアス是正の検証（バックテストでのSELLシグナル増加確認）
- [ ] 単体テスト: レジーム分類精度70%以上の確認

#### Week 3-4: Bull/Bearトレンドレジームの実装
- [ ] STRONG_BULL/MODERATE_BULL/WEAK_BULLの実装とチューニング
- [ ] STRONG_BEAR/MODERATE_BEAR/WEAK_BEARの実装とチューニング
- [ ] トレンド相場でのパフォーマンス改善検証
- [ ] 統合テスト: トレンドレジームでの適応効果測定

#### Week 5-6: レンジ相場レジームの実装と最適化
- [ ] HIGH/MODERATE/LOW_VOLATILITY_RANGEの実装
- [ ] レンジ相場でのノイズシグナル低減検証
- [ ] ボリンジャー/ストキャスティクス適応の最適化
- [ ] パフォーマンステスト: レンジ相場での勝率向上確認

#### Week 7-8: 特殊条件レジームの実装完了
- [ ] EXTREME_VOLATILITY, CONSOLIDATIONの実装
- [ ] BREAKOUT_SETUP, BREAKDOWN_SETUPの実装
- [ ] 特殊相場でのリスク管理効果検証
- [ ] エンドツーエンドテスト: 全16レジームの適応機能

#### Week 9-10: 包括的検証とチューニング
- [ ] 複数レジームでのクロスバリデーション
- [ ] Optunaによるレジーム別パラメータ最適化
- [ ] 実データでのバックテスト（2020-2024年）
- [ ] パフォーマンス指標: 勝率50%+, ドローダウン-0.2%以下

#### Week 11-12: 実装完了とPhase 3移行準備
- [ ] 最終的なパラメータセットの確定
- [ ] Phase 3 Enhancedバックテスト環境への統合
- [ ] 包括的ドキュメント更新
- [ ] Phase 3: アンサンブル手法の設計開始

### レジーム適応の検証フレームワーク

#### RegimeAdaptiveBacktesterクラス
```python
class RegimeAdaptiveBacktester:
    """16レジーム適応の包括的検証"""

    def __init__(self, enhanced_system: EnhancedSignalGuidanceSystem):
        self.system = enhanced_system
        self.regime_performance = defaultdict(list)
        self.adaptation_metrics = defaultdict(list)

    def run_comprehensive_regime_validation(self, market_data: pd.DataFrame,
                                          test_periods: List[Tuple[str, str]]) -> Dict[str, Any]:
        """包括的なレジーム適応検証"""

        results = {
            'regime_detection_accuracy': {},
            'adaptation_effectiveness': {},
            'performance_by_regime': {},
            'cross_regime_stability': {},
            'overall_improvement': {}
        }

        for period_name, (start_date, end_date) in test_periods:
            period_data = market_data[start_date:end_date]

            # レジーム分類精度検証
            regime_accuracy = self._validate_regime_detection_accuracy(period_data)
            results['regime_detection_accuracy'][period_name] = regime_accuracy

            # 適応効果の測定
            adaptation_effect = self._measure_adaptation_effectiveness(period_data)
            results['adaptation_effectiveness'][period_name] = adaptation_effect

            # レジーム別パフォーマンス
            regime_performance = self._analyze_regime_specific_performance(period_data)
            results['performance_by_regime'][period_name] = regime_performance

        # クロスレジーム安定性分析
        results['cross_regime_stability'] = self._analyze_cross_regime_stability(results)

        # 全体改善度の計算
        results['overall_improvement'] = self._calculate_overall_improvement(results)

        return results

    def _validate_regime_detection_accuracy(self, data: pd.DataFrame) -> Dict[str, float]:
        """レジーム分類精度の検証"""
        # 実際の市場データでの分類精度測定
        # 専門家判断との比較や統計的妥当性チェック
        pass

    def _measure_adaptation_effectiveness(self, data: pd.DataFrame) -> Dict[str, float]:
        """適応パラメータの有効性測定"""
        # 適応前後のパフォーマンス比較
        # 各レジームでの改善度計算
        pass

    def _analyze_regime_specific_performance(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """レジーム別パフォーマンス分析"""
        # 各レジームでの勝率、ドローダウン、リターン分析
        # レジーム遷移時のパフォーマンス変動分析
        pass
```

#### 検証指標の定義
```python
@dataclass
class RegimeAdaptationMetrics:
    """レジーム適応の検証指標"""

    # 分類精度指標
    regime_detection_accuracy: float      # レジーム分類精度（0-1）
    false_positive_rate: float           # 誤分類率
    regime_transition_accuracy: float    # レジーム遷移検出精度

    # 適応効果指標
    adaptation_improvement: float        # 適応による改善度（%）
    parameter_stability: float          # パラメータ安定性スコア
    regime_specific_win_rate: Dict[RegimeType, float]  # レジーム別勝率

    # パフォーマンス指標
    overall_win_rate: float             # 全体勝率
    max_drawdown: float                 # 最大ドローダウン
    sharpe_ratio: float                 # シャープレシオ
    regime_consistency_score: float     # レジーム間一貫性スコア

    # リスク指標
    volatility_adjusted_return: float   # ボラティリティ調整リターン
    regime_risk_adjustment: float       # レジーム別リスク調整スコア
    tail_risk_measure: float           # テールリスク指標
```

### 期待される検証結果

#### レジーム分類精度目標
- **全体精度**: 75%以上のレジーム正分類率
- **SELL特化レジーム**: 80%以上の検出精度
- **トレンドレジーム**: 70%以上の検出精度
- **レンジレジーム**: 65%以上の検出精度

#### 適応効果目標
- **SELLバイアス是正**: SELLシグナル比率が40%→50%へ改善
- **勝率向上**: 非適応比で+5-10ptの改善
- **ドローダウン低減**: 最大ドローダウン-0.3%→-0.2%へ改善
- **安定性向上**: レジーム間パフォーマンス変動の低減

#### パフォーマンス目標（Phase 2完了時）
| レジームタイプ | 目標勝率 | 目標ドローダウン | 特徴 |
|---------------|---------|----------------|------|
| SELL特化レジーム | 60-70% | -0.15% | 高精度SELL |
| Bullトレンド | 55-65% | -0.18% | トレンド追従 |
| Bearトレンド | 55-65% | -0.18% | トレンド追従 |
| 高ボラティリティ | 45-55% | -0.25% | リスク管理 |
| 中ボラティリティ | 50-60% | -0.20% | 安定収益 |
| 低ボラティリティ | 55-65% | -0.15% | 高頻度 |
| 特殊条件 | 40-50% | -0.30% | リスク回避 |

### 継続的改善プロセス

#### 定期検証サイクル
1. **日次モニタリング**: レジーム分類精度と適応効果の追跡
2. **週次レビュー**: パフォーマンス指標の分析とパラメータ調整
3. **月次最適化**: Optunaによる自動パラメータチューニング
4. **四半期レビュー**: アーキテクチャの見直しと改善策立案

#### 改善策の優先順位付け
1. **高影響・低リスク**: レジーム分類精度の改善
2. **高影響・中リスク**: パラメータ適応ロジックの最適化
3. **中影響・低リスク**: 新規指標の追加検討
4. **低影響・高リスク**: 根本的なアーキテクチャ変更
```python
class RegimeAdaptiveBacktester:
    """レジーム適応バックテスト検証"""

    def __init__(self, enhanced_system: EnhancedSignalGuidanceSystem):
        self.system = enhanced_system
        self.regime_performance = defaultdict(list)

    def run_regime_specific_backtest(self, market_data: pd.DataFrame,
                                   regime_labels: List[str]) -> Dict[str, Dict]:
        """レジーム別バックテスト実行"""
        results = {}

        for regime in set(regime_labels):
            regime_data = market_data[regime_labels == regime]
            if len(regime_data) < 100:  # 最小データ要件
                continue

            # レジーム別パフォーマンス評価
            performance = self._evaluate_regime_performance(regime_data, regime)
            results[regime] = performance

            # 適応パラメータの有効性検証
            adaptation_effectiveness = self._measure_adaptation_effectiveness(
                regime_data, regime
            )
            results[regime]['adaptation_score'] = adaptation_effectiveness

        return results

    def _measure_adaptation_effectiveness(self, data: pd.DataFrame, regime: str) -> float:
        """適応パラメータの有効性測定"""
        # 適応前後のパフォーマンス比較
        pre_adaptation_score = self._calculate_base_performance(data)
        post_adaptation_score = self._calculate_adaptive_performance(data, regime)

        # 改善度の計算
        improvement = (post_adaptation_score - pre_adaptation_score) / pre_adaptation_score
        return max(0, improvement)  # 負の改善は0にクリップ
```

#### パフォーマンス指標
```python
def calculate_regime_adaptation_metrics(self) -> Dict[str, float]:
    """レジーム適応のパフォーマンス指標計算"""
    return {
        'regime_detection_accuracy': self._calculate_regime_accuracy(),
        'parameter_adaptation_effectiveness': self._calculate_adaptation_effectiveness(),
        'cross_regime_performance_stability': self._calculate_stability_score(),
        'overall_adaptation_improvement': self._calculate_overall_improvement()
    }
```

### 2.5 実装ロードマップと検証計画

#### Week 1-2: 基本統合
- [ ] MarketRegimeClassifierのSIGNAL_GUIDANCE統合
- [ ] 基本的なレジーム分類機能の実装
- [ ] 静的パラメータ適応のテスト

#### Week 3-4: 動的適応
- [ ] リアルタイムレジーム検出の実装
- [ ] 動的パラメータ調整ロジックの開発
- [ ] レジーム遷移時のスムーズな適応

#### Week 5-6: 最適化と検証
- [ ] レジーム別パラメータのOptuna最適化
- [ ] 包括的なバックテスト検証
- [ ] 適応効果の定量評価

#### Week 7-8: 実装完了と統合テスト
- [ ] Phase 3 Enhancedバックテスト環境への統合
- [ ] エンドツーエンドの適応性テスト
- [ ] パフォーマンス改善の最終検証

## 期待される改善効果

### Phase 2完了後の目標達成度
| 項目 | Phase 1後 | Phase 2目標 | 期待改善 |
|------|----------|------------|----------|
| 勝率 | 40-45% | 50-55% | +5-10pt |
| ドローダウン | -0.3% | -0.2% | 33%低減 |
| レジーム適応性 | なし | 動的適応 | 新機能 |
| シグナル品質 | 中程度 | 文脈適応 | 向上 |

### レジーム別パフォーマンス目標
- **低ボラティリティ**: 勝率55%+（高頻度取引）
- **高ボラティリティ**: 勝率45%+（高品質シグナル）
- **トレンド相場**: 勝率60%+（トレンド追従）
- **サイドウェイズ**: 勝率40%+（レンジ取引）

## 技術的実装詳細

### 依存関係
- `ztb.trading.signal.quality_scorer.SignalQualityScorer` - ベースシグナル品質計算
- `ztb.analysis.market_regime.MarketRegimeClassifier` - 市場レジーム分類
- `pandas` - データ処理
- `numpy` - 数値計算

### 設定パラメータ
```python
regime_adaptation_config = {
    'adaptation_enabled': True,
    'regime_detection_window': 50,  # レジーム検出ウィンドウ
    'adaptation_smoothing': 0.8,    # パラメータ変更のスムージング
    'min_regime_duration': 20,      # 最小レジーム持続期間
    'regime_confidence_threshold': 0.7  # レジーム分類信頼度閾値
}
```

### テストと検証
- **単体テスト**: レジーム分類とパラメータ適応のテスト
- **統合テスト**: SIGNAL_GUIDANCE + MarketRegimeClassifierの連携テスト
- **バックテスト**: 複数レジームでのパフォーマンス検証
- **A/Bテスト**: 適応型 vs 非適応型の比較

## 次のステップ

### Phase 3: アンサンブル手法の導入 (FUTURE)
- 多ソースシグナル統合
- 信頼度計算の実装
- アンサンブルウェイト最適化

### Phase 4: 分足対応アーキテクチャ (FUTURE)
- AdaptiveTimeframeManager実装
- MultiTimeframeSignalValidator開発
- 分足データパイプライン構築

## 成功基準

1. **レジーム分類精度**: 70%以上の市場レジーム正分類率
2. **適応効果**: 非適応比10%以上のパフォーマンス改善
3. **安定性**: 異なるレジームでの一貫したパフォーマンス
4. **リアルタイム性**: 許容可能な計算コストでの適応
5. **堅牢性**: エッジケースでの適切なフォールバック動作

## 関連ファイル

- `ztb/trading/signal/quality_scorer.py` - SignalQualityScorerベースクラス
- `ztb/analysis/market_regime.py` - MarketRegimeClassifier
- `tests/unit/trading/signal/quality_scorer/` - テストスイート
- `docs/analysis/SIGNAL_GUIDANCE_COMPREHENSIVE_IMPROVEMENT_PROPOSAL.md` - 詳細設計

## まとめ

Phase 2: 市場レジーム適応の実装により、SIGNAL_GUIDANCEシステムは16種類の市場レジームを考慮した動的適応能力を獲得します。各レジームに対して最適化されたパラメータと戦略により、様々な市場条件での安定したパフォーマンスを実現し、全体的な勝率とリスク管理を大幅に改善することが期待されます。

**最終更新**: 2025-11-12
**ステータス**: Phase 2 IN PROGRESS (16レジーム詳細設計完了、実装開始)
**次回更新予定**: Phase 2実装完了後（Week 12）
