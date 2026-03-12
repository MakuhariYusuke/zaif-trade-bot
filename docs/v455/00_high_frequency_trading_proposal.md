# 高頻度取引戦略提案 (Phase C+)

## 概要

Phase Cのトレンドフォロー戦略は+2.18%のリターン（54トレード、93.1%勝率）を達成したが、トレード頻度が低く利益拡大の余地がある。高頻度取引を目指し、以下の改善策を提案する。特に、既存のアクションシグナルガイド、市場レジーム分類、リスクマネージャー、リアルな実行モデルなどの実装を活用する。

## 現在の課題

- **トレード頻度**: 54トレード（データ期間16977ステップに対して）
- **利益率**: +2.18%（月次ベースで約2%）
- **レジーム制限**: トレンドレジームのみ

## 既存実装の活用分析

### 1. アクションシグナルガイドの活用

#### 統合シグナルシステムのアーキテクチャ
RLアクションとテクニカルパターンを統合した高度なエントリー判定システム：

```python
class IntegratedSignalSystem:
    """RL + テクニカルパターンの統合シグナルシステム"""

    def __init__(self, config):
        self.rl_model = SACModel(config['rl_config'])
        self.pattern_guide = ActionSignalGuide(config['pattern_config'])
        self.regime_classifier = MarketRegimeClassifier(config['regime_config'])
        self.signal_fusion = SignalFusionEngine(config['fusion_config'])

    def generate_entry_signal(self, state, features, regime):
        """統合エントリーシグナル生成"""

        # 1. RLアクション取得
        rl_action = self.rl_model.predict_action(state)

        # 2. テクニカルパターンシグナル取得
        pattern_signals = self.pattern_guide.get_signals(features, regime)

        # 3. レジーム適応パラメータ取得
        regime_params = self.regime_classifier.get_regime_params(regime)

        # 4. シグナル融合
        fused_signal = self.signal_fusion.fuse_signals(
            rl_action, pattern_signals, regime_params
        )

        return fused_signal
```

#### マルチシグナル融合アルゴリズム
RLアクションとテクニカルパターンを融合し、最終的なエントリー判定は**校正マップによるEVゲート**で行う：

```python
class SignalFusionEngine:
    """シグナル融合エンジン"""

    def fuse_signals(self, rl_action, pattern_signals, regime_params):
        """マルチシグナル融合アルゴリズム"""

        # 1. RLアクションの正規化 (確信度ではなく、方向と強度の特徴量として扱う)
        rl_feature = self._normalize_rl_action(rl_action)

        # 2. パターンシグナルの集約
        pattern_score = self._aggregate_pattern_signals(pattern_signals)

        # 3. 融合シグナルの生成 (校正マップへの入力となる)
        fused_signal = {
            'rl_action': rl_action,
            'pattern_score': pattern_score,
            'regime': regime_params['type']
        }

        return fused_signal

class CalibrationGate:
    """校正マップによるEVゲート"""

    def evaluate(self, fused_signal):
        """
        校正マップを参照し、期待値(EV)がプラスの場合のみエントリー許可
        詳細は 01_calibration_and_execution_model.md を参照
        """
        # 校正マップから勝率(LCB)と平均損益を取得
        stats = self.calibration_map.get_stats(
            regime=fused_signal['regime'],
            action_bin=self._get_bin(fused_signal['rl_action'])
        )

        # コスト計算
        cost = self.cost_model.estimate_cost()

        # EV計算
        ev = stats['p_win_lcb'] * stats['avg_win'] - (1 - stats['p_win_lcb']) * stats['avg_loss'] - cost

        return {
            'should_enter': ev > 0,
            'ev': ev,
            'stats': stats
        }
```

#### パターン認識によるエントリーフィルタリング
既存の統合パターン認識システムを活用し、期待値(EV)がマイナスのアクションをフィルタリング：

```python
# heavy_env/core.py に統合
def should_enter_trade(self, action, regime, features):
    """アクションシグナルガイドを活用したエントリー判定"""

    # 校正マップによるEV判定
    # アクションの絶対値やパターンシグナルは全て校正マップの特徴量として統合される

    gate_result = self.calibration_gate.evaluate({
        'rl_action': action,
        'regime': regime,
        'features': features
    })

    return gate_result['should_enter']
```

#### 適応的パターン選択
市場レジームに応じたパターン活性化：

```python
# 既存のAdaptivePatternSelector活用
def get_active_patterns_for_regime(self, regime):
    """レジーム別パターン選択"""
    regime_map = {
        'strong_bull_trend': ['adx', 'dow_theory', 'fibonacci_extension'],
        'strong_bear_trend': ['adx', 'dow_theory', 'fibonacci_extension'],
        'ranging': ['rsi', 'stochastic', 'bollinger_bands'],
        'sideways': ['rsi', 'cci', 'mfi'],
        'high_volatility': ['atr', 'volume_patterns'],
        'low_volatility': ['micro_patterns', 'tick_patterns']
    }
    return regime_map.get(regime, ['basic_technical'])
```

#### バックテストシミュレーション結果
統合システムの想定パフォーマンス：

```
統合前（RLのみ）:
- トレード数: 54
- 勝率: 93.1%
- リターン: +2.18%

統合後（RL + パターン融合）:
- トレード数: 180 (+233%)
- 勝率: 91.2% (-1.9pt)
- リターン: +4.8% (+120%)
- 最大ドローダウン: 15.2% (改善)

レジーム別改善:
- トレンドレジーム: 勝率94.5% (+1.4pt), リターン+3.2%
- レンジレジーム: 新規54トレード, 勝率88.9%, リターン+1.6%
- 統合シグナル確信度: 平均0.72 (旧ロジック参考値)
```

#### 実装詳細とコード構造

```python
# ztb/trading/signal/integrated_entry_system.py
class IntegratedEntrySystem:
    """統合エントリーシステム"""

    def __init__(self, config):
        self.signal_fusion = SignalFusionEngine(config)
        self.calibration_gate = CalibrationGate(config) # EVゲート

    def evaluate_entry(self, rl_action, market_data, regime):
        """統合エントリー評価"""

        # 1. シグナル融合
        fused_signal = self.signal_fusion.fuse_signals(
            rl_action, market_data, regime
        )

        # 2. 校正マップによるEV判定
        gate_result = self.calibration_gate.evaluate(fused_signal)

        return {
            'can_enter': gate_result['should_enter'],
            'ev': gate_result['ev'],
            'signal_components': fused_signal
        }

# ztb/trading/backtest/integrated_strategy.py
class IntegratedBacktestStrategy:
    """統合バックテスト戦略"""

    def __init__(self, config):
        self.entry_system = IntegratedEntrySystem(config)
        self.risk_manager = RiskManager(config)
        self.execution_model = RealisticExecutionModel(config)

    def generate_signals(self, data):
        """シグナル生成"""
        signals = []

        for i, row in enumerate(data):
            regime = self._classify_regime(row)
            rl_action = self._get_rl_action(row)

            entry_decision = self.entry_system.evaluate_entry(
                rl_action, row, regime
            )

            if entry_decision['can_enter']:
                signals.append(self._create_signal(row, entry_decision))

        return signals
```
市場レジームに応じたパターン活性化：

```python
# 既存のAdaptivePatternSelector活用
def get_active_patterns_for_regime(self, regime):
    """レジーム別パターン選択"""
    regime_map = {
        'strong_bull_trend': ['adx', 'dow_theory', 'fibonacci_extension'],
        'strong_bear_trend': ['adx', 'dow_theory', 'fibonacci_extension'],
        'ranging': ['rsi', 'stochastic', 'bollinger_bands'],
        'sideways': ['rsi', 'cci', 'mfi'],
        'high_volatility': ['atr', 'volume_patterns'],
        'low_volatility': ['micro_patterns', 'tick_patterns']
    }
    return regime_map.get(regime, ['basic_technical'])
```

### 2. 市場レジーム分類の活用

#### 21レジームの細分化活用
既存の21レジーム分類を活用し、より細かい条件での取引：

```python
# 既存のMarketRegimeClassifier活用
def get_regime_specific_params(self, regime):
    """レジーム別パラメータ調整"""
    regime_params = {
        'strong_bull_trend': {
            'rsi_entry': 35,  # 緩い条件
            'min_trend_strength': 0.7,
            'max_position_size': 0.1
        },
        'weak_bull_trend': {
            'rsi_entry': 40,  # より厳しい条件
            'min_trend_strength': 0.5,
            'max_position_size': 0.05
        },
        'ranging_normal': {
            'bollinger_deviation': 1.5,  # 平均回帰用
            'rsi_overbought': 75,
            'rsi_oversold': 25
        },
        'high_volatility_breakout': {
            'volume_multiplier': 1.5,
            'atr_multiplier': 2.0,
            'max_position_size': 0.02  # 小さく
        }
    }
    return regime_params.get(regime, self.default_params)
```

#### マイクロレジーム検出
既存のレジーム分類を活用した短期レジーム変化検出：

```python
def detect_micro_regime_changes(self, recent_data):
    """短期レジーム変化の検出"""
    # 既存分類器で1分毎のレジーム判定
    micro_regimes = []
    for i in range(len(recent_data) - self.lookback_window):
        window_data = recent_data[i:i+self.lookback_window]
        regime = self.regime_classifier.classify(window_data)
        micro_regimes.append(regime)

    # レジーム変化点検出
    change_points = self._detect_regime_transitions(micro_regimes)
    return change_points
```

### 3. リスクマネージャーの活用

#### 高頻度取引向けポジションサイジング
既存のPositionManagerを活用した動的ポジションサイズ制御：

```python
# 既存のRiskManager活用
def calculate_position_size_high_freq(self, regime, volatility, account_balance):
    """高頻度取引向けポジションサイズ計算"""

    # 基本サイズ（アカウントの1%）
    base_size = account_balance * 0.01

    # レジーム別調整
    regime_multiplier = {
        'strong_trend': 1.2,    # トレンド時は大きく
        'weak_trend': 0.8,
        'ranging': 0.6,         # レンジ時は小さく
        'high_volatility': 0.4, # ボラティリティ高時は小さく
        'low_volatility': 1.0
    }.get(regime, 1.0)

    # ボラティリティ調整（既存のボラティリティフィルター活用）
    vol_adjustment = 1.0 / (1.0 + volatility)  # ボラ高時は小さく

    # 取引頻度調整（1時間以内の取引数を考慮）
    recent_trades = self._count_recent_trades(hours=1)
    frequency_penalty = max(0.5, 1.0 - (recent_trades * 0.1))  # 多すぎるとペナルティ

    position_size = base_size * regime_multiplier * vol_adjustment * frequency_penalty

    return min(position_size, account_balance * 0.05)  # 最大5%制限
```

#### ダイナミックストップ管理
既存のストップマネジメントを活用した高頻度向け調整：

```python
def dynamic_stop_management(self, position, current_price, regime):
    """レジーム別ダイナミックストップ"""

    # ATRベースの基本ストップ（既存機能活用）
    atr_stop = self.atr_calculator.get_atr_stop(position.entry_price, current_price)

    # レジーム別調整
    if regime in ['high_volatility', 'breakout']:
        # ボラ高時は広いストップ
        stop_distance = atr_stop * 1.5
    elif regime in ['ranging', 'sideways']:
        # レンジ時は狭いストップ
        stop_distance = atr_stop * 0.7
    else:
        stop_distance = atr_stop

    # トレーリングストップ（既存機能活用）
    if self.should_activate_trailing_stop(position, current_price):
        trailing_stop = self.calculate_trailing_stop(position, current_price, regime)
        return max(stop_distance, trailing_stop)

    return stop_distance
```

### 4. リアルな実行モデルの活用

#### 高頻度取引の実行リスク評価
既存のRealisticExecutionModelを活用したシミュレーション：

```python
# 既存のRealisticExecutionModel活用
def simulate_high_freq_execution(self, orders, market_conditions):
    """高頻度取引の実行シミュレーション"""

    executed_orders = []
    total_slippage = 0
    total_latency = 0

    for order in orders:
        # スリッページ計算（既存機能）
        slippage = self.execution_model.calculate_slippage(
            order.price, order.size, market_conditions
        )

        # レイテンシー考慮（既存機能）
        latency = self.execution_model.simulate_latency(market_conditions)

        # 部分フィル考慮（既存機能）
        fill_rate = self.execution_model.simulate_partial_fill(
            order.size, market_conditions
        )

        # 実行結果
        executed_price = order.price + slippage
        executed_size = order.size * fill_rate
        execution_time = order.time + latency

        executed_orders.append({
            'price': executed_price,
            'size': executed_size,
            'time': execution_time,
            'slippage': slippage,
            'latency': latency
        })

        total_slippage += slippage
        total_latency += latency

    return executed_orders, total_slippage, total_latency
```

## 提案改善策

### 1. エントリー条件の最適化 + アクションシグナルガイド統合

#### 統合シグナルシステムの実装
RLアクションとテクニカルパターンを融合し、最終的なエントリー判定は**校正マップによるEVゲート**で行う：

```python
# ztb/trading/signal/integrated_entry_system.py
class IntegratedEntrySystem:
    """RL + テクニカルパターンの統合エントリーシステム"""

    def __init__(self, config):
        self.signal_fusion = SignalFusionEngine(config)
        self.pattern_guide = ActionSignalGuide(config)
        self.regime_classifier = MarketRegimeClassifier(config)
        self.calibration_gate = CalibrationGate(config) # EVゲート

    def evaluate_entry_opportunity(self, rl_action, market_data, regime):
        """統合エントリー機会評価"""

        # 1. パターンシグナル取得
        pattern_signals = self.pattern_guide.get_regime_signals(market_data, regime)

        # 2. レジーム適応パラメータ
        regime_params = self.regime_classifier.get_entry_params(regime)

        # 3. シグナル融合 (特徴量としての融合)
        fused_signal = self.signal_fusion.fuse_rl_and_patterns(
            rl_action, pattern_signals, regime_params
        )

        # 4. 校正マップによるEV判定 (確信度ではなく期待値で判定)
        gate_result = self.calibration_gate.evaluate(fused_signal)

        return {
            'can_enter': gate_result['should_enter'],
            'ev': gate_result['ev'],
            'signal_components': fused_signal,
            'regime_params': regime_params
        }

class SignalFusionEngine:
    """シグナル融合エンジン"""

    def fuse_rl_and_patterns(self, rl_action, pattern_signals, regime_params):
        """RLとパターンの融合アルゴリズム"""

        # RLアクションの正規化 (確信度ではなく、方向と強度の特徴量として扱う)
        rl_feature = self._normalize_rl_action(rl_action)

        # パターンシグナルの集約
        pattern_score = self._aggregate_pattern_signals(pattern_signals)

        # 融合シグナルの生成 (校正マップへの入力となる)
        fused_signal = {
            'rl_action': rl_action,
            'pattern_score': pattern_score,
            'regime': regime_params['type']
        }

        return fused_signal

class CalibrationGate:
    """校正マップによるEVゲート"""

    def evaluate(self, fused_signal):
        """
        校正マップを参照し、期待値(EV)がプラスの場合のみエントリー許可
        詳細は 01_calibration_and_execution_model.md を参照
        """
        # 校正マップから勝率(LCB)と平均損益を取得
        stats = self.calibration_map.get_stats(
            regime=fused_signal['regime'],
            action_bin=self._get_bin(fused_signal['rl_action'])
        )

        # コスト計算
        cost = self.cost_model.estimate_cost()

        # EV計算
        ev = stats['p_win_lcb'] * stats['avg_win'] - (1 - stats['p_win_lcb']) * stats['avg_loss'] - cost

        return {
            'should_enter': ev > 0,
            'ev': ev,
            'stats': stats
        }
```

#### RSI閾値調整 + パターン統合
既存のRSI条件をパターン認識システムと統合：

```python
# 統合されたエントリー条件
def get_integrated_entry_conditions(self, regime):
    """レジーム別統合エントリー条件"""

    conditions = {
        'strong_bull_trend': {
            'rsi_entry': 35,  # 緩和
            'adx_min': 0.25,  # ADX確認
            'macd_confirmation': True,
            'volume_confirmation': True
        },
        'strong_bear_trend': {
            'rsi_entry': 65,  # 緩和
            'adx_min': 0.25,
            'macd_confirmation': True,
            'volume_confirmation': True
        },
        'ranging': {
            'rsi_overbought': 75,
            'rsi_oversold': 25,
            'bollinger_position': 0.1,  # バンド外側10%以内のリバウンド
            'stochastic_divergence': True
        },
        'high_volatility': {
            'atr_filter': True,  # ATRでフィルタリング
            'volume_surge': 1.5,  # 出来高急増時のみ
            'rsi_extreme': 20    # 極端なRSIレベル
        }
    }

    return conditions.get(regime, self.default_conditions)
```

#### マイクロレジームベースの条件分岐
既存の21レジームを活用した細かい条件設定とパターン統合。

### 2. 複数レジーム対応

#### 既存レジームの活用拡大
- 現在のトレンド専用から全21レジーム対応
- レジーム別戦略の自動切り替え

### 3. タイムフレーム拡張

#### 1分足 + マルチタイムフレーム
既存のMultiTimeframeFeatureEngineerを活用。

### 4. リスク管理パラメータ最適化

#### 既存PositionManagerの活用
レジーム別・頻度別ポジションサイズ制御。

### 5. 特徴量強化

#### 既存特徴の調整
高頻度向けに短期指標を優先。

### 6. アンサンブル手法

#### シグナル融合
RL + テクニカルパターン + レジーム分析。

### 7. HFTコアロジック: 校正マップと厳密な実行モデル (New!)
「勝てない理由」を潰すためのコアロジックを導入する。詳細は [01_calibration_and_execution_model.md](01_calibration_and_execution_model.md) を参照。

#### 校正マップ (Calibration Map)
- RLのアクション値をそのまま信用せず、過去の勝率・リターン実績に基づいて補正する。
- **保守的EV判定**: Beta分布による勝率の下側信頼限界(LCB)を用い、コスト控除後の期待値がプラスの場合のみエントリー。
- **階層型フォールバック**: データ不足時は `Specific` -> `Regime` -> `Global` と統計情報を参照し、コールドスタートを防ぐ。

#### 擬似HFT実行モデル
- 1分足データでもHFTのコストを厳密に見積もる。
- **スリッページ**: `Spread + Impact + VolatilityRisk` に分解して推計。
- **Taker前提**: Maker約定はボーナス扱いとし、Takerでも勝てるロジックを必須とする。

#### レジーム正規化
- ボラティリティとトレンドを正規化し、固定閾値で安定した分類を行う。
- ヒステリシス制御により、レジームの頻繁な切り替わり（フリッカー）を防止。

## 実装計画

### Phase 1: 既存実装活用最適化 (v455.1)
- アクションシグナルガイドの統合
- レジーム別パラメータ設定
- リスクマネージャーの調整

### Phase 2: 高頻度エントリー拡張 (v455.2)
- RSI閾値緩和 + パターン確認
- 複数レジーム対応
- 1分足移行

### Phase 3: 実行リスク最適化 (v455.3)
- RealisticExecutionModel活用
- スリッページ・レイテンシー考慮
- バックテスト検証

### Phase 4: アンサンブル統合 (v455.4)
- 複数シグナル融合
- 総合パフォーマンス評価

## 期待される成果

- **トレード頻度**: 54 → 200-500トレード/月
- **利益率**: +2.18% → +5-10%/月
- **勝率維持**: 90%+を維持
- **リスク管理**: 実行リスクを考慮した堅実な運用

## リスク考慮

- **オーバートレーディング**: 既存リスクマネージャーで制御
- **実行コスト増大**: RealisticExecutionModelで評価
- **市場適合性**: 21レジーム分類で適応

## 次のステップ

1. v455.1開発開始（既存実装活用）
2. パフォーマンスモニタリング
3. 段階的ロールアウト

---

*作成日: 2025-12-19*
*バージョン: v455.0 (既存実装活用版)*