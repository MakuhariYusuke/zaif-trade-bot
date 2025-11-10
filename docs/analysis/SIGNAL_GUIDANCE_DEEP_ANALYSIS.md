# アクションシグナルガイドシステムの深掘り分析と改善提案

**作成日**: 2025年11月10日  
**対象**: SAC v445 アクションシグナルガイド  
**目標**: スキャルピングレベルの高頻度取引で単体採算性を実現  

---

## Executive Summary

### 現状の課題

現在の `SignalGuidanceSystem` は以下の構造上の問題により、スキャルピング目標（20-50回/日）から**7-17倍の乖離**が生じています：

| 項目 | 現状 | 目標 | ギャップ |
|------|------|------|---------|
| 1日当たりの平均シグナル数 | ~2.9回 | 20-50回 | 7-17倍 |
| 決定方式 | 確率的 | 決定論的スコア | 定性→定量 |
| テクニカル指標数 | 1個（価格トレンド） | 5-6個 | 5-6倍 |
| 信頼度スコア | なし | 70-100 | 新規追加 |
| ポジション管理 | 固定閾値（80%, 10%） | リスク適応 | 静的→動的 |

### 根本原因

1. **確率ベースの設計欠陥**
   - `sell_injection_base_probability = 0.15`で基本確率15%
   - 複数の確率判定が直列化され、有効確率が指数関数的に低下
   - 実際のシグナル発生確率 ≈ 0.15 × 0.3 × 0.25 ≈ **1.125%** に低下

2. **市場状態の不十分な分析**
   - トレンド判定: 直近5期間で ±0.2% のみ（threshold固定）
   - RSI、MACD、ボリンジャーバンド等の指標が未活用
   - ボラティリティ、出来高の重要性が無視されている

3. **ポジション管理の静的設計**
   - 「overexposed = position_ratio > 80%」という硬直的判定
   - リスク指標（Sharpe, Sortino, VaR）が計算されていない
   - ケリー基準によるポジションサイジングが未実装

4. **シグナルの信頼度評価がない**
   - 発生したシグナルにスコア（0-100）がない
   - 複数指標の組み合わせ根拠が明示されていない
   - バックテスト時の精度検証ができない

---

## 深掘り分析

### 1. 確率ベース設計の数学的問題

#### 現在の実装フロー

```
Base Threshold (0.33)
        ↓
Theme Probability Chain:
  - Bearish market? × 1.5
  - Overexposed? × 1.8
  - No recent SELL? × 2.0
  - Streak penalty? × 0.3
        ↓
Final Probability = Cap at 50%
```

#### 問題点の計算

**シナリオ**: 通常の市場、ポジション中程度、ストリーク0

```
計算フロー:
1. apply_guidance() の各段で確率判定が発生
   - base_action 決定: P(SELL|threshold) ≈ 15%
   - position_guidance: P(不変|has_position) ≈ 80%
   - trend_guidance: P(BUY|BULLISH) ≈ 30%
   - signal_guidance: P(SELL|rare) ≈ 25%

2. 複合確率:
   P(有効SELL) = 0.15 × 0.8 × 1.0 × 0.25 ≈ 3%

3. 実測: ~1/3.5日に1SELL = 0.29/日
   理論値: 3% × (テストケース数) の範囲

→ **結論**: 確率チェーンが長すぎて実効確率が極小化
```

### 2. テクニカル指標の不足分析

#### 現在の実装
- **価格トレンド**: 直近5期間の単純比較
- **その他**: ほぼなし

#### 必要な指標と計算

| 指標 | 計算式 | 適用用途 | 重要度 |
|------|--------|---------|--------|
| **RSI** | $RSI = 100 - \frac{100}{1 + \frac{AU}{AD}}$ | 過買い/過売り判定 | ★★★★★ |
| **MACD** | $MACD = EMA_{12} - EMA_{26}$ | トレンド強度 | ★★★★★ |
| **ボリンジャーバンド** | $BB_{upper} = MA_{20} + 2\sigma$ | サポート/レジスタンス | ★★★★ |
| **ATR（真の値幅）** | $ATR = \text{True Range}_n$ | ボラティリティ | ★★★★ |
| **出来高比率** | $\frac{V_t}{V_{MA20}}$ | 上昇・下降の確実性 | ★★★ |
| **ストキャスティクス** | $\%K = \frac{C - L14}{H14 - L14}$ | 相対強度 | ★★★ |

### 3. ポジション管理の問題点

#### 現状の固定的設定
```python
is_overexposed = position_ratio > 0.8      # 80%以上でOVER
is_underexposed = position_ratio < 0.1     # 10%以下でUNDER
```

#### 問題
- **市場ボラティリティを無視**: 高ボラティリティ時は30%がリスクレベル
- **ドローダウン無視**: 最大ドローダウン20%時は50%ポジションが危険
- **資金効率無視**: 小資金なら60%でも十分なポジション

#### ケリー基準による改善
$$f^* = \frac{p \cdot b - q}{b}$$

ここで：
- $p$ = 勝率
- $q$ = 敗率
- $b$ = 平均利益/平均損失

**例**:
- 勝率55%, 平均利益/損失比 1.2
- $f^* = \frac{0.55 \times 1.2 - 0.45}{1.2} = 0.208$ (20.8%)
- フル・ケリー = 20.8%、フラクショナル（1/2） = 10.4%

---

## Phase 1: スコアベース決定論的シグナル生成システム

### アーキテクチャ

```
Market Data (OHLCV)
    ↓
[Advanced Market Analyzer]
  ├─ RSI Calculator
  ├─ MACD Calculator
  ├─ Bollinger Bands Calculator
  ├─ ATR Calculator
  └─ Volume Analyzer
    ↓
[Micro Trend Detector]
  ├─ 1-min trend
  ├─ 5-min trend
  └─ 15-min trend
    ↓
[Signal Quality Scorer]
  ├─ Buy Score (0-100)
  ├─ Sell Score (0-100)
  └─ Hold Score (0-100)
    ↓
[Risk-Based Position Manager]
  ├─ Kelly Criterion Calculator
  ├─ VaR Calculator
  └─ Position Sizing
    ↓
[Final Decision]
  └─ Action + Confidence + Reason
```

### 1-1. Advanced Market Analyzer 実装

```python
class AdvancedMarketAnalyzer:
    """高度な市場分析エンジン"""
    
    def __init__(self, lookback_period: int = 50):
        self.lookback = lookback_period
        self.price_history = deque(maxlen=lookback_period)
        self.volume_history = deque(maxlen=lookback_period)
        self.indicators_cache = {}
        
    def calculate_rsi(self, period: int = 14) -> float:
        """RSI計算: 過買い/過売り判定"""
        if len(self.price_history) < period + 1:
            return 50.0  # 中立値
            
        prices = list(self.price_history)
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        """MACD計算: トレンド強度判定"""
        # EMA計算
        # ...
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_line - signal_line
        }
    
    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> dict:
        """ボリンジャーバンド計算"""
        # ...
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band,
            'width': (upper_band - lower_band) / middle_band,  # 正規化幅
            'position': (current_price - lower_band) / (upper_band - lower_band)  # 0-1位置
        }
    
    def calculate_atr(self, period: int = 14) -> float:
        """ATR計算: ボラティリティ指標"""
        # ...
        return atr_value
    
    def analyze_volume(self, period: int = 20) -> dict:
        """出来高分析"""
        # ...
        return {
            'volume_ratio': current_volume / avg_volume,
            'trend': volume_trend,  # 'increasing', 'decreasing', 'stable'
            'strength': volume_strength  # 0-1
        }
```

### 1-2. Signal Quality Scorer 実装

```python
class SignalQualityScorer:
    """シグナル品質評価システム"""
    
    def __init__(self, analyzer: AdvancedMarketAnalyzer):
        self.analyzer = analyzer
        self.weights = {
            'rsi': 0.25,
            'macd': 0.25,
            'bb': 0.15,
            'atr': 0.15,
            'volume': 0.10,
            'price_momentum': 0.10
        }
    
    def calculate_signal_score(self, 
                              direction: str,  # 'buy' or 'sell'
                              config: SignalConfig) -> dict:
        """
        方向別シグナルスコア計算
        
        Returns:
            {
                'score': 0-100,  # 総合スコア
                'confidence': 0-1,  # 信頼度
                'component_scores': {...},  # 各指標スコア
                'reason': '...',  # 根拠説明
                'strength': 0-1  # シグナル強度
            }
        """
        rsi = self.analyzer.calculate_rsi()
        macd = self.analyzer.calculate_macd()
        bb = self.analyzer.calculate_bollinger_bands()
        atr = self.analyzer.calculate_atr()
        volume = self.analyzer.analyze_volume()
        
        # 各指標のスコア計算
        scores = {}
        
        # RSI スコア
        if direction == 'buy':
            # RSIが低い（過売り）ほど買いシグナル強い
            scores['rsi'] = max(0, 100 - rsi) if rsi < 50 else 0
        else:  # sell
            # RSIが高い（過買い）ほど売りシグナル強い
            scores['rsi'] = max(0, rsi - 50) if rsi > 50 else 0
        
        # MACD スコア
        if direction == 'buy':
            # ヒストグラムがプラスで増加中
            scores['macd'] = 100 if macd['histogram'] > 0 and macd['macd'] > macd['signal'] else 0
        else:  # sell
            # ヒストグラムがマイナスで減少中
            scores['macd'] = 100 if macd['histogram'] < 0 and macd['macd'] < macd['signal'] else 0
        
        # ボリンジャーバンド スコア
        bb_position = bb['position']  # 0-1
        if direction == 'buy':
            # 下部バンド近辺（position < 0.3）で買いシグナル強い
            scores['bb'] = 100 * max(0, 0.3 - bb_position) / 0.3
        else:  # sell
            # 上部バンド近辺（position > 0.7）で売りシグナル強い
            scores['bb'] = 100 * max(0, bb_position - 0.7) / 0.3
        
        # ATR スコア（ボラティリティが高いほど取引活発）
        atr_score = min(100, self.analyzer.atr_ratio * 50)  # ATR比率を50倍で正規化
        scores['atr'] = atr_score
        
        # 出来高 スコア
        volume_score = min(100, volume['volume_ratio'] * 50)
        scores['volume'] = volume_score
        
        # 価格モメンタム
        momentum = self._calculate_price_momentum()
        if direction == 'buy':
            scores['price_momentum'] = momentum if momentum > 0 else 0
        else:  # sell
            scores['price_momentum'] = -momentum if momentum < 0 else 0
        
        # 総合スコア計算（加重平均）
        total_score = sum(scores.get(k, 0) * v for k, v in self.weights.items())
        
        # 信頼度計算（各指標が同じ方向を指しているか）
        alignment = self._calculate_alignment(scores, direction)
        confidence = alignment
        
        return {
            'score': total_score,
            'confidence': confidence,
            'component_scores': scores,
            'reason': self._generate_reason(scores, direction),
            'strength': total_score / 100.0
        }
    
    def should_execute_signal(self, 
                             score_result: dict,
                             min_confidence: float = 0.70) -> bool:
        """
        シグナル実行判定
        
        Args:
            score_result: calculate_signal_score() の戻り値
            min_confidence: 最小信頼度閾値
        
        Returns:
            True if confidence >= min_confidence and score >= 50
        """
        return (score_result['confidence'] >= min_confidence and 
                score_result['score'] >= 50)
```

### 1-3. Risk-Based Position Manager 実装

```python
class RiskBasedPositionManager:
    """リスクベースのポジション管理"""
    
    def __init__(self, win_rate: float = 0.55, avg_profit_ratio: float = 1.2):
        """
        Args:
            win_rate: 過去の勝率
            avg_profit_ratio: 平均利益 / 平均損失
        """
        self.win_rate = win_rate
        self.avg_profit_ratio = avg_profit_ratio
    
    def calculate_kelly_position_size(self, 
                                     total_portfolio: float,
                                     fractional: float = 0.5) -> float:
        """
        ケリー基準によるポジションサイジング
        
        f* = (p * b - q) / b
        ここで p=勝率, q=敗率, b=平均利益/損失比
        """
        p = self.win_rate
        q = 1 - p
        b = self.avg_profit_ratio
        
        if b == 0:
            return 0.02  # デフォルト: 2%
        
        f_star = (p * b - q) / b
        
        # フラクショナル・ケリー（1/2推奨）
        position_ratio = max(0.01, min(0.25, f_star * fractional))
        
        return total_portfolio * position_ratio
    
    def calculate_var(self,
                     portfolio_value: float,
                     confidence_level: float = 0.95,
                     lookback_days: int = 30) -> float:
        """
        VaR（バリュー・アット・リスク）計算
        
        Returns:
            最大予想損失額
        """
        # 過去リターン分析
        returns = self._get_historical_returns(lookback_days)
        
        # パーセンタイル計算
        var_percentile = np.percentile(returns, (1 - confidence_level) * 100)
        
        return portfolio_value * var_percentile
    
    def calculate_position_size_with_risk(self,
                                         portfolio_value: float,
                                         entry_price: float,
                                         stop_loss_price: float,
                                         risk_percent: float = 0.02) -> float:
        """
        リスク額ベースのポジションサイジング
        
        Args:
            portfolio_value: 総資産
            entry_price: エントリー価格
            stop_loss_price: ストップロス価格
            risk_percent: リスク許容度（資産の何%まで）
        
        Returns:
            ポジションサイズ（通貨単位）
        """
        risk_amount = portfolio_value * risk_percent
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        
        return position_size
```

---

## Phase 2: マイクロトレンド検出とスキャルピング最適化

### 2-1. Multi-Timeframe Analyzer

```python
class MicroTrendDetector:
    """マイクロトレンド検出システム"""
    
    def __init__(self):
        self.ma1_short = deque(maxlen=2)    # 1分 短期MA
        self.ma5_short = deque(maxlen=10)   # 5分 短期MA
        self.ma15_short = deque(maxlen=30)  # 15分 短期MA
        
    def detect_microtrend(self, 
                         ohlcv_data: Dict[str, list]) -> dict:
        """
        複数時間軸でトレンド検出
        
        Returns:
            {
                '1min': {'trend': 'up'/'down', 'strength': 0-1},
                '5min': {...},
                '15min': {...},
                'convergence': 0-1  # 時間軸の一致度
            }
        """
        # 時間軸ごとにトレンド計算
        trends = {}
        
        # 1分足
        if len(ohlcv_data['1m_close']) >= 5:
            trend_1m = self._calculate_trend(ohlcv_data['1m_close'][-5:])
            trends['1min'] = trend_1m
        
        # 5分足
        if len(ohlcv_data['5m_close']) >= 5:
            trend_5m = self._calculate_trend(ohlcv_data['5m_close'][-5:])
            trends['5min'] = trend_5m
        
        # 15分足
        if len(ohlcv_data['15m_close']) >= 5:
            trend_15m = self._calculate_trend(ohlcv_data['15m_close'][-5:])
            trends['15min'] = trend_15m
        
        # 収束度計算
        convergence = self._calculate_convergence(trends)
        
        return {
            'trends': trends,
            'convergence': convergence,
            'recommendation': self._get_multi_timeframe_signal(trends, convergence)
        }
    
    def _calculate_trend(self, prices: list) -> dict:
        """トレンド計算"""
        if len(prices) < 2:
            return {'trend': 'neutral', 'strength': 0}
        
        change = prices[-1] - prices[0]
        avg_price = sum(prices) / len(prices)
        strength = abs(change) / avg_price
        
        trend = 'up' if change > 0 else 'down' if change < 0 else 'neutral'
        
        return {'trend': trend, 'strength': min(1.0, strength)}
    
    def _calculate_convergence(self, trends: dict) -> float:
        """時間軸の一致度計算"""
        if len(trends) < 2:
            return 0.0
        
        # 全指標が同じ方向を指しているか
        trend_values = []
        for timeframe_trend in trends.values():
            if isinstance(timeframe_trend, dict):
                trend_values.append(1 if timeframe_trend['trend'] == 'up' else -1 if timeframe_trend['trend'] == 'down' else 0)
        
        # 一致度: 1.0 = 完全一致, 0.0 = 分散
        if all(v == trend_values[0] for v in trend_values):
            return 1.0
        else:
            return abs(sum(trend_values)) / len(trend_values)
```

### 2-2. スキャルピング最適化ロジック

```python
class ScalpingOptimizer:
    """スキャルピング取引最適化"""
    
    def __init__(self):
        self.min_hold_time = 60  # 最小保有時間（秒）
        self.max_daily_trades = 100  # 最大日次取引数
        self.daily_trade_count = 0
        self.last_trade_time = None
        
    def should_scalp(self,
                    signal_score: float,
                    microtrend: dict,
                    volatility: float) -> bool:
        """
        スキャルピング実行判定
        
        条件:
        1. シグナルスコア >= 65
        2. マイクロトレンド収束度 >= 0.7
        3. ボラティリティ > 低下閾値
        4. 最小保有時間経過済み
        5. 日次上限未到達
        """
        # 基本条件
        if signal_score < 65:
            return False
        
        # マイクロトレンド条件
        if microtrend['convergence'] < 0.7:
            return False
        
        # ボラティリティ条件（高すぎず低すぎず）
        if volatility < 0.001 or volatility > 0.05:
            return False
        
        # 最小保有時間条件
        if self.last_trade_time:
            time_since_trade = time.time() - self.last_trade_time
            if time_since_trade < self.min_hold_time:
                return False
        
        # 日次上限条件
        if self.daily_trade_count >= self.max_daily_trades:
            return False
        
        return True
    
    def get_optimal_position_size(self,
                                 portfolio_value: float,
                                 signal_strength: float,
                                 volatility: float) -> float:
        """
        ボラティリティ適応的なポジションサイジング
        
        高ボラティリティ時は小さく、低ボラティリティ時は大きく
        """
        # ベースサイズ: ポートフォリオの1-2%
        base_size = portfolio_value * 0.015
        
        # シグナル強度による調整
        size_by_strength = base_size * (0.5 + signal_strength * 0.5)
        
        # ボラティリティによる調整（逆相関）
        vol_factor = min(1.0, 0.02 / (volatility + 0.001))
        
        final_size = size_by_strength * vol_factor
        
        return final_size
```

---

## Phase 3: 信頼度向上と統計的検証

### 3-1. Performance Metrics System

```python
class PerformanceAnalyzer:
    """パフォーマンス分析システム"""
    
    def __init__(self):
        self.trades = []
        self.equity_curve = []
        
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.03) -> float:
        """シャープレシオ計算"""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0
        
        returns = np.diff(np.log(self.equity_curve))
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        sharpe = (mean_return - risk_free_rate / 252) / std_return * np.sqrt(252)
        return sharpe
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.03) -> float:
        """ソルティノレシオ計算（下方リスクのみ考慮）"""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0
        
        returns = np.diff(np.log(self.equity_curve))
        mean_return = np.mean(returns)
        
        # 下方リターンのみを抽出
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        if downside_std == 0:
            return 0.0
        
        sortino = (mean_return - risk_free_rate / 252) / downside_std * np.sqrt(252)
        return sortino
    
    def calculate_max_drawdown(self) -> float:
        """最大ドローダウン計算"""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0
        
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        
        return np.min(drawdown)
    
    def calculate_win_rate(self) -> float:
        """勝率計算"""
        if not self.trades:
            return 0.0
        
        winning_trades = sum(1 for t in self.trades if t['profit'] > 0)
        return winning_trades / len(self.trades)
    
    def calculate_profit_factor(self) -> float:
        """プロフィットファクター計算"""
        if not self.trades:
            return 0.0
        
        gross_profit = sum(t['profit'] for t in self.trades if t['profit'] > 0)
        gross_loss = abs(sum(t['profit'] for t in self.trades if t['profit'] < 0))
        
        if gross_loss == 0:
            return 0.0
        
        return gross_profit / gross_loss
    
    def get_performance_summary(self) -> dict:
        """パフォーマンスサマリー"""
        return {
            'total_trades': len(self.trades),
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': self.calculate_sortino_ratio(),
            'max_drawdown': self.calculate_max_drawdown(),
            'annual_return': self._calculate_annual_return(),
            'monthly_return': self._calculate_monthly_return()
        }
```

### 3-2. Statistical Significance Testing

```python
class SignificanceValidator:
    """統計的有意性検証"""
    
    def t_test_returns(self, returns: np.ndarray, 
                       null_hypothesis_mean: float = 0.0) -> dict:
        """
        t検定によるリターンの有意性検証
        
        帰無仮説: リターンの平均 = null_hypothesis_mean
        """
        t_stat = (np.mean(returns) - null_hypothesis_mean) / (np.std(returns) / np.sqrt(len(returns)))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(returns) - 1))
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'mean_return': np.mean(returns),
            'confidence_level': 1 - p_value
        }
    
    def bootstrap_confidence_interval(self, 
                                     trades: list,
                                     metric: str = 'profit',
                                     n_bootstrap: int = 10000,
                                     ci: float = 0.95) -> dict:
        """
        ブートストラップ法による信頼区間推定
        
        オーバーフィッティングを考慮した信頼性評価
        """
        metric_values = [t[metric] for t in trades]
        
        bootstrap_samples = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(metric_values, size=len(metric_values), replace=True)
            bootstrap_samples.append(np.mean(sample))
        
        lower = np.percentile(bootstrap_samples, (1 - ci) / 2 * 100)
        upper = np.percentile(bootstrap_samples, (1 + ci) / 2 * 100)
        
        return {
            'lower_bound': lower,
            'upper_bound': upper,
            'point_estimate': np.mean(metric_values),
            'std_error': np.std(bootstrap_samples)
        }
```

---

## 実装ロードマップ

### Week 1-2: Phase 1 実装
- [ ] AdvancedMarketAnalyzer クラス実装
  - RSI, MACD, BB, ATR計算ロジック
  - 単体テスト（各指標の数値検証）
  
- [ ] SignalQualityScorer クラス実装
  - 複数指標のスコアリング
  - 加重平均計算
  - 信頼度アルゴリズム
  
- [ ] RiskBasedPositionManager クラス実装
  - ケリー基準計算
  - VaR計算

### Week 3: Phase 2 実装
- [ ] MicroTrendDetector クラス実装
  - 複数時間軸分析
  - トレンド収束度計算
  
- [ ] ScalpingOptimizer クラス実装
  - スキャルピング判定ロジック
  - ボラティリティ適応的ポジションサイジング

### Week 4: Phase 3 実装
- [ ] PerformanceAnalyzer クラス実装
  - Sharpe, Sortino比率計算
  - ドローダウン分析
  
- [ ] SignificanceValidator クラス実装
  - t検定
  - ブートストラップ検証

### Week 5-6: 統合テストと検証
- [ ] バックテスト検証（過去3ヶ月データ）
- [ ] パフォーマンス目標達成確認
- [ ] ウォークフォワード分析
- [ ] ライブ取引テスト開始（小規模）

---

## 成功指標

| 項目 | 現状 | 目標 | 期限 |
|------|------|------|------|
| 1日平均シグナル数 | 2.9 | 30-40 | Week 2 |
| 信頼度スコア（平均） | - | 75+ | Week 2 |
| Sharpe比率 | - | 2.0+ | Week 4 |
| 勝率 | - | 55%+ | Week 4 |
| 最大ドローダウン | - | 10%以下 | Week 4 |
| 年率リターン（シミュレ） | - | 50%+ | Week 5 |

---

## 技術的考慮事項

### 1. 計算効率
- テクニカル指標計算の最適化（Cython化等）
- インクリメンタル計算による逐次更新
- キャッシング戦略の導入

### 2. メモリ管理
- `deque` による固定サイズバッファ
- 不要な履歴データの自動削除
- NumPy配列による効率的計算

### 3. 信頼性
- テストカバレッジ: 90%以上
- エッジケーステスト（急変動、出来高ゼロ等）
- ライブ市場データでの検証

### 4. スケーラビリティ
- 複数時間軸同時処理
- マルチ銘柄対応
- リアルタイム処理への最適化

---

## 参考リソース

### テクニカル指標計算
- RSI: Wilder's Smoothing method
- MACD: Exponential Moving Average
- BB: Standard Deviation bands
- ATR: True Range calculation

### ポジションサイジング
- Kelly Criterion: https://en.wikipedia.org/wiki/Kelly_criterion
- Risk Parity: バランスド・リスク配分
- VaR: Historical simulation method

### 統計検証
- t-test: Student's distribution
- Bootstrap: Resampling method
- Walkforward Analysis: In-sample/Out-of-sample split
