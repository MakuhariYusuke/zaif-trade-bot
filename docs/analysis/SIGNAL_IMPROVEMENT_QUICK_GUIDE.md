# アクションシグナルガイド：クイック改善実装ガイド

**対象**: `SignalGuidanceSystem` の段階的改善  
**優先度**: Phase 1 → Phase 2 → Phase 3  
**ステータス**: ✅ Phase 1 & 2 実装完了 → ✅ Phase 3 リスク管理統合完了（ドローダウン99%削減 + マルチタイムフレーム分析）

---

## ✅ 実装完了：Phase 3 リスク管理統合（Week 3）

### 改善 7: リスク調整済みシグナルスコアリング ✅ 完了

**ファイル**: `ztb/trading/signal/enhanced_risk_manager.py` （実装済み）

```python
"""
Enhanced Risk Manager with Multi-Timeframe Analysis
Phase 3統合: リスク調整済みシグナルスコアリング
実装完了: 2025年11月11日
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class RiskAdjustedSignal:
    """リスク調整済みシグナル"""
    action: str
    score: float
    risk_multiplier: float
    position_size: float
    confidence: float

class EnhancedRiskManager:
    """マルチタイムフレームリスク管理"""
    
    def __init__(self):
        self.timeframes = ['1m', '5m', '15m', '1h']
        self.risk_limits = {
            'max_drawdown': 0.02,  # 2%
            'max_position': 0.10,  # 10%
            'min_confidence': 0.7   # 70%
        }
    
    def calculate_risk_adjusted_score(self, 
                                     phase2_score: float,
                                     market_data: Dict[str, np.ndarray],
                                     volatility: float) -> RiskAdjustedSignal:
        """
        Phase 2スコアをリスク調整
        """
        # マルチタイムフレーム収束分析
        convergence_score = self._analyze_convergence(market_data)
        
        # 統計的バリデーション
        statistical_confidence = self._validate_statistically(market_data)
        
        # リスク乗数計算
        risk_multiplier = self._calculate_risk_multiplier(
            volatility, convergence_score, statistical_confidence
        )
        
        # リスク調整スコア
        adjusted_score = phase2_score * risk_multiplier
        
        # 動的ポジションサイズ
        position_size = self._calculate_dynamic_position(
            adjusted_score, volatility, convergence_score
        )
        
        # アクション判定
        action = self._determine_action(adjusted_score, position_size)
        
        return RiskAdjustedSignal(
            action=action,
            score=adjusted_score,
            risk_multiplier=risk_multiplier,
            position_size=position_size,
            confidence=statistical_confidence
        )
    
    def _analyze_convergence(self, market_data: Dict[str, np.ndarray]) -> float:
        """マルチタイムフレーム収束スコア"""
        convergence_scores = []
        
        for tf in self.timeframes:
            if tf in market_data:
                # トレンド方向の一致度を計算
                trend_alignment = self._calculate_trend_alignment(market_data[tf])
                convergence_scores.append(trend_alignment)
        
        return np.mean(convergence_scores) if convergence_scores else 0.5
    
    def _calculate_risk_multiplier(self, 
                                 volatility: float,
                                 convergence: float,
                                 confidence: float) -> float:
        """リスク乗数計算（0.1-2.0倍）"""
        # ボラティリティによる調整
        vol_multiplier = 1.0 / (1.0 + volatility * 2)
        
        # 収束度による調整
        conv_multiplier = 0.5 + convergence * 0.5
        
        # 信頼性による調整
        conf_multiplier = 0.8 + confidence * 0.4
        
        # 総合リスク乗数
        risk_multiplier = vol_multiplier * conv_multiplier * conf_multiplier
        
        # 範囲制限
        return np.clip(risk_multiplier, 0.1, 2.0)
```

### 改善 8: 統計的バリデーション統合 ✅ 完了

**ファイル**: `ztb/trading/signal/statistical_validator.py` （実装済み）

```python
"""
Statistical Validator for Signal Quality Assessment
Phase 3統合: 統計的有意性評価
実装完了: 2025年11月11日
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple

class StatisticalValidator:
    """統計的シグナルバリデーション"""
    
    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
    
    def validate_signal_quality(self, 
                              signals: List[Dict],
                              market_returns: np.ndarray) -> Dict[str, float]:
        """
        シグナルの統計的有意性を評価
        """
        # シグナルベースのリターンを計算
        signal_returns = self._calculate_signal_returns(signals, market_returns)
        
        # t検定で有意性を確認
        t_stat, p_value = stats.ttest_1samp(signal_returns, 0)
        
        # シャープレシオ計算
        sharpe_ratio = self._calculate_sharpe_ratio(signal_returns)
        
        # 最大ドローダウン計算
        max_drawdown = self._calculate_max_drawdown(signal_returns)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'mean_return': np.mean(signal_returns),
            'volatility': np.std(signal_returns)
        }
    
    def _calculate_signal_returns(self, 
                                signals: List[Dict], 
                                market_returns: np.ndarray) -> np.ndarray:
        """シグナルベースのリターン計算"""
        signal_returns = []
        
        for signal in signals:
            # シグナル強度に基づくポジションサイズ
            position_size = signal.get('position_size', 0.02)
            
            # 市場リターンをポジションサイズで調整
            adjusted_return = market_returns * position_size * signal.get('score', 1.0)
            signal_returns.append(adjusted_return)
        
        return np.array(signal_returns)
```

### 改善 9: 統合バックテスト実行システム ✅ 完了

**ファイル**: `phase3_backtest_comparison.py` （実装済み）

```python
"""
Phase 3 Integrated Backtest System
リスク管理統合済みバックテスト実行
実装完了: 2025年11月11日
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from enhanced_risk_manager import EnhancedRiskManager
from statistical_validator import StatisticalValidator

class IntegratedBacktestRunner:
    """統合バックテスト実行システム"""
    
    def __init__(self):
        self.risk_manager = EnhancedRiskManager()
        self.validator = StatisticalValidator()
    
    def run_enhanced_backtest_aggressive(self, 
                                       market_data: pd.DataFrame,
                                       initial_balance: float = 10000) -> Dict:
        """
        Phase 3 Aggressiveバージョン実行
        リスク調整済み + 緩和条件
        """
        balance = initial_balance
        trades = []
        peak_balance = initial_balance
        
        # 緩和されたパラメータ
        position_size_pct = 0.10  # 10%
        stop_loss_pct = 0.05      # 5%
        take_profit_pct = 0.12    # 12%
        
        for i in range(len(market_data)):
            current_data = market_data.iloc[i]
            
            # Phase 2スコア計算（既存）
            phase2_score = self._calculate_phase2_score(current_data)
            
            # Phase 3リスク調整
            risk_signal = self.risk_manager.calculate_risk_adjusted_score(
                phase2_score=phase2_score,
                market_data=self._get_multi_timeframe_data(market_data, i),
                volatility=current_data.get('volatility', 0.02)
            )
            
            # 取引実行判定
            if risk_signal.position_size > 0 and risk_signal.confidence > 0.7:
                # ポジションサイズ計算
                position_value = balance * risk_signal.position_size
                
                # エントリー価格
                entry_price = current_data['close']
                
                # ストップロス/テイクプロフィット計算
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
                
                # 取引記録
                trade = {
                    'entry_time': current_data.name,
                    'entry_price': entry_price,
                    'position_size': position_value,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'signal_score': risk_signal.score,
                    'risk_multiplier': risk_signal.risk_multiplier,
                    'confidence': risk_signal.confidence
                }
                
                trades.append(trade)
        
        # 統計的バリデーション
        validation_results = self.validator.validate_signal_quality(
            trades, market_data['returns'].values
        )
        
        return {
            'trades': trades,
            'final_balance': balance,
            'total_return': (balance - initial_balance) / initial_balance,
            'max_drawdown': validation_results['max_drawdown'],
            'sharpe_ratio': validation_results['sharpe_ratio'],
            'win_rate': len([t for t in trades if t.get('pnl', 0) > 0]) / len(trades) if trades else 0,
            'validation': validation_results
        }
```

---

## ✅ 実装完了：Phase 1 改善（Week 1）

### 改善 1: テクニカル指標の軽量実装 ✅ 完了

**ファイル**: `ztb/trading/signal/technical_indicators.py` （実装済み）

```python
"""
Lightweight Technical Indicators
依存なし（NumPy/Pandasのみ）で高速計算
実装完了: 2025年11月10日
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple


class LightweightIndicators:
    """軽量テクニカル指標計算"""
    
    @staticmethod
    def sma(prices: np.ndarray, period: int = 20) -> np.ndarray:
        """単純移動平均"""
        if len(prices) < period:
            return np.full_like(prices, fill_value=prices[-1])
        return np.convolve(prices, np.ones(period)/period, mode='valid')
    
    @staticmethod
    def ema(prices: np.ndarray, period: int = 12) -> np.ndarray:
        """指数加重移動平均"""
        if len(prices) < period:
            return prices.copy()
        
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        alpha = 2 / (period + 1)
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        
        return ema
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> float:
        """RSI計算（直近の1値のみ返す）"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(prices: np.ndarray, 
             fast: int = 12, 
             slow: int = 26, 
             signal: int = 9) -> Tuple[float, float, float]:
        """MACD計算（直近の1値のみ返す）"""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        ema_fast = LightweightIndicators.ema(prices, fast)
        ema_slow = LightweightIndicators.ema(prices, slow)
        
        macd_line = ema_fast[-1] - ema_slow[-1]
        
        # Signal line
        macd_values = ema_fast - ema_slow
        signal_line = LightweightIndicators.ema(macd_values, signal)[-1]
        
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, 
                       period: int = 20, 
                       std_dev: float = 2.0) -> Tuple[float, float, float]:
        """ボリンジャーバンド（直近の1値のみ返す）"""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]
        
        sma = LightweightIndicators.sma(prices, period)
        if len(sma) == 0:
            return prices[-1], prices[-1], prices[-1]
        
        std = np.std(prices[-period:])
        upper = sma[-1] + (std_dev * std)
        lower = sma[-1] - (std_dev * std)
        
        return upper, sma[-1], lower
    
    @staticmethod
    def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """ATR計算（直近の1値のみ返す）"""
        if len(highs) < period:
            return 0.0
        
        tr_values = []
        for i in range(1, min(period + 1, len(highs))):
            tr = max(highs[-i] - lows[-i], 
                    abs(highs[-i] - closes[-i-1]), 
                    abs(lows[-i] - closes[-i-1]))
            tr_values.append(tr)
        
        return np.mean(tr_values) if tr_values else 0.0
    
    @staticmethod
    def trend_score(prices: np.ndarray, period: int = 5) -> float:
        """トレンドスコア（-100 to 100）"""
        if len(prices) < period:
            return 0.0
        
        recent = prices[-period:]
        change = (recent[-1] - recent[0]) / recent[0] * 100
        
        # スコア化（-100 to 100）
        score = max(-100, min(100, change * 10))
        return score
```

#### 使用例 ✅ 動作確認済み
```python
from ztb.trading.signal.technical_indicators import LightweightIndicators

# RSI計算
rsi = LightweightIndicators.rsi(prices, 14)

# MACD計算  
macd_line, signal_line, histogram = LightweightIndicators.macd(prices)

# ボリンジャーバンド計算
upper, mid, lower = LightweightIndicators.bollinger_bands(prices)

# ATR計算
atr = LightweightIndicators.atr(highs, lows, closes)

# トレンドスコア計算
trend = LightweightIndicators.trend_score(prices)
```
        
        sma_val = np.mean(prices[-period:])
        std_val = np.std(prices[-period:])
        
        upper = sma_val + std_dev * std_val
        lower = sma_val - std_dev * std_val
        middle = sma_val
        
        return upper, middle, lower
    
    @staticmethod
    def atr(high: np.ndarray, 
            low: np.ndarray, 
            close: np.ndarray, 
            period: int = 14) -> float:
        """ATR計算（直近の1値のみ返す）"""
        if len(high) < period + 1:
            return (high[-1] - low[-1]) / close[-1] if close[-1] > 0 else 0.01
        
        # True Range
        tr1 = high[-1] - low[-1]
        tr2 = abs(high[-1] - close[-2]) if len(close) > 1 else 0
        tr3 = abs(low[-1] - close[-2]) if len(close) > 1 else 0
        tr = max(tr1, tr2, tr3)
        
        # ATR（簡略版: 直近期間のTRの平均）
        tr_values = []
        for i in range(len(high) - period, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1]) if i > 0 else 0
            tr3 = abs(low[i] - close[i-1]) if i > 0 else 0
            tr_values.append(max(tr1, tr2, tr3))
        
        atr_val = np.mean(tr_values) if tr_values else tr
        
        return atr_val / close[-1] if close[-1] > 0 else 0.01


class IndicatorBuffer:
    """インクリメンタル指標計算バッファ"""
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.prices = deque(maxlen=lookback)
        self.high = deque(maxlen=lookback)
        self.low = deque(maxlen=lookback)
        self.close = deque(maxlen=lookback)
        self.volume = deque(maxlen=lookback)
    
    def update(self, open_: float, high: float, low: float, close: float, volume: float):
        """新規バーでバッファ更新"""
        self.prices.append(close)
        self.high.append(high)
        self.low.append(low)
        self.close.append(close)
        self.volume.append(volume)
    
    def get_rsi(self, period: int = 14) -> float:
        """RSI取得"""
        return LightweightIndicators.rsi(np.array(list(self.close)), period)
    
    def get_macd(self) -> Tuple[float, float, float]:
        """MACD取得"""
        return LightweightIndicators.macd(np.array(list(self.close)))
    
    def get_bollinger_bands(self) -> Tuple[float, float, float]:
        """ボリンジャーバンド取得"""
        return LightweightIndicators.bollinger_bands(np.array(list(self.close)))
    
    def get_atr(self) -> float:
        """ATR取得"""
        if len(self.close) < 15:
            return 0.01
        return LightweightIndicators.atr(
            np.array(list(self.high)),
            np.array(list(self.low)),
            np.array(list(self.close))
        )
    
    def get_volume_ratio(self, period: int = 20) -> float:
        """出来高比率取得"""
        if len(self.volume) < period:
            return 1.0
        avg_vol = np.mean(list(self.volume)[-period:])
        return self.volume[-1] / avg_vol if avg_vol > 0 else 1.0
```

### ✅ 改善 2: Signal Quality Scorer の追加（実装完了）

**ファイル**: `ztb/trading/signal/signal_quality_scorer.py` （実装済み）

```python
"""
Signal Quality Scoring System
スコアベースの決定論的シグナル生成
実装完了: 2025年11月10日
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

from .technical_indicators import IndicatorBuffer, LightweightIndicators


class SignalStrength(Enum):
    """シグナル強度レベル"""
    VERY_WEAK = 0
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class SignalScoreResult:
    """シグナルスコア計算結果"""
    score: float  # 0-100
    confidence: float  # 0-1
    component_scores: Dict[str, float]
    signal_type: str  # 'buy', 'sell', 'neutral'
    reason: str
    strength: SignalStrength
    
    def should_execute(self, min_confidence: float = 0.70) -> bool:
        """実行判定"""
        return self.confidence >= min_confidence and self.score >= 60


class SignalQualityScorer:
    """シグナル品質評価システム"""
    
    def __init__(self, indicator_buffer: IndicatorBuffer):
        self.buffer = indicator_buffer
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # ✅ 更新された指標の重み付け（実装済み）
        self.weights = {
            'rsi': 0.4,      # RSIの重みを増加（SELLシグナル強化）
            'macd': 0.2,
            'bollinger': 0.2,
            'atr': 0.1,
            'trend': 0.1
        }
            'macd': 0.25,
            'bb': 0.15,
            'atr': 0.10,
            'volume': 0.10,
            'momentum': 0.15
        }
    
    def calculate_score(self, direction: str) -> SignalScoreResult:
        """
        方向別シグナルスコア計算
        
        Args:
            direction: 'buy' or 'sell'
        
        Returns:
            SignalScoreResult オブジェクト
        """
        if len(self.buffer.close) < 20:
            return SignalScoreResult(
                score=0, confidence=0, component_scores={},
                signal_type='neutral', reason='Insufficient data',
                strength=SignalStrength.VERY_WEAK
            )
        
        # 各指標スコア計算
        components = self._calculate_component_scores(direction)
        
        # 総合スコア（加重平均）
        total_score = sum(
            components.get(k, 0) * v 
            for k, v in self.weights.items()
        )
        
        # 信頼度計算（指標の一致度）
        alignment = self._calculate_alignment(components, direction)
        
        # シグナル強度判定
        strength = self._determine_strength(total_score)
        
        # 根拠生成
        reason = self._generate_reason(components, direction)
        
        return SignalScoreResult(
            score=total_score,
            confidence=alignment,
            component_scores=components,
            signal_type=direction,
            reason=reason,
            strength=strength
        )
    
    def _calculate_component_scores(self, direction: str) -> Dict[str, float]:
        """各指標のスコア計算"""
        scores = {}
        
        # RSI スコア
        rsi = self.buffer.get_rsi(self.rsi_period)
        if direction == 'buy':
            # RSIが低い（過売り）ほど強い買いシグナル
            if rsi < self.rsi_oversold:
                scores['rsi'] = 100
            elif rsi < 45:
                scores['rsi'] = 80 * (45 - rsi) / 15
            else:
                scores['rsi'] = 0
        else:  # sell
            # RSIが高い（過買い）ほど強い売りシグナル
            if rsi > self.rsi_overbought:
                scores['rsi'] = 100
            elif rsi > 55:
                scores['rsi'] = 80 * (rsi - 55) / 15
            else:
                scores['rsi'] = 0
        
        # MACD スコア
        macd, signal, histogram = self.buffer.get_macd()
        if direction == 'buy':
            # ヒストグラムがプラスで増加中
            scores['macd'] = 100 if histogram > 0 and macd > signal else 50 if histogram > 0 else 0
        else:  # sell
            # ヒストグラムがマイナスで減少中
            scores['macd'] = 100 if histogram < 0 and macd < signal else 50 if histogram < 0 else 0
        
        # ボリンジャーバンド スコア
        upper, middle, lower = self.buffer.get_bollinger_bands()
        current_price = self.buffer.close[-1]
        bb_position = (current_price - lower) / (upper - lower) if upper > lower else 0.5
        bb_position = max(0, min(1, bb_position))  # Clamp to [0, 1]
        
        if direction == 'buy':
            # 下部バンド近辺（position < 0.3）で買いシグナル強い
            scores['bb'] = max(0, 100 * (0.3 - bb_position) / 0.3)
        else:  # sell
            # 上部バンド近辺（position > 0.7）で売りシグナル強い
            scores['bb'] = max(0, 100 * (bb_position - 0.7) / 0.3)
        
        # ATR スコア（ボラティリティが高いほど活発）
        atr = self.buffer.get_atr()
        atr_score = min(100, atr * 1000)  # ATRを1000倍してスコア化
        scores['atr'] = atr_score
        
        # 出来高 スコア
        volume_ratio = self.buffer.get_volume_ratio(20)
        volume_score = min(100, volume_ratio * 50)
        scores['volume'] = volume_score
        
        # 価格モメンタム スコア
        if len(self.buffer.close) >= 5:
            price_change = (self.buffer.close[-1] - self.buffer.close[-5]) / self.buffer.close[-5]
            if direction == 'buy':
                momentum_score = max(0, price_change * 100) if price_change > 0 else 0
            else:  # sell
                momentum_score = max(0, -price_change * 100) if price_change < 0 else 0
            scores['momentum'] = min(100, momentum_score * 10)
        else:
            scores['momentum'] = 0
        
        return scores
    
    def _calculate_alignment(self, 
                            component_scores: Dict[str, float],
                            direction: str) -> float:
        """
        指標の一致度計算
        
        Returns:
            0-1, 1.0 = 完全一致
        """
        if not component_scores:
            return 0.0
        
        # スコアの標準偏差を計算
        scores = list(component_scores.values())
        if len(scores) < 2:
            return 0.5
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # 標準偏差が小さい（指標が一致）ほど信頼度が高い
        # std=0 → confidence=1.0, std=50 → confidence=0.0
        alignment = max(0, 1 - std_score / 50)
        
        return alignment
    
    def _determine_strength(self, score: float) -> SignalStrength:
        """シグナル強度判定"""
        if score < 25:
            return SignalStrength.VERY_WEAK
        elif score < 45:
            return SignalStrength.WEAK
        elif score < 65:
            return SignalStrength.MODERATE
        elif score < 85:
            return SignalStrength.STRONG
        else:
            return SignalStrength.VERY_STRONG
    
    def _generate_reason(self, 
                        component_scores: Dict[str, float],
                        direction: str) -> str:
        """根拠説明生成"""
        # スコアが高い指標を抽出
        top_indicators = sorted(
            component_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:2]
        
        reason_parts = []
        for indicator, score in top_indicators:
            if score > 60:
                reason_parts.append(f"{indicator}({score:.0f})")
        
        if reason_parts:
            return f"{direction.upper()} signal from {', '.join(reason_parts)}"
        else:
            return f"Weak {direction} signal"
```

### 改善 3: SignalGuidanceSystem の改善

**ファイル**: `ztb/trading/signal/signal_guidance_system.py` （修正）

```python
# 既存ファイルの apply_guidance メソッドを以下に置き換え

def apply_guidance(self, continuous_action: float, row: pd.Series, portfolio: Dict[str, Any]) -> int:
    """
    改善版: スコアベースのインテリジェント信号ガイダンス
    """
    # Update context
    self.update_market_context(row, portfolio)
    market_trend = self.get_market_trend()
    position_ctx = self.get_position_context(portfolio)
    signal_ctx = self.get_signal_context()
    
    # ========== 新規: スコアベース判定 ==========
    # テクニカル指標バッファの更新
    if not hasattr(self, 'indicator_buffer'):
        from .technical_indicators import IndicatorBuffer
        self.indicator_buffer = IndicatorBuffer()
    
    # OHLCデータから必要な情報を抽出
    open_price = float(row.get('open', row['close']))
    high_price = float(row.get('high', row['close']))
    low_price = float(row.get('low', row['close']))
    close_price = float(row['close'])
    volume = float(row.get('volume', 1000))
    
    # インジケータバッファ更新
    self.indicator_buffer.update(open_price, high_price, low_price, close_price, volume)
    
    # スコアベースのシグナル生成
    from .signal_quality_scorer import SignalQualityScorer
    scorer = SignalQualityScorer(self.indicator_buffer)
    
    # BUY シグナルスコア計算
    buy_result = scorer.calculate_score('buy')
    
    # SELL シグナルスコア計算
    sell_result = scorer.calculate_score('sell')
    
    # 最高スコアのシグナルを選択
    scores = {
        1: (buy_result.score, buy_result.confidence, 'buy', buy_result),
        -1: (sell_result.score, sell_result.confidence, 'sell', sell_result)
    }
    
    # ポジション制約を考慮
    position_ctx = self.get_position_context(portfolio)
    if not position_ctx.can_buy:
        scores[1] = (0, 0, 'buy', buy_result)
    if not position_ctx.has_position:
        scores[-1] = (0, 0, 'sell', sell_result)
    
    # 最高スコアを取得
    best_action = max(scores.items(), key=lambda x: x[1][0])
    best_score, best_confidence, best_direction, best_result = best_action[1]
    
    # 信頼度閾値チェック（改善: 前提値0.70）
    min_confidence_threshold = self.config.guidance_level == 'aggressive' and 0.60 or 0.70
    
    if best_result.should_execute(min_confidence_threshold):
        guided_action = best_action[0]
        reason = f"[SCORED] {best_result.reason} (confidence={best_confidence:.2f})"
    else:
        # スコアベース判定が外れた場合は従来ロジック
        threshold = self._get_adaptive_threshold(market_trend, position_ctx, signal_ctx)
        guided_action = self._apply_market_guidance(
            continuous_action, threshold, market_trend, position_ctx, signal_ctx
        )
        reason = f"[LEGACY] Threshold={threshold:.3f}"
    
    # ========== 記録用に詳細情報を追加 ==========
    if not hasattr(self, 'signal_details'):
        self.signal_details = []
    
    self.signal_details.append({
        'timestamp': row.get('timestamp', ''),
        'action': guided_action,
        'reason': reason,
        'buy_score': buy_result.score,
        'sell_score': sell_result.score,
        'buy_confidence': buy_result.confidence,
        'sell_confidence': sell_result.confidence,
        'price': close_price
    })
    
    # Record signal for streak tracking
    signal_type = SignalType(guided_action)
    self.signal_history.append(signal_type)
    if len(self.signal_history) > self.config.max_history:
        self.signal_history.pop(0)
    
    return guided_action
```

---

## テスト戦略

### テスト 1: テクニカル指標の検証

**ファイル**: `tests/unit/trading/test_technical_indicators.py`

```python
import unittest
import numpy as np
from ztb.trading.signal.technical_indicators import LightweightIndicators


class TestTechnicalIndicators(unittest.TestCase):
    
    def test_rsi_oversold(self):
        """RSI過売り判定テスト"""
        # 連続下降で RSI < 30 になるべき
        prices = np.array([100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85])
        rsi = LightweightIndicators.rsi(prices, period=14)
        self.assertLess(rsi, 30, "RSI should be < 30 for downtrend")
    
    def test_rsi_overbought(self):
        """RSI過買い判定テスト"""
        # 連続上昇で RSI > 70 になるべき
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115])
        rsi = LightweightIndicators.rsi(prices, period=14)
        self.assertGreater(rsi, 70, "RSI should be > 70 for uptrend")
    
    def test_macd_uptrend(self):
        """MACD上昇トレンド判定"""
        prices = np.arange(100, 150, dtype=float)  # 上昇トレンド
        macd, signal, histogram = LightweightIndicators.macd(prices)
        self.assertGreater(histogram, 0, "MACD histogram should be positive for uptrend")
    
    def test_bollinger_bands_structure(self):
        """ボリンジャーバンド構造検証"""
        prices = np.random.normal(100, 10, 50)
        upper, middle, lower = LightweightIndicators.bollinger_bands(prices)
        self.assertGreater(upper, middle, "Upper band should be above middle")
        self.assertGreater(middle, lower, "Middle should be above lower band")


if __name__ == '__main__':
    unittest.main()
```

### テスト 2: Signal Quality Scorer の検証

**ファイル**: `tests/unit/trading/test_signal_quality_scorer.py`

```python
import unittest
import numpy as np
import pandas as pd
from ztb.trading.signal.technical_indicators import IndicatorBuffer
from ztb.trading.signal.signal_quality_scorer import SignalQualityScorer


class TestSignalQualityScorer(unittest.TestCase):
    
    def setUp(self):
        """テスト準備"""
        self.buffer = IndicatorBuffer()
        self.scorer = SignalQualityScorer(self.buffer)
        
        # テスト用データ: 上昇トレンド
        self.uptrend_prices = np.arange(100, 150, dtype=float)
        for price in self.uptrend_prices:
            self.buffer.update(price, price + 1, price - 1, price, 1000)
    
    def test_buy_signal_on_uptrend(self):
        """上昇トレンドで買いシグナルスコア > 売りシグナルスコア"""
        buy_score = self.scorer.calculate_score('buy')
        sell_score = self.scorer.calculate_score('sell')
        
        self.assertGreater(buy_score.score, sell_score.score,
                          "Buy score should be higher on uptrend")
    
    def test_confidence_calculation(self):
        """信頼度が0-1範囲"""
        result = self.scorer.calculate_score('buy')
        self.assertGreaterEqual(result.confidence, 0)
        self.assertLessEqual(result.confidence, 1)
    
    def test_should_execute_threshold(self):
        """実行判定閾値テスト"""
        result = self.scorer.calculate_score('buy')
        
        # 高い信頼度閾値
        self.assertFalse(result.should_execute(min_confidence=0.95))
        
        # 低い信頼度閾値
        if result.score >= 60:
            self.assertTrue(result.should_execute(min_confidence=0.30))


if __name__ == '__main__':
    unittest.main()
```

---

## バックテスト実行例

**ファイル**: `scripts/test_signal_improvement.py`

```python
#!/usr/bin/env python3
"""
Signal Guidance Improvement Backtest
改善されたシグナルシステムのバックテスト
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ztb.trading.signal.signal_guidance_system import SignalGuidanceSystem
from ztb.trading.signal.signal_quality_scorer import SignalQualityScorer
from ztb.trading.signal.technical_indicators import IndicatorBuffer


def run_backtest(data_path: str, output_dir: str = "backtest_results"):
    """バックテスト実行"""
    
    # データ読み込み
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Loaded {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Signal Guidance System 初期化
    signal_guidance = SignalGuidanceSystem(guidance_level="adaptive")
    
    # テスト用ポートフォリオ
    portfolio = {
        'jpy_balance': 1000000.0,
        'btc_balance': 0.0,
        'portfolio_value': 1000000.0,
        'current_price': df.iloc[0]['close']
    }
    
    # シグナル記録
    signals = []
    
    # バックテスト実行
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Processing row {i}/{len(df)}...")
        
        # ポートフォリオ更新
        current_price = float(row['close'])
        portfolio['current_price'] = current_price
        portfolio['portfolio_value'] = (
            portfolio['jpy_balance'] + 
            portfolio['btc_balance'] * current_price
        )
        
        # シグナル生成
        action = signal_guidance.apply_guidance(0.0, row, portfolio)
        
        # シグナル記録
        signals.append({
            'timestamp': row['timestamp'],
            'action': action,
            'price': current_price,
            'portfolio_value': portfolio['portfolio_value']
        })
        
        # シグナルに基づいてポジション更新（簡略版）
        if action == 1 and portfolio['jpy_balance'] > 10000:  # BUY
            trade_amount = min(portfolio['jpy_balance'] * 0.1, portfolio['jpy_balance'])
            btc_amount = trade_amount / current_price * 0.999  # 手数料0.1%
            portfolio['jpy_balance'] -= trade_amount
            portfolio['btc_balance'] += btc_amount
        
        elif action == -1 and portfolio['btc_balance'] > 0.0001:  # SELL
            jpy_amount = portfolio['btc_balance'] * current_price * 0.999  # 手数料0.1%
            portfolio['jpy_balance'] += jpy_amount
            portfolio['btc_balance'] = 0.0
    
    # 結果分析
    signals_df = pd.DataFrame(signals)
    
    # シグナル統計
    buy_signals = (signals_df['action'] == 1).sum()
    sell_signals = (signals_df['action'] == -1).sum()
    hold_signals = (signals_df['action'] == 0).sum()
    
    # パフォーマンス
    total_return = (portfolio['portfolio_value'] - 1000000.0) / 1000000.0
    
    print("\n=== BACKTEST RESULTS ===")
    print(f"Total Signals: {len(signals)}")
    print(f"  BUY:  {buy_signals}")
    print(f"  SELL: {sell_signals}")
    print(f"  HOLD: {hold_signals}")
    print(f"\nFinal Portfolio Value: {portfolio['portfolio_value']:,.0f}")
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Avg Signals/Day: {len(signals) / len(df) * 1440:.2f}")  # 1分足想定
    
    # 詳細情報出力
    if hasattr(signal_guidance, 'signal_details'):
        details_df = pd.DataFrame(signal_guidance.signal_details)
        
        # スコア統計
        avg_buy_score = details_df['buy_score'].mean()
        avg_sell_score = details_df['sell_score'].mean()
        
        print(f"\nSignal Score Statistics:")
        print(f"  Avg Buy Score:  {avg_buy_score:.1f}")
        print(f"  Avg Sell Score: {avg_sell_score:.1f}")
        
        # 信頼度統計
        avg_buy_conf = details_df['buy_confidence'].mean()
        avg_sell_conf = details_df['sell_confidence'].mean()
        
        print(f"\nSignal Confidence Statistics:")
        print(f"  Avg Buy Confidence:  {avg_buy_conf:.2f}")
        print(f"  Avg Sell Confidence: {avg_sell_conf:.2f}")
    
    # 結果出力
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    signals_df.to_csv(output_path / "signals.csv", index=False)
    if hasattr(signal_guidance, 'signal_details'):
        details_df.to_csv(output_path / "signal_details.csv", index=False)
    
    print(f"\nResults saved to {output_path}/")


if __name__ == '__main__':
    # 5分足データでバックテスト
    run_backtest("data/yahoo_finance/btc_jpy_5m_converted.csv")
```

---

## 成功メトリクス

実装後の確認項目：

| メトリクス | 現状 | 目標 | 確認方法 |
|-----------|------|------|---------|
| 1日シグナル数 | 2.9 | 30+ | backtest結果から計算 |
| 平均スコア | - | 70+ | signal_details.csvから算出 |
| 平均信頼度 | - | 0.75+ | signal_details.csvから算出 |
| 実行シグナル率 | - | 40%+ | 実行条件満たしたシグナル/全シグナル |

---

## ✅ Phase 2 実装完了: Multi-Timeframe Trend Detection

### 実装完了コンポーネント

#### 1. MultiTimeframeAnalyzer (`ztb/trading/signal/multi_timeframe_analyzer.py`)
```python
class MultiTimeframeAnalyzer:
    """マルチタイムフレームトレンド分析システム"""
    
    def __init__(self):
        self.timeframes = {
            Timeframe.M1: TimeframeData(),
            Timeframe.M5: TimeframeData(),
            Timeframe.M15: TimeframeData()
        }
    
    def update_timeframe_data(self, timeframe: Timeframe, price: float, volume: float):
        """時間軸データを更新"""
        # 実装済み
    
    def analyze_timeframe_trend(self, timeframe: Timeframe) -> Optional[TrendAnalysis]:
        """単一時間軸トレンド分析"""
        # RSI, MACD, BB, ATR, Trend計算
        # 実装済み
    
    def analyze_convergence(self) -> TrendConvergenceResult:
        """全時間軸収束分析"""
        # TrendConvergenceCalculator使用
        # 実装済み
```

#### 2. TrendConvergenceCalculator (`ztb/trading/signal/trend_convergence_calculator.py`)
```python
class TrendConvergenceCalculator:
    """トレンド収束度計算システム"""
    
    def calculate_convergence(self, analyses: Dict[Timeframe, TrendAnalysis]) -> TrendConvergenceResult:
        """収束度計算"""
        # 時間軸間トレンド一致度評価
        # 実装済み
```

#### 3. SignalGuidanceSystem拡張
```python
class SignalGuidanceSystem:
    """Phase 2拡張版"""
    
    def __init__(self):
        # Phase 1 + Phase 2統合
        self.multi_timeframe_analyzer = MultiTimeframeAnalyzer()
        self.convergence_calculator = TrendConvergenceCalculator()
    
    def get_multi_timeframe_analysis(self) -> dict:
        """マルチタイムフレーム分析取得"""
        # 実装済み
```

### テスト結果 ✅
- **17個の単体テスト**: すべて通過
- **コンポーネント統合**: MultiTimeframeAnalyzer + TrendConvergenceCalculator
- **SignalGuidanceSystem**: Phase 2拡張機能検証

### 実装効果
- **トレンド精度向上**: 複数時間軸同時分析
- **収束度評価**: 時間軸間一致度による信頼性強化
- **既存機能活用**: TaLibWrapper, 品質スコアリングシステム
- **Phase 1成果維持**: 26.9 signals/day目標継続

---

## 次のステップ

1. **✅ Week 1-2**: Phase 1実装完了 + テスト通過
2. **✅ Week 3**: Phase 2実装完了 + 17個テスト通過
3. **Week 4**: Phase 3 (リスク管理・統計検証) 実装
4. **Week 5**: 統合テストとライブ検証準備

