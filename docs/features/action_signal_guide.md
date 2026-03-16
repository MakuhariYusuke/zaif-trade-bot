# Action Signal Guide Documentation

## Overview

The Action Signal Guide is a critical component of the SAC reinforcement learning system, providing classical technical analysis signals to guide the agent's decision-making process during training. This system integrates traditional Japanese candlestick patterns and Western technical indicators to enhance the RL agent's learning efficiency and market understanding.

## Core Functionality

### Signal Generation
- **Classical Technical Signals**: Provides buy/sell/hold signals based on established technical analysis patterns
- **Signal Strength Evaluation**: Each signal includes a confidence score (0.0-1.0) indicating reliability
- **Multi-timeframe Analysis**: Supports analysis across different timeframes for comprehensive market assessment

### Integration with RL Training
- **Guidance Mode**: Configurable guidance levels (none, weak, strong) for different training phases
- **Signal Integration**: Seamlessly integrates with the reward system to reinforce correct market behaviors
- **Adaptive Learning**: Signals evolve as the agent learns, reducing guidance over time

## Current Implementation Status

### Basic Signals Implemented
- Moving Average Crossovers
- RSI Divergence
- MACD Signals
- Bollinger Band Breakouts
- Volume-based Signals

### Enhancement Plan (Phase 3)

#### High Priority Patterns (Immediate Implementation)
The following patterns have been identified as high priority based on statistical significance, implementation feasibility, and market relevance:

1. **Sakata's Five Methods** (酒田五法)
   - Core Japanese trading methodology
   - High statistical validity in trend continuation/ reversal
   - Implementation: Pattern recognition engine

2. **Morning Star** (明けの明星)
   - Three-candle bullish reversal pattern
   - Strong statistical performance in downtrends
   - Implementation: Candlestick pattern detector

3. **Evening Star** (宵の明星)
   - Three-candle bearish reversal pattern
   - Reliable reversal signal in uptrends
   - Implementation: Candlestick pattern detector

4. **Hammer** (捨て子底)
   - Single-candle bullish reversal
   - Effective at market bottoms
   - Implementation: Candlestick morphology analysis

5. **Hanging Man** (首吊り線)
   - Single-candle bearish reversal
   - Effective at market tops
   - Implementation: Candlestick morphology analysis

6. **Three Black Crows** (三羽ガラス)
   - Three-candle bearish reversal
   - Strong reversal signal
   - Implementation: Sequential pattern recognition

7. **Three White Soldiers** (赤三兵)
   - Three-candle bullish reversal
   - Reliable uptrend signal
   - Implementation: Sequential pattern recognition

8. **Rising Three Methods** (上げ三法)
   - Continuation pattern in uptrends
   - High success rate in trending markets
   - Implementation: Complex pattern recognition

#### Medium Priority Patterns (Conditional Implementation)
Selected medium-priority patterns for enhanced signal diversity:

1. **Bullish Engulfing** (抱き陽線)
   - Two-candle bullish reversal
   - Good statistical performance
   - Implementation: Simple engulfing pattern

2. **Bearish Engulfing** (抱き陰線)
   - Two-candle bearish reversal
   - Reliable reversal signal
   - Implementation: Simple engulfing pattern

3. **Piercing Pattern** (差し込み線)
   - Two-candle bullish reversal
   - Effective in downtrends
   - Implementation: Price penetration analysis

4. **Three Black Crows (Bearish)** (三空叩き込み)
   - Three-candle bearish pattern
   - Strong momentum signal
   - Implementation: Sequential analysis

5. **Three White Soldiers (Bullish)** (三空踏み上げ)
   - Three-candle bullish pattern
   - Reliable uptrend continuation
   - Implementation: Sequential analysis

#### Elliot Wave Integration
Advanced wave pattern recognition for comprehensive market cycle analysis:

- **Wave I**: Initial impulse wave
- **Wave V**: Final impulse wave
- **Wave Y**: Terminal wave in complex corrections
- **Wave P**: Irregular correction wave
- **Wave N**: Complex correction wave
- **Wave S**: Secondary correction wave

#### Advanced Technical Analysis Integration

##### Fibonacci Extensions and Retracements
Mathematical ratios for price projection and support/resistance levels:

- **Fibonacci Retracements**: 0.236, 0.382, 0.5, 0.618, 0.786 levels for pullback identification
- **Fibonacci Extensions**: 1.272, 1.414, 1.618, 2.618 levels for price targets
- **Implementation**: Automated level calculation and signal generation at key ratios
- **Signal Types**: Entry signals at retracement levels, exit signals at extension targets

##### Gann Analysis
Geometric and mathematical analysis for time and price relationships:

- **Gann Angles**: 1:1, 2:1, 4:1, 8:1 angles for trend analysis
- **Gann Squares**: Square of nine for price and time projections
- **Gann Fans**: Support/resistance lines based on geometric angles
- **Implementation**: Angle calculation and intersection signal detection
- **Signal Types**: Trend continuation/reversal signals at angle intersections

##### Wave Counting
Advanced market structure analysis beyond basic Elliot waves:

- **Degree Labels**: Micro, Minor, Intermediate, Major wave classifications
- **Wave Structure Validation**: Proper alternation and proportion analysis
- **Fractal Wave Analysis**: Multi-timeframe wave relationships
- **Implementation**: Automated wave labeling and structure validation
- **Signal Types**: Wave completion signals, invalidation alerts

##### Harmonic Patterns
Geometric price patterns based on Fibonacci ratios:

- **Gartley Pattern**: Bullish/bearish 5-point harmonic pattern
- **Butterfly Pattern**: Extended harmonic pattern with 1.618 BC projection
- **Bat Pattern**: 0.886 XA retracement harmonic structure
- **Crab Pattern**: Extreme harmonic pattern with 1.618 AB=CD
- **Implementation**: Pattern recognition using Fibonacci ratios and geometric relationships
- **Signal Types**: High-probability reversal signals at pattern completion

## Configuration

### Signal Guide Configuration
```python
from ztb.trading.strategies.action_signal_guide import ActionSignalGuide, GuidanceMode

# Initialize with different guidance levels
signal_guide = ActionSignalGuide(
    guidance_mode=GuidanceMode.WEAK,  # NONE, WEAK, STRONG
    enabled_patterns=['sakata_five_methods', 'morning_star', 'hammer'],
    signal_weights={'bullish': 0.7, 'bearish': 0.8}
)
```

### Pattern-Specific Settings
```python
# Configure pattern recognition parameters
pattern_config = {
    'sakata_five_methods': {
        'min_trend_length': 20,
        'confirmation_threshold': 0.75
    },
    'morning_star': {
        'body_ratio_threshold': 0.6,
        'shadow_ratio_threshold': 0.3
    }
}
```

## Usage Examples

### Basic Signal Generation
```python
# Get signals for current market state
signals = signal_guide.get_signals(market_data, current_position)

# Process signals
for signal in signals:
    if signal.strength > 0.7:
        print(f"Strong {signal.type} signal: {signal.description}")
```

### Integration with Training Loop
```python
# During RL training
state = env.get_state()
action_signals = signal_guide.get_action_signals(state)

# Modify rewards based on signal alignment
if action_signals.preferred_action == agent_action:
    reward += signal_alignment_bonus
```

## Performance Metrics

### Signal Quality Metrics
- **Accuracy**: Percentage of correct signals
- **Precision**: True positive rate
- **Recall**: Signal detection completeness
- **F1-Score**: Harmonic mean of precision and recall

### Training Impact Metrics
- **Convergence Speed**: Training episodes to reach target performance
- **Stability**: Variance in final policy performance
- **Generalization**: Performance on unseen market conditions

## Implementation Roadmap

### Phase 1: Core Pattern Implementation (Current)
- [x] Basic signal framework
- [x] Moving average and momentum signals
- [ ] High-priority candlestick patterns

### Phase 2: Advanced Patterns (Next)
- [ ] Elliot wave recognition
- [ ] Complex multi-candle patterns
- [ ] Volume-price analysis integration

### Phase 3: Machine Learning Enhancement (Future)
- [ ] Pattern effectiveness learning
- [ ] Adaptive signal weighting
- [ ] Market regime-specific patterns

## Testing and Validation

### Unit Tests
- Pattern recognition accuracy
- Signal strength calculation
- Edge case handling

### Integration Tests
- Signal integration with reward system
- Training loop compatibility
- Performance impact assessment

### Backtesting Validation
- Historical pattern performance
- Signal contribution to returns
- Risk-adjusted performance metrics

## Dependencies

- `ztb.trading.strategies.signal_definitions`
- `pandas` for data manipulation
- `numpy` for numerical computations
- `scipy` for statistical analysis

## References

1. "Japanese Candlestick Charting Techniques" by Steve Nison
2. "The Elliott Wave Principle" by Robert Prechter
3. "Sakata's Five Methods" - Traditional Japanese trading methodology
4. Internal pattern evaluation results (2024 analysis)</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\docs\features\action_signal_guide.md
