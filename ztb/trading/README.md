# Trading Engine Developer Guide

This guide covers the trading components in the `ztb/t**Usage Example**:

```python
from ztb.live.symbols import SymbolNormalizer
from ztb.live.precision_policy import PrecisionPolicyManager

normalizer = SymbolNormalizer()
base, quote = normalizer.normalize("coincheck", "BTC/JPY")  # ("BTC", "JPY")

manager = PrecisionPolicyManager()
price = manager.quantize_price("coincheck", "BTC/JPY", Decimal("12345.67"))  # 12345.67
```

## Risk Profile Presets

**Role**: Predefined risk management configurations for different trading strategies.

**Key Components**:

- `ztb.config.schema.RiskProfileConfig`: Risk management parameters
- `ztb.live.risk_profiles.RiskProfileManager`: Profile management and validation
- Built-in presets: conservative, moderate, aggressive

**Risk Parameters**:

- Position sizing limits and daily loss thresholds
- Stop loss and take profit percentages
- Maximum leverage and open positions
- Cooldown periods after losses

**Built-in Presets**:

- **Conservative**: Low risk, small positions, tight stops
- **Moderate**: Balanced risk-reward, standard parameters
- **Aggressive**: Higher risk tolerance, larger positions

**Usage Example**:

```python
from ztb.live.risk_profiles import get_risk_manager

manager = get_risk_manager()
profile = manager.get_profile("moderate")

# Validate position size
is_valid = manager.validate_position_size("moderate", position_size, portfolio_value)

# Calculate recommended position size
size = manager.calculate_position_size("moderate", portfolio_value, volatility)
```

## Run Seal for Reproducibility

**Role**: Ensures deterministic training runs with environment tracking and seed management.

**Key Components**:

- `ztb.training.run_seal.RunSeal`: Reproducibility metadata container
- `ztb.training.run_seal.RunSealManager`: Seal creation, loading, and validation
- Environment snapshots with Python version, git commit, dependencies

**Features**:

- Automatic seed generation or manual seed setting
- Environment validation against saved seals
- Config and metadata storage
- JSON serialization for persistence

**Usage Example**:

```python
from ztb.training import create_run_seal, get_run_seal_manager

# Create run seal at training start
seal = create_run_seal(
    seed=42,
    config={"learning_rate": 0.001},
    metadata={"experiment": "baseline"}
)

# Later, validate environment
manager = get_run_seal_manager()
validation = manager.validate_environment(seal)
if not all(validation.values()):
    print("Environment mismatch - reproducibility not guaranteed")
```

## Baseline Comparison Output

**Role**: Compares trained model performance against traditional trading strategies.

**Key Components**:

- `ztb.evaluation.baseline_comparison.BaselineComparisonEngine`: Comparison engine
- Built-in strategies: Buy & Hold, SMA Crossover
- `ztb.evaluation.baseline_comparison.ComparisonReport`: Structured comparison results

**Features**:

- Superiority metrics (return difference, Sharpe ratio comparison)
- Risk-adjusted return analysis
- Statistical significance testing
- Automated report generation

**Built-in Strategies**:

- **Buy & Hold**: Passive strategy holding throughout period
- **SMA Crossover**: Moving average crossover signals

**Usage Example**:

```python
from ztb.evaluation import get_baseline_comparison_engine, BaselineResult

engine = get_baseline_comparison_engine()

# Model performance result
model_result = BaselineResult(
    strategy_name="RL Agent",
    total_return=0.25,
    sharpe_ratio=2.1,
    max_drawdown=-0.12,
    win_rate=0.58,
    total_trades=150,
    metrics={}
)

# Compare against baselines
comparison = engine.compare(model_result, price_data)

# Generate report
report = engine.generate_report(comparison, "baseline_comparison.md")
```

## Market Regime Evaluation

**Role**: Evaluates trading performance across different market regimes (bull, bear, sideways) and compares against baselines.

**Key Components**:

- `ztb.evaluation.regime_eval.RegimeDetector`: Detects market regimes using volatility and trend analysis
- `ztb.evaluation.regime_eval.RegimeEvaluator`: Evaluates performance across regime segments
- `scripts/run_regime_eval.py`: CLI tool for regime evaluation and reporting

**Regime Detection**:

- **Bull Market**: Strong upward trends with low volatility
- **Bear Market**: Strong downward trends with high volatility
- **Sideways Market**: Low trend strength with moderate volatility

**Features**:

- Automatic regime segmentation from price data
- Performance metrics per regime (returns, Sharpe ratio, win rate)
- Baseline comparison (Buy & Hold, SMA Crossover) per regime
- JSON and Markdown report generation

**Usage Example**:

```bash
# Run regime evaluation
python scripts/run_regime_eval.py \
  --price-data data/price_history.csv \
  --trade-log logs/trade_log.json \
  --output-dir reports
```

**Sample Output** (regime_report.json):

```json
{
  "bull": {
    "metrics": {
      "total_return": 0.1567,
      "sharpe_ratio": 1.234,
      "win_rate": 0.62,
      "total_trades": 45,
      "avg_trade_return": 0.0035
    },
    "trade_count": 45,
    "regime_confidence": 0.85
  },
  "bear": {
    "metrics": {
      "total_return": -0.0345,
      "sharpe_ratio": -0.456,
      "win_rate": 0.48,
      "total_trades": 32,
      "avg_trade_return": -0.0011
    },
    "trade_count": 32,
    "regime_confidence": 0.78
  },
  "baseline_comparison": {
    "bull": {
      "rl_vs_buy_hold": {
        "return_diff": 0.0567,
        "sharpe_diff": 0.434,
        "win_rate_diff": 0.12
      },
      "rl_vs_sma": {
        "return_diff": 0.0767,
        "sharpe_diff": 0.634,
        "win_rate_diff": 0.07
      }
    }
  }
}
```

## Checkpoint Manager

**Role**: Model and experiment state persistence with async operations, compression, and generation management.

**Key Classes/Functions**:

- `CheckpointManager`: Asynchronous checkpoint management
- `save_async()`, `load()`, `cleanup_old_generations()`

**Usage Example**:

```python
from ztb.utils import CheckpointManager

manager = CheckpointManager(base_path="checkpoints", max_generations=5)
await manager.save_async(model_state, "exp_001")
```

## PPO Trainer

**Role**: Proximal Policy Optimization implementation for trading strategy learning.

**Key Components**:

- `ppo_trainer.py`: Main PPO training implementation
- Environment integration with trading features
- Policy optimization with actor-critic architecture

**Training Features**:

- Asynchronous checkpointing during training
- Memory-efficient batch processing
- Configurable hyperparameters for different market conditions

## Trading Environment

**Role**: Reinforcement learning environment for trading strategy development.

**Key Classes**:

- `environment.py`: Trading environment implementation
- `bridge.py`: Interface between trading logic and RL framework

**Environment Features**:

- Realistic market simulation with historical data
- Reward functions based on Sharpe ratio and risk metrics
- Action space for position sizing and timing decisions

## Integration with Data Pipeline

The trading components integrate with the data pipeline for:

- Real-time feature computation
- Historical data replay for training
- Live trading execution with learned policies

## Configuration

Training and checkpoint parameters are configured through environment variables and config files:

- `ZTB_CHECKPOINT_INTERVAL`: Checkpoint frequency during training
- `ZTB_MAX_MEMORY_GB`: Memory limits for training
- Training hyperparameters in `trade-config.json`

## Symbol Normalization & Precision Policies

**Role**: Cross-exchange symbol standardization and precision enforcement for order accuracy.

**Key Components**:

- `ztb.live.symbols.SymbolNormalizer`: Normalizes trading symbols across venues (Zaif, Coincheck)
- `ztb.live.precision_policy.PrecisionPolicyManager`: Applies venue-specific rounding policies
- `ztb.config.schema.VenuePrecisionConfig`: Configuration for tick sizes and step sizes

**Symbol Formats Supported**:

- BTC/JPY, BTC_JPY, btcjpy (normalized to BTC/JPY)
- Venue-specific mappings (btcfx_jpy → BTCFX/JPY)

**Precision Enforcement**:

- Price quantization using exchange tick sizes
- Quantity quantization using step sizes
- Automatic application in order creation flow

**Usage Example**:

```python
from ztb.live.symbols import SymbolNormalizer
from ztb.live.precision_policy import PrecisionPolicyManager

normalizer = SymbolNormalizer()
base, quote = normalizer.normalize("coincheck", "BTC/JPY")  # ("BTC", "JPY")

manager = PrecisionPolicyManager()
price = manager.quantize_price("coincheck", "BTC/JPY", Decimal("12345.67"))  # 12346
```
