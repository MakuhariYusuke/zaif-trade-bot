# Zaif Trade Bot 🤖

A production-ready reinforcement learning-based trading bot for cryptocurrency markets, built with modern Python practices and comprehensive testing.

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Type Checking](https://img.shields.io/badge/mypy-strict-green.svg)](https://mypy-lang.org)
[![Security](https://img.shields.io/badge/security-bandit+safety-green.svg)](https://github.com/PyCQA/bandit)
[![Coverage](https://img.shields.io/badge/coverage-40%25-yellow.svg)](https://coverage.readthedocs.io)

## 🚀 Features

- **Reinforcement Learning**: PPO and SAC algorithms for trading strategies
- **Multi-Modal Learning**: Integrated price, news sentiment, and economic indicators
- **Risk Management Integration**: SAC v435 with dynamic position sizing, drawdown control, and market adaptation
- **Configurable Feature Sets**: Flexible feature engineering with preset configurations (minimal, high-quality, full)
- **Production Ready**: Comprehensive type checking, security scanning, and CI/CD
- **Modular Architecture**: Clean separation of concerns with well-defined interfaces
- **Extensive Testing**: Unit tests with coverage reporting and integration tests
- **Security First**: Automated vulnerability scanning and secure coding practices
- **Performance Optimized**: Memory-efficient data processing and CUDA optimizations
- **Comprehensive Monitoring**: Logging, metrics, and alerting systems
- **Advanced Reward System**: Modular reward calculator with clear bonus/penalty separation
- **Market Regime Adaptation**: SAC v444 with 12-regime classification and adaptive strategies

## 🆕 Recent Updates (2025-11-02)

### SAC v444 Advanced Market Regime Adaptation System 🚀
- **12-Regime Classification**: Enhanced market state detection (strong/moderate/weak bull/bear trends, ranging markets, special states)
- **Dynamic Threshold Adaptation**: Volatility-based regime threshold adjustment
- **Multi-Timeframe Integration**: Hierarchical analysis across short/medium/long-term timeframes
- **Regime-Specific Optimization**: Adaptive action balance, entropy regularization, and risk management per regime

### Backtest Fixes and Normalization Improvements 📊
- **Action Distribution Balance**: Fixed persistent single-action issues through normalization statistics regeneration
- **Stochastic Prediction**: Implemented `deterministic=False` for balanced BUY/SELL/HOLD distributions (28.3%/36.6%/35.1%)
- **Environment Consistency**: Unified training and backtest configurations with proper VecNormalize application
- **Feature Count Alignment**: Resolved 68→212 feature mismatch through environment warmup and stat regeneration

### Code Organization and Cleanup 🧹
- **Directory Structure**: Organized root-level files into appropriate subdirectories (analysis/, debug/, scripts/, tests/)
- **Documentation**: Updated CHANGELOG.md and README.md with comprehensive change history
- **Type Safety**: Improved type annotations and error handling across backtest scripts

## 🏗️ Reward System Architecture

The trading bot features a sophisticated reward system designed for optimal learning:

### Reward Components
- **Profit Bonuses**: Multipliers for BUY/SELL/HOLD actions with ATR and portfolio coefficients
- **Action Bonuses**: Balanced incentives for different trading actions
- **Behavior Penalties**: Penalties for suboptimal trading patterns
- **Risk Penalties**: Risk management through volatility and position controls

### Configuration Structure
```json
{
  "reward_settings": {
    "profit_bonuses": {
      "profit_multipliers": [2.0, 0.6, 0.4]  // [BUY, SELL, HOLD]
    },
    "action_bonuses": {
      "buy_action_bonus": -0.01,
      "sell_action_bonus": 0.02,
      "hold_action_bonus": 0.0
    },
    "behavior_penalties": {
      "loss_penalty_multiplier": 3.0,
      "action_frequency_penalty": 0.005
    },
    "risk_penalties": {
      "volatility_penalty": 0.02
    }
  }
}
```

### Action Distribution Balance
Recent improvements achieved balanced BUY/SELL/HOLD distributions:
- **BUY**: 31.2% | **SELL**: 53.1% | **HOLD**: 15.8%
- Fixed SELL bias by correcting action bonus parameters
- Modular reward calculation for maintainable code

## 🧠 Multi-Modal Learning System

Advanced multi-modal learning integration for SAC v421, combining multiple data sources for enhanced trading decisions.

### Architecture Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    MultiModal Trading AI                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │  Price      │ │  Text       │ │  Economic   │           │
│  │  Encoder    │ │  Encoder    │ │  Encoder    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│           │             │             │                     │
│           └──────┬──────┼──────┬──────┘                     │
│                  │      │      │                            │
│           ┌──────▼──────▼──────▼──────┐                     │
│           │   Cross-Modal Attention   │                     │
│           └───────────────────────────┘                     │
│                           │                                 │
│           ┌───────────────▼───────────────┐                 │
│           │   Temporal Integration       │                 │
│           │   (BiLSTM + Transformer)     │                 │
│           └───────────────────────────┬───┘                 │
│                           │           │                     │
│           ┌───────────────▼───────────▼───────────────┐     │
│           │         SAC Agent Core                     │     │
│           │  (Actor + Twin Critics + Auto Entropy)    │     │
│           └───────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Multi-Modal Components
- **Price Encoder**: 156 technical indicators processing
- **Text Encoder**: News sentiment analysis with BERT
- **Economic Encoder**: FRED economic indicators integration
- **Cross-Modal Attention**: Multi-head attention for modality interaction
- **Temporal Integration**: BiLSTM + Transformer for time series modeling
- **Model Optimization**: Pruning, Quantization, Knowledge Distillation
- **Inference Acceleration**: JIT Compilation, ONNX, TensorRT support
- **Memory Management**: Advanced memory monitoring and batch processing

### Key Features
- **Enhanced Prediction**: +15-25% accuracy improvement expected
- **Performance Optimization**: 3-5x inference speed improvement with optimization
- **Robustness**: Multi-source validation and risk diversification
- **Explainability**: Attention weights for decision transparency
- **Scalability**: Modular architecture for easy extension
- **Production Ready**: Comprehensive optimization for deployment

### Optimization Features
- **Model Compression**: Pruning (6% sparsity), Dynamic Quantization, Knowledge Distillation
- **Inference Optimization**: JIT compilation, ONNX export, TensorRT acceleration
- **Memory Management**: Intelligent memory monitoring, batch processing optimization
- **Integration Testing**: 100% test coverage with 5 comprehensive test suites

### Usage Example
```python
from ztb.multimodal import create_multimodal_agent, get_default_config
from ztb.multimodal.optimization import InferenceOptimizer

# 設定の読み込み
config = get_default_config()

# マルチモーダルSACエージェントの作成
agent = create_multimodal_agent(
    price_dim=156,
    text_dim=768,
    economic_dim=20,
    action_dim=3
)

# 推論最適化の適用
optimizer = InferenceOptimizer(agent.model)
optimizer.enable_jit_compilation()
optimizer.enable_onnx_optimization()

# データパイプラインの作成
pipeline = create_data_pipeline()
```

## 🎯 Action Signal Guide

Advanced technical analysis pattern recognition system with performance-optimized configuration.

### Pattern Recognition Capabilities
- **Fibonacci Patterns**: Retracement, Extension, Projection with deviation-based strength
- **Harmonic Patterns**: Gartley, Butterfly, Bat, Crab with ratio tolerance
- **Wave Patterns**: Elliott Wave counting with multi-timeframe validation
- **Candlestick Patterns**: 11 traditional Japanese patterns
- **Oscillator Patterns**: RSI, Stochastic, MACD with adaptive thresholds
- **ADX Patterns**: Trend strength with correlation optimization
- **Bollinger Bands**: Volatility-based signals with dynamic width

### Performance Optimization Results
- **Signal Generation**: 1,563 signals analyzed across 7 pattern types
- **Top Performers**: ADX (0.54), Wave (0.63), Oscillator (0.72)
- **Correlation Analysis**: Profit correlation up to 0.106 (ADX)
- **Optimization**: Parallel processing, caching, signal limits (5/bar)

### Usage Example
```python
from ztb.trading.strategies.action_signal_guide import ActionSignalGuide
from ztb.tests.unit.trading.strategies.action_signal_guide import get_optimized_config

# 最適化設定の取得
config = get_optimized_config()

# Action Signal Guideの初期化
guide = ActionSignalGuide(config)

# シグナル生成
signals = guide.generate_signals(market_data)
```

## 📁 Project Structure

```
ztb/                          # Main Python package
├── analysis/                 # Analysis and diagnostic tools
│   ├── unified_analyze.py    # Unified analysis interface
│   ├── core/                 # Core analysis functionality
│   │   ├── model/           # Model analysis (SAC, PPO, etc.)
│   │   ├── data/            # Data quality and feature analysis
│   │   ├── training/        # Training process analysis
│   │   └── performance/     # System and memory performance
│   ├── comparative/         # Version comparison and statistical tests
│   ├── diagnostic/          # System diagnosis and debugging tools
│   ├── specialized/         # Specialized analysis (features, rewards, risk)
│   │   ├── features/        # Feature quality analysis
│   │   ├── rewards/         # Reward function analysis
│   │   ├── risk/            # Risk metrics analysis
│   │   └── market/          # Market regime analysis
│   ├── sessions/            # Session-specific analysis
│   └── utilities/           # Utility analysis tools
├── config/                   # Configuration management
├── data/                     # Data processing utilities
├── evaluation/               # Backtesting and evaluation
├── features/                 # Feature engineering
├── metrics/                  # Performance metrics
├── ops/                      # Operational utilities
├── trading/                  # Trading environments and live trading
│   ├── algorithms/          # RL algorithm implementations
│   ├── core/                # Core training infrastructure
│   ├── evaluation/          # Model evaluation tools
│   └── utils/               # Training utilities
├── types/                    # Type definitions and protocols
└── utils/                    # General utilities

tests/                        # Comprehensive test suite
├── unit/                     # Unit tests
├── integration/             # Integration tests
└── training/                # Training-specific tests

docs/                         # Documentation
config/                       # Configuration files
scripts/                      # Utility scripts
```

## 🎯 Feature Set Management

The bot includes a sophisticated feature set management system that allows flexible configuration of trading features:

### Available Feature Sets
- **minimal**: Core features only (30-50 dimensions) - fastest processing
- **no_harmful** (default): Full features with critical harmful features removed
- **high_quality**: Only correlation-filtered high-quality features
- **full**: Complete feature set (150+ dimensions) - maximum information

### Usage
```python
from ztb.features.sac_v427_feature_engineering import SACv427FeatureEngineer

# Use default (no_harmful) set
engineer = SACv427FeatureEngineer()
features = engineer.generate_v427_features(data)

# Use specific set
features = engineer.generate_v427_features(data, feature_set='high_quality')
```

📖 **[Complete Documentation](docs/features/feature_set_management.md)**

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Git LFS (for large model files)

### Setup
```bash
# Clone repository
git clone https://github.com/MakuhariYusuke/zaif-trade-bot.git
cd zaif-trade-bot

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e .[dev]
```

## 🧪 Testing & Quality

### Test Coverage
- **Unit Tests**: Comprehensive unit test suite covering all core components
- **Integration Tests**: End-to-end testing of complete workflows
- **Trainer Tests**: Specialized tests for SAC trainers including:
  - `OnlineLearningSACTrainer`: Real-time adaptation and streaming learning
  - `MultimodalSACTrainer`: Multi-modal feature integration (price, text, economic data)
- **Coverage Target**: >80% code coverage maintained

### Run Tests
```bash
# Unit tests with coverage
pytest tests/unit/ -v --cov=ztb --cov-report=html

# Integration tests
pytest tests/integration/ -v

# Trainer-specific tests
pytest ztb/adaptation/online_learning/tests.py::TestOnlineLearningSACTrainer -v
pytest tests/test_multimodal_core.py::TestMultimodalSACTrainer -v

# Reward Function & Parameter Tuning Tests
pytest ztb/tests/test_advanced_features.py::TestContinualLearning -v
pytest ztb/tests/test_advanced_features.py::TestContinualLearningIntegration -v

# Parameter sweep tests (SAC baseline tuning)
python -m ztb.training.unified_trainer.trainer --config configs/sac_v420_baseline.json
python -m ztb.training.unified_trainer.trainer --config configs/sac_v420_lr_sweep_0001.json
python -m ztb.training.unified_trainer.trainer --config configs/sac_v420_lr_sweep_0003.json
python -m ztb.training.unified_trainer.trainer --config configs/sac_v420_lr_sweep_0010.json
python -m ztb.training.unified_trainer.trainer --config configs/sac_v420_buffer_sweep_200k.json

# All tests
pytest
```

### Code Quality Checks
```bash
# Type checking
mypy ztb/

# Security scanning
bandit -r ztb/
safety check

# Linting
black ztb/
isort ztb/
```

## 🚀 Usage

### SAC Suite - Unified CLI
The project provides a unified command-line interface for all SAC-related operations:

```bash
# Analyze SAC model performance
python sac.py analyze --model models/sac_model.zip --config configs/analysis.yaml

# Run backtesting simulations
python sac.py backtest --model models/sac_model.zip --data data/test.csv --episodes 100

# Train SAC models (integrated with unified trainer)
python sac.py train --config configs/sac_training.yaml --timesteps 100000

# Parallel training with multiple algorithms
python sac.py train --config configs/sac_training.yaml --parallel --include-ppo

# Utility functions
python sac.py utils config  # Check configuration
python sac.py utils data    # Validate data files
python sac.py utils clean   # Clean project files
```

## 🎯 V4XX Unified Training System

A comprehensive, structured training and analysis system for all V4XX series SAC models, providing unified interfaces, automatic configuration conversion, and improved maintainability.

### Key Features

- **Unified Configuration**: Automatic detection and conversion between legacy (v427) and unified (v435+) formats
- **Consistent Training**: Single trainer interface supporting all V4XX versions
- **Unified Analysis**: Version-agnostic analysis framework with advanced statistical metrics
- **PowerShell Support**: Native Windows command-line interface
- **Modular Architecture**: Clean separation enabling easy extension and maintenance

### Supported Versions

| Version | Status | Training Support | Analysis Support | Notes |
|---------|--------|------------------|------------------|-------|
| V427 | ✅ Unified | ✅ Full | ✅ Full | Legacy format auto-conversion |
| V435 | ✅ Unified | ✅ Full | ✅ Full | Native unified format |
| V437 | ✅ Unified | ✅ Full | ✅ Full | Unified trainer integration |
| V440 | ✅ Unified | ✅ Full | ✅ Full | Enhanced reward function |
| V441+ | 🚀 Planned | ✅ Ready | ✅ Ready | New versions use unified system |

### Quick Start

#### Training a Model
```bash
# Using Python directly
python scripts/training/train_sac_v435_unified.py --config config/sac_v435_7a_config.json

# Using PowerShell (Windows)
.\scripts\run_training.ps1 -Action train -Version v435
```

#### Analyzing Results
```bash
# Using Python directly
python -c "from ztb.analysis.v4xx_unified_analyzer import analyze_v4xx_results; analyze_v4xx_results('results/v440/backtest_results_v440.json', version='440')"

# Using PowerShell (Windows)
.\scripts\run_training.ps1 -Action analyze -Version v440
```

#### Converting Legacy Configurations
```bash
# Convert v427 format to unified format
python -c "from ztb.utils.v4xx_config_converter import convert_config_file; convert_config_file('config/sac_v427_default_config.json')"

# Using PowerShell
.\scripts\run_training.ps1 -Action convert -Config config/sac_v427_default_config.json
```

### Architecture Overview

```
V4XX Unified System
├── Configuration Layer
│   ├── V4XXConfigConverter    # Auto-detect and convert config formats
│   └── Validation             # Unified configuration validation
├── Training Layer
│   ├── V4XXUnifiedTrainer     # Single interface for all versions
│   ├── Unified Trainer Core   # SAC/PPO algorithm abstraction
│   └── Environment Integration # HeavyTradingEnv compatibility
├── Analysis Layer
│   ├── V4XXUnifiedAnalyzer    # Version-agnostic analysis
│   ├── Statistical Metrics    # P-mean, Sharpe, Drawdown analysis
│   └── Report Generation      # Structured analysis reports
└── Interface Layer
    ├── PowerShell Scripts     # Windows-native CLI
    ├── Python API             # Direct programmatic access
    └── Migration Tools        # Legacy script conversion
```

### Configuration Conversion

The system automatically handles configuration differences:

#### Legacy Format (v427)
```json
{
  "model_name": "sac_v427_market_adaptive",
  "algorithm": "sac",
  "total_timesteps": 10000,
  "sac_hyperparameters": {
    "learning_rate": 0.0003,
    "buffer_size": 50000
  },
  "environment": {
    "initial_balance": 200000.0,
    "transaction_cost": 1e-05
  }
}
```

#### Unified Format (v435+)
```json
{
  "algorithm": "sac",
  "model_name": "sac_v427_converted",
  "version": "4.2.7",
  "training": {
    "total_timesteps": 10000,
    "sac_hyperparameters": {
      "learning_rate": 0.0003,
      "buffer_size": 50000
    },
    "environment": {
      "initial_balance": 200000.0,
      "transaction_cost": 1e-05
    },
    "data_config": {
      "data_path": "data/btc_jpy_real_dataset.csv"
    }
  }
}
```

### Advanced Analysis Features

The unified analyzer provides comprehensive statistical analysis:

- **Basic Metrics**: Total episodes, average reward, win rate, total return
- **Risk Metrics**: Sharpe ratio, maximum drawdown, volatility analysis
- **Advanced Statistics**: P-mean method (geometric/arithmetic), Sortino ratio
- **Performance Scoring**: Overall performance score (0-1 scale)
- **Structured Reports**: JSON reports with analysis metadata

#### Analysis Output Example
```
📊 V440 Analysis Report
============================================================
Results Path: results\v440\backtest_results_v440.json
Analysis Time: 2025-10-29T01:47:43.130394

==================================================
 V440 Performance Metrics
==================================================
Total Episodes: 10
Average Reward: -6480.0000
Average Trades: 0.0000
Win Rate: 0.00%
Total Return: -49.7789
Sharpe Ratio: -4977886795.4659
Max Drawdown: -99.7968
P Mean Arithmetic: -49.7789
P Mean Geometric: 0.0000
Average Drawdown: -88.8130
==================================================

📈 Summary:
  - Metrics Calculated: 10
  - Advanced Metrics: Yes
  - Performance Score: 0.00
```

### Migration Guide

#### From Individual Scripts to Unified System

**Before (v437):**
```python
# Individual training script
from stable_baselines3 import SAC
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv

# Manual setup for each version
config = load_config("config/sac_v427_default_config.json")
env = HeavyTradingEnv(...)
model = SAC(...)
model.learn(total_timesteps=10000)
```

**After (Unified):**
```python
# Single unified interface
from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer

trainer = V4XXUnifiedTrainer("config/sac_v427_config.json", version="v427")
trainer.train()  # Handles all versions automatically
```

#### PowerShell Integration

For Windows users, the PowerShell script provides a native interface:

```powershell
# Activate virtual environment and run training
.\scripts\run_training.ps1 -Action train -Version v435 -Config config/custom.json

# Analyze multiple versions
.\scripts\run_training.ps1 -Action analyze -Version v427
.\scripts\run_training.ps1 -Action analyze -Version v440

# Convert legacy configurations
.\scripts\run_training.ps1 -Action convert -Config config/legacy_config.json
```

### API Reference

#### V4XXUnifiedTrainer

```python
class V4XXUnifiedTrainer:
    def __init__(self, config_path: str, version: Optional[str] = None)
    def validate_config(self) -> bool
    def initialize_trainer(self)
    def train(self)
    def save_config(self, output_path: Optional[str] = None)
```

#### V4XXUnifiedAnalyzer

```python
class V4XXUnifiedAnalyzer:
    def __init__(self, results_path: str, version: Optional[str] = None)
    def calculate_basic_metrics(self) -> Dict[str, Any]
    def calculate_advanced_metrics(self) -> Dict[str, Any]
    def generate_report(self) -> Dict[str, Any]
    def print_report(self)
    def save_report(self, output_path: Optional[str] = None)

# Convenience function
def analyze_v4xx_results(results_path: str, version: Optional[str] = None, save_report: bool = True)
```

#### V4XXConfigConverter

```python
class V4XXConfigConverter:
    @staticmethod
    def convert_to_unified(config: Dict[str, Any]) -> Dict[str, Any]
    @staticmethod
    def detect_config_version(config: Dict[str, Any]) -> str
    @classmethod
    def load_and_convert_config(cls, config_path: str) -> Dict[str, Any]

def convert_config_file(input_path: str, output_path: Optional[str] = None) -> str
```

### Benefits

- **Reduced Code Duplication**: Common components shared across all V4XX versions
- **Improved Maintainability**: Changes to core functionality benefit all versions
- **Consistent Interfaces**: Same API regardless of model version
- **Automatic Compatibility**: Legacy configurations work without modification
- **Enhanced Analysis**: Comprehensive statistical analysis for all versions
- **Windows Support**: Native PowerShell integration for Windows development

### Future Development

- **v441+**: All new versions will use this unified system by default
- **Plugin Architecture**: Easy addition of new algorithms and features
- **Configuration Templates**: Pre-built configurations for common use cases
- **Automated Testing**: Comprehensive test suite for all unified components
- **Performance Monitoring**: Integrated performance tracking and optimization

### Troubleshooting

#### Common Issues

1. **Configuration Validation Errors**
   ```bash
   # Validate configuration before training
   python ztb/training/v4xx_unified_trainer.py --config your_config.json --validate-only
   ```

2. **Version Auto-Detection Issues**
   ```python
   # Explicitly specify version
   trainer = V4XXUnifiedTrainer("config.json", version="v427")
   ```

3. **PowerShell Execution Policy**
   ```powershell
   # Allow script execution
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

4. **Memory Issues During Training**
   - Reduce batch size in configuration
   - Enable gradient accumulation if supported
   - Use smaller buffer sizes for memory-constrained systems

### File Structure

```
ztb/
├── training/
│   └── v4xx_unified_trainer.py     # Main unified trainer
├── analysis/
│   └── v4xx_unified_analyzer.py    # Unified analysis framework
├── utils/
│   ├── v4xx_config_converter.py    # Configuration conversion
│   └── safety.py                   # Safe file operations
├── config/
│   ├── sac_v427_config.json        # v427 unified config
│   └── sac_v440_config.json        # v440 unified config
└── training/
    └── unified_trainer/
        └── algorithms/             # SAC/PPO implementations

scripts/
├── run_training.ps1                # PowerShell interface
└── training/
    ├── train_sac_v437_unified.py   # v437 unified script
    └── train_sac_v440_unified.py   # v440 unified script

v435/
└── V4XX_CONFIGURATION_GUIDE.md     # Detailed configuration guide
```

### Contributing to V4XX System

When adding new V4XX versions:

1. **Create unified configuration** in `config/` directory
2. **Add version support** to `V4XXConfigConverter.detect_config_version()`
3. **Test with unified trainer** and analyzer
4. **Update documentation** in `V4XX_CONFIGURATION_GUIDE.md`
5. **Add PowerShell support** if needed

The unified system ensures that all V4XX versions maintain consistent behavior and interfaces while allowing for version-specific optimizations and features.

---

### Unified Analysis Suite
Comprehensive analysis toolkit for all trading system components:

```bash
# Model analysis
python ztb/analysis/unified_analyze.py model sac --model models/sac_model.zip
python ztb/analysis/unified_analyze.py model validate --model model.zip --data test.csv

# Data analysis
python ztb/analysis/unified_analyze.py data quality --dataset data/train.csv
python ztb/analysis/unified_analyze.py data correlation --dataset data.csv --threshold 0.8

# Training analysis
python ztb/analysis/unified_analyze.py training tensorboard --logdir logs/
python ztb/analysis/unified_analyze.py training metrics --logdirs logs/v1 logs/v2

# Performance analysis
python ztb/analysis/unified_analyze.py performance memory --pid 1234 --duration 60
python ztb/analysis/unified_analyze.py performance profile --code script.py

# Comparative analysis
python ztb/analysis/unified_analyze.py comparative versions --versions v378 v381 v384
python ztb/analysis/unified_analyze.py comparative statistical --data-a results1.csv --data-b results2.csv

# Diagnostic tools
python ztb/analysis/unified_analyze.py diagnostic environment --config config.yaml
python ztb/analysis/unified_analyze.py diagnostic simple --model model.zip

# Specialized analysis
python ztb/analysis/unified_analyze.py specialized features quality --data data.csv
python ztb/analysis/unified_analyze.py specialized reward function --config reward.yaml

# Show available tools in any category
python ztb/analysis/unified_analyze.py model  # Shows: extract, features, sac, validate
python ztb/analysis/unified_analyze.py data   # Shows: correlation, quality, schema, timeseries
```

### Unified Trainer Integration
SAC training is now fully integrated with the unified trainer system, providing:
- **Unified Configuration**: Consistent configuration across all algorithms
- **Parallel Training**: Train multiple algorithms simultaneously
- **Advanced Features**: Mixed precision, federated learning, ensemble systems
- **Horizontal Scaling**: Support for multiple model variations and hyperparameter sweeps

### Training a Model
```python
from ztb.training.algorithms import AlgorithmFactory
from ztb.training.core import PPOTrainerAutoHalt

# Create algorithm
ppo = AlgorithmFactory.create("ppo")

# Train model
trainer = PPOTrainerAutoHalt(config)
trained_model = trainer.train(session_id="my_session")
```

### Configuration
```python
from ztb.training.config.ppo_config import get_ppo_config

config = get_ppo_config({
    "learning_rate": 3e-4,
    "total_timesteps": 1000000,
    "batch_size": 64
})
```

### Evaluating a Model
```python
from ztb.evaluation.evaluate import main

# Run evaluation with default config
main()

# Or specify custom config
main(config_path="config/custom_evaluation.yaml")
```

### Backtesting a Model
```python
from ztb.evaluation.backtest_model import run_backtest

# Run backtest with default config
run_backtest()

# Or specify custom config
run_backtest(config_path="config/custom_backtest.yaml")
```

### Command Line Usage
```bash
# Evaluate a trained model
python -m ztb.evaluation.evaluate --config config/evaluation.yaml

# Run backtest simulation
python -m ztb.evaluation.backtest_model --config config/backtest.yaml
```

## 🔒 Security

This project implements multiple security measures:

- **Dependency Scanning**: Automated vulnerability detection with `safety`
- **Code Security**: Static analysis with `bandit`
- **Type Safety**: Strict type checking with `mypy`
- **Secure Defaults**: Conservative configuration defaults

## 📊 Monitoring & Observability

- **Comprehensive Logging**: Structured logging with configurable levels
- **Metrics Collection**: Performance metrics and trading statistics
- **Health Checks**: System health monitoring and alerting
- **TensorBoard Integration**: Training visualization and monitoring

## 📚 Documentation

### Analysis Reports
- **[SAC v424 Deep Analysis Report](docs/SAC_V424_DEEP_ANALYSIS_REPORT.md)**: Comprehensive analysis revealing critical strategy weaknesses including 67% SELL bias, market non-correlation (0.019), and robustness collapse (0.262 score)
- **[SAC v425 Improvement Plan](docs/SAC_V425_IMPROVEMENT_PLAN.md)**: 5-phase improvement strategy leveraging 85% existing systems to address fundamental issues over 10-15 days

### Key Findings
- **SELL Bias Overlearning**: Training 26.8% → Test 67% indicates data leakage or reward design flaws
- **Market Disconnection**: Price correlation 0.019, β-value 0.017 shows strategy ignores BTC price movements
- **Adaptation Failure**: Learning efficiency 0.000, adaptation ratio -1.755 demonstrates inability to learn
- **Robustness Breakdown**: Score 0.262 with 0.000 regime consistency across market conditions

### Improvement Strategy
- **Data Foundation**: BTCDataAugmentor for balanced market condition datasets (50k samples)
- **Feature Engineering**: Correlation-aware features for market connectivity
- **Adaptive Rewards**: Dynamic penalty adjustment based on action distribution
- **Curriculum Learning V2**: 4-stage progressive learning (bias awareness → correlation optimization → scalping)
- **Comprehensive Validation**: Enhanced analyze_backtest.py with correlation, stress testing, and walk-forward analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Run tests: `pytest`
4. Check code quality: `mypy ztb/ && bandit -r ztb/`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Development Guidelines

- **Type Hints**: All code must have proper type annotations
- **Testing**: Maintain >80% test coverage
- **Security**: No high/critical security issues
- **Documentation**: Update docs for API changes
- **Style**: Follow PEP 8 with Black formatting

## 📈 Performance

- **Memory Optimization**: Efficient data structures and garbage collection
- **CUDA Support**: GPU acceleration for training
- **Parallel Processing**: Multi-threaded data loading and processing
- **Caching**: Intelligent data caching and reuse

## 🐳 Docker

```bash
# Build image
docker build -t zaif-trade-bot .

# Run training
docker run -v $(pwd)/data:/app/data zaif-trade-bot train --config config/training.json
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Stable Baselines3 for RL algorithms
- PyTorch for deep learning infrastructure
- The open-source community for invaluable tools and libraries

---

**⚠️ Disclaimer**: This software is for educational and research purposes. Cryptocurrency trading involves significant risk. Always test thoroughly and never risk more than you can afford to lose.

## 🔍 Advanced ML Features for SAC v421

SAC v421 introduces cutting-edge machine learning capabilities for enhanced trading intelligence and data quality management.

### Anomaly Detection System
Comprehensive data quality monitoring and outlier detection for robust trading signals.

**Key Components:**
- **Statistical Methods**: Z-score, IQR, MAD-based anomaly detection
- **Machine Learning**: Isolation Forest, Elliptic Envelope algorithms
- **Neural Networks**: Autoencoder-based unsupervised anomaly detection
- **Voting System**: Multi-method consensus for high-confidence anomaly detection

**Usage:**
```python
from ztb.data.anomaly_detection import ComprehensiveAnomalyDetector

detector = ComprehensiveAnomalyDetector(
    statistical_methods=['zscore', 'iqr'],
    ml_methods=['isolation_forest'],
    voting_threshold=0.5
)

# Fit on training data
detector.fit_ml_detectors(training_data)

# Detect anomalies
is_anomaly, results = detector.detect_anomalies(new_data)
```

### Meta Learning for Rapid Adaptation
MAML and Reptile algorithms for quick adaptation to new market conditions.

**Key Features:**
- **MAML**: Model-Agnostic Meta-Learning for few-shot adaptation
- **Reptile**: First-order meta-learning for efficient knowledge transfer
- **Market-Specific Adaptation**: Specialized models for different exchanges
- **Cross-Market Knowledge**: Transfer learning between correlated markets

**Usage:**
```python
from ztb.adaptation.meta_learning import MarketMetaLearner

meta_learner = MarketMetaLearner(state_dim=10, action_dim=4)

# Add market data
meta_learner.add_market_data('BTC_JPY', states, actions, rewards, next_states, dones)

# Train meta-learner
history = meta_learner.train_on_markets(num_epochs=100)

# Adapt to new market
adapted_model = meta_learner.adapt_to_market('ETH_JPY', market_data)
```

### Federated Learning with Privacy
Privacy-preserving distributed training across multiple exchanges and data sources.

**Key Features:**
- **FedAvg Algorithm**: Federated Averaging with differential privacy
- **Privacy Protection**: Opacus integration for ε-differential privacy
- **Market-Based Federation**: Exchange-specific model training
- **Cross-Market Aggregation**: Knowledge synthesis across privacy boundaries

**Usage:**
```python
from ztb.training.federated_learning import MarketFederatedLearner, FederatedConfig

# Configure federated learning
market_configs = {
    'exchange_A': FederatedConfig(num_clients=5, enable_privacy=True),
    'exchange_B': FederatedConfig(num_clients=3, enable_privacy=True)
}

federated_learner = MarketFederatedLearner(base_model, market_configs)

# Add clients with private data
federated_learner.add_market_client('exchange_A', client_data_loader)

# Train federated models
results = federated_learner.train_all_markets(loss_fn)

# Aggregate cross-market knowledge
global_model = federated_learner.aggregate_cross_market_knowledge()
```

### Unified Integration
All advanced features are seamlessly integrated into the UnifiedTrainer.

**Configuration:**
```json
{
  "enable_anomaly_detection": true,
  "anomaly_statistical_methods": ["zscore", "iqr"],
  "anomaly_ml_methods": ["isolation_forest"],
  "enable_anomaly_autoencoder": false,
  "anomaly_voting_threshold": 0.5,

  "enable_meta_learning": true,
  "meta_algorithm": "maml",
  "meta_batch_size": 4,

  "enable_federated": true,
  "federated_markets": true,
  "markets": ["exchange_A", "exchange_B"],
  "num_clients": 5,
  "federated_rounds": 10,
  "enable_privacy": true,
  "privacy_budget": 1.0
}
```

**Training with Advanced Features:**
```bash
# Enable all advanced features
python -m ztb.training.unified_trainer.trainer \\
  --config config/advanced_sac_v421.yaml \\
  --enable_anomaly_detection \\
  --enable_meta_learning \\
  --enable_federated \\
  --federated_markets
```

## 📚 Continual Learning for Knowledge Accumulation
Long-term knowledge accumulation and catastrophic forgetting prevention for sustained trading performance.

### Key Techniques
- **EWC (Elastic Weight Consolidation)**: Protects important parameters from being overwritten
- **Rehearsal Methods**: Maintains past knowledge through data replay
- **Progressive Networks**: Expands network capacity for new tasks while preserving old knowledge

### Usage
```python
from ztb.adaptation.continual_learning import ContinualLearner, ContinualLearningConfig

config = ContinualLearningConfig(
    method='ewc',
    ewc_lambda=0.1,
    max_tasks_in_memory=5
)

continual_learner = ContinualLearner(model, config)

# Learn new task while preserving previous knowledge
stats = continual_learner.learn_task(task_data, loss_fn, optimizer)
```

### Configuration
```json
{
  "enable_continual_learning": true,
  "continual_method": "ewc",
  "continual_ewc_lambda": 0.1,
  "continual_buffer_size": 1000,
  "continual_max_tasks": 5
}
```

## ### SAC Parameter Tuning

For systematic parameter optimization before implementing advanced features:

```bash
# Run complete parameter tuning suite (learning rate, buffer size, batch size, etc.)
python scripts/run_sac_v420_parameter_tuning.py

# Or use the batch file on Windows
./run_sac_v420_tuning.bat

# Analyze tuning results and get recommendations
python scripts/analyze_sac_v420_tuning_results.py
```

**Tuning Features:**
- **Short Test Runs**: 1k-5k steps for efficient parameter validation
- **Comprehensive Sweeps**: Learning rate, buffer size, batch size, entropy, reward scale, gamma
- **Automated Analysis**: Performance comparison and optimal parameter recommendations
- **Results Storage**: Structured results in `results/sac_v420_tuning/`

### Individual Parameter Tests

Test specific parameters with individual config files:

```bash
# Test learning rates
python -m ztb.training.unified_trainer.main configs/sac_v420_lr_sweep_0.0001_1k.json
python -m ztb.training.unified_trainer.main configs/sac_v420_lr_sweep_0.001_1k.json

# Test buffer sizes
python -m ztb.training.unified_trainer.main configs/sac_v420_buffer_sweep_100k_1k.json
python -m ztb.training.unified_trainer.main configs/sac_v420_buffer_sweep_200k_1k.json
```
