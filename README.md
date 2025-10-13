# Zaif Trade Bot 🤖

A production-ready reinforcement learning-based trading bot for cryptocurrency markets, built with modern Python practices and comprehensive testing.

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Type Checking](https://img.shields.io/badge/mypy-strict-green.svg)](https://mypy-lang.org)
[![Security](https://img.shields.io/badge/security-bandit+safety-green.svg)](https://github.com/PyCQA/bandit)
[![Coverage](https://img.shields.io/badge/coverage-40%25-yellow.svg)](https://coverage.readthedocs.io)

## 🚀 Features

- **Reinforcement Learning**: PPO and SAC algorithms for trading strategies
- **Production Ready**: Comprehensive type checking, security scanning, and CI/CD
- **Modular Architecture**: Clean separation of concerns with well-defined interfaces
- **Extensive Testing**: Unit tests with coverage reporting and integration tests
- **Security First**: Automated vulnerability scanning and secure coding practices
- **Performance Optimized**: Memory-efficient data processing and CUDA optimizations
- **Comprehensive Monitoring**: Logging, metrics, and alerting systems

## 📁 Project Structure

```
ztb/                          # Main Python package
├── analysis/                 # Analysis and diagnostic tools
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

### Run Tests
```bash
# Unit tests with coverage
pytest tests/unit/ -v --cov=ztb --cov-report=html

# Integration tests
pytest tests/integration/ -v

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