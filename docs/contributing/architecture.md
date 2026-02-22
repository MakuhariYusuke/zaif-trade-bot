# Architecture Overview

This document provides an overview of the Zaif Trade Bot architecture, covering both the TypeScript/Node.js layer and the Python/ML layer.

## System Architecture

The project follows a layered architecture with clear separation of concerns between trading logic, data processing, and machine learning components.

## TypeScript/Node.js Layer

### Core Components

- **`core/`**: Pure business logic (independent of I/O, filesystem, environment)
- **`adapters/`**: I/O and external API delegation (filesystem, HTTP, signing, rate limiting)
- **`application/`**: Strategy and use case orchestration (event subscription, aggregation)
- **`app/`**: Main execution loop (`npm start`)
- **`tools/`**: CLI tools for live/paper/ml/stats modes

### Design Principles

- **Dependency Injection**: Adapters injected into core logic for testability
- **Pure Functions**: Core logic is deterministic and side-effect free
- **Interface Segregation**: Clean contracts between layers

## Python/ML Layer (ztb/)

### Directory Structure

- **`ztb/core/`**: Coroutines (trade execution, market data, risk management)
- **`ztb/features/`**: Technical indicator features (trend, volatility, momentum)
- **`ztb/evaluation/`**: Quality assessment, promotion engine, benchmarks
- **`ztb/trading/`**: Reinforcement learning environment, PPO trainer
- **`ztb/ml/`**: ML pipeline, data transformation, export
- **`ztb/experiments/`**: Experiment code (scaling tests, validation scripts)
- **`ztb/utils/`**: Common utilities (notifications, caching, logging)
- **`ztb/tests/`**: Unit tests, integration tests
- **`ztb/tools/`**: Operational tools, report generation

### Key Components

#### Trading Engine (`ztb/trading/`)

- **PPO Trainer**: Proximal Policy Optimization for strategy learning
- **Environment**: RL environment with realistic market simulation
- **Bridge**: Interface between trading logic and RL framework

#### Data Pipeline (`ztb/data/`)

- **Streaming**: Real-time data ingestion and processing
- **CoinGecko Integration**: External market data API client
- **Caching**: Price and feature data caching for performance

#### Feature Engineering (`ztb/features/`)

- **Technical Indicators**: RSI, MACD, Bollinger Bands, etc.
- **Statistical Features**: Volatility, momentum, trend analysis
- **Quality Gates**: Feature validation and statistical checks

#### Evaluation Framework (`ztb/evaluation/`)

- **Quality Gates**: NaN rates, correlations, skewness, kurtosis
- **Benchmarking**: Performance comparison across experiments
- **Promotion Engine**: Automated model advancement based on criteria

#### Utilities (`ztb/utils/`)

- **LoggerManager**: Unified logging, notifications, experiment tracking
- **CheckpointManager**: Async model persistence with compression
- **Stats Utils**: Statistical analysis functions
- **Cache Manager**: Multi-level caching (memory, disk, Redis)

## Data Flow

```text
Market Data → Streaming Pipeline → Feature Engineering → Trading Environment
                                                            ↓
Evaluation Framework ← Checkpoint Manager ← PPO Training ← ↑
                                                            ↓
Notification System ← Experiment Tracking ← Quality Gates ← ↑
```

## Configuration Management

- **Environment Variables**: Runtime overrides and secrets
- **YAML/JSON Config Files**: Default settings and complex configurations
- **ZTBConfig Class**: Centralized configuration access

## Testing Strategy

- **Unit Tests**: Isolated component testing with mocks
- **Integration Tests**: End-to-end pipeline validation
- **Performance Tests**: Memory profiling and timing analysis
- **Quality Gates**: Automated validation before promotion

## Migration Notes

- Legacy `src/services/*` has been removed - use `@adapters/*` for new code
- Types consolidated in `src/contracts` - import via `@contracts`
- Python code centralized under `ztb/` directory

## Development Workflow

1. **Feature Development**: Create branch, implement with tests
2. **Quality Checks**: Type checking, linting, unit tests
3. **Integration**: Run integration tests and performance validation
4. **Review**: Code review and automated quality gates
5. **Deployment**: Automated deployment with rollback capabilities
