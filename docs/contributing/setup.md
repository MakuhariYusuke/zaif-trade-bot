# Development Environment Setup

This guide covers setting up the development environment for contributing to the Zaif Trade Bot project.

## Prerequisites

### System Requirements

- **Python**: 3.11 or higher
- **Node.js**: 18.x or higher (for testing framework)
- **Git**: Latest version
- **VS Code**: Recommended IDE with extensions

### Hardware Requirements

- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: 10GB free space for datasets and checkpoints
- **GPU**: Optional but recommended for ML training

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/zaif-trade-bot.git
cd zaif-trade-bot
```

### 2. Python Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Node.js Dependencies (for testing)

```bash
npm install
```

### 4. VS Code Extensions

Install these recommended extensions:

- Python (Microsoft)
- Pylance
- Jupyter
- GitLens
- Markdown Preview Enhanced
- Code Runner

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Copy template
cp .env.example .env

# Edit with your settings
ZTB_LOG_LEVEL=DEBUG
ZTB_MEM_PROFILE=true
ZTB_CUDA_WARN_GB=8.0
```

### VS Code Settings

Add to `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.mypyEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false
}
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### 3. Run Tests

```bash
# Unit tests
npm run test:unit

# Integration tests
npm run test:int-fast

# All tests
npm run test
```

### 4. Code Quality Checks

```bash
# Type checking
mypy ztb/

# Linting
flake8 ztb/

# Formatting
black ztb/
isort ztb/
```

## Testing Data Setup

### Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

### Download Market Data

```bash
python -c "from ztb.data.coin_gecko import fetch_btc_jpy; fetch_btc_jpy(days=365)"
```

## Troubleshooting

### Common Issues

**Import Errors**: Ensure virtual environment is activated
**Memory Issues**: Reduce batch sizes in config
**GPU Issues**: Check CUDA installation with `nvidia-smi`

### Getting Help

- Check existing issues on GitHub
- Review documentation in `docs/`
- Ask in project discussions

## Next Steps

After setup, review:

- [Architecture Overview](architecture.md)
- [Testing Guide](testing.md)
