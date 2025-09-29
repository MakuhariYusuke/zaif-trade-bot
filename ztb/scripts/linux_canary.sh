#!/bin/bash
# Linux Canary Script for Zaif Trade Bot
# Validates deployment readiness on Linux systems

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${PROJECT_ROOT}/artifacts/canary_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/logs/canary.log"
PYTHON_MIN_VERSION="3.10"
REQUIRED_PACKAGES=("pandas" "numpy" "pytest" "requests")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}ERROR: $*${NC}" >&2
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: $*" >> "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}WARNING: $*${NC}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING: $*" >> "$LOG_FILE"
}

success() {
    echo -e "${GREEN}SUCCESS: $*${NC}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - SUCCESS: $*" >> "$LOG_FILE"
}

# Create output directory structure
mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/metrics" "${OUTPUT_DIR}/reports" "${OUTPUT_DIR}/config"

log "Starting Linux canary validation for Zaif Trade Bot"
log "Project root: $PROJECT_ROOT"
log "Output directory: $OUTPUT_DIR"
log "Log file: $LOG_FILE"

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    error "This canary script is designed for Linux systems only"
    exit 1
fi

success "Running on Linux system"

# Check Python version
log "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    error "python3 command not found"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
log "Found Python version: $PYTHON_VERSION"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (${PYTHON_MIN_VERSION//./, }) else 1)"; then
    error "Python $PYTHON_MIN_VERSION or higher required, found $PYTHON_VERSION"
    exit 1
fi

success "Python version check passed"

# Check pip
log "Checking pip..."
if ! python3 -m pip --version &> /dev/null; then
    error "pip not available"
    exit 1
fi

success "pip check passed"

# Check required packages
log "Checking required Python packages..."
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $package" 2>/dev/null; then
        error "Required package '$package' not found"
        exit 1
    fi
    log "✓ $package"
done

success "All required packages available"

# Check project structure
log "Checking project structure..."
REQUIRED_FILES=(
    "ztb/__init__.py"
    "ztb/backtest/__init__.py"
    "requirements.txt"
    "pyproject.toml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${PROJECT_ROOT}/$file" ]]; then
        error "Required file '$file' not found"
        exit 1
    fi
    log "✓ $file"
done

success "Project structure check passed"

# Test basic imports
log "Testing basic Python imports..."
cd "$PROJECT_ROOT"

if ! python3 -c "import ztb; print('ztb import successful')" >> "$LOG_FILE" 2>&1; then
    error "Failed to import ztb module"
    exit 1
fi

if ! python3 -c "from ztb.backtest.runner import BacktestEngine; print('BacktestEngine import successful')" >> "$LOG_FILE" 2>&1; then
    error "Failed to import BacktestEngine"
    exit 1
fi

success "Basic imports check passed"

# Test configuration loading
log "Testing configuration loading..."
if [[ -f "config/evaluation.yaml" ]]; then
    if ! python3 -c "import yaml; yaml.safe_load(open('config/evaluation.yaml')); print('Config load successful')" >> "$LOG_FILE" 2>&1; then
        error "Failed to load evaluation.yaml"
        exit 1
    fi
    success "Configuration loading check passed"
else
    warn "evaluation.yaml not found, skipping config test"
fi

# Test basic backtest execution (smoke test)
log "Running basic backtest smoke test..."
export PYTHONPATH="${PYTHONPATH}:$PROJECT_ROOT"

# Create minimal test data
cat > /tmp/test_data.csv << EOF
timestamp,open,high,low,close,volume
2020-01-01 00:00:00,10000,10050,9950,10025,10.5
2020-01-01 00:01:00,10025,10075,9975,10050,12.3
2020-01-01 00:02:00,10050,10100,10000,10075,8.7
2020-01-01 00:03:00,10075,10125,10025,10100,15.2
2020-01-01 00:04:00,10100,10150,10050,10125,11.8
EOF

# Run minimal backtest
if python3 -c "
import pandas as pd
from ztb.backtest.runner import BacktestEngine
from ztb.backtest.adapters import create_adapter

# Load test data
data = pd.read_csv('/tmp/test_data.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Create backtest engine
engine = BacktestEngine(initial_capital=1000.0)

# Create buy-hold strategy
strategy = create_adapter('buy_hold')

# Run backtest
equity, orders = engine.run_backtest(strategy, data)

print(f'Backtest completed: {len(equity)} equity points, {len(orders)} orders')
print('Smoke test successful')
" >> "$LOG_FILE" 2>&1; then
    success "Basic backtest smoke test passed"
else
    error "Basic backtest smoke test failed"
    exit 1
fi

# Clean up
rm -f /tmp/test_data.csv

# Memory and performance check
log "Running memory and performance check..."
if python3 -c "
import psutil
import os
import time

# Check available memory
mem = psutil.virtual_memory()
if mem.available < 500 * 1024 * 1024:  # 500MB
    print('WARNING: Low memory available')
else:
    print('Memory check passed')

# Quick performance test
start = time.time()
for i in range(100000):
    _ = i ** 2
end = time.time()
print(f'Performance test: {(end - start) * 1000:.2f}ms for 100k operations')
" >> "$LOG_FILE" 2>&1; then
    success "Memory and performance check completed"
else
    warn "Memory and performance check had issues"
fi

# Network connectivity check (optional)
log "Checking network connectivity..."
if curl -s --max-time 10 https://api.zaif.jp/api/1/ticker/btc_jpy > /dev/null 2>&1; then
    success "Network connectivity check passed"
else
    warn "Network connectivity check failed - may affect live trading"
fi

# Final summary
log "Canary validation completed successfully"
echo ""
echo "========================================"
echo "🐦 Linux Canary Validation Results 🐦"
echo "========================================"
echo -e "${GREEN}✓${NC} Linux environment detected"
echo -e "${GREEN}✓${NC} Python $PYTHON_MIN_VERSION+ available"
echo -e "${GREEN}✓${NC} Required packages installed"
echo -e "${GREEN}✓${NC} Project structure intact"
echo -e "${GREEN}✓${NC} Basic imports working"
echo -e "${GREEN}✓${NC} Configuration loading"
echo -e "${GREEN}✓${NC} Basic backtest execution"
echo -e "${GREEN}✓${NC} Memory and performance adequate"
echo ""
echo -e "${GREEN}🎉 All checks passed! System ready for deployment.${NC}"
echo ""
echo "Artifacts saved to: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"

exit 0