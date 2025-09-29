#!/bin/bash
# Stakeholder Evidence Bundle Generator
#
# Generates comprehensive evidence package for real-trading readiness:
# - Backtest results (SMA baseline + RL policy)
# - Short paper trading replay
# - Performance comparison report

set -e

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BUNDLE_DIR="artifacts/stakeholder_bundle_${TIMESTAMP}"
RESULTS_DIR="results"
BACKTEST_DIR="${RESULTS_DIR}/backtest"
PAPER_DIR="${RESULTS_DIR}/paper"
CANARY_DIR="${RESULTS_DIR}/canary"

echo "Generating stakeholder evidence bundle..."
echo "Output directory: ${BUNDLE_DIR}"

# Create bundle directory
mkdir -p "${BUNDLE_DIR}"

# Function to run backtest
run_backtest() {
    local policy=$1
    local output_subdir="${BACKTEST_DIR}/${policy}_${TIMESTAMP}"

    echo "Running backtest for ${policy}..."

    # Run backtest (using python module syntax)
    python -m ztb.backtest.runner \
        --policy "${policy}" \
        --output-dir "${output_subdir}" \
        --slippage-bps 5

    # Generate run metadata
    python -m ztb.utils.run_metadata --output "${output_subdir}/run_metadata.json"

    # Copy results to bundle
    if [ -d "${output_subdir}" ]; then
        cp -r "${output_subdir}" "${BUNDLE_DIR}/backtest_${policy}/"
    fi
}

# Function to run paper trading
run_paper_trading() {
    local mode=$1
    local policy=$2
    local output_subdir="${PAPER_DIR}/${mode}_${TIMESTAMP}"

    echo "Running paper trading ${mode} for ${policy}..."

    # Run short paper trading session
    python -m ztb.live.paper_trader \
        --mode "${mode}" \
        --policy "${policy}" \
        --duration-minutes 30 \
        --output-dir "${output_subdir}"

    # Copy results to bundle
    if [ -d "${output_subdir}" ]; then
        cp -r "${output_subdir}" "${BUNDLE_DIR}/paper_${mode}_${policy}/"
    fi
}

# Function to run canary
run_canary() {
    local output_subdir="${CANARY_DIR}/canary_${TIMESTAMP}"

    echo "Running canary..."

    # Run canary script
    ./scripts/linux_canary.sh "${output_subdir}"

    # Generate run metadata for canary
    python -m ztb.utils.run_metadata --output "${output_subdir}/run_metadata.json"

    # Copy canary artifacts to bundle
    if [ -d "${output_subdir}" ]; then
        cp -r "${output_subdir}" "${BUNDLE_DIR}/canary/"
        # Copy venue.yaml used in canary
        cp "config/venue.yaml" "${BUNDLE_DIR}/canary/venue.yaml"
        # Copy canary log
        cp "${output_subdir}/canary_log.txt" "${BUNDLE_DIR}/canary/canary_log.txt"
    fi
}

# Run evidence generation
echo "1. Running baseline backtest (SMA crossover)..."
run_backtest "sma_fast_slow"

echo "2. Running RL policy backtest..."
run_backtest "rl"

echo "3. Running short paper trading replay..."
run_paper_trading "replay" "sma_fast_slow"

echo "4. Running short live-lite paper trading..."
run_paper_trading "live-lite" "rl"

echo "5. Running canary deployment validation..."
run_canary

# Generate comparison report
echo "6. Generating comparison report..."
cat > "${BUNDLE_DIR}/README.md" << 'EOF'
# Trading Bot Real-Trading Readiness Evidence

This bundle contains evidence demonstrating the bot's practical usability for real trading.

## Contents

### Backtest Results
- `backtest_sma_fast_slow/`: SMA crossover strategy baseline
- `backtest_rl/`: RL policy strategy results

### Paper Trading Results
- `paper_replay_sma_fast_slow/`: 30-minute replay simulation with SMA strategy
- `paper_live-lite_rl/`: Live-lite simulation with RL strategy

### Deployment Validation
- `canary/`: Deployment readiness validation including connectivity tests, configuration validation, and system health checks

## Key Metrics Comparison

| Strategy | Sharpe Ratio | Max Drawdown | Total Return | Win Rate | Total Trades |
|----------|-------------|--------------|--------------|----------|--------------|
| SMA Baseline | [See metrics.json] | [See metrics.json] | [See metrics.json] | [See metrics.json] | [See metrics.json] |
| RL Policy | [See metrics.json] | [See metrics.json] | [See metrics.json] | [See metrics.json] | [See metrics.json] |

## Statistical Significance

| Test | SMA Baseline | RL Policy | Significance |
|------|--------------|-----------|--------------|
| Deflated Sharpe Ratio (DSR) | [See metrics.json] | [See metrics.json] | [See significance.json] |
| Bootstrap p-value | [See metrics.json] | [See metrics.json] | [See significance.json] |
| Out-of-Sample Performance | N/A | [See metrics.json] | [See significance.json] |

## Transition to Real Trading

The system is designed for seamless transition to live trading:

1. **Replace SimBroker with ZaifAdapter**: Same interface, just change broker implementation
2. **Risk Controls**: Already integrated with configurable profiles (conservative/balanced/aggressive)
3. **Monitoring**: Comprehensive logging and metrics collection
4. **Safety**: Pre-trade validation and position limits prevent catastrophic losses

## Risk Management

- **Daily Loss Limits**: Configurable percentage limits on daily losses
- **Position Sizing**: Maximum position and trade size controls
- **Stop Loss/Take Profit**: Automatic exit rules
- **Trade Frequency**: Rate limiting to prevent over-trading
- **Volatility Checks**: Portfolio volatility monitoring

## Next Steps for Production

1. **API Credentials**: Obtain Zaif exchange API keys
2. **ZaifAdapter Implementation**: Complete the broker adapter (currently stub)
3. **Extended Testing**: Run longer paper trading sessions
4. **Performance Monitoring**: Set up alerting and dashboards
5. **Capital Allocation**: Start with small position sizes

## Files to Review

- `backtest_*/metrics.json`: Detailed performance metrics
- `backtest_*/report.md`: Executive summary reports
- `backtest_*/run_metadata.json`: Environment and reproducibility information
- `paper_*/pnl.csv`: P&L time series
- `paper_*/trade_log.json`: Individual trade records
- `paper_*/summary.json`: Session summaries
- `canary/canary_log.txt`: Deployment validation log
- `canary/venue.yaml`: Configuration used for validation
- `canary/run_metadata.json`: Canary execution environment
- `canary/config_validation.json`: Configuration compliance check
- `canary/connectivity_test.json`: Exchange connectivity verification

---
*Generated automatically on $(date)*
EOF

# Create zip archive
echo "7. Creating zip archive..."
cd artifacts
zip -r "stakeholder_bundle_${TIMESTAMP}.zip" "stakeholder_bundle_${TIMESTAMP}"
cd ..

echo ""
echo "✅ Stakeholder bundle generated successfully!"
echo "📁 Bundle location: ${BUNDLE_DIR}"
echo "📦 Archive: artifacts/stakeholder_bundle_${TIMESTAMP}.zip"
echo ""
echo "Contents:"
find "${BUNDLE_DIR}" -type f -name "*.json" -o -name "*.csv" -o -name "*.md" | head -10

echo ""
echo "🚀 Ready for stakeholder review!"