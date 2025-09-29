#!/bin/bash
# Canary test script for Linux
# Runs a short replay simulation to verify system integrity

set -e

DURATION_MINUTES=${1:-3}
POLICY=${2:-sma_fast_slow}
ENABLE_RISK=${3:-false}

echo "Running canary test (Linux)..."

# Set environment variables
export QUIET=1
export PYTHONPATH="$(dirname "$0")/.."

# Create temp directory
TEMP_DIR=$(mktemp -d -t ztb_canary_XXXXXX)

cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

cd "$TEMP_DIR"

# Run paper trader replay
ARGS=(
    -m ztb.live.paper_trader
    --mode replay
    --policy "$POLICY"
    --duration-minutes "$DURATION_MINUTES"
)

if [ "$ENABLE_RISK" = "true" ]; then
    ARGS+=(--enable-risk)
fi

if ! python "${ARGS[@]}"; then
    echo "Canary test failed with exit code $?"
    exit 1
fi

# Verify artifacts
EXPECTED_FILES=(
    "run_metadata.json"
    "orders.csv"
    "stats.json"
)

for file in "${EXPECTED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Missing expected artifact: $file"
        exit 1
    fi
done

echo "Canary test passed!"
exit 0