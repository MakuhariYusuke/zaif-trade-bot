#!/bin/bash
# Canary test script for Linux
# Runsif ! python "${ARGS[@]}"; then
    echo "Canary test failed with exit code $?"
    exit 1
fi

# Find artifacts directory
ARTIFACTS_DIR=$(find "$TEMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -1)
if [ -z "$ARTIFACTS_DIR" ]; then
    echo "No artifacts directory found"
    exit 1
fi

# Verify artifacts
EXPECTED_FILES=(
    "run_metadata.json"
    "orders.csv"
    "stats.json"
)

for file in "${EXPECTED_FILES[@]}"; do
    if [ ! -f "$ARTIFACTS_DIR/$file" ]; then
        echo "Missing expected artifact: $file"
        exit 1
    fi
donemulation to verify system integrity

set -e

DURATION_MINUTES=${1:-3}
POLICY=${2:-sma_fast_slow}
ENABLE_RISK=${3:-false}
OUTPUT_DIR=${4}

echo "Running canary test (Linux)..."

# Set environment variables
export QUIET=1
export PYTHONPATH="$(dirname "$0")"

# Create temp directory
if [ -n "$OUTPUT_DIR" ]; then
    TEMP_DIR="$OUTPUT_DIR"
else
    TEMP_DIR=$(mktemp -d -t ztb_canary_XXXXXX)
fi

# Copy venues directory to temp dir
cp -r "$(dirname "$0")/venues" "$TEMP_DIR/"

cleanup() {
    if [ -z "$OUTPUT_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
}
trap cleanup EXIT

cd "$TEMP_DIR"

# Run paper trader replay
ARGS=(
    -m ztb.live.paper_trader
    --mode replay
    --policy "$POLICY"
    --duration-minutes "$DURATION_MINUTES"
    --output-dir "$TEMP_DIR"
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