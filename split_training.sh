#!/bin/bash
# Split Training Script for Zaif Trade Bot Scalping Mode
# This script runs training in small chunks to avoid memory issues

echo "Zaif Trade Bot Split Training Script"
echo "===================================="

# Configuration
TOTAL_STEPS=1000000  # 1M total steps
STEPS_PER_CHUNK=10000  # 10k steps per chunk
SESSION_PREFIX="scalping_split"

# Set environment variables for PyTorch optimization
export PYTORCH_DISABLE_TORCH_DYNAMO=1
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Disable CUDA to reduce memory usage
export CUDA_VISIBLE_DEVICES=""

# Calculate number of chunks needed
NUM_CHUNKS=$((TOTAL_STEPS / STEPS_PER_CHUNK))

echo "Total steps: $TOTAL_STEPS"
echo "Steps per chunk: $STEPS_PER_CHUNK"
echo "Number of chunks: $NUM_CHUNKS"
echo ""

# Run training in chunks
for ((i=1; i<=NUM_CHUNKS; i++))
do
    echo "Running chunk $i/$NUM_CHUNKS..."

    # Update session ID for each chunk
    SESSION_ID="${SESSION_PREFIX}_chunk_${i}"

    # Modify config for this chunk
    sed -i "s/\"session_id\": \"[^\"]*\"/\"session_id\": \"$SESSION_ID\"/" unified_training_config.json
    sed -i "s/\"total_timesteps\": [0-9]*/\"total_timesteps\": $STEPS_PER_CHUNK/" unified_training_config.json

    echo "Session ID: $SESSION_ID"
    echo "Steps: $STEPS_PER_CHUNK"

    # Run training
    if python -m ztb.training.unified_trainer --config unified_training_config.json --force; then
        echo "✓ Chunk $i completed successfully"
    else
        echo "✗ Chunk $i failed"
        exit 1
    fi

    echo "Completed chunk $i at $(date)"
    echo "----------------------------------------"
done

echo "All chunks completed!"
echo "Check checkpoints/ for results"