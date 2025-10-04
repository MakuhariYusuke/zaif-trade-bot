#!/bin/bash
# Cloud Training Script for Zaif Trade Bot Scalping Mode
# This script is optimized for cloud environments with sufficient memory

echo "Starting Zaif Trade Bot Scalping Training on Cloud Environment"
echo "================================================================"

# Set environment variables for PyTorch optimization
export PYTORCH_DISABLE_TORCH_DYNAMO=1
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Enable CUDA if available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, enabling GPU support"
    export CUDA_VISIBLE_DEVICES=0
else
    echo "CUDA not detected, using CPU mode"
    export CUDA_VISIBLE_DEVICES=""
fi

# Run the training
echo "Starting training with optimized scalping configuration..."
python -m ztb.training.unified_trainer --config unified_training_config.json --force

echo "Training completed!"
echo "Check checkpoints/scalping_training_v2/ for results"