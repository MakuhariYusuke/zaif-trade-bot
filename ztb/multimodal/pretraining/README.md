# Self-Supervised Pre-training for SAC v421

This module implements comprehensive self-supervised learning techniques specifically adapted for financial time series data in the SAC v421 trading AI system.

## Overview

The pretraining module provides three main self-supervised learning approaches:

1. **Masked Price Modeling (MPM)** - BERT-style masked prediction for financial data
2. **Contrastive Learning** - SimCLR-style representation learning for time series
3. **Anomaly Detection Pre-training** - Hybrid reconstruction and prediction-based anomaly detection

## Features

- **Financial Domain Adaptation**: All techniques are specifically designed for financial time series characteristics
- **Multi-Modal Integration**: Seamlessly integrates with the existing multimodal architecture
- **Scalable Training**: Support for different model sizes from lightweight to production-scale
- **Comprehensive Evaluation**: Built-in anomaly detection and representation quality metrics

## Installation

The module is part of the ZTB (Zaif Trade Bot) multimodal package:

```python
from ztb.multimodal.pretraining import SelfSupervisedTrainer
```

## Quick Start

### Basic Usage

```python
import torch
from ztb.multimodal.pretraining import SelfSupervisedTrainer
from ztb.multimodal.pretraining.config import get_config

# Initialize trainer
trainer = SelfSupervisedTrainer(input_dim=156, device='cuda')

# Generate sample financial data (batch_size, seq_len, features)
train_data = torch.randn(100, 100, 156)
val_data = torch.randn(20, 100, 156)

# Load configuration
config = get_config('lightweight')  # or 'default', 'production'

# Train all stages
trainer.train_all_stages(train_data, val_data, config)

# Get pretrained encoders for downstream tasks
encoders = trainer.get_pretrained_encoders()

# Extract embeddings
embeddings = trainer.get_embeddings(test_data, method='contrastive')

# Compute anomaly scores
anomaly_scores = trainer.compute_anomaly_scores(test_data)
```

### Individual Component Training

#### Masked Price Modeling

```python
# Initialize MPM
trainer.initialize_masked_price_model(
    hidden_dim=512,
    num_layers=6,
    num_heads=8,
    max_seq_len=100,
    mask_prob=0.15
)

# Train MPM
trainer.train_masked_price_modeling(
    train_data, val_data,
    epochs=100,
    batch_size=32
)
```

#### Contrastive Learning

```python
# Initialize Contrastive Learning
trainer.initialize_contrastive_model(
    hidden_dim=512,
    projection_dim=128,
    temperature=0.5
)

# Train Contrastive Learning
trainer.train_contrastive_learning(
    train_data, val_data,
    epochs=100,
    batch_size=32
)
```

#### Anomaly Detection

```python
# Initialize Anomaly Detection
trainer.initialize_anomaly_model(
    hidden_dims=[256, 128, 64],
    latent_dim=32,
    alpha=0.5  # Balance between reconstruction and prediction
)

# Train Anomaly Detection
trainer.train_anomaly_detection(
    train_data, val_data,
    epochs=100,
    batch_size=32
)
```

## Configuration

The module provides three predefined configurations:

- **lightweight**: For quick testing and development
- **default**: Balanced configuration for standard use
- **production**: High-performance configuration for production deployment

```python
from ztb.multimodal.pretraining.config import (
    get_config,
    LIGHTWEIGHT_CONFIG,
    SELF_SUPERVISED_CONFIG,
    PRODUCTION_CONFIG
)

# Get predefined config
config = get_config('production')

# Customize configuration
custom_config = {
    'mpm': {
        'hidden_dim': 768,
        'learning_rate': 1e-4
    },
    'training': {
        'epochs': 150,
        'batch_size': 64
    }
}
```

## API Reference

### SelfSupervisedTrainer

Main trainer class that integrates all pretraining techniques.

#### Methods

- `initialize_masked_price_model(**kwargs)`: Initialize MPM model
- `initialize_contrastive_model(**kwargs)`: Initialize contrastive learning model
- `initialize_anomaly_model(**kwargs)`: Initialize anomaly detection model
- `train_all_stages(train_data, val_data, config)`: Train all stages sequentially
- `get_pretrained_encoders()`: Get trained encoders for downstream tasks
- `get_embeddings(data, method)`: Extract embeddings using specified method
- `compute_anomaly_scores(data)`: Compute anomaly scores for data

### Individual Models

#### MaskedPriceModel
BERT-style masked prediction for financial time series.

#### ContrastiveLearningModel
SimCLR-style contrastive learning with time series augmentations.

#### HybridAnomalyDetector
Combines reconstruction-based and prediction-based anomaly detection.

## Data Format

All models expect financial time series data in the format:
```
(batch_size, sequence_length, feature_dim)
```

Where `feature_dim` is typically 156 for the multimodal financial features.

## Training Tips

1. **Data Quality**: Ensure training data represents normal market conditions
2. **Sequence Length**: Longer sequences (100+) provide better context
3. **Batch Size**: Larger batches improve contrastive learning performance
4. **Early Stopping**: Use patience-based early stopping to prevent overfitting
5. **Multi-GPU**: For production training, consider using multiple GPUs

## Integration with Unified Trainer

The self-supervised pre-training can be seamlessly integrated with the Unified Trainer system:

```python
from ztb.training.unified_trainer import UnifiedTrainer
from ztb.training.unified_trainer.config import UnifiedAlgorithm

# Configuration for self-supervised pre-training
config = {
    'algorithm': 'self_supervised',
    'input_dim': 156,  # Financial feature dimension
    'device': 'cuda',
    'config_type': 'lightweight',  # or 'default', 'production'
    'checkpoint_dir': 'checkpoints/pretraining',
    'synthetic_batch_size': 100,  # For synthetic data generation
    'seq_len': 100
}

# Optional: specify data paths
config.update({
    'train_data_path': 'data/train_timeseries.csv',
    'val_data_path': 'data/val_timeseries.csv'
})

# Optional: custom configuration override
config['custom_config'] = {
    'mpm': {'hidden_dim': 512, 'learning_rate': 1e-4},
    'contrastive': {'temperature': 0.5},
    'anomaly': {'alpha': 0.3}
}

# Initialize and run training
trainer = UnifiedTrainer(config)
success = trainer.train()

if success:
    print("Self-supervised pre-training completed successfully!")
    stats = trainer.get_training_stats()
    print(f"Available encoders: {stats['encoders_available']}")
```

## Evaluation

Monitor these metrics during training:

- **MPM**: Masked prediction loss, validation loss
- **Contrastive**: Contrastive loss, embedding quality
- **Anomaly**: Reconstruction loss, prediction loss, anomaly score distribution

## Checkpointing

Models are automatically saved during training:

```python
# Save checkpoint
trainer.save_checkpoint('pretraining_checkpoint')

# Load checkpoint
trainer.load_checkpoint('pretraining_checkpoint')

# Save training history
trainer.save_training_history('training_history.json')
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or model dimensions
2. **Poor Convergence**: Check data quality and normalization
3. **High Anomaly Scores**: Verify training data represents normal conditions

### Performance Optimization

- Use mixed precision training for faster convergence
- Implement gradient accumulation for larger effective batch sizes
- Use data parallelism for multi-GPU training

## Contributing

When adding new self-supervised techniques:

1. Follow the existing model-trainer pattern
2. Include comprehensive tests
3. Update configuration files
4. Document integration points with SAC v421

## License

Part of the Zaif Trade Bot (ZTB) system.
