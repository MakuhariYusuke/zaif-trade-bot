#!/usr/bin/env python3
"""
Advanced neural network architectures for SAC algorithm.

This module provides LSTM and Transformer-based network architectures
for improved temporal pattern recognition in trading environments.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    LSTM-based feature extractor for SAC.

    Processes sequential trading data using LSTM layers to capture
    temporal dependencies and patterns.
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        sequence_length: int = 10,
    ):
        """
        Initialize LSTM feature extractor.

        Args:
            observation_space: Gym observation space
            features_dim: Output feature dimension
            lstm_hidden_size: LSTM hidden state size
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
            sequence_length: Length of input sequences
        """
        super().__init__(observation_space, features_dim)

        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size

        # Input projection to match LSTM input expectations
        input_size = observation_space.shape[0] // sequence_length
        self.input_projection = nn.Linear(input_size, lstm_hidden_size)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(lstm_hidden_size, features_dim)
        self.layer_norm = nn.LayerNorm(features_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM network.

        Args:
            observations: Input tensor of shape (batch_size, sequence_length * features)

        Returns:
            Features tensor of shape (batch_size, features_dim)
        """
        batch_size = observations.shape[0]

        # Reshape to sequence format
        # observations: (batch_size, sequence_length * input_size)
        x = observations.view(batch_size, self.sequence_length, -1)

        # Project input to LSTM dimension
        x = self.input_projection(x)  # (batch_size, sequence_length, lstm_hidden_size)

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        features = h_n[-1]  # (batch_size, lstm_hidden_size)

        # Output projection
        features = self.dropout(features)
        features = self.output_layer(features)
        features = self.layer_norm(features)

        return features


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer architecture.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with multi-head attention.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor
            mask: Attention mask

        Returns:
            Output tensor
        """
        # Multi-head attention with residual connection
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Transformer-based feature extractor for SAC.

    Uses transformer architecture to capture long-range dependencies
    and complex patterns in trading sequences.
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        sequence_length: int = 10,
    ):
        """
        Initialize Transformer feature extractor.

        Args:
            observation_space: Gym observation space
            features_dim: Output feature dimension
            d_model: Model dimension for transformer
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            sequence_length: Length of input sequences
        """
        super().__init__(observation_space, features_dim)

        self.sequence_length = sequence_length
        self.d_model = d_model

        # Input projection
        input_size = observation_space.shape[0] // sequence_length
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, sequence_length)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, features_dim)
        self.layer_norm = nn.LayerNorm(features_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer network.

        Args:
            observations: Input tensor of shape (batch_size, sequence_length * features)

        Returns:
            Features tensor of shape (batch_size, features_dim)
        """
        batch_size = observations.shape[0]

        # Reshape to sequence format
        x = observations.view(batch_size, self.sequence_length, -1)

        # Project input to model dimension
        x = self.input_projection(x)  # (batch_size, sequence_length, d_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Use the output of the last position (or mean pooling)
        # For sequence classification, we typically use the last position
        features = x[:, -1, :]  # (batch_size, d_model)

        # Output projection
        features = self.dropout(features)
        features = self.output_layer(features)
        features = self.layer_norm(features)

        return features


class LSTMPolicy(ActorCriticPolicy):
    """
    SAC policy using LSTM feature extractor.
    """

    def __init__(self, *args, **kwargs):
        # Extract LSTM-specific parameters
        lstm_kwargs = {}
        lstm_params = ['lstm_hidden_size', 'lstm_layers', 'dropout', 'sequence_length']
        for param in lstm_params:
            if param in kwargs:
                lstm_kwargs[param] = kwargs.pop(param)

        # Set default values if not provided
        lstm_kwargs.setdefault('lstm_hidden_size', 128)
        lstm_kwargs.setdefault('lstm_layers', 2)
        lstm_kwargs.setdefault('dropout', 0.1)
        lstm_kwargs.setdefault('sequence_length', 10)

        # Configure features_extractor_kwargs
        if 'features_extractor_kwargs' not in kwargs:
            kwargs['features_extractor_kwargs'] = {}
        kwargs['features_extractor_kwargs'].update(lstm_kwargs)

        # Set features extractor class
        kwargs['features_extractor_class'] = LSTMFeatureExtractor

        super().__init__(*args, **kwargs)


class TransformerPolicy(ActorCriticPolicy):
    """
    SAC policy using Transformer feature extractor.
    """

    def __init__(self, *args, **kwargs):
        # Extract Transformer-specific parameters
        transformer_kwargs = {}
        transformer_params = ['d_model', 'n_heads', 'n_layers', 'd_ff', 'dropout', 'sequence_length']
        for param in transformer_params:
            if param in kwargs:
                transformer_kwargs[param] = kwargs.pop(param)

        # Set default values if not provided
        transformer_kwargs.setdefault('d_model', 128)
        transformer_kwargs.setdefault('n_heads', 8)
        transformer_kwargs.setdefault('n_layers', 4)
        transformer_kwargs.setdefault('d_ff', 512)
        transformer_kwargs.setdefault('dropout', 0.1)
        transformer_kwargs.setdefault('sequence_length', 10)

        # Configure features_extractor_kwargs
        if 'features_extractor_kwargs' not in kwargs:
            kwargs['features_extractor_kwargs'] = {}
        kwargs['features_extractor_kwargs'].update(transformer_kwargs)

        # Set features extractor class
        kwargs['features_extractor_class'] = TransformerFeatureExtractor

        super().__init__(*args, **kwargs)