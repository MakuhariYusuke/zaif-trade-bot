#!/usr/bin/env python3
"""
Advanced neural network architectures for SAC algorithm.

This module provides LSTM and Transformer-based network architectures
for improved temporal pattern recognition in trading environments.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
try:
    import torch.nn.functional as F
except Exception:

    F = _F
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
            bidirectional=False,
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
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
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
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with multi-head attention.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

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
        lstm_params = ["lstm_hidden_size", "lstm_layers", "dropout", "sequence_length"]
        for param in lstm_params:
            if param in kwargs:
                lstm_kwargs[param] = kwargs.pop(param)

        # Set default values if not provided
        lstm_kwargs.setdefault("lstm_hidden_size", 128)
        lstm_kwargs.setdefault("lstm_layers", 2)
        lstm_kwargs.setdefault("dropout", 0.1)
        lstm_kwargs.setdefault("sequence_length", 10)

        # Configure features_extractor_kwargs
        if "features_extractor_kwargs" not in kwargs:
            kwargs["features_extractor_kwargs"] = {}
        kwargs["features_extractor_kwargs"].update(lstm_kwargs)

        # Set features extractor class
        kwargs["features_extractor_class"] = LSTMFeatureExtractor

        super().__init__(*args, **kwargs)


class TransformerPolicy(ActorCriticPolicy):
    """
    SAC policy using Transformer feature extractor.
    """

    def __init__(self, *args, **kwargs):
        # Extract Transformer-specific parameters
        transformer_kwargs = {}
        transformer_params = [
            "d_model",
            "n_heads",
            "n_layers",
            "d_ff",
            "dropout",
            "sequence_length",
        ]
        for param in transformer_params:
            if param in kwargs:
                transformer_kwargs[param] = kwargs.pop(param)

        # Set default values if not provided
        transformer_kwargs.setdefault("d_model", 128)
        transformer_kwargs.setdefault("n_heads", 8)
        transformer_kwargs.setdefault("n_layers", 4)
        transformer_kwargs.setdefault("d_ff", 512)
        transformer_kwargs.setdefault("dropout", 0.1)
        transformer_kwargs.setdefault("sequence_length", 10)

        # Configure features_extractor_kwargs
        if "features_extractor_kwargs" not in kwargs:
            kwargs["features_extractor_kwargs"] = {}
        kwargs["features_extractor_kwargs"].update(transformer_kwargs)

        # Set features extractor class
        kwargs["features_extractor_class"] = TransformerFeatureExtractor

        super().__init__(*args, **kwargs)


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Separable Convolution for efficient 1D convolution.
    Reduces parameters while maintaining representational power.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EfficientAttention(nn.Module):
    """
    Efficient attention mechanism using Linformer/Performer approach.
    Reduces complexity from O(n^2) to O(n) or O(n log n).
    """

    def __init__(
        self, embed_dim: int, num_heads: int, seq_len: int, method: str = "linformer"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.method = method

        if method == "linformer":
            # Linformer: project to lower dimension
            self.proj_k = nn.Linear(seq_len, seq_len // 4)
            self.proj_v = nn.Linear(seq_len, seq_len // 4)
        elif method == "performer":
            # Performer: use random features
            self.random_features = nn.Linear(seq_len, seq_len // 2)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape

        # Linear projections
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        if self.method == "linformer":
            # Apply dimension reduction
            k = self.proj_k(k.transpose(-2, -1)).transpose(-2, -1)
            v = self.proj_v(v.transpose(-2, -1)).transpose(-2, -1)

        # Efficient attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if self.method == "performer":
            # Apply random features for O(n) complexity
            attn_weights = self.random_features(
                attn_weights.transpose(-2, -1)
            ).transpose(-2, -1)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, embed_dim)
        )
        return self.out_proj(attn_output)


class DynamicNetwork(nn.Module):
    """
    Dynamic network that adjusts computation based on input complexity.
    Uses conditional computation to skip unnecessary operations.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

        # Main processing layers
        self.main_layers = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            ]
        )

        # Lightweight layers for simple inputs
        self.lightweight_layers = nn.ModuleList([nn.Linear(input_dim, output_dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input for complexity estimation and linear layers
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        # Estimate input complexity
        complexity = self.complexity_estimator(x_flat)  # Use flattened input

        # Conditional computation
        if complexity.mean() > 0.5:  # Complex input
            x_out = x_flat
            for layer in self.main_layers:
                x_out = layer(x_out)
            x = x_out
        else:  # Simple input
            x_out = x_flat
            for layer in self.lightweight_layers:
                x_out = layer(x_out)
            x = x_out

        return x


class EfficientFeatureExtractor(BaseFeaturesExtractor):
    """
    Efficient feature extractor combining multiple optimization techniques:
    - Depthwise separable convolutions
    - Efficient attention (Linformer/Performer)
    - Dynamic networks with conditional computation
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        use_depthwise_conv: bool = True,
        use_efficient_attention: bool = True,
        use_dynamic_network: bool = True,
        attention_method: str = "linformer",
        sequence_length: int = 10,
    ):
        super().__init__(observation_space, features_dim)

        self.use_depthwise_conv = use_depthwise_conv
        self.use_efficient_attention = use_efficient_attention
        self.use_dynamic_network = use_dynamic_network
        self.sequence_length = sequence_length

        input_dim = observation_space.shape[0]

        # Depthwise separable convolution layers
        if use_depthwise_conv:
            self.conv_layers = nn.Sequential(
                DepthwiseSeparableConv1d(input_dim, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                DepthwiseSeparableConv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(sequence_length),
            )
            conv_output_dim = 128
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(sequence_length),
            )
            conv_output_dim = 128

        # Efficient attention layer
        if use_efficient_attention:
            self.attention = EfficientAttention(
                embed_dim=conv_output_dim,
                num_heads=8,
                seq_len=sequence_length,
                method=attention_method,
            )
            attention_output_dim = conv_output_dim
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=conv_output_dim, num_heads=8, batch_first=True
            )
            attention_output_dim = conv_output_dim

        # Dynamic network
        if use_dynamic_network:
            self.dynamic_net = DynamicNetwork(
                input_dim=attention_output_dim * sequence_length,
                hidden_dim=256,
                output_dim=features_dim,
            )
        else:
            self.dynamic_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(attention_output_dim * sequence_length, 256),
                nn.ReLU(),
                nn.Linear(256, features_dim),
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Reshape for convolution (batch, channels, seq_len)
        x = observations.unsqueeze(1)  # Add channel dimension

        # Apply convolution
        x = self.conv_layers(x)

        # Reshape for attention (batch, seq_len, channels)
        x = x.transpose(1, 2)

        # Apply attention
        if hasattr(self.attention, "forward"):
            if isinstance(self.attention, EfficientAttention):
                x = self.attention(x)
            else:
                # Standard multihead attention
                x, _ = self.attention(x, x, x)

        # Apply dynamic network
        x = self.dynamic_net(x)
        return x


class EfficientSACPolicy(ActorCriticPolicy):
    """
    SAC policy with efficient network architectures.
    """

    def __init__(self, *args, **kwargs):
        # Extract custom kwargs
        efficient_kwargs = {
            "use_depthwise_conv": kwargs.pop("use_depthwise_conv", True),
            "use_efficient_attention": kwargs.pop("use_efficient_attention", True),
            "use_dynamic_network": kwargs.pop("use_dynamic_network", True),
            "attention_method": kwargs.pop("attention_method", "linformer"),
            "sequence_length": kwargs.pop("sequence_length", 10),
        }

        # Set features extractor
        kwargs["features_extractor_class"] = EfficientFeatureExtractor
        kwargs["features_extractor_kwargs"] = efficient_kwargs

        super().__init__(*args, **kwargs)
