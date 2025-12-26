import numpy as np
from typing import Optional, Union, Tuple

class OnlineScaler:
    """
    Welford's algorithm implementation for online standardization of vector data.
    Computes running mean and variance without storing the full history.
    
    Suitable for preventing data leakage in reinforcement learning environments
    by scaling observations based only on past data.
    """
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-5, clip: float = 10.0):
        """
        Initialize the OnlineScaler.

        Args:
            shape: Shape of the data vector (e.g., (n_features,)).
            epsilon: Small constant to prevent division by zero.
            clip: Maximum absolute value for scaled output (to prevent outliers).
        """
        self.shape = shape
        self.epsilon = epsilon
        self.clip = clip
        
        # Statistics
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float32)
        self.M2 = np.zeros(shape, dtype=np.float32)  # Sum of squares of differences from the current mean
        self.var = np.ones(shape, dtype=np.float32)   # Running variance

    def update(self, x: np.ndarray) -> None:
        """
        Update statistics with a new sample x.
        
        Args:
            x: Input vector matching self.shape.
        """
        if x.shape != self.shape:
            # Try to reshape if it's a flat array matching the size
            if x.size == np.prod(self.shape):
                x = x.reshape(self.shape)
            else:
                raise ValueError(f"Shape mismatch: expected {self.shape}, got {x.shape}")
            
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        
        if self.n >= 2:
            self.var = self.M2 / (self.n - 1)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Standardize x using current statistics.
        
        Args:
            x: Input vector or batch of vectors.
            
        Returns:
            Scaled vector (Z-score).
        """
        # Handle batch/window input (N, *shape)
        is_batch = False
        if x.shape != self.shape:
            if x.ndim == len(self.shape) + 1 and x.shape[1:] == self.shape:
                is_batch = True
            elif x.size == np.prod(self.shape):
                x = x.reshape(self.shape)
            # If neither, let numpy broadcasting try or fail, or we could raise here
        
        if self.n < 2:
            return np.zeros_like(x, dtype=np.float32)  # Not enough data to scale
            
        std = np.sqrt(self.var)
        
        # Z-score normalization
        # Broadcasting handles both single vector (features,) and batch (N, features)
        # provided self.mean/std are (features,)
        scaled = (x - self.mean) / (std + self.epsilon)
        
        # Clip to prevent extreme outliers from destabilizing the model
        if self.clip > 0:
            scaled = np.clip(scaled, -self.clip, self.clip)
            
        return scaled.astype(np.float32)

    def partial_fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Update stats and then transform (convenience method)."""
        self.update(x)
        return self.transform(x)

    def get_params(self) -> dict:
        """Get current statistics for saving."""
        return {
            "n": self.n,
            "mean": self.mean.tolist(),
            "M2": self.M2.tolist(),
            "var": self.var.tolist()
        }

    def load_params(self, params: dict) -> None:
        """Load statistics from saved dictionary."""
        self.n = params["n"]
        self.mean = np.array(params["mean"], dtype=np.float32)
        self.M2 = np.array(params["M2"], dtype=np.float32)
        self.var = np.array(params["var"], dtype=np.float32)
