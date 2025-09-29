"""
Feature registry for trading features.
特徴量レジストリ
"""

import os
import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


class FeatureRegistry:
    """全特徴量関数を一元管理するレジストリ"""

    _registry: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}
    _cache_enabled: bool = True
    _parallel_enabled: bool = True
    _initialized: bool = False
    _config: Dict[str, Any] = {}
    _base_seed: int = 42

    @classmethod
    def initialize(
        cls,
        seed: Optional[int] = None,
        cache_enabled: Optional[bool] = None,
        parallel_enabled: Optional[bool] = None,
    ) -> None:
        """Initialize the registry with seed, cache and parallel settings"""
        if cls._initialized:
            return

        # Set seed from parameter or default
        final_seed = seed if seed is not None else 42

        # Fix seeds for reproducibility
        np.random.seed(final_seed)
        random.seed(final_seed)
        os.environ["PYTHONHASHSEED"] = str(final_seed)

        # Set PyTorch seed if available
        try:
            import torch

            torch.manual_seed(final_seed)
            torch.cuda.manual_seed_all(final_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass  # PyTorch not available

        # Set BLAS thread limits for optimal parallel performance
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"

        # Store base seed for parallel processing
        cls._base_seed = final_seed

        # Set cache enabled from parameter or default
        if cache_enabled is not None:
            cls._cache_enabled = cache_enabled
        else:
            cls._cache_enabled = True

        # Set parallel enabled from parameter or default
        if parallel_enabled is not None:
            cls._parallel_enabled = parallel_enabled
        else:
            cls._parallel_enabled = True

        cls._initialized = True

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reset registry state for testing purposes"""
        cls._initialized = False
        cls._registry.clear()
        cls._config.clear()

    @classmethod
    def set_worker_seed(cls, worker_id: int) -> None:
        """Set deterministic seed for parallel worker processes"""
        if not cls._initialized:
            raise RuntimeError(
                "FeatureRegistry must be initialized before setting worker seed"
            )

        worker_seed = cls._base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

        # Set PyTorch seed if available
        try:
            import torch

            torch.manual_seed(worker_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(worker_seed)
        except ImportError:
            pass  # PyTorch not available

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get current configuration"""
        from copy import deepcopy

        return deepcopy(cls._config)

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[
        [Callable[[pd.DataFrame], pd.Series]], Callable[[pd.DataFrame], pd.Series]
    ]:
        def decorator(
            func: Callable[[pd.DataFrame], pd.Series],
        ) -> Callable[[pd.DataFrame], pd.Series]:
            cls._registry[name] = func
            return func

        return decorator

    @classmethod
    def get(cls, name: str) -> Callable[[pd.DataFrame], pd.Series]:
        if name not in cls._registry:
            raise KeyError(
                f"Feature '{name}' is not registered in the FeatureRegistry."
            )
        return cls._registry[name]

    @classmethod
    def list(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def is_cache_enabled(cls) -> bool:
        """Check if caching is enabled"""
        return cls._cache_enabled

    @classmethod
    def is_parallel_enabled(cls) -> bool:
        """Check if parallel processing is enabled"""
        return cls._parallel_enabled

    def get_enabled_features(self, wave: Optional[int] = None) -> List[str]:
        """Get enabled features for the given wave"""
        return type(self).list()

    def compute_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute features for the given dataframe"""
        features = self.get_enabled_features()
        for feature in features:
            func = type(self).get(feature)
            df[feature] = func(df)
        return df