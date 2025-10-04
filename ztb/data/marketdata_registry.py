"""
Market data source registry for unified data source management.

Provides a centralized registry for different market data sources (cached, streaming, replay)
with factory pattern for instantiation and configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol

from ztb.data.coin_gecko_stream import CoinGeckoStream
from ztb.data.streaming_pipeline import StreamingPipeline
from ztb.trading.live.replay_market import ReplayMarket

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MarketDataSource(Protocol):
    """Protocol for market data sources."""

    def get_data(self, **kwargs: Any) -> Any:
        """Get market data from the source."""
        ...


class MarketDataSourceFactory(ABC):
    """Abstract factory for creating market data sources."""

    @abstractmethod
    def create(self, *args: Any, **kwargs: Any) -> MarketDataSource:
        """Create a market data source instance."""
        ...


class CachedMarketDataFactory(MarketDataSourceFactory):
    """Factory for cached market data sources."""

    def create(self, *args: Any, **kwargs: Any) -> MarketDataSource:
        """Create a cached market data source."""
        cache_path = args[0] if args else None
        if not cache_path:
            raise ValueError("cache_path is required")
        # Import here to avoid circular imports
        from ztb.cache.price_cache import PriceCache  # type: ignore[import]

        cache = PriceCache(cache_path)
        return cache


class StreamingMarketDataFactory(MarketDataSourceFactory):
    """Factory for streaming market data sources."""

    def create(self, *args: Any, **kwargs: Any) -> MarketDataSource:
        """Create a streaming market data pipeline."""
        # Extract streaming-specific parameters
        buffer_capacity = kwargs.get("buffer_capacity", 1_000_000)
        feature_names = kwargs.get("feature_names")
        lookback_rows = kwargs.get("lookback_rows", 512)

        # Create CoinGecko stream client
        stream_client = CoinGeckoStream()

        # Create streaming pipeline
        pipeline = StreamingPipeline(
            stream_client,
            buffer_capacity=buffer_capacity,
            feature_names=feature_names,
            lookback_rows=lookback_rows,
        )

        return pipeline


class ReplayMarketDataFactory(MarketDataSourceFactory):
    """Factory for replay market data sources."""

    def create(self, *args: Any, **kwargs: Any) -> MarketDataSource:
        """Create a replay market data source."""
        data_path = args[0] if args else None
        if not data_path:
            raise ValueError("data_path is required")
        # For now, return a simple replay source
        # This would be expanded to support actual replay functionality

        replay = ReplayMarket(str(data_path), **kwargs)
        return replay


class MarketDataSourceRegistry:
    """Registry for market data source factories."""

    def __init__(self) -> None:
        super().__init__()
        self._factories: Dict[str, MarketDataSourceFactory] = {}
        self._register_default_factories()

    def _register_default_factories(self) -> None:
        """Register default factory implementations."""
        self.register_factory("cached", CachedMarketDataFactory())
        self.register_factory("streaming", StreamingMarketDataFactory())
        self.register_factory("replay", ReplayMarketDataFactory())

    def register_factory(
        self, source_type: str, factory: MarketDataSourceFactory
    ) -> None:
        """Register a factory for a source type."""
        self._factories[source_type] = factory
        logger.info(f"Registered market data source factory: {source_type}")

    def get_factory(self, source_type: str) -> MarketDataSourceFactory:
        """Get factory for a source type."""
        if source_type not in self._factories:
            available = list(self._factories.keys())
            raise ValueError(
                f"Unknown market data source type: {source_type}. Available: {available}"
            )

        return self._factories[source_type]

    def create_source(self, source_type: str, **kwargs: Any) -> MarketDataSource:
        """Create a market data source instance."""
        factory = self.get_factory(source_type)
        return factory.create(**kwargs)

    def list_available_sources(self) -> list[str]:
        """List all available source types."""
        return list(self._factories.keys())


# Global registry instance
_registry = None


def get_market_data_registry() -> MarketDataSourceRegistry:
    """Get the global market data source registry."""
    global _registry
    if _registry is None:
        _registry = MarketDataSourceRegistry()
    return _registry


def create_market_data_source(source_type: str, **kwargs: Any) -> MarketDataSource:
    """Convenience function to create a market data source."""
    registry = get_market_data_registry()
    return registry.create_source(source_type, **kwargs)
