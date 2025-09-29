"""
Unit tests for market data source registry.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

"""
Unit tests for market data source registry.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock all problematic imports at the module level
sys.modules.update(
    {
        "ztb.cache": MagicMock(),
        "ztb.cache.price_cache": MagicMock(),
        "ztb.data.coin_gecko_stream": MagicMock(),
        "ztb.data.streaming_pipeline": MagicMock(),
        "ztb.live.replay_market": MagicMock(),
        "ztb.features": MagicMock(),
        "ztb.features.feature_engine": MagicMock(),
        "ztb.utils.observability": MagicMock(),
    }
)

from ztb.data.marketdata_registry import (
    CachedMarketDataFactory,
    MarketDataSourceRegistry,
    ReplayMarketDataFactory,
    StreamingMarketDataFactory,
    create_market_data_source,
    get_market_data_registry,
)


class TestMarketDataSourceRegistry:
    """Test cases for MarketDataSourceRegistry class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = MarketDataSourceRegistry()

    def test_init_registers_default_factories(self):
        """Test that default factories are registered on initialization."""
        assert "cached" in self.registry._factories
        assert "streaming" in self.registry._factories
        assert "replay" in self.registry._factories

    def test_register_factory(self):
        """Test registering a custom factory."""
        mock_factory = Mock()
        self.registry.register_factory("custom", mock_factory)

        assert "custom" in self.registry._factories
        assert self.registry._factories["custom"] is mock_factory

    def test_get_factory_known_type(self):
        """Test getting a factory for known type."""
        factory = self.registry.get_factory("cached")
        assert isinstance(factory, CachedMarketDataFactory)

    def test_get_factory_unknown_type(self):
        """Test getting a factory for unknown type raises error."""
        with pytest.raises(ValueError, match="Unknown market data source type"):
            self.registry.get_factory("unknown")

    def test_create_source_cached(self):
        """Test creating a cached data source."""
        with patch("ztb.cache.price_cache.PriceCache") as mock_cache:
            mock_instance = Mock()
            mock_cache.return_value = mock_instance

            source = self.registry.create_source("cached", cache_path="/tmp/cache")

            mock_cache.assert_called_once_with("/tmp/cache")
            assert source is mock_instance

    def test_create_source_streaming(self):
        """Test creating a streaming data source."""
        with (
            patch("ztb.data.marketdata_registry.CoinGeckoStream") as mock_stream,
            patch("ztb.data.marketdata_registry.StreamingPipeline") as mock_pipeline,
        ):
            mock_stream_instance = Mock()
            mock_stream.return_value = mock_stream_instance
            mock_pipeline_instance = Mock()
            mock_pipeline.return_value = mock_pipeline_instance

            source = self.registry.create_source("streaming", buffer_capacity=50000)

            mock_stream.assert_called_once()
            mock_pipeline.assert_called_once()
            assert source is mock_pipeline_instance

    def test_create_source_replay(self):
        """Test creating a replay data source."""
        with patch("ztb.data.marketdata_registry.ReplayMarket") as mock_replay:
            mock_instance = Mock()
            mock_replay.return_value = mock_instance

            source = self.registry.create_source("replay", data_path="/tmp/data.csv")

            mock_replay.assert_called_once_with(Path("/tmp/data.csv"))
            assert source is mock_instance

    def test_list_available_sources(self):
        """Test listing available source types."""
        sources = self.registry.list_available_sources()
        assert "cached" in sources
        assert "streaming" in sources
        assert "replay" in sources


class TestGlobalRegistry:
    """Test cases for global registry functions."""

    def test_get_market_data_registry(self):
        """Test getting the global registry."""
        registry = get_market_data_registry()
        assert isinstance(registry, MarketDataSourceRegistry)

    def test_get_market_data_registry_singleton(self):
        """Test that global registry is a singleton."""
        registry1 = get_market_data_registry()
        registry2 = get_market_data_registry()
        assert registry1 is registry2

    def test_create_market_data_source(self):
        """Test convenience function for creating sources."""
        with patch("ztb.cache.price_cache.PriceCache") as mock_cache:
            mock_instance = Mock()
            mock_cache.return_value = mock_instance

            source = create_market_data_source("cached", cache_path="/tmp/cache")

            assert source is mock_instance


class TestCachedMarketDataFactory:
    """Test cases for CachedMarketDataFactory."""

    def test_create(self):
        """Test creating cached data source."""
        factory = CachedMarketDataFactory()

        with patch("ztb.cache.price_cache.PriceCache") as mock_cache:
            mock_instance = Mock()
            mock_cache.return_value = mock_instance

            source = factory.create(cache_path="/tmp/cache")

            mock_cache.assert_called_once_with("/tmp/cache")
            assert source is mock_instance


class TestStreamingMarketDataFactory:
    """Test cases for StreamingMarketDataFactory."""

    def test_create(self):
        """Test creating streaming data source."""
        factory = StreamingMarketDataFactory()

        with (
            patch("ztb.data.marketdata_registry.CoinGeckoStream") as mock_stream,
            patch("ztb.data.marketdata_registry.StreamingPipeline") as mock_pipeline,
        ):
            mock_stream_instance = Mock()
            mock_stream.return_value = mock_stream_instance
            mock_pipeline_instance = Mock()
            mock_pipeline.return_value = mock_pipeline_instance

            source = factory.create(buffer_capacity=50000, symbols=["btc"])

            mock_stream.assert_called_once()
            mock_pipeline.assert_called_once()
            assert source is mock_pipeline_instance


class TestReplayMarketDataFactory:
    """Test cases for ReplayMarketDataFactory."""

    def test_create(self):
        """Test creating replay data source."""
        factory = ReplayMarketDataFactory()

        with patch("ztb.data.marketdata_registry.ReplayMarket") as mock_replay:
            mock_instance = Mock()
            mock_replay.return_value = mock_instance

            source = factory.create(data_path="/tmp/data.csv", speed_multiplier=2.0)

            mock_replay.assert_called_once_with(
                Path("/tmp/data.csv"), speed_multiplier=2.0
            )
            assert source is mock_instance
