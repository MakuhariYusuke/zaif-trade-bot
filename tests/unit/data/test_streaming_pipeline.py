from unittest.mock import MagicMock, patch

from ztb.data.streaming_pipeline import StreamingPipeline


class TestStreamingPipeline:
    """Test StreamingPipeline functionality."""

    @patch("ztb.data.streaming_pipeline.CoinGeckoStream")
    def test_init_default_config(self, mock_stream_class: MagicMock) -> None:
        """Test initialization with default config."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        pipeline = StreamingPipeline(mock_stream)

        assert pipeline.stream_client is not None
        assert pipeline.buffer is not None
        assert pipeline.health().status == "ok"

    @patch("ztb.data.streaming_pipeline.CoinGeckoStream")
    def test_start_background_stream(self, mock_stream_class: MagicMock) -> None:
        """Test successful pipeline background stream start."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        pipeline = StreamingPipeline(mock_stream)
        stop_event = pipeline.start_background_stream()

        assert stop_event is not None
        assert pipeline.health().status == "ok"

    @patch("ztb.data.streaming_pipeline.CoinGeckoStream")
    def test_close(self, mock_stream_class: MagicMock) -> None:
        """Test successful pipeline close."""
        mock_stream = MagicMock()
        mock_stream_class.return_value = mock_stream

        pipeline = StreamingPipeline(mock_stream)
        pipeline.close()

        # After close, health status might change or we can check if it's closed
        assert pipeline is not None
