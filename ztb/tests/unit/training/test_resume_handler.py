"""
Unit tests for resume_handler.py module.
"""

from unittest.mock import Mock, patch

from ztb.training.resume_handler import ResumeHandler, ResumeState


class TestResumeHandler:
    """Test cases for ResumeHandler class."""

    def test_init(self):
        """Test ResumeHandler initialization."""
        mock_checkpoint_manager = Mock()
        mock_streaming_pipeline = Mock()

        handler = ResumeHandler(
            checkpoint_manager=mock_checkpoint_manager,
            streaming_pipeline=mock_streaming_pipeline,
        )

        assert handler.checkpoint_manager == mock_checkpoint_manager
        assert handler.streaming_pipeline == mock_streaming_pipeline

    def test_init_without_streaming_pipeline(self):
        """Test ResumeHandler initialization without streaming pipeline."""
        mock_checkpoint_manager = Mock()

        handler = ResumeHandler(checkpoint_manager=mock_checkpoint_manager)

        assert handler.checkpoint_manager == mock_checkpoint_manager
        assert handler.streaming_pipeline is None

    def test_resume_no_checkpoint(self):
        """Test resume when no checkpoint is available."""
        mock_checkpoint_manager = Mock()
        mock_checkpoint_manager.load_latest.return_value = None

        handler = ResumeHandler(checkpoint_manager=mock_checkpoint_manager)
        mock_model = Mock()

        result = handler.resume(mock_model)

        assert result is None
        mock_checkpoint_manager.load_latest.assert_called_once()

    def test_resume_with_checkpoint_and_apply_snapshot(self):
        """Test resume with checkpoint and custom apply_snapshot function."""
        # Mock checkpoint manager and snapshot
        mock_checkpoint_manager = Mock()
        mock_snapshot = Mock()
        mock_snapshot.step = 1000
        mock_snapshot.metrics = {"loss": 0.5}
        mock_snapshot.metadata = {"epoch": 10}
        mock_snapshot.payload = {"stream_state": {"buffer": None}}
        mock_checkpoint_manager.load_latest.return_value = mock_snapshot

        # Mock streaming pipeline
        mock_streaming_pipeline = Mock()

        handler = ResumeHandler(
            checkpoint_manager=mock_checkpoint_manager,
            streaming_pipeline=mock_streaming_pipeline,
        )

        # Mock model and custom apply function
        mock_model = Mock()
        mock_apply_snapshot = Mock()

        result = handler.resume(mock_model, apply_snapshot=mock_apply_snapshot)

        # Verify calls
        mock_checkpoint_manager.load_latest.assert_called_once()
        mock_apply_snapshot.assert_called_once_with(mock_snapshot)

        # Verify result
        assert isinstance(result, ResumeState)
        assert result.step == 1000
        assert result.metrics == {"loss": 0.5}
        assert result.metadata == {"epoch": 10}
        assert result.streaming_state == {"buffer": None}

    def test_resume_with_checkpoint_default_apply(self):
        """Test resume with checkpoint using default apply_snapshot."""
        # Mock checkpoint manager and snapshot
        mock_checkpoint_manager = Mock()
        mock_snapshot = Mock()
        mock_snapshot.step = 500
        mock_snapshot.metrics = {"reward": 1.2}
        mock_snapshot.metadata = {"batch": 5}
        mock_snapshot.payload = {"stream_state": None}
        mock_checkpoint_manager.load_latest.return_value = mock_snapshot

        handler = ResumeHandler(checkpoint_manager=mock_checkpoint_manager)
        mock_model = Mock()

        result = handler.resume(mock_model)

        # Verify calls
        mock_checkpoint_manager.load_latest.assert_called_once()
        mock_checkpoint_manager.apply_snapshot.assert_called_once_with(
            mock_model, mock_snapshot
        )

        # Verify result
        assert isinstance(result, ResumeState)
        assert result.step == 500
        assert result.metrics == {"reward": 1.2}
        assert result.metadata == {"batch": 5}
        assert result.streaming_state is None

    @patch("ztb.training.resume_handler.logger")
    def test_restore_streaming_state_with_data(self, mock_logger):
        """Test _restore_streaming_state with buffer data."""
        import pandas as pd

        # Mock streaming pipeline
        mock_streaming_pipeline = Mock()
        mock_stats = Mock()
        mock_stats.buffer.rows = 100
        mock_stats.buffer.capacity = 1000
        mock_streaming_pipeline.stats.return_value = mock_stats

        # Mock buffer data
        buffer_data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
        stream_state = {"buffer": buffer_data}

        handler = ResumeHandler(
            checkpoint_manager=Mock(), streaming_pipeline=mock_streaming_pipeline
        )

        handler._restore_streaming_state(stream_state)

        # Verify buffer was cleared and extended
        mock_streaming_pipeline.buffer.clear.assert_called_once()
        mock_streaming_pipeline.buffer.extend.assert_called_once()
        extended_df = mock_streaming_pipeline.buffer.extend.call_args[0][0]
        assert isinstance(extended_df, pd.DataFrame)
        assert len(extended_df) == 3

        # Verify logging
        mock_logger.info.assert_called_once_with(
            "Streaming pipeline restored with %s rows (capacity %s)", 100, 1000
        )

    def test_restore_streaming_state_no_pipeline(self):
        """Test _restore_streaming_state with no streaming pipeline."""
        handler = ResumeHandler(checkpoint_manager=Mock())

        # Should not raise exception
        handler._restore_streaming_state({"buffer": "some_data"})

    def test_restore_streaming_state_no_state(self):
        """Test _restore_streaming_state with no stream state."""
        mock_streaming_pipeline = Mock()

        handler = ResumeHandler(
            checkpoint_manager=Mock(), streaming_pipeline=mock_streaming_pipeline
        )

        handler._restore_streaming_state(None)

        # Should not interact with pipeline
        mock_streaming_pipeline.buffer.clear.assert_not_called()

    def test_restore_streaming_state_empty_buffer(self):
        """Test _restore_streaming_state with empty buffer."""
        import pandas as pd

        # Mock streaming pipeline
        mock_streaming_pipeline = Mock()

        # Empty DataFrame
        empty_df = pd.DataFrame()
        stream_state = {"buffer": empty_df}

        handler = ResumeHandler(
            checkpoint_manager=Mock(), streaming_pipeline=mock_streaming_pipeline
        )

        handler._restore_streaming_state(stream_state)

        # Should clear buffer but not extend
        mock_streaming_pipeline.buffer.clear.assert_called_once()
        mock_streaming_pipeline.buffer.extend.assert_not_called()
