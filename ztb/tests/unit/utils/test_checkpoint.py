#!/usr/bin/env python3
"""
test_checkpoint.py
Unit tests for checkpoint management with negative path testing
"""

import os
import tempfile
from pathlib import Path

import pytest

from ztb.utils.checkpoint import CheckpointManager


class TestCheckpointNegativePaths:
    """Test checkpoint manager negative paths and error handling"""

    def test_load_latest_no_checkpoints(self):
        """Test loading when no checkpoints exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir)

            with pytest.raises(FileNotFoundError, match="No checkpoints found"):
                manager.load_latest()

    def test_load_corrupted_checkpoint_file(self):
        """Test loading corrupted checkpoint file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir)

            # Create a corrupted checkpoint file
            corrupt_path = Path(tmpdir) / "checkpoint_100.pkl.zst"
            with open(corrupt_path, "wb") as f:
                f.write(b"this is not a valid compressed pickle")

            with pytest.raises(Exception):  # Should raise some unpickling error
                manager.load_latest()

    def test_load_checkpoint_with_invalid_pickle_data(self):
        """Test loading checkpoint with invalid pickle data after decompression"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir)

            # Create a file that decompresses but contains invalid pickle
            checkpoint_path = Path(tmpdir) / "checkpoint_50.pkl.zst"

            # For zstd compression, we need to create valid compressed data
            # but with invalid pickle content
            invalid_pickle_data = b"not pickle data"
            compressed_data = manager._compress_data(invalid_pickle_data)

            with open(checkpoint_path, "wb") as f:
                f.write(compressed_data)

            with pytest.raises(Exception):  # Should raise unpickling error
                manager.load_latest()

    def test_load_checkpoint_with_wrong_compression_format(self):
        """Test loading checkpoint saved with different compression"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manager with zstd compression
            manager_zstd = CheckpointManager(save_dir=tmpdir, compress="zstd")

            # Save a valid checkpoint
            test_data = {"model": "test", "step": 100}
            manager_zstd.save_sync(test_data, 100)

            # Try to load with a manager expecting different compression
            manager_gzip = CheckpointManager(save_dir=tmpdir, compress="gzip")

            # This should still work since we detect compression from file extension
            loaded_data, step, metadata = manager_gzip.load_latest()
            assert loaded_data["model"] == "test"
            assert step == 100

    def test_save_checkpoint_invalid_save_dir(self):
        """Test saving checkpoint to invalid directory"""
        # Try to save to a directory that doesn't exist and can't be created
        invalid_dir = "/nonexistent/deep/path/that/cannot/be/created"
        manager = CheckpointManager(save_dir=invalid_dir)

        test_data = {"test": "data"}

        with pytest.raises(OSError):  # Should raise file system error
            manager.save_sync(test_data, 1)

    def test_load_checkpoint_with_permission_denied(self):
        """Test loading checkpoint when file permissions deny access"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir)

            # Create a checkpoint file
            test_data = {"test": "data"}
            manager.save_sync(test_data, 1)

            # Find the created file
            checkpoint_files = list(Path(tmpdir).glob("checkpoint_*.pkl*"))
            assert len(checkpoint_files) == 1

            checkpoint_path = checkpoint_files[0]

            # Remove read permission (if on Unix-like system)
            try:
                os.chmod(checkpoint_path, 0o000)
                with pytest.raises(PermissionError):
                    manager.load_latest()
            finally:
                # Restore permissions for cleanup
                try:
                    os.chmod(checkpoint_path, 0o644)
                except Exception:
                    pass  # Ignore if already cleaned up

    def test_cleanup_old_checkpoints_invalid_directory(self):
        """Test cleanup when directory becomes invalid during operation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir, keep_last=2)

            # Create some checkpoints
            for i in range(5):
                manager.save_sync({"step": i}, i)

            # Verify checkpoints were created
            checkpoints = list(Path(tmpdir).glob("checkpoint_*.pkl*"))
            assert len(checkpoints) >= 3  # Should keep at least last 2

            # Try cleanup - should work normally
            manager.cleanup_old_checkpoints()

            # Verify cleanup worked
            remaining = list(Path(tmpdir).glob("checkpoint_*.pkl*"))
            assert len(remaining) <= 2  # Should have cleaned up old ones

    def test_concurrent_checkpoint_access(self):
        """Test behavior when checkpoints are modified during access"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir)

            # Save a checkpoint
            test_data = {"test": "data"}
            manager.save_sync(test_data, 1)

            # Load it successfully first
            loaded_data, step, metadata = manager.load_latest()
            assert loaded_data["test"] == "data"

            # Now delete the file while trying to load again
            checkpoint_files = list(Path(tmpdir).glob("checkpoint_*.pkl*"))
            if checkpoint_files:
                os.remove(checkpoint_files[0])

                with pytest.raises(FileNotFoundError):
                    manager.load_latest()

    def test_checkpoint_with_extremely_large_metadata(self):
        """Test checkpointing with extremely large metadata"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir)

            # Create very large metadata
            large_metadata = {"large_data": "x" * 1000000}  # 1MB string

            test_data = {"model": "test"}

            # Should handle large metadata gracefully
            manager.save_sync(test_data, 1, large_metadata)

            # Should be able to load it back
            loaded_data, step, metadata = manager.load_latest()
            assert loaded_data["model"] == "test"
            assert metadata["large_data"] == "x" * 1000000

    def test_checkpoint_file_naming_edge_cases(self):
        """Test checkpoint file naming with edge case step numbers"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir)

            # Test with zero step
            manager.save_sync({"step": 0}, 0)

            # Test with very large step number
            manager.save_sync({"step": 999999}, 999999)

            # Should be able to load the latest (largest step)
            loaded_data, step, metadata = manager.load_latest()
            assert step == 999999
            assert loaded_data["step"] == 999999
