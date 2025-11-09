from pathlib import Path
from unittest.mock import Mock

from stable_baselines3.common.callbacks import CheckpointCallback


def test_checkpoint_callback_creates_checkpoint(tmp_path: Path):
    # Prepare a small checkpoint directory
    cp_dir = tmp_path / "checkpoints"
    cp_dir.mkdir()

    # Create a mock model with save method
    mock_model = Mock()
    mock_model.save = Mock()

    callback = CheckpointCallback(
        save_freq=1, save_path=str(cp_dir), name_prefix="test_model"
    )

    # Attach the mock model
    callback.model = mock_model

    # Simulate a couple of steps; _on_step uses n_calls internally
    # We manually increment n_calls by calling _on_step repeatedly
    for i in range(3):
        callback.n_calls = i + 1
        callback._on_step()

    # Check that save was called at least once (at step 1, 2, 3)
    assert mock_model.save.call_count >= 1, "CheckpointCallback did not save the model"
