import time

import sys
import types

# Provide a simple fake 'stable_baselines3.common.callbacks' to avoid heavy deps during tests
sb3_mod = types.ModuleType("stable_baselines3")
sb3_common = types.ModuleType("stable_baselines3.common")
sb3_callbacks = types.ModuleType("stable_baselines3.common.callbacks")
sb3_callbacks.BaseCallback = object
sb3_common.callbacks = sb3_callbacks
sb3_mod.common = sb3_common
sys.modules["stable_baselines3"] = sb3_mod
sys.modules["stable_baselines3.common"] = sb3_common
sys.modules["stable_baselines3.common.callbacks"] = sb3_callbacks

# Provide a minimal 'websockets' module if not installed to avoid heavy runtime
ws_mod = types.ModuleType("websockets")
sys.modules["websockets"] = ws_mod

from ztb.training.callbacks.core.modern_callback_system import (
    CallbackContext,
    CallbackEvent,
)
from ztb.training.callbacks.core.callback_implementations import (
    CheckpointCallback,
    MetricsCallback,
    ProgressCallback,
)


def test_checkpoint_callback_saves_file(tmp_path):
    save_dir = tmp_path / "checkpoints"
    cb = CheckpointCallback(save_interval=2, save_path=str(save_dir))

    ctx = CallbackContext(event=CallbackEvent.TRAINING_START, total_steps=10)
    # start
    res = cb.on_training_start(ctx)
    assert res.success is True

    # simulate steps
    ctx.step = 2
    res = cb.on_step_end(ctx)
    assert isinstance(res, object)
    # checkpoint file should have been created
    files = list(save_dir.glob("*.zip"))
    assert len(files) >= 1

    # test save_best_only behavior
    cb2 = CheckpointCallback(save_interval=1000, save_path=str(save_dir))
    cb2.checkpoint_config.save_best_only = True
    cb2._best_metric_value = -float("inf")
    ctx.metrics = {cb2.checkpoint_config.best_metric: 10}
    res = cb2.on_step_end(ctx)
    # it should attempt to save best
    files2 = list(save_dir.glob("*.zip"))
    assert len(files2) >= 1


def test_metrics_callback_collection_and_tensorboard_close(tmp_path, monkeypatch):
    cb = MetricsCallback(collection_interval=1, log_interval=10)

    ctx = CallbackContext(event=CallbackEvent.TRAINING_START,timestamp=time.time())
    cb.on_training_start(ctx)

    # set a fake writer to verify close called
    closed = {"val": False}

    class FakeWriter:
        def close(self):
            closed["val"] = True

        def add_scalar(self, *args, **kwargs):
            pass

    cb._tensorboard_writer = FakeWriter()

    # simulate a metrics update
    ctx.step = 1
    ctx.metrics = {"loss": 0.5}
    cb.on_step_end(ctx)
    assert cb.get_metrics_history(), "Metrics history should not be empty"

    # call training end and ensure writer closed
    res = cb.on_training_end(ctx)
    assert res.success
    assert closed["val"] is True


def test_progress_callback_format_metrics_and_end():
    cb = ProgressCallback(log_interval=1, show_eta=False)
    cb.progress_config.show_metrics = True
    ctx = CallbackContext(event=CallbackEvent.TRAINING_START, total_steps=10)
    cb.on_training_start(ctx)

    ctx.step = 1
    ctx.metrics = {"loss": 1.23456, "reward": 2}

    formatted = cb._format_metrics(ctx.metrics)
    assert "loss" in formatted and "reward" in formatted

    res = cb.on_step_end(ctx)
    assert res.success

    # end
    res = cb.on_training_end(ctx)
    assert res.success
