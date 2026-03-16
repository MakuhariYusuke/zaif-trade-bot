from ztb.utils.training_utils import display_training_complete, get_safe_loss_function


def test_display_training_complete_success(capsys):
    metrics = {"reward": 1.2345, "loss": 0.1234}
    display_training_complete(metrics, 123.4)
    captured = capsys.readouterr()
    assert "Training completed successfully" in captured.out
    assert "Total training time" in captured.out
    assert "reward" in captured.out


def test_display_training_complete_failure(capsys):
    display_training_complete({}, 0.0)
    captured = capsys.readouterr()
    assert "Training failed" in captured.out


class _BadLoss:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("can't instantiate")


def test_get_safe_loss_function_fallback():
    loss = get_safe_loss_function(_BadLoss)
    # Should be callable and return a scalar-like zero
    assert callable(loss)
    assert hasattr(loss, "__call__")
    val = loss()
    # Dummy fallback returns 0
    assert val == 0
