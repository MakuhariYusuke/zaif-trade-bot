import json

import numpy as np
import pytest
from gymnasium import Env, spaces

from ztb.utils.training_utils import create_eval_callback, save_training_results


class DummyEnv(Env):
    """Minimal gymnasium env for EvalCallback instantiation."""

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))

    def reset(self, *, seed=None, options=None):
        return np.zeros(1), {}

    def step(self, action):
        return np.zeros(1), 0.0, True, False, {}


def test_create_eval_callback_renders_and_creates_dirs(tmp_path):
    eval_env = DummyEnv()
    best_model_dir = tmp_path / "models"
    eval_log_dir = tmp_path / "logs"

    callback = create_eval_callback(
        eval_env=eval_env,
        eval_freq=1,
        render=True,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
    )

    assert callback.render is True
    assert best_model_dir.exists(), "Best model directory should be created"
    assert eval_log_dir.exists(), "Eval log directory should be created"


def test_save_training_results_writes_json(tmp_path):
    results = {"accuracy": 0.88, "params": {"lr": 1e-4}}
    output_file = tmp_path / "training_results.json"

    assert save_training_results(results, str(output_file)) is True

    with open(output_file, encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded == results