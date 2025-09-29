import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ztb.training.checkpoint_manager import TrainingCheckpointConfig, TrainingCheckpointManager


class _DummyOptimizer:
    def __init__(self) -> None:
        self.state = {"lr": 0.001}

    def state_dict(self) -> Dict[str, Any]:
        return {"state": dict(self.state)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.state.update(state.get("state", {}))


class _DummyPolicy:
    def __init__(self) -> None:
        self.values = {"weight": 1.0}
        self.optimizer = _DummyOptimizer()

    def state_dict(self) -> Dict[str, Any]:
        return dict(self.values)

    def load_state_dict(self, state: Dict[str, Any], strict: bool = False) -> None:
        self.values.update(state)


class _DummyModel:
    def __init__(self) -> None:
        self.policy = _DummyPolicy()
        self.replay_buffer: Dict[str, Any] = {"items": [1, 2, 3]}

    def save_replay_buffer(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self.replay_buffer, fh)

    def load_replay_buffer(self, path: str) -> None:
        with open(path, "rb") as fh:
            self.replay_buffer = pickle.load(fh)



def test_checkpoint_manager_roundtrip(tmp_path: Path) -> None:
    config = TrainingCheckpointConfig(
        interval_steps=10,
        async_save=False,
        include_rng_state=False,
        compress='zlib',
    )
    manager = TrainingCheckpointManager(save_dir=str(tmp_path), config=config)

    model = _DummyModel()
    model.policy.values["weight"] = 42.0
    model.replay_buffer = {"items": [5]}

    stream_state = {
        "buffer": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
                "price": [1.23],
                "market_cap": [100.0],
                "volume": [10.0],
            }
        ),
        "stats": {"rows": 1},
    }

    manager.save(
        step=10,
        model=model,
        metrics={"reward": 5.0},
        stream_state=stream_state,
        async_save=False,
    )

    snapshot = manager.load_latest()
    assert snapshot is not None
    assert snapshot.step == 10
    assert snapshot.payload["metrics"]["reward"] == 5.0
    assert snapshot.payload["stream_state"]["stats"]["rows"] == 1

    restored = _DummyModel()
    restored.policy.values["weight"] = 0.0
    restored.replay_buffer = {"items": []}
    manager.apply_snapshot(restored, snapshot)

    assert restored.policy.values["weight"] == 42.0
    assert restored.replay_buffer == {"items": [5]}

    manager.shutdown()
