import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import types

if "ztb.features" not in sys.modules:
    fake_features = types.ModuleType("ztb.features")

    class _FakeRegistry:
        @classmethod
        def initialize(cls) -> None:
            return None

        @classmethod
        def list(cls):
            return []

    fake_features.FeatureRegistry = _FakeRegistry
    sys.modules["ztb.features"] = fake_features

    # Use common test utility for feature engine mock
    from ztb.tests.test_utils import create_mock_feature_engine
    create_mock_feature_engine()

from ztb.data.coin_gecko_stream import MarketDataBatch, StreamConfig
from ztb.data.streaming_pipeline import StreamingPipeline
from ztb.trading.ppo_trainer import PPOTrainer


class SyntheticStream:
    def __init__(self, batch_rows: int = 64) -> None:
        self.batch_rows = batch_rows
        self._counter = 0

    def _frame(self) -> pd.DataFrame:
        base_ts = pd.Timestamp.utcnow().floor("s")
        offsets = pd.to_timedelta(range(self.batch_rows), unit="s")
        prices = 100 + self._counter + pd.Series(range(self.batch_rows)) * 0.01
        df = pd.DataFrame(
            {
                "timestamp": base_ts + offsets,
                "open": (prices - 0.05).astype("float32"),
                "high": (prices + 0.1).astype("float32"),
                "low": (prices - 0.1).astype("float32"),
                "close": prices.astype("float32"),
                "price": prices.astype("float32"),
                "market_cap": (prices * 1000).astype("float32"),
                "volume": (prices * 10).astype("float32"),
            }
        )
        self._counter += 1
        return df

    def fetch_range(
        self,
        coin_id: str,
        vs_currency: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        granularity: str = "1m",
        page_size: int = 200,
    ) -> MarketDataBatch:
        return MarketDataBatch(self._frame(), pd.Timestamp.utcnow(), {})

    def stream(
        self,
        config: StreamConfig,
        *,
        start_at: pd.Timestamp | None = None,
        stop_event: object | None = None,
    ):
        while True:
            yield self.fetch_range(
                config.coin_id,
                config.vs_currency,
                pd.Timestamp.utcnow(),
                pd.Timestamp.utcnow(),
            )


@pytest.mark.parametrize("total_steps,partial_steps", [(128, 64)])
def test_streaming_checkpoint_kill_and_resume(
    tmp_path: Path, total_steps: int, partial_steps: int
) -> None:
    checkpoint_dir = tmp_path / "ckpt"
    log_dir = tmp_path / "logs"
    model_dir = tmp_path / "models"
    tensorboard_dir = tmp_path / "tensorboard"

    for path in (checkpoint_dir, log_dir, model_dir, tensorboard_dir):
        path.mkdir(parents=True, exist_ok=True)

    for key in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "8")

    stream = SyntheticStream(batch_rows=64)
    pipeline = StreamingPipeline(
        stream_client=stream,
        buffer_capacity=4096,
        lookback_rows=128,
        buffer_compression="zlib",
    )
    now = pd.Timestamp.utcnow()
    pipeline.prime(now - pd.Timedelta(minutes=10), now)

    base_training_config = {
        "training": {
            "total_timesteps": total_steps,
            "eval_freq": 1000,
            "n_eval_episodes": 1,
            "batch_size": 32,
            "n_steps": 64,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "n_epochs": 2,
            "ent_coef": 0.0,
            "max_grad_norm": 0.5,
            "log_dir": str(log_dir),
            "model_dir": str(model_dir),
            "tensorboard_log": str(tensorboard_dir),
            "stream": {"batch_size": 64},
            "checkpoint": {
                "keep_last": 2,
                "compress": "zlib",
                "async_save": False,
                "include_optimizer": True,
                "include_replay_buffer": False,
                "include_rng_state": False,
            },
        }
    }

    trainer_a = PPOTrainer(
        data_path=None,
        config=base_training_config,
        checkpoint_interval=16,
        checkpoint_dir=str(checkpoint_dir),
        streaming_pipeline=pipeline,
    )
    trainer_a.config["total_timesteps"] = partial_steps
    model_a = trainer_a.train()

    partial_step_count = int(getattr(model_a, "num_timesteps", partial_steps))
    trainer_a.training_checkpoint_manager.save(
        step=partial_step_count,
        model=model_a,
        metrics={"phase": "partial"},
        stream_state=trainer_a._stream_state_snapshot(),
        async_save=False,
    )
    checkpoint_config = trainer_a.training_checkpoint_config
    trainer_a.training_checkpoint_manager.shutdown()
    pipeline.close()

    pipeline_resume = StreamingPipeline(
        stream_client=SyntheticStream(batch_rows=64),
        buffer_capacity=4096,
        lookback_rows=128,
        buffer_compression="zlib",
    )
    pipeline_resume.prime(now - pd.Timedelta(minutes=10), now)

    trainer_b = PPOTrainer(
        data_path=None,
        config=base_training_config,
        checkpoint_interval=16,
        checkpoint_dir=str(checkpoint_dir),
        streaming_pipeline=pipeline_resume,
    )
    trainer_b.config["total_timesteps"] = total_steps
    model_b = trainer_b.train()
    trainer_b.training_checkpoint_manager.shutdown()
    pipeline_resume.close()

    final_timesteps = int(getattr(model_b, "num_timesteps", total_steps))
    assert final_timesteps >= total_steps

    latest_snapshot = trainer_b.training_checkpoint_manager.load_latest()
    assert latest_snapshot.step >= total_steps
    assert latest_snapshot.metadata.get("metrics") is not None
