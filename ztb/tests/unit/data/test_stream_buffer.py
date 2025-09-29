import sys
import types

import pandas as pd
import pytest

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

    fake_feature_engine = types.ModuleType("ztb.features.feature_engine")

    def _compute_features_batch(df, feature_names=None, **_kwargs):
        return pd.DataFrame(index=df.index)

    fake_feature_engine.compute_features_batch = _compute_features_batch
    sys.modules["ztb.features.feature_engine"] = fake_feature_engine

from ztb.data.stream_buffer import SchemaMismatchError, StreamBuffer


def test_stream_buffer_capacity_trim() -> None:
    buffer = StreamBuffer(capacity=3, columns=("timestamp", "price"))

    df1 = pd.DataFrame({"timestamp": [1, 2], "price": [10.0, 11.0]})
    buffer.extend(df1)
    assert len(buffer) == 2

    df2 = pd.DataFrame({"timestamp": [3, 4], "price": [12.0, 13.0]})
    buffer.extend(df2)
    assert len(buffer) == 3

    latest = buffer.latest(2)
    assert latest["timestamp"].tolist() == [3, 4]

    stats = buffer.stats()
    assert stats.rows == 3
    assert stats.capacity == 3


def test_stream_buffer_schema_validation() -> None:
    buffer = StreamBuffer(capacity=2, columns=("a", "b"))
    buffer.extend(pd.DataFrame({"a": [1], "b": [2]}))

    with pytest.raises(SchemaMismatchError):
        buffer.extend(pd.DataFrame({"a": [3], "c": [4]}))
