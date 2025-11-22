import csv
import os
from pathlib import Path
import pandas as pd
import pytest

from ztb.training import unified_trainer

from tools.data.convert_timeframe import resample_ohlcv, map_freq
from tools.data.validate_dataset import main as validate_main


def make_minute_df(start_ts, periods=5):
    idx = pd.date_range(start=start_ts, periods=periods, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": range(100, 100 + periods),
            "high": range(100, 100 + periods),
            "low": range(100, 100 + periods),
            "close": range(100, 100 + periods),
            "volume": [10] * periods,
        }
    )
    return df


def test_resample_5m_basic(tmp_path):
    # Create minute dataframe
    df = make_minute_df('2025-11-01T00:00:00', periods=5)
    df = df.set_index('timestamp')

    # 5m resample
    resampled = resample_ohlcv(df, '5T')

    assert len(resampled) == 1
    assert resampled.iloc[0]['open'] == 100
    assert resampled.iloc[0]['close'] == 104
    assert resampled.iloc[0]['high'] == 104
    assert resampled.iloc[0]['low'] == 100
    assert resampled.iloc[0]['volume'] == 10 * 5


def test_map_freq_alias():
    assert map_freq('1m') == '1T'
    assert map_freq('5m') == '5T'
    assert map_freq('1h') == '1H'


def test_validate_dataset_missing_col(tmp_path, capsys, monkeypatch):
    # Create CSV missing volume column
    csv_path = tmp_path / "test_missing.csv"
    df = make_minute_df('2025-11-01T00:00:00', periods=3)
    df = df.drop(columns=['volume'])
    df.to_csv(csv_path, index=False)

    # Monkeypatch argv for validate_main
    monkeypatch.setattr('sys.argv', ["validate_dataset.py", "--path", str(csv_path)])

    # Run validator
    # Call validator. It should print the missing columns message and return normally
    validate_main()
    # The script returns None but may exit; check output
    captured = capsys.readouterr()
    assert "Missing required columns" in captured.out


def test_validate_dataset_and_resample(tmp_path, capsys, monkeypatch):
    csv_path = tmp_path / "test_minute.csv"
    df = make_minute_df('2025-11-01T00:00:00', periods=6)
    df.to_csv(csv_path, index=False)

    # Call validator with resample targets
    monkeypatch.setattr('sys.argv', ["validate_dataset.py", "--path", str(csv_path), "--resample-to", "5m", "15m"])  

    # Should invoke convert_timeframe via subprocess (we assume success)
    # Running in tests: expect it not to raise
    validate_main()
    captured = capsys.readouterr()
    assert "Dataset validation complete" in captured.out

