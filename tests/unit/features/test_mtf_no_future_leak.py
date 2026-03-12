"""
MTF (Multi-Timeframe) Future Leak Detection Tests

検証項目:
1. クローズドバーのみ使用（未来データ未含）
2. asof()による欠損データ処理の正確性
3. タイムゾーン一貫性
4. 境界条件（10:00、週末など）
"""

import logging
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TestMTFClosedBarBoundary:
    """MTFクローズドバー境界テスト"""

    @pytest.fixture
    def sample_1m_data(self):
        """1分足テストデータ生成"""
        start = pd.Timestamp("2025-01-10 09:50:00", tz="UTC")
        dates = pd.date_range(start=start, periods=150, freq="1min", tz="UTC")
        
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.randn(150).cumsum() + 100,
            "high": np.random.randn(150).cumsum() + 101,
            "low": np.random.randn(150).cumsum() + 99,
            "close": np.random.randn(150).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 150),
        })
        df = df.set_index("timestamp")
        return df

    def normalize_timeframe(self, tf: str) -> str:
        """タイムフレーム正規化"""
        tf = str(tf).lower().strip()
        mapping = {
            "5m": "5min", "5min": "5min",
            "15m": "15min", "15min": "15min",
            "1h": "1h", "60m": "1h", "60min": "1h",
        }
        return mapping.get(tf, tf)

    def get_mtf_closed_bar(self, df: pd.DataFrame, current_timestamp: pd.Timestamp, mtf: str) -> pd.DataFrame | None:
        """
        MTFクローズドバーを取得
        
        Args:
            df: 1分足OHLCV DataFrame (index: timestamp with tz-aware)
            current_timestamp: 現在時刻 (tz-aware, UTC)
            mtf: タイムフレーム ("5min", "15min", "1h")
        
        Returns:
            クローズドバーのOHLCV、またはNone
        
        ロジック:
        - バーインデックスはバーの開始時刻を表す
        - 5minバー: 10:00-10:05のバーは index=10:00
        - 現在時刻 10:07 のとき:
          - floor(10:07, 5min) = 10:05（現在進行中のバー）
          - 確定バーは 10:00-10:05（index=10:00）
        """
        # タイムフレーム正規化
        mtf = self.normalize_timeframe(mtf)
        
        # 現在時刻をフロア（バーの開始時刻）
        current_bar_start = current_timestamp.floor(mtf)
        
        # 現在進行中のバーなので、1つ前のバーを使用
        closed_bar_start = current_bar_start - pd.Timedelta(mtf)
        
        # closed_bar_start のインデックスで検索
        mask = df.index == closed_bar_start
        
        if mask.any():
            return df[mask].copy()
        
        # 正確なマッチなければ、asof()で最新バーを取得
        mask = df.index <= closed_bar_start
        if not mask.any():
            return None
        
        bar = df[mask].iloc[-1:].copy()
        return bar

    def test_mtf_5m_10_07_closed_bar(self, sample_1m_data):
        """10:07時点でのMTF 5分足クローズドバー確認"""
        current = pd.Timestamp("2025-01-10 10:07:00", tz="UTC")
        bar = self.get_mtf_closed_bar(sample_1m_data, current, "5min")
        
        assert bar is not None
        expected_bar_time = pd.Timestamp("2025-01-10 10:00:00", tz="UTC")
        assert bar.index[0] == expected_bar_time

    def test_mtf_5m_10_00_boundary(self, sample_1m_data):
        """10:00境界でのMTF確認"""
        # 10:00:00 時点
        current_open = pd.Timestamp("2025-01-10 10:00:00", tz="UTC")
        bar_open = self.get_mtf_closed_bar(sample_1m_data, current_open, "5min")
        assert bar_open is not None
        expected_prev = pd.Timestamp("2025-01-10 09:55:00", tz="UTC")
        assert bar_open.index[0] == expected_prev

    def test_mtf_15m_closed_bar(self, sample_1m_data):
        """15分足クローズドバー確認"""
        current = pd.Timestamp("2025-01-10 10:20:00", tz="UTC")
        bar = self.get_mtf_closed_bar(sample_1m_data, current, "15min")
        
        assert bar is not None
        expected_bar_time = pd.Timestamp("2025-01-10 10:00:00", tz="UTC")
        assert bar.index[0] == expected_bar_time

    def test_mtf_1h_closed_bar(self, sample_1m_data):
        """1時間足クローズドバー確認"""
        current = pd.Timestamp("2025-01-10 11:30:00", tz="UTC")
        bar = self.get_mtf_closed_bar(sample_1m_data, current, "1h")
        
        assert bar is not None
        expected_bar_time = pd.Timestamp("2025-01-10 10:00:00", tz="UTC")
        assert bar.index[0] == expected_bar_time

    def test_no_future_data_leak(self, sample_1m_data):
        """未来データリークなし確認"""
        current = pd.Timestamp("2025-01-10 10:10:00", tz="UTC")
        bar = self.get_mtf_closed_bar(sample_1m_data, current, "5min")
        
        assert bar.index[0] < current


class TestMTFAsofMissingData:
    """asof()による欠損データ処理テスト"""

    @pytest.fixture
    def sparse_5m_data(self):
        """5分足欠損データを含むテストデータ"""
        dates = [
            pd.Timestamp("2025-01-10 10:00:00", tz="UTC"),
            pd.Timestamp("2025-01-10 10:05:00", tz="UTC"),
            pd.Timestamp("2025-01-10 10:15:00", tz="UTC"),
        ]
        
        df = pd.DataFrame({
            "timestamp": dates,
            "close": [100.0, 101.0, 102.0],
            "volume": [1000, 1100, 1200],
        })
        df = df.set_index("timestamp")
        return df

    def test_asof_forward_fill(self, sparse_5m_data):
        """asof()での前方フォワードフィル確認"""
        query_time = pd.Timestamp("2025-01-10 10:12:00", tz="UTC")
        result = sparse_5m_data.asof(query_time)
        assert result["close"] == 101.0

    def test_asof_exact_match(self, sparse_5m_data):
        """asof()での正確マッチ確認"""
        query_time = pd.Timestamp("2025-01-10 10:05:00", tz="UTC")
        result = sparse_5m_data.asof(query_time)
        assert result["close"] == 101.0

    def test_asof_no_prior_data(self, sparse_5m_data):
        """asof()で以前のデータなし時の処理"""
        query_time = pd.Timestamp("2025-01-10 09:50:00", tz="UTC")
        result = sparse_5m_data.asof(query_time)
        assert pd.isna(result["close"])


class TestTimestampValidation:
    """タイムゾーン検証テスト"""

    def validate_and_convert_timestamp(
        self,
        timestamp: pd.Timestamp,
        require_tz: bool = True,
        target_tz: str = "UTC"
    ) -> pd.Timestamp:
        """タイムゾーン検証と変換"""
        if timestamp.tzinfo is None:
            if require_tz:
                raise ValueError(
                    f"Naive timestamp not allowed. "
                    f"All timestamps must be timezone-aware. "
                    f"Got: {timestamp}"
                )
            timestamp = timestamp.tz_localize("UTC")
        
        return timestamp.tz_convert(target_tz)

    def test_naive_timestamp_rejected(self):
        """Naive timestampが拒否されることを確認"""
        naive_ts = pd.Timestamp("2025-01-10 10:00:00")
        
        with pytest.raises(ValueError, match="Naive timestamp not allowed"):
            self.validate_and_convert_timestamp(naive_ts, require_tz=True)

    def test_utc_aware_timestamp_accepted(self):
        """UTC timezone-aware timestampが受け入れられることを確認"""
        utc_ts = pd.Timestamp("2025-01-10 10:00:00", tz="UTC")
        result = self.validate_and_convert_timestamp(utc_ts)
        assert result.tzinfo is not None
        assert str(result.tz) == "UTC"

    def test_jst_to_utc_conversion(self):
        """JST → UTC変換確認"""
        jst_ts = pd.Timestamp("2025-01-10 10:00:00", tz="Asia/Tokyo")
        result = self.validate_and_convert_timestamp(jst_ts, target_tz="UTC")
        # JST 10:00 = UTC 01:00（同一日）
        assert result.hour == 1
        assert result.day == 10

    def test_timezone_consistency(self):
        """複数タイムゾーンでの一貫性確認"""
        utc_ts = pd.Timestamp("2025-01-10 10:00:00", tz="UTC")
        jst_ts = pd.Timestamp("2025-01-10 19:00:00", tz="Asia/Tokyo")
        
        utc_result = self.validate_and_convert_timestamp(utc_ts, target_tz="UTC")
        jst_to_utc = self.validate_and_convert_timestamp(jst_ts, target_tz="UTC")
        
        assert utc_result == jst_to_utc


class TestMTFTimeframeNormalization:
    """タイムフレーム正規化テスト"""

    def normalize_timeframe(self, tf: str) -> str:
        """タイムフレーム正規化"""
        tf = str(tf).lower().strip()
        mapping = {
            "5m": "5min", "5min": "5min",
            "15m": "15min", "15min": "15min",
            "1h": "1h", "60m": "1h", "60min": "1h",
        }
        return mapping.get(tf, tf)

    def test_normalize_5m(self):
        assert self.normalize_timeframe("5m") == "5min"

    def test_normalize_15m(self):
        assert self.normalize_timeframe("15m") == "15min"

    def test_normalize_1h(self):
        assert self.normalize_timeframe("1h") == "1h"
        assert self.normalize_timeframe("60m") == "1h"

    def test_case_insensitive(self):
        assert self.normalize_timeframe("5M") == "5min"
        assert self.normalize_timeframe("1H") == "1h"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
