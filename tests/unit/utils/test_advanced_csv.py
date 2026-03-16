from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from ztb.io.advanced_csv import (
    clear_read_csv_cache,
    get_read_csv_cache_stats,
    read_csv_cached,
)


class TestAdvancedCsvCache:
    def test_read_csv_cache_is_bounded(self, tmp_path) -> None:
        clear_read_csv_cache()
        df = pd.DataFrame({"x": [1, 2, 3]})

        with patch("ztb.io.advanced_csv.read_csv_mmap", return_value=df):
            for i in range(4):
                read_csv_cached(
                    tmp_path / f"sample_{i}.csv",
                    max_cache_entries=2,
                )

        assert get_read_csv_cache_stats()["entries"] == 2

    def test_clear_read_csv_cache_empties_entries(self, tmp_path) -> None:
        clear_read_csv_cache()
        df = pd.DataFrame({"x": [1]})

        with patch("ztb.io.advanced_csv.read_csv_mmap", return_value=df):
            read_csv_cached(tmp_path / "sample.csv")

        assert get_read_csv_cache_stats()["entries"] == 1
        clear_read_csv_cache()
        assert get_read_csv_cache_stats()["entries"] == 0
