from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.v460.ml.cache_cleanup import (
    clear_ml_data_caches,
    clear_ml_data_caches_with_log,
    get_ml_data_cache_stats,
)


class TestMlCacheCleanup:
    def test_get_ml_data_cache_stats_combines_sources(self) -> None:
        with patch(
            "scripts.v460.ml.cache_cleanup.get_fill_records_cache_stats",
            return_value={"fill_records_cache_entries": 2},
        ), patch(
            "scripts.v460.ml.cache_cleanup.get_raw_load_cache_stats",
            return_value={"orderbook_cache_entries": 3, "trades_cache_entries": 4},
        ):
            stats = get_ml_data_cache_stats()

        assert stats["fill_records_cache_entries"] == 2
        assert stats["orderbook_cache_entries"] == 3
        assert stats["trades_cache_entries"] == 4
        assert stats["total_ml_cache_entries"] == 9

    def test_clear_ml_data_caches_runs_gc_when_requested(self) -> None:
        with patch("scripts.v460.ml.cache_cleanup.clear_fill_records_cache") as clear_fill, patch(
            "scripts.v460.ml.cache_cleanup.clear_raw_load_caches",
        ) as clear_raw, patch(
            "scripts.v460.ml.cache_cleanup.get_ml_data_cache_stats",
            return_value={"total_ml_cache_entries": 0},
        ), patch(
            "scripts.v460.ml.cache_cleanup.gc.collect",
            return_value=11,
        ):
            stats = clear_ml_data_caches(collect_garbage=True)

        clear_fill.assert_called_once()
        clear_raw.assert_called_once()
        assert stats["gc_collected"] == 11

    def test_clear_ml_data_caches_with_log_emits_info_for_non_empty_stats(self) -> None:
        fake_logger = MagicMock()

        with patch(
            "scripts.v460.ml.cache_cleanup.clear_ml_data_caches",
            return_value={"total_ml_cache_entries": 2, "gc_collected": 3},
        ):
            stats = clear_ml_data_caches_with_log(
                fake_logger,
                context="unit",
                collect_garbage=True,
            )

        fake_logger.info.assert_called_once()
        assert stats["total_ml_cache_entries"] == 2


class TestMlCacheCleanupIntegration:
    _ENTRYPOINTS = (
        "scripts/v460/ml/run_ml_pipeline.py",
        "scripts/v460/ml/train_sg_v2.py",
        "scripts/v460/ml/train_sg_v3.py",
        "scripts/v460/ml/train_alt_horizon.py",
        "scripts/v460/ml/tune_as_classifier.py",
        "scripts/v460/ml/walk_forward_as.py",
        "scripts/v460/ml/deploy_sg_v3.py",
        "scripts/v460/ml/deploy_sg_v4.py",
        "scripts/v460/ml/run_070_deep_analysis.py",
        "scripts/v460/ml/run_070_final_analysis.py",
        "scripts/v460/ml/run_070_model_search.py",
        "scripts/v460/ml/retrain_scheduler.py",
    )

    def test_all_ml_entrypoints_clear_caches(self) -> None:
        for rel_path in self._ENTRYPOINTS:
            source = Path(rel_path).read_text(encoding="utf-8")
            assert "clear_ml_data_caches_with_log(" in source, rel_path
