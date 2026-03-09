"""Tests for 169# Config Hot-Reload mechanism."""

from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from scripts.v460.lib.config_hot_reload import (
    ConfigHotReloader,
    _HOT_RELOADABLE_FIELDS,
    _resolve_time_filter_cls,
)
from scripts.v460.lib.fill_config import FillTestConfig


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def yaml_content_base() -> str:
    """Base YAML content for fill_test config."""
    return """\
symbol: btc_jpy
spread_offset_ratio: 0.05
cycle_interval_sec: 120.0
order_timeout_sec: 90.0
min_spread_jpy: 0.0

side_offset:
  buy: 0.04
  sell: 0.06

skip_gate:
  enabled: true
  as_threshold: 0.52

sell_dynamic_kill:
  enabled: true
  threshold_bps: -0.5

daily_drawdown:
  enabled: true
  hard_limit_bps: -50.0

time_filter:
  enabled: true
  skip_utc_hours_buy: []
  skip_utc_hours_sell: []

regime:
  enabled: true
  low_vol_offset_boost_enabled: false
  low_vol_offset_boost: 1.4
  skip_ranging_buy_low_vol: false
"""


@pytest.fixture
def yaml_content_updated() -> str:
    """Updated YAML with changed values."""
    return """\
symbol: btc_jpy
spread_offset_ratio: 0.08
cycle_interval_sec: 60.0
order_timeout_sec: 45.0
min_spread_jpy: 5.0

side_offset:
  buy: 0.07
  sell: 0.09

skip_gate:
  enabled: false
  as_threshold: 0.60

sell_dynamic_kill:
  enabled: false
  threshold_bps: -1.0

daily_drawdown:
  enabled: false
  hard_limit_bps: -80.0

time_filter:
  enabled: false
  skip_utc_hours_buy: [8]
  skip_utc_hours_sell: [16]

regime:
  enabled: true
  low_vol_offset_boost_enabled: true
  low_vol_offset_boost: 1.6
  skip_ranging_buy_low_vol: true
"""


@pytest.fixture
def temp_yaml(tmp_path: Path, yaml_content_base: str) -> Path:
    """Temporary YAML file that can be modified."""
    path = tmp_path / "fill_test.yaml"
    path.write_text(yaml_content_base, encoding="utf-8")
    return path


@pytest.fixture
def base_config() -> FillTestConfig:
    """A config with known initial values."""
    return FillTestConfig(
        spread_offset_ratio=0.05,
        spread_offset_ratio_buy=0.04,
        spread_offset_ratio_sell=0.06,
        cycle_interval_sec=120.0,
        min_spread_jpy=0.0,
        skip_gate_enabled=True,
        skip_gate_as_threshold=0.52,
        sell_dynamic_kill_enabled=True,
        sell_dynamic_kill_threshold_bps=-0.5,
        daily_drawdown_enabled=True,
        daily_drawdown_hard_limit_bps=-50.0,
    )


@pytest.fixture(autouse=True)
def _stub_git_sha() -> Iterator[None]:
    """Reload tests do not need a real git subprocess."""
    with patch("ztb.utils.git_utils.get_git_sha", return_value="abc123"):
        yield


@pytest.fixture(autouse=True)
def _stub_time_filter() -> Iterator[None]:
    """Hot-reload tests do not need the real TimeFilter import graph."""
    with patch(
        "scripts.v460.lib.config_hot_reload._resolve_time_filter_cls",
        return_value=MagicMock,
    ):
        yield


def _make_mock_runner(config: FillTestConfig) -> MagicMock:
    """Create a mock runner with required attributes."""
    runner = MagicMock()
    runner.config = config
    runner._git_sha = "abc123"
    runner._maker_price = MagicMock()
    runner._time_filter = MagicMock()
    runner._sell_kill_mgr = MagicMock()
    runner._buy_kill_mgr = MagicMock()
    runner._daily_drawdown_guard = MagicMock()
    runner._daily_drawdown_guard.export_state.return_value = {}
    runner._fast_fill_defense = MagicMock()
    runner._config_reloader = MagicMock()
    return runner


def _make_reloader(
    config: FillTestConfig,
    yaml_path: Path | str | None,
    *,
    yaml_cfg: dict[str, object] | None = None,
    check_interval_sec: float = 0.0,
) -> ConfigHotReloader:
    """ConfigHotReloader の標準構築."""
    return ConfigHotReloader(
        config=config,
        yaml_path=yaml_path,
        yaml_cfg={} if yaml_cfg is None else yaml_cfg,
        check_interval_sec=check_interval_sec,
    )


def _write_yaml_with_updated_mtime(path: Path, content: str) -> None:
    """mtime 差分を sleep なしで保証して YAML を更新する."""
    path.write_text(content, encoding="utf-8")
    current = path.stat().st_mtime
    bumped = max(time.time(), current + 1.0)
    os.utime(path, (bumped, bumped))


# ======================================================================
# Tests: 基本動作
# ======================================================================


class TestConfigHotReloaderBasic:
    """基本的なリロードメカニズムのテスト."""

    def test_no_reload_before_interval(
        self, base_config: FillTestConfig, temp_yaml: Path,
    ) -> None:
        """check_interval_sec 未満ではリロードしない."""
        reloader = _make_reloader(base_config, temp_yaml, check_interval_sec=300.0)
        runner = _make_mock_runner(base_config)

        # すぐに呼んでもリロードされない
        result = reloader.maybe_reload(runner)
        assert result is False

    def test_no_reload_without_mtime_change(
        self, base_config: FillTestConfig, temp_yaml: Path,
    ) -> None:
        """mtime が変わらなければリロードしない."""
        reloader = _make_reloader(base_config, temp_yaml)
        runner = _make_mock_runner(base_config)

        # mtime 変更なし → no reload
        reloader._last_check_time = 0.0  # force check
        result = reloader.maybe_reload(runner)
        assert result is False

    def test_reload_on_mtime_change(
        self,
        base_config: FillTestConfig,
        temp_yaml: Path,
        yaml_content_updated: str,
    ) -> None:
        """mtime 変更 + フィールド差分ありでリロード実行."""
        reloader = _make_reloader(base_config, temp_yaml)
        runner = _make_mock_runner(base_config)

        # YAML を更新
        _write_yaml_with_updated_mtime(temp_yaml, yaml_content_updated)

        reloader._last_check_time = 0.0  # force check
        with patch("scripts.v460.lib.config_hot_reload.ConfigHotReloader._do_reload") as mock_reload:
            mock_reload.return_value = True
            result = reloader.maybe_reload(runner)
        assert result is True

    def test_reload_count_increments(
        self, base_config: FillTestConfig, temp_yaml: Path,
    ) -> None:
        """リロードごとにカウンタがインクリメントされる."""
        reloader = _make_reloader(base_config, temp_yaml)
        assert reloader.reload_count == 0

    def test_none_yaml_path_returns_false(
        self, base_config: FillTestConfig,
    ) -> None:
        """YAML パスが None ならリロードしない."""
        reloader = _make_reloader(base_config, None)
        runner = _make_mock_runner(base_config)
        reloader._last_check_time = 0.0
        result = reloader.maybe_reload(runner)
        assert result is False


# ======================================================================
# Tests: フィールド差分更新
# ======================================================================


class TestConfigFieldUpdate:
    """フィールド差分更新の正確性."""

    def test_do_reload_updates_reloadable_fields(
        self,
        base_config: FillTestConfig,
        temp_yaml: Path,
        yaml_content_updated: str,
    ) -> None:
        """_do_reload がホットリロード対象フィールドを更新する."""
        reloader = _make_reloader(base_config, temp_yaml)
        runner = _make_mock_runner(base_config)

        # YAML を更新して直接 _do_reload
        _write_yaml_with_updated_mtime(temp_yaml, yaml_content_updated)

        result = reloader._do_reload(runner)
        assert result is True

        # 更新された値を確認
        assert base_config.spread_offset_ratio == 0.08
        assert base_config.spread_offset_ratio_buy == 0.07
        assert base_config.spread_offset_ratio_sell == 0.09
        assert base_config.cycle_interval_sec == 60.0
        assert base_config.min_spread_jpy == 5.0
        assert base_config.skip_gate_enabled is False
        assert base_config.skip_gate_as_threshold == 0.60

    def test_non_reloadable_fields_preserved(
        self,
        base_config: FillTestConfig,
        temp_yaml: Path,
    ) -> None:
        """ホットリロード対象外フィールドは変更されない."""
        original_results_dir = base_config.results_dir
        original_symbol = base_config.symbol

        # results_dir を変更した YAML
        updated_yaml = """\
symbol: btc_jpy
results_dir: results/different_dir
spread_offset_ratio: 0.05
"""
        reloader = _make_reloader(base_config, temp_yaml)
        runner = _make_mock_runner(base_config)

        _write_yaml_with_updated_mtime(temp_yaml, updated_yaml)

        reloader._do_reload(runner)

        # results_dir は変更されない (非リロード対象)
        assert base_config.results_dir == original_results_dir
        assert base_config.symbol == original_symbol


# ======================================================================
# Tests: コンポーネント再構築
# ======================================================================


class TestComponentRebuild:
    """設定変更時のコンポーネント再構築."""

    def test_sell_kill_mgr_rebuild_on_threshold_change(
        self,
        base_config: FillTestConfig,
        temp_yaml: Path,
        yaml_content_updated: str,
    ) -> None:
        """sell_dynamic_kill 設定変更でマネージャが再構築される."""
        reloader = _make_reloader(base_config, temp_yaml)
        runner = _make_mock_runner(base_config)

        _write_yaml_with_updated_mtime(temp_yaml, yaml_content_updated)

        reloader._do_reload(runner)

        # sell_dynamic_kill_enabled changed → rebuild callback called
        runner._rebuild_sell_kill_mgr.assert_called_once()

    def test_daily_drawdown_rebuild_on_change(
        self,
        base_config: FillTestConfig,
        temp_yaml: Path,
        yaml_content_updated: str,
    ) -> None:
        """daily_drawdown 設定変更でガードが再構築される."""
        reloader = _make_reloader(base_config, temp_yaml)
        runner = _make_mock_runner(base_config)

        _write_yaml_with_updated_mtime(temp_yaml, yaml_content_updated)

        reloader._do_reload(runner)

        runner._rebuild_daily_drawdown_guard.assert_called_once()

    def test_maker_price_offsets_updated(
        self,
        base_config: FillTestConfig,
        temp_yaml: Path,
        yaml_content_updated: str,
    ) -> None:
        """offset 変更で MakerPriceCalculator のオフセットが更新される."""
        reloader = _make_reloader(base_config, temp_yaml)
        runner = _make_mock_runner(base_config)

        _write_yaml_with_updated_mtime(temp_yaml, yaml_content_updated)

        reloader._do_reload(runner)

        # MakerPriceCalculator の offset が更新されたことを確認
        assert runner._maker_price.base_offset_ratio == 0.08
        assert runner._maker_price.base_offset_ratio_buy == 0.07
        assert runner._maker_price.base_offset_ratio_sell == 0.09


# ======================================================================
# Tests: エラー耐性
# ======================================================================


class TestReloadErrorHandling:
    """リロード失敗時の防御的動作."""

    def test_invalid_yaml_preserves_old_config(
        self,
        base_config: FillTestConfig,
        temp_yaml: Path,
    ) -> None:
        """不正 YAML でもクラッシュせず旧設定を維持."""
        original_offset = base_config.spread_offset_ratio

        reloader = _make_reloader(base_config, temp_yaml)
        runner = _make_mock_runner(base_config)

        # 不正 YAML を書き込み
        _write_yaml_with_updated_mtime(temp_yaml, "invalid: [yaml: {broken")

        reloader._last_check_time = 0.0
        with patch("scripts.v460.lib.config_hot_reload.logger.error"):
            result = reloader.maybe_reload(runner)

        # エラーでもクラッシュしない
        assert result is False
        # 旧設定が維持される
        assert base_config.spread_offset_ratio == original_offset

    def test_missing_yaml_file_no_crash(
        self,
        base_config: FillTestConfig,
    ) -> None:
        """YAML ファイルが存在しなくてもクラッシュしない."""
        reloader = _make_reloader(base_config, "/nonexistent/path.yaml")
        runner = _make_mock_runner(base_config)
        reloader._last_check_time = 0.0
        # mtime が 0.0 で固定 → リロードしない
        result = reloader.maybe_reload(runner)
        assert result is False

    def test_reload_error_preserves_config(
        self,
        base_config: FillTestConfig,
        temp_yaml: Path,
    ) -> None:
        """_do_reload が例外を投げても旧設定を維持."""
        original_offset = base_config.spread_offset_ratio
        reloader = _make_reloader(base_config, temp_yaml)
        runner = _make_mock_runner(base_config)

        # mtime を人為的に変更して _do_reload を発動
        _write_yaml_with_updated_mtime(temp_yaml, "spread_offset_ratio: 0.10")
        reloader._last_check_time = 0.0

        with patch.object(reloader, "_do_reload", side_effect=RuntimeError("test")):
            result = reloader.maybe_reload(runner)

        assert result is False
        assert base_config.spread_offset_ratio == original_offset


# ======================================================================
# Tests: _HOT_RELOADABLE_FIELDS の整合性
# ======================================================================


class TestReloadableFieldsConsistency:
    """ホットリロード対象フィールドが FillTestConfig に実在することを検証."""

    def test_all_reloadable_fields_exist_in_config(self) -> None:
        """_HOT_RELOADABLE_FIELDS の全フィールドが FillTestConfig に存在."""
        config_fields = {f.name for f in dataclasses.fields(FillTestConfig)}
        missing = _HOT_RELOADABLE_FIELDS - config_fields
        assert missing == set(), (
            f"_HOT_RELOADABLE_FIELDS contains fields not in FillTestConfig: {missing}"
        )

    def test_immutable_fields_not_in_reloadable(self) -> None:
        """安全でないフィールドがリロード対象に含まれていない."""
        forbidden = {
            "symbol",          # exchange adapter と密結合
            "results_dir",     # ファイルパス変更は危険
            "enable_regime",   # RegimeDetector 構造体の再構築が必要
        }
        overlap = forbidden & _HOT_RELOADABLE_FIELDS
        assert overlap == set(), (
            f"These fields should NOT be hot-reloadable: {overlap}"
        )

    def test_recent_soft_guard_fields_are_reloadable(self) -> None:
        """後発追加の soft-guard 関連は reload 対象から漏らさない."""
        expected = {
            "skip_gate_ev_as_offset_enabled",
            "skip_gate_ev_offset_sensitivity",
            "skip_gate_ev_offset_min_mult",
            "skip_gate_ev_offset_max_mult",
            "skip_gate_ev_emergency_skip_threshold",
            "velocity_skip_as_offset_enabled",
            "velocity_offset_boost_factor",
            "velocity_offset_proportional",
            "velocity_offset_max_mult",
            "trending_sell_as_offset_enabled",
            "trending_sell_offset_boost_factor",
            # 253# 削除済: "balance_forced_apply_trending_offset",
        }
        missing = expected - _HOT_RELOADABLE_FIELDS
        assert missing == set(), (
            f"Recent soft-guard fields should be hot-reloadable: {missing}"
        )
