from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import MakerPriceCalculator
from tests.unit.v460._yaml_test_helpers import (
    clone_fill_test_config,
    load_fill_test_config_from_path,
    load_yaml_mapping,
)
from ztb.metrics.fill_quality import FillRecord
from ztb.trading.risk.fast_fill_defense import FastFillDefense, FastFillDefenseConfig


# ─── 共通ヘルパー (非-fixture) ─────────────────────────────────────────
# テスト中に任意の overrides で呼べるためヘルパー関数として公開。
# pytest fixture から使ってもよいし、テスト本文から直接呼んでもよい。


def make_gate_config(**overrides: object) -> FillTestConfig:
    """CycleGateAggregator テスト用の FillTestConfig.

    全ゲート有効状態をデフォルトとし、overrides で任意フィールドを差し替え可能。
    """
    defaults: dict[str, object] = {
        "skip_buy_unknown_regime": True,
        "skip_ranging_buy_low_vol": True,
        "low_vol_threshold": 0.75,
        "skip_sell_trending": True,
        "skip_sell_trending_up_only": False,
        "max_consecutive_trending_sell_skip": 30,
        "sell_guard_inv_bypass_threshold": 0.3,
        "buy_dynamic_kill_enabled": True,
        "sell_dynamic_kill_enabled": True,
        "buy_dynamic_kill_threshold_bps": -5.0,
        "sell_dynamic_kill_threshold_bps": -5.0,
        "sell_velocity_skip_enabled": True,
        "sell_velocity_skip_threshold_bps": 8.0,
        "buy_velocity_skip_enabled": True,
        "buy_velocity_skip_threshold_bps": -8.0,
        "skip_sell_unknown_regime": True,
    }
    defaults.update(overrides)
    return FillTestConfig(**defaults)


def make_maker_price_config(**overrides: object) -> FillTestConfig:
    """MakerPriceCalculator テスト用の最小 FillTestConfig.

    max > min offset を保証するデフォルト値を持つ。
    """
    defaults: dict[str, object] = {
        "max_offset_ratio": 0.02,
        "min_offset_ratio": 0.001,
    }
    defaults.update(overrides)
    return FillTestConfig(**defaults)


@pytest.fixture(scope="session")
def v460_fill_test_yaml_path() -> Path:
    """fill_test.yaml の絶対パス."""
    return Path(__file__).resolve().parents[3] / "configs" / "v460" / "fill_test.yaml"


@pytest.fixture(scope="session")
def v460_fill_test_yaml_base(v460_fill_test_yaml_path: Path) -> dict[str, object]:
    """fill_test.yaml の session キャッシュ."""
    return load_yaml_mapping(v460_fill_test_yaml_path)


@pytest.fixture
def v460_fill_test_yaml(v460_fill_test_yaml_base: dict[str, object]) -> dict[str, object]:
    """fill_test.yaml のテストごと deepcopy."""
    return copy.deepcopy(v460_fill_test_yaml_base)


@pytest.fixture(scope="session")
def v460_fill_test_config_base(v460_fill_test_yaml_path: Path) -> FillTestConfig:
    """fill_test.yaml 由来の FillTestConfig を session キャッシュ."""
    return load_fill_test_config_from_path(v460_fill_test_yaml_path)


@pytest.fixture
def v460_fill_test_config_yaml(v460_fill_test_config_base: FillTestConfig) -> FillTestConfig:
    """共有 FillTestConfig のテストごと copy."""
    return clone_fill_test_config(v460_fill_test_config_base)


@pytest.fixture
def v460_tmp_results_dir(tmp_path: Path) -> Path:
    """v460 tests 用の共通 results ディレクトリ."""
    path = tmp_path / "results" / "v460"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def v460_fill_test_config(v460_tmp_results_dir: Path) -> FillTestConfig:
    """共通の高速テスト向け FillTestConfig."""
    return FillTestConfig(
        results_dir=str(v460_tmp_results_dir),
        order_timeout_sec=1.0,
        poll_interval_sec=0.01,
        post_fill_wait_sec=0.01,
        status_unknown_retry_delays=[0.0, 0.0, 0.0],
    )


@pytest.fixture
def v460_fill_record_factory() -> Callable[..., FillRecord]:
    """FillRecord の共通ファクトリ.

    必須引数は cycle_id / order_price / order_quantity。
    """

    def _factory(
        *,
        cycle_id: str,
        order_price: float,
        order_quantity: float,
        **overrides: object,
    ) -> FillRecord:
        payload: dict[str, object] = {
            "cycle_id": cycle_id,
            "timestamp": float(overrides.pop("timestamp", 1_700_000_000.0)),
            "side": str(overrides.pop("side", "buy")),
            "order_price": order_price,
            "order_quantity": order_quantity,
            "filled": bool(overrides.pop("filled", False)),
            "cancelled": bool(overrides.pop("cancelled", False)),
            "fill_price": overrides.pop("fill_price", None),
            "queue_wait_sec": float(overrides.pop("queue_wait_sec", 0.0)),
        }
        payload.update(overrides)
        return FillRecord(**payload)

    return _factory


@pytest.fixture
def v460_fast_fill_defense_config() -> FastFillDefenseConfig:
    """FastFillDefense の共通設定."""
    return FastFillDefenseConfig(
        enabled=True,
        threshold_sec=5.0,
        offset_boost=2.0,
        max_offset_ratio=0.30,
        min_offset_ratio=0.01,
    )


@pytest.fixture
def v460_fast_fill_defense(
    v460_fast_fill_defense_config: FastFillDefenseConfig,
) -> FastFillDefense:
    """FastFillDefense の標準インスタンス."""
    return FastFillDefense(
        config=v460_fast_fill_defense_config,
        base_offset_ratio=0.05,
    )


@pytest.fixture
def v460_maker_price_calculator(
    v460_fill_test_config: FillTestConfig,
    v460_fast_fill_defense: FastFillDefense,
) -> MakerPriceCalculator:
    """MakerPriceCalculator の標準インスタンス."""
    return MakerPriceCalculator(
        config=v460_fill_test_config,
        fast_fill_defense=v460_fast_fill_defense,
        regime_detector=None,
        base_offset_ratio=v460_fill_test_config.spread_offset_ratio,
        base_offset_ratio_buy=v460_fill_test_config.spread_offset_ratio_buy,
        base_offset_ratio_sell=v460_fill_test_config.spread_offset_ratio_sell,
    )
