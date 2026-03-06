from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from scripts.v460.lib.fast_fill_defense import FastFillDefense, FastFillDefenseConfig
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.maker_price import MakerPriceCalculator
from ztb.metrics.fill_quality import FillRecord


@pytest.fixture(scope="session")
def v460_fill_test_yaml_path() -> Path:
    """fill_test.yaml の絶対パス."""
    return Path(__file__).resolve().parents[3] / "configs" / "v460" / "fill_test.yaml"


@pytest.fixture(scope="session")
def v460_fill_test_yaml_base(v460_fill_test_yaml_path: Path) -> dict[str, object]:
    """fill_test.yaml の session キャッシュ."""
    with open(v460_fill_test_yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError("fill_test.yaml must deserialize to dict")
    return raw


@pytest.fixture
def v460_fill_test_yaml(v460_fill_test_yaml_base: dict[str, object]) -> dict[str, object]:
    """fill_test.yaml のテストごと deepcopy."""
    return copy.deepcopy(v460_fill_test_yaml_base)


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
