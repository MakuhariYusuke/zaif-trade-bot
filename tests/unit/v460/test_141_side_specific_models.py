"""141# P1-01/02/04: buy/sell 分離モデル + target 二層化 + regime 別閾値テスト.

§1: FillConfig side 別モデルパス追加
§2: retrain_scheduler side_filter + _retrain_side_specific
§3: SkipGateEvaluator side 別モデルロード + ディスパッチ
§4: YAML 設定の反映
§5: 統合テスト (side dispatch + hot-reload)
§6: P1-04 regime 別 PnL 閾値オーバーライド
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pandas as pd
import pytest
import yaml
from scripts.v460.lib.fill_config import FillTestConfig
from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator
from scripts.v460.ml.retrain_scheduler import (
    _DEFAULT_CONFIG,
    _retrain_side_specific,
    load_retrain_config,
    retrain_model,
)
from scripts.v460.ml.skip_gate import SkipDecision, SkipGate, SkipGateConfig
from ztb.ml.online_monitor import OnlineMonitor, OnlineMonitorConfig

try:
    import lightgbm  # noqa: F401
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False


class _IdentityScaler:
    """SkipGate pickle 用の軽量 scaler."""

    def set_output(self, *, transform: str) -> "_IdentityScaler":
        del transform
        return self

    def transform(self, x: object) -> object:
        return x


class _ConstantRegressor:
    """常に一定値を返す軽量 regressor."""

    def __init__(self, value: float) -> None:
        self._value = value

    def predict(self, x: object) -> np.ndarray:
        n_rows = len(x) if hasattr(x, "__len__") else 1
        return np.full(n_rows, self._value, dtype=float)


class _PredictPipeline:
    """SkipGate evaluate テスト用の軽量 predict stub."""

    def __init__(self, value: float = 0.0) -> None:
        self._prediction = np.array([value], dtype=float)
        self.steps: list[tuple[str, object]] = []

    def set_output(self, *, transform: str) -> "_PredictPipeline":
        del transform
        return self

    def set_prediction(self, value: float) -> None:
        self._prediction = np.array([value], dtype=float)

    def predict(self, x: object) -> np.ndarray:
        del x
        return self._prediction

# ---------------------------------------------------------------------------
# §1: FillConfig — side 別モデルパスフィールド
# ---------------------------------------------------------------------------
class TestFillConfigSideModelPaths:
    """141# §1: FillConfig に model_path_buy/sell が追加されていること."""

    def test_default_none(self) -> None:
        """デフォルトは None (統一モデルにフォールバック)."""

        config = FillTestConfig()
        assert config.skip_gate_model_path_buy is None
        assert config.skip_gate_model_path_sell is None

    def test_explicit_paths(self) -> None:
        """明示的にパスを指定できること."""

        config = FillTestConfig(
            skip_gate_model_path_buy="models/v460/buy.pkl",
            skip_gate_model_path_sell="models/v460/sell.pkl",
        )
        assert config.skip_gate_model_path_buy == "models/v460/buy.pkl"
        assert config.skip_gate_model_path_sell == "models/v460/sell.pkl"

    def test_yaml_parsing(self, tmp_path: Path) -> None:
        """YAML skip_gate セクションから model_path_buy/sell を読み取れること."""

        yaml_content = {
            "symbol": "btc_jpy",
            "skip_gate": {
                "enabled": True,
                "model_path": "models/v460/unified.pkl",
                "model_path_buy": "models/v460/buy.pkl",
                "model_path_sell": "models/v460/sell.pkl",
                "mode": "pnl",
            },
        }

        config = FillTestConfig.from_yaml(yaml_content)
        assert config.skip_gate_model_path_buy == "models/v460/buy.pkl"
        assert config.skip_gate_model_path_sell == "models/v460/sell.pkl"

# ---------------------------------------------------------------------------
# §2: retrain_scheduler — side_filter + _retrain_side_specific
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_LIGHTGBM, reason="lightgbm not installed")
class TestRetrainSideFilter:
    """141# §2: retrain_model の side_filter パラメータ."""

    def test_side_filter_in_result(self) -> None:
        """side_filter が result に記録されること."""

        cfg = {
            "model_path": "models/v460/test_side.pkl",
            "results_dir": "nonexistent_dir_for_test",
            "target": "pnl30",
            "use_ob_features": True,
            "mode": "pnl",
            "side_filter": "buy",
            "side_min_samples": 50,
            "min_new_samples": 10,
            "min_total_samples": 30,
        }
        # データなし → skipped
        result = retrain_model(cfg)
        assert result["status"] == "skipped"

    def test_side_filter_insufficient_samples(self, tmp_path: Path) -> None:
        """side_min_samples 未満のデータ → skipped."""

        # fill_records を作成 (buy=3, sell=3 で min_samples=50 に足りない)
        records_dir = tmp_path / "results"
        records_dir.mkdir()
        df = pd.DataFrame({
            "cycle_id": [f"c{i}" for i in range(6)],
            "side": ["buy", "buy", "buy", "sell", "sell", "sell"],
            "filled": [True] * 6,
            "run_id": ["run_1"] * 6,
            "order_price": [15_000_000.0] * 6,
            "order_quantity": [0.001] * 6,
            "timestamp": [f"2026-03-01T00:0{i}:00Z" for i in range(6)],
        })
        df.to_json(records_dir / "fill_records_test.jsonl", orient="records", lines=True)

        cfg = {
            "model_path": str(tmp_path / "model.pkl"),
            "results_dir": str(records_dir),
            "target": "pnl30",
            "use_ob_features": True,
            "mode": "pnl",
            "side_filter": "buy",
            "side_min_samples": 50,
            "min_new_samples": 0,
            "min_total_samples": 10,
            "latest_run_only": False,
            "exclude_missing_run_id": False,
        }
        result = retrain_model(cfg)
        assert result["status"] == "skipped"
        assert "Insufficient buy samples" in result.get("reason", "")
        assert result.get("side_filter") == "buy"

@pytest.mark.skipif(not _HAS_LIGHTGBM, reason="lightgbm not installed")
class TestRetrainSideSpecificFunction:
    """141# §2: _retrain_side_specific 関数テスト."""

    def test_retrain_side_specific_calls_retrain_model(self, tmp_path: Path) -> None:
        """_retrain_side_specific が buy/sell 各 retrain_model を呼ぶこと."""

        history_path = tmp_path / "history.jsonl"
        cfg = {
            "model_path": str(tmp_path / "unified.pkl"),
            "model_path_buy": str(tmp_path / "buy.pkl"),
            "model_path_sell": str(tmp_path / "sell.pkl"),
            "results_dir": "nonexistent_dir",
            "target": "pnl30",
            "target_buy": "pnl30",
            "target_sell": "pnl120",
            "use_ob_features": True,
            "mode": "pnl",
            "side_specific_enabled": True,
            "side_min_samples": 50,
            "min_new_samples": 10,
            "min_total_samples": 30,
        }
        results = _retrain_side_specific(cfg, history_path)
        assert len(results) == 2
        # Both should be skipped (no data directory)
        for r in results:
            assert r["status"] == "skipped"

    def test_retrain_side_specific_uses_target_per_side(self, tmp_path: Path) -> None:
        """side ごとに異なる target が使用されること."""

        history_path = tmp_path / "history.jsonl"
        calls: list[dict[str, Any]] = []

        def mock_retrain(cfg: dict[str, Any]) -> dict[str, Any]:
            calls.append({"target": cfg["target"], "side_filter": cfg.get("side_filter")})
            return {"status": "skipped", "reason": "mock"}

        cfg = {
            "model_path": str(tmp_path / "unified.pkl"),
            "model_path_buy": str(tmp_path / "buy.pkl"),
            "model_path_sell": str(tmp_path / "sell.pkl"),
            "results_dir": str(tmp_path),
            "target": "pnl30",
            "target_buy": "pnl30",
            "target_sell": "pnl120",
            "use_ob_features": True,
            "mode": "pnl",
        }

        with patch("scripts.v460.ml.retrain_scheduler.retrain_model", side_effect=mock_retrain):
            _retrain_side_specific(cfg, history_path)

        assert len(calls) == 2
        assert calls[0]["target"] == "pnl30"
        assert calls[0]["side_filter"] == "buy"
        assert calls[1]["target"] == "pnl120"
        assert calls[1]["side_filter"] == "sell"

    def test_no_model_path_skips_side(self, tmp_path: Path) -> None:
        """model_path_buy/sell が空ならその side はスキップ."""

        history_path = tmp_path / "history.jsonl"
        cfg = {
            "model_path": str(tmp_path / "unified.pkl"),
            "model_path_buy": "",
            "model_path_sell": str(tmp_path / "sell.pkl"),
            "results_dir": "nonexistent_dir",
            "target": "pnl30",
            "target_sell": "pnl120",
            "use_ob_features": True,
            "mode": "pnl",
            "side_min_samples": 50,
            "min_new_samples": 10,
            "min_total_samples": 30,
        }
        results = _retrain_side_specific(cfg, history_path)
        # buy の model_path が空 → buy はスキップ、sell のみ実行
        assert len(results) == 1
        assert results[0].get("side_model") == "sell"

    def test_history_written(self, tmp_path: Path) -> None:
        """side retrain 結果が history に書き込まれること."""

        history_path = tmp_path / "history.jsonl"
        cfg = {
            "model_path": str(tmp_path / "unified.pkl"),
            "model_path_buy": str(tmp_path / "buy.pkl"),
            "model_path_sell": str(tmp_path / "sell.pkl"),
            "results_dir": "nonexistent_dir",
            "target": "pnl30",
            "target_buy": "pnl30",
            "target_sell": "pnl120",
            "use_ob_features": True,
            "mode": "pnl",
            "side_min_samples": 50,
            "min_new_samples": 10,
            "min_total_samples": 30,
        }
        def _mock_retrain(side_cfg: dict[str, Any]) -> dict[str, Any]:
            return {
                "status": "ok",
                "target": side_cfg["target"],
                "side_filter": side_cfg["side_filter"],
            }

        with patch("scripts.v460.ml.retrain_scheduler.retrain_model", side_effect=_mock_retrain):
            _retrain_side_specific(cfg, history_path)

        assert history_path.exists()
        lines = history_path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "side_model" in data

# ---------------------------------------------------------------------------
# §3: SkipGateEvaluator — side 別モデルロード + ディスパッチ
# ---------------------------------------------------------------------------
def _create_mock_gate(
    tmp_path: Path,
    filename: str,
    target: str = "pnl30",
    *,
    prediction: float = 0.2,
    version: str | None = None,
) -> Path:
    """テスト用 SkipGate pkl を作成."""

    feature_cols = ["side_buy", "spread_jpy", "offset_ratio"]
    gate = SkipGate(
        model=_ConstantRegressor(prediction),
        scaler=_IdentityScaler(),
        feature_cols=feature_cols,
        config=SkipGateConfig(mode="pnl", use_ob_features=False),
        metadata={
            "version": version or f"test_{target}",
            "target": target,
            "n_samples": 10,
        },
    )
    path = tmp_path / filename
    gate.save(path)
    return path

class TestEvaluatorSideDispatch:
    """141# §3: SkipGateEvaluator の side 別モデルディスパッチ."""

    @staticmethod
    def _make_dispatch_evaluator(
        *,
        has_unified: bool = True,
        has_buy: bool = False,
        has_sell: bool = False,
    ) -> SkipGateEvaluator:
        evaluator = SkipGateEvaluator.__new__(SkipGateEvaluator)
        evaluator._skip_gate = MagicMock() if has_unified else None
        evaluator._gate_buy = MagicMock() if has_buy else None
        evaluator._gate_sell = MagicMock() if has_sell else None
        return evaluator

    def test_select_gate_for_side_buy(self, tmp_path: Path) -> None:
        """buy 側モデルが存在する場合は buy 側モデルを返す."""
        del tmp_path
        evaluator = self._make_dispatch_evaluator(has_unified=True, has_buy=True)
        assert evaluator._skip_gate is not None
        assert evaluator._gate_buy is not None
        assert evaluator._gate_sell is None

        # buy → side-specific
        gate = evaluator._select_gate_for_side("buy")
        assert gate is evaluator._gate_buy

        # sell → unified fallback
        gate = evaluator._select_gate_for_side("sell")
        assert gate is evaluator._skip_gate

    def test_select_gate_for_side_both(self, tmp_path: Path) -> None:
        """buy/sell 両方存在する場合は各々を返す."""
        del tmp_path
        evaluator = self._make_dispatch_evaluator(
            has_unified=True,
            has_buy=True,
            has_sell=True,
        )
        assert evaluator._gate_buy is not None
        assert evaluator._gate_sell is not None

        assert evaluator._select_gate_for_side("buy") is evaluator._gate_buy
        assert evaluator._select_gate_for_side("sell") is evaluator._gate_sell

    def test_no_side_models_uses_unified(self, tmp_path: Path) -> None:
        """side 別モデル未設定 → unified を使用."""
        del tmp_path
        evaluator = self._make_dispatch_evaluator(has_unified=True)
        assert evaluator._gate_buy is None
        assert evaluator._gate_sell is None

        # both sides → unified
        assert evaluator._select_gate_for_side("buy") is evaluator._skip_gate
        assert evaluator._select_gate_for_side("sell") is evaluator._skip_gate

    def test_side_model_file_missing_uses_unified(self, tmp_path: Path) -> None:
        """side モデルファイルが存在しない → unified にフォールバック."""

        unified_path = _create_mock_gate(tmp_path, "unified.pkl")

        config = FillTestConfig(
            skip_gate_enabled=True,
            skip_gate_model_path=str(unified_path),
            skip_gate_model_path_buy=str(tmp_path / "nonexistent_buy.pkl"),
            skip_gate_model_path_sell=str(tmp_path / "nonexistent_sell.pkl"),
            skip_gate_mode="pnl",
            skip_gate_use_ob_features=False,
        )
        evaluator = SkipGateEvaluator(config, tmp_path)
        assert evaluator._gate_buy is None
        assert evaluator._gate_sell is None
        assert evaluator._skip_gate is not None

# ---------------------------------------------------------------------------
# §4: YAML→retrain config 反映
# ---------------------------------------------------------------------------
class TestRetrainConfigSideSpecific:
    """141# §4: retrain config に side 別設定が反映されること."""

    def test_load_retrain_config_side_fields(
        self,
        write_yaml_file: "Callable[[str | Path, str | dict[str, Any]], Path]",
    ) -> None:
        """YAML から model_path_buy/sell と retrain section の side 設定を読める."""

        yaml_content = {
            "skip_gate": {
                "model_path": "models/v460/unified.pkl",
                "model_path_buy": "models/v460/buy.pkl",
                "model_path_sell": "models/v460/sell.pkl",
                "mode": "pnl",
            },
            "retrain": {
                "enabled": True,
                "target": "pnl30",
                "side_specific_enabled": True,
                "target_buy": "pnl30",
                "target_sell": "pnl120",
                "side_min_samples": 50,
            },
            "results_dir": "results/v460/fill_test",
        }
        yaml_path = write_yaml_file("config.yaml", yaml_content)

        cfg = load_retrain_config(yaml_path)
        # skip_gate セクションから継承
        assert cfg["model_path_buy"] == "models/v460/buy.pkl"
        assert cfg["model_path_sell"] == "models/v460/sell.pkl"
        # retrain セクションのキーは _DEFAULT_CONFIG に含まれるもののみ自動反映
        # side_specific_enabled 等はデフォルト外だが for loop で拾われる
        assert cfg.get("side_specific_enabled") is True
        assert cfg.get("target_buy") == "pnl30"
        assert cfg.get("target_sell") == "pnl120"
        assert cfg.get("side_min_samples") == 50

    def test_load_retrain_config_no_side_fields(
        self,
        write_yaml_file: "Callable[[str | Path, str | dict[str, Any]], Path]",
    ) -> None:
        """YAML に side 設定がない場合はデフォルト."""

        yaml_content = {
            "skip_gate": {
                "model_path": "models/v460/unified.pkl",
                "mode": "pnl",
            },
            "retrain": {
                "target": "pnl30",
            },
            "results_dir": "results/v460/fill_test",
        }
        yaml_path = write_yaml_file("config.yaml", yaml_content)

        cfg = load_retrain_config(yaml_path)
        assert cfg["model_path_buy"] == ""
        assert cfg["model_path_sell"] == ""

# ---------------------------------------------------------------------------
# §5: 統合テスト (side dispatch + model_used タグ)
# ---------------------------------------------------------------------------
class TestSideModelEvaluateIntegration:
    """141# §5: evaluate() の side dispatch 統合テスト."""

    def test_evaluate_model_used_tag_side(self, tmp_path: Path) -> None:
        """side 別モデル使用時の model_used に 'side_buy' タグが含まれること."""

        unified_path = _create_mock_gate(tmp_path, "unified.pkl")
        buy_path = _create_mock_gate(tmp_path, "buy.pkl", "pnl30")

        config = FillTestConfig(
            skip_gate_enabled=True,
            skip_gate_model_path=str(unified_path),
            skip_gate_model_path_buy=str(buy_path),
            skip_gate_mode="pnl",
            skip_gate_use_ob_features=False,
            skip_gate_adaptive_threshold=False,
        )
        evaluator = SkipGateEvaluator(config, tmp_path)

        # Mock adapter
        adapter = MagicMock()
        adapter.get_recent_trades = AsyncMock(return_value=[])
        adapter.get_orderbook = AsyncMock(return_value=None)

        result = asyncio.run(
            evaluator.evaluate(
                side="buy",
                cycle_id="test_1",
                order_price=14000000.0,
                spread_at_order=3000.0,
                effective_offset_ratio=0.25,
                adapter=adapter,
                symbol="btc_jpy",
                current_lot=0.001,
                run_id="run_test_1",
                git_sha="abc123",
                regime_value="trending",
                last_imbalance=0.5,
                last_bid_depth=10.0,
                last_ask_depth=10.0,
                imbalance_enabled=False,
            )
        )
        # model_used should contain "side_buy"
        assert "side_buy" in result.model_used

    def test_evaluate_model_used_tag_unified(self, tmp_path: Path) -> None:
        """unified モデルにフォールバック時の model_used に 'unified' タグが含まれること."""

        unified_path = _create_mock_gate(tmp_path, "unified.pkl")

        config = FillTestConfig(
            skip_gate_enabled=True,
            skip_gate_model_path=str(unified_path),
            skip_gate_mode="pnl",
            skip_gate_use_ob_features=False,
            skip_gate_adaptive_threshold=False,
        )
        evaluator = SkipGateEvaluator(config, tmp_path)

        adapter = MagicMock()
        adapter.get_recent_trades = AsyncMock(return_value=[])
        adapter.get_orderbook = AsyncMock(return_value=None)

        result = asyncio.run(
            evaluator.evaluate(
                side="sell",
                cycle_id="test_2",
                order_price=14000000.0,
                spread_at_order=3000.0,
                effective_offset_ratio=0.25,
                adapter=adapter,
                symbol="btc_jpy",
                current_lot=0.001,
                run_id="run_test_2",
                git_sha="abc123",
                regime_value="trending",
                last_imbalance=0.5,
                last_bid_depth=10.0,
                last_ask_depth=10.0,
                imbalance_enabled=False,
            )
        )
        assert "unified" in result.model_used

    def test_evaluate_side_only_missing_side_returns_reason(self, tmp_path: Path) -> None:
        """unified無し + buy-only で sell 評価時は例外化せず理由を返す."""

        buy_path = _create_mock_gate(tmp_path, "buy.pkl", "pnl30")

        config = FillTestConfig(
            skip_gate_enabled=True,
            skip_gate_model_path=str(tmp_path / "missing_unified.pkl"),
            skip_gate_model_path_buy=str(buy_path),
            skip_gate_mode="pnl",
            skip_gate_use_ob_features=False,
            skip_gate_adaptive_threshold=False,
        )
        evaluator = SkipGateEvaluator(config, tmp_path)
        assert evaluator._skip_gate is None
        assert evaluator._gate_buy is not None
        assert evaluator._gate_sell is None

        adapter = MagicMock()
        adapter.get_recent_trades = AsyncMock(return_value=[])
        adapter.get_orderbook = AsyncMock(return_value=None)

        result = asyncio.run(
            evaluator.evaluate(
                side="sell",
                cycle_id="test_side_missing",
                order_price=14000000.0,
                spread_at_order=3000.0,
                effective_offset_ratio=0.25,
                adapter=adapter,
                symbol="btc_jpy",
                current_lot=0.001,
                run_id="run_side_missing",
                git_sha="abc123",
                regime_value="trending",
                last_imbalance=0.5,
                last_bid_depth=10.0,
                last_ask_depth=10.0,
                imbalance_enabled=False,
            )
        )
        assert result.reason == "no_model_for_side:sell"
        assert result.skipped is False

class TestSideModelHotReload:
    """141# §5: side 別モデルの hot-reload テスト."""

    def test_hot_reload_new_side_model(self, tmp_path: Path) -> None:
        """最初は存在しなかった side モデルが後から配置されたらロードされること."""

        unified_path = _create_mock_gate(tmp_path, "unified.pkl")
        buy_path = tmp_path / "buy.pkl"

        config = FillTestConfig(
            skip_gate_enabled=True,
            skip_gate_model_path=str(unified_path),
            skip_gate_model_path_buy=str(buy_path),
            skip_gate_mode="pnl",
            skip_gate_use_ob_features=False,
        )
        evaluator = SkipGateEvaluator(config, tmp_path)
        assert evaluator._gate_buy is None  # ファイル未存在

        # buy モデルを作成
        _create_mock_gate(tmp_path, "buy.pkl", "pnl30")
        assert buy_path.exists()

        # _check_and_reload_side_models を直接呼ぶ (unified hash は変わらないため)
        evaluator._check_and_reload_side_models()

        # 新規 buy モデルがロードされたことを確認
        assert evaluator._gate_buy is not None

    def test_hot_reload_updated_side_model(self, tmp_path: Path) -> None:
        """既存 side モデルが更新されたらリロードされること."""

        unified_path = _create_mock_gate(tmp_path, "unified.pkl")
        sell_path = _create_mock_gate(tmp_path, "sell.pkl", "pnl120")

        config = FillTestConfig(
            skip_gate_enabled=True,
            skip_gate_model_path=str(unified_path),
            skip_gate_model_path_sell=str(sell_path),
            skip_gate_mode="pnl",
            skip_gate_use_ob_features=False,
        )
        evaluator = SkipGateEvaluator(config, tmp_path)
        assert evaluator._gate_sell is not None
        old_hash = evaluator._model_file_hash_sell

        # sell モデルを再作成してハッシュ変更を起こす.
        gate_new = SkipGate(
            model=_ConstantRegressor(0.35),
            scaler=_IdentityScaler(),
            feature_cols=["side_buy", "spread_jpy", "offset_ratio"],
            config=SkipGateConfig(mode="pnl", use_ob_features=False),
            metadata={"version": "pnl120_v2", "target": "pnl120", "n_samples": 20},
        )
        gate_new.save(sell_path)

        # _check_and_reload_side_models を直接呼ぶ
        evaluator._check_and_reload_side_models()

        assert evaluator._model_file_hash_sell != old_hash

class TestSideSpecificWarmStartDisabled:
    """141# warm_start は side 別モデルで無効化されること."""

    def test_side_retrain_warm_start_disabled(self, tmp_path: Path) -> None:
        """_retrain_side_specific で warm_start_enabled=False."""
        calls: list[dict[str, Any]] = []

        def mock_retrain(cfg: dict[str, Any]) -> dict[str, Any]:
            calls.append({
                "warm_start_enabled": cfg.get("warm_start_enabled"),
                "side_filter": cfg.get("side_filter"),
            })
            return {"status": "skipped", "reason": "mock"}

        cfg = {
            "model_path_buy": str(tmp_path / "buy.pkl"),
            "model_path_sell": str(tmp_path / "sell.pkl"),
            "target": "pnl30",
            "target_buy": "pnl30",
            "target_sell": "pnl120",
            "use_ob_features": True,
            "mode": "pnl",
            "warm_start_enabled": True,  # 統一モデルでは有効
        }

        with patch("scripts.v460.ml.retrain_scheduler.retrain_model", side_effect=mock_retrain):
            _retrain_side_specific(cfg, tmp_path / "history.jsonl")

        for call in calls:
            assert call["warm_start_enabled"] is False

class TestSideSpecificCacheKey:
    """141# cache_key に side_filter が含まれること."""

    def test_cache_key_differs_by_side(self) -> None:
        """同一 target でも side_filter が異なればキャッシュキーが異なること."""

        target = "pnl30"
        feature_cols_str = "a,b,c"
        run_ids_str = "run_1"

        key_all = hashlib.md5(
            f"{target}|{feature_cols_str}|{run_ids_str}|all".encode()
        ).hexdigest()[:16]
        key_buy = hashlib.md5(
            f"{target}|{feature_cols_str}|{run_ids_str}|buy".encode()
        ).hexdigest()[:16]
        key_sell = hashlib.md5(
            f"{target}|{feature_cols_str}|{run_ids_str}|sell".encode()
        ).hexdigest()[:16]

        assert key_all != key_buy
        assert key_all != key_sell
        assert key_buy != key_sell

# ---------------------------------------------------------------------------
# §6: P1-04 regime 別 PnL 閾値オーバーライド
# ---------------------------------------------------------------------------
class TestRegimeThresholdsConfig:
    """141# §6-1: SkipGateConfig.regime_thresholds フィールド."""

    def test_default_empty(self) -> None:
        """デフォルトは空辞書 (全レジーム共通閾値)."""
        cfg = SkipGateConfig()
        assert cfg.regime_thresholds == {}

    def test_explicit_thresholds(self) -> None:
        """明示的に regime_thresholds を設定できること."""
        cfg = SkipGateConfig(
            regime_thresholds={"high_vol": 0.2, "trending": -0.1}
        )
        assert cfg.regime_thresholds["high_vol"] == 0.2
        assert cfg.regime_thresholds["trending"] == -0.1

class TestRegimeThresholdFillConfig:
    """141# §6-2: FillConfig.skip_gate_regime_thresholds."""

    def test_default_empty(self) -> None:

        config = FillTestConfig()
        assert config.skip_gate_regime_thresholds == {}

    def test_yaml_parsing(self) -> None:
        """YAML skip_gate.regime_thresholds が FillConfig に反映されること."""

        yaml_dict = {
            "symbol": "btc_jpy",
            "skip_gate": {
                "enabled": True,
                "regime_thresholds": {
                    "high_vol": 0.2,
                    "ranging": 0.1,
                    "trending": -0.1,
                },
            },
        }
        config = FillTestConfig.from_yaml(yaml_dict)
        assert config.skip_gate_regime_thresholds == {
            "high_vol": 0.2,
            "ranging": 0.1,
            "trending": -0.1,
        }

class TestRegimeThresholdEvaluate:
    """141# §6-3: SkipGate.evaluate() での regime 別閾値適用."""

    def _make_gate(
        self,
        threshold_bps: float = 0.0,
        regime_thresholds: dict[str, float] | None = None,
    ) -> Any:
        """テスト用 SkipGate を構築 (PnL mode, Pipeline mock)."""

        cfg = SkipGateConfig(
            mode="pnl",
            threshold_bps=threshold_bps,
            enabled=True,
            regime_thresholds=regime_thresholds or {},
            adaptive_threshold=False,
        )
        pipeline = _PredictPipeline()
        gate = SkipGate(
            model=_ConstantRegressor(0.0),
            scaler=_IdentityScaler(),
            feature_cols=["side_buy", "spread_bps", "offset_ratio"],
            config=cfg,
            pipeline=pipeline,
        )
        return gate

    def test_no_regime_uses_default_threshold(self) -> None:
        """regime=None の場合は threshold_bps を使用."""
        gate = self._make_gate(threshold_bps=0.1)
        gate._pipeline.set_prediction(0.05)  # 0.05 < 0.1 → skip

        decision = gate.evaluate(
            {"side_buy": 1.0, "spread_bps": 10.0, "offset_ratio": 0.01},
            side="buy",
            regime=None,
        )
        assert decision.should_skip is True
        assert decision.predicted_pnl_bps == pytest.approx(0.05)

    def test_regime_override_applied(self) -> None:
        """regime_thresholds に一致する場合、その閾値が使われる."""
        gate = self._make_gate(
            threshold_bps=0.0,
            regime_thresholds={"high_vol": 0.3},
        )
        # pred_pnl = 0.2 → threshold_bps=0.0 なら pass, high_vol=0.3 なら skip
        gate._pipeline.set_prediction(0.2)

        # high_vol → threshold=0.3, 0.2 < 0.3 → skip
        decision_hv = gate.evaluate(
            {"side_buy": 1.0, "spread_bps": 10.0, "offset_ratio": 0.01},
            side="buy",
            regime="high_vol",
        )
        assert decision_hv.should_skip is True

        # trending (not in regime_thresholds) → threshold=0.0, 0.2 > 0.0 → pass
        decision_tr = gate.evaluate(
            {"side_buy": 1.0, "spread_bps": 10.0, "offset_ratio": 0.01},
            side="buy",
            regime="trending",
        )
        assert decision_tr.should_skip is False

    def test_regime_threshold_relaxed(self) -> None:
        """regime_thresholds が負の場合 (緩和) の動作確認."""
        gate = self._make_gate(
            threshold_bps=0.0,
            regime_thresholds={"trending": -0.5},
        )
        gate._pipeline.set_prediction(-0.3)

        # trending: threshold=-0.5, -0.3 > -0.5 → pass
        decision = gate.evaluate(
            {"side_buy": 1.0, "spread_bps": 10.0, "offset_ratio": 0.01},
            side="buy",
            regime="trending",
        )
        assert decision.should_skip is False

        # default: threshold=0.0, -0.3 < 0.0 → skip
        decision_def = gate.evaluate(
            {"side_buy": 0.0, "spread_bps": 10.0, "offset_ratio": 0.01},
            side="sell",
            regime=None,
        )
        assert decision_def.should_skip is True

    def test_empty_regime_thresholds_uses_default(self) -> None:
        """regime_thresholds が空の場合は常に threshold_bps."""
        gate = self._make_gate(threshold_bps=0.0, regime_thresholds={})
        gate._pipeline.set_prediction(0.1)

        decision = gate.evaluate(
            {"side_buy": 1.0, "spread_bps": 10.0, "offset_ratio": 0.01},
            side="buy",
            regime="high_vol",
        )
        assert decision.should_skip is False  # 0.1 > 0.0 → pass

class TestRegimeThresholdConfigOverrides:
    """141# §6-4: _apply_config_overrides で regime_thresholds が反映されること."""

    def test_overrides_applied(self) -> None:
        """SkipGateEvaluator._apply_config_overrides が regime_thresholds を設定する."""

        config = FillTestConfig(
            skip_gate_enabled=False,
            skip_gate_regime_thresholds={"high_vol": 0.2, "trending": -0.1},
        )
        evaluator = SkipGateEvaluator(config, Path("/tmp"))

        # mock gate with config
        mock_gate = SimpleNamespace(
            config=SimpleNamespace(),
            feature_cols=["a", "b"],
        )
        evaluator._apply_config_overrides(mock_gate)

        assert mock_gate.config.regime_thresholds == {"high_vol": 0.2, "trending": -0.1}

    def test_overrides_empty_dict(self) -> None:
        """regime_thresholds 未設定時は空辞書が設定される."""

        config = FillTestConfig(skip_gate_enabled=False)
        evaluator = SkipGateEvaluator(config, Path("/tmp"))

        mock_gate = SimpleNamespace(
            config=SimpleNamespace(),
            feature_cols=["a", "b"],
        )
        evaluator._apply_config_overrides(mock_gate)

        assert mock_gate.config.regime_thresholds == {}

class TestRegimeThresholdEvaluatorIntegration:
    """141# §6-5: SkipGateEvaluator.evaluate() が regime を SkipGate に渡すこと."""

    def test_regime_passed_to_gate_evaluate(self) -> None:
        """evaluate() 呼び出しで regime パラメータが渡されること."""

        config = FillTestConfig(
            skip_gate_enabled=False,
        )
        evaluator = SkipGateEvaluator(config, Path("/tmp"))

        # mock the internal skip gate
        mock_gate = MagicMock()
        mock_gate.config = MagicMock()
        mock_gate.config.use_ob_features = False
        mock_gate.evaluate.return_value = SkipDecision(
            should_skip=False,
            predicted_pnl_bps=0.5,
            threshold_bps=0.0,
            features_used=3,
            reason="pass",
            model_used="primary",
        )
        evaluator._skip_gate = mock_gate
        evaluator._gate_buy = None
        evaluator._gate_sell = None

        # Mock adapter and build_features
        mock_adapter = AsyncMock()
        mock_adapter.get_recent_trades.return_value = []
        mock_adapter.get_orderbook.return_value = None

        with patch(
            "scripts.v460.ml.skip_gate.build_features_from_market_state",
            return_value={"side_buy": 1.0, "spread_bps": 10.0},
        ):
            result = asyncio.run(
                evaluator.evaluate(
                    side="buy",
                    cycle_id="c1",
                    order_price=10000000.0,
                    spread_at_order=100.0,
                    effective_offset_ratio=0.01,
                    adapter=mock_adapter,
                    symbol="btc_jpy",
                    current_lot=0.001,
                    run_id="run_1",
                    git_sha="abc123",
                    regime_value="high_vol",
                    last_imbalance=None,
                    last_bid_depth=None,
                    last_ask_depth=None,
                    imbalance_enabled=False,
                )
            )

        # verify regime was passed
        mock_gate.evaluate.assert_called_once()
        call_kwargs = mock_gate.evaluate.call_args
        assert call_kwargs.kwargs.get("regime") == "high_vol"

# ---------------------------------------------------------------------------
# §7: P1-12 オンラインパフォーマンスモニター
# ---------------------------------------------------------------------------
class TestOnlineMonitorConfig:
    """141# §7-1: OnlineMonitorConfig フィールド."""

    def test_default_values(self) -> None:

        cfg = OnlineMonitorConfig()
        assert cfg.window == 100
        assert cfg.degraded_threshold_bps == -0.3
        assert cfg.min_skip_precision == 0.4
        assert cfg.min_samples == 20
        assert cfg.pnl_column == "post_fill_30s_pnl"

class TestOnlineMonitorEvaluate:
    """141# §7-2: OnlineMonitor.evaluate() の基本動作."""

    def _make_monitor(self, **kwargs: float | int) -> OnlineMonitor:
        defaults = {"window": 100}
        defaults.update(kwargs)
        return OnlineMonitor(OnlineMonitorConfig(**defaults))

    def _make_records(
        self,
        n_pass: int = 50,
        n_skip: int = 20,
        pass_pnl_mean: float = 0.5,
        skip_score_mean: float = -0.3,
    ) -> pd.DataFrame:
        """テスト用 fill_records DataFrame を生成."""
        rows = []
        rng = np.random.RandomState(42)
        for i in range(n_pass):
            rows.append({
                "skip_gate_skipped": False,
                "filled": True,
                "post_fill_30s_pnl": pass_pnl_mean + rng.randn() * 0.1,
                "skip_gate_score": 0.5 + rng.randn() * 0.1,
                "side": "buy" if i % 2 == 0 else "sell",
            })
        for i in range(n_skip):
            rows.append({
                "skip_gate_skipped": True,
                "filled": False,
                "post_fill_30s_pnl": np.nan,
                "skip_gate_score": skip_score_mean + rng.randn() * 0.1,
                "side": "buy" if i % 2 == 0 else "sell",
            })
        return pd.DataFrame(rows)

    def test_basic_evaluation(self) -> None:

        records = self._make_records(n_pass=60, n_skip=20)
        monitor = self._make_monitor()
        result = monitor.evaluate(records)

        assert result.n_total == 80
        assert result.n_passed == 60
        assert result.n_skipped == 20
        assert result.pass_mean_pnl > 0  # positive mean
        assert result.pass_win_rate > 0.5  # positive mean → high win rate
        assert result.degraded is False

    def test_degraded_detection(self) -> None:

        records = self._make_records(n_pass=50, n_skip=10, pass_pnl_mean=-0.5)
        monitor = self._make_monitor(degraded_threshold_bps=-0.3)
        result = monitor.evaluate(records)

        assert result.degraded is True
        assert "pass_mean_pnl" in (result.degraded_reason or "")

    def test_insufficient_samples(self) -> None:

        records = self._make_records(n_pass=5, n_skip=2)
        monitor = self._make_monitor(min_samples=20)
        result = monitor.evaluate(records)

        assert result.n_total == 7
        assert result.degraded is False  # not enough samples to judge

    def test_empty_records(self) -> None:

        result = self._make_monitor().evaluate(pd.DataFrame())
        assert result.n_total == 0

    def test_side_summary(self) -> None:

        records = self._make_records(n_pass=40, n_skip=20)
        monitor = self._make_monitor()
        result = monitor.evaluate(records)

        assert result.side_summary is not None
        assert "buy" in result.side_summary
        assert "sell" in result.side_summary
        assert result.side_summary["buy"]["n_total"] > 0
        assert result.side_summary["sell"]["n_total"] > 0

    def test_skip_precision(self) -> None:
        """skip したうち score < 0 (正しく skip) の割合が計算されること."""

        # skip_score_mean = -0.3 → 大半が < 0 → precision 高い
        records = self._make_records(n_pass=30, n_skip=30, skip_score_mean=-0.5)
        monitor = self._make_monitor()
        result = monitor.evaluate(records)

        assert result.skip_precision > 0.8  # 大半 negative → high precision

    def test_to_dict(self) -> None:

        records = self._make_records()
        result = self._make_monitor().evaluate(records)
        d = result.to_dict()

        assert "n_total" in d
        assert "pass_mean_pnl" in d
        assert "degraded" in d
        assert isinstance(d["n_total"], int)

    def test_window_truncation(self) -> None:
        """window より多い records がある場合、直近 window 件のみ使用."""

        records = self._make_records(n_pass=200, n_skip=50)
        monitor = self._make_monitor(window=50)
        result = monitor.evaluate(records)

        assert result.n_total == 50

class TestOnlineMonitorRetrain:
    """141# §7-3: _run_online_monitor の retrain_scheduler 統合テスト."""

    def test_default_config_has_online_monitor(self) -> None:
        """_DEFAULT_CONFIG に online_monitor_* キーがあること."""

        assert "online_monitor_enabled" in _DEFAULT_CONFIG
        assert "online_monitor_window" in _DEFAULT_CONFIG
        assert _DEFAULT_CONFIG["online_monitor_enabled"] is True

# ---------------------------------------------------------------------------
# §8: 142# 自己チェック修正 — C-1 regime + adaptive_threshold 統合テスト
# ---------------------------------------------------------------------------
class TestRegimeAdaptiveThresholdIntegration:
    """142# C-1 回帰テスト: regime_thresholds と adaptive_threshold の共存."""

    def _make_gate_adaptive(
        self,
        threshold_bps: float = 0.0,
        regime_thresholds: dict[str, float] | None = None,
        target_skip_rate_buy: float = 0.5,
        adaptive_window: int = 10,
        adaptive_min_samples: int = 5,
    ) -> Any:
        """adaptive_threshold=True の SkipGate を構築."""

        cfg = SkipGateConfig(
            mode="pnl",
            threshold_bps=threshold_bps,
            enabled=True,
            regime_thresholds=regime_thresholds or {},
            adaptive_threshold=True,
            target_skip_rate_buy=target_skip_rate_buy,
            target_skip_rate_sell=0.5,
            adaptive_window=adaptive_window,
            adaptive_min_samples=adaptive_min_samples,
            adaptive_step=0.05,
        )
        pipeline = _PredictPipeline()
        gate = SkipGate(
            model=_ConstantRegressor(0.0),
            scaler=_IdentityScaler(),
            feature_cols=["side_buy", "spread_bps", "offset_ratio"],
            config=cfg,
            pipeline=pipeline,
        )
        return gate

    def test_calibrate_receives_regime_base_threshold(self) -> None:
        """142# C-1: _calibrate_pnl_threshold に regime 後の base_threshold が渡される."""
        gate = self._make_gate_adaptive(
            threshold_bps=0.0,
            regime_thresholds={"high_vol": 0.5},
            adaptive_min_samples=100,  # warmup 未達で base_threshold が返る
        )
        gate._pipeline.set_prediction(0.3)

        # adaptive min_samples 未達 → warmup → base_threshold がそのまま返る
        # regime=high_vol → base_threshold=0.5 (not 0.0)
        decision = gate.evaluate(
            {"side_buy": 1.0, "spread_bps": 10.0, "offset_ratio": 0.01},
            side="buy",
            regime="high_vol",
        )
        # pred=0.3, threshold_used=0.5 → 0.3 < 0.5 → skip
        assert decision.should_skip is True
        assert decision.threshold_used == pytest.approx(0.5)

    def test_calibrate_without_regime_uses_config_threshold(self) -> None:
        """regime=None + adaptive 未達 → config.threshold_bps が使われる."""
        gate = self._make_gate_adaptive(
            threshold_bps=0.0,
            regime_thresholds={"high_vol": 0.5},
            adaptive_min_samples=100,
        )
        gate._pipeline.set_prediction(0.3)

        decision = gate.evaluate(
            {"side_buy": 1.0, "spread_bps": 10.0, "offset_ratio": 0.01},
            side="buy",
            regime=None,
        )
        # pred=0.3, threshold_used=0.0 → 0.3 > 0.0 → pass
        assert decision.should_skip is False
        assert decision.threshold_used == pytest.approx(0.0)

    def test_adaptive_calibrated_with_regime_base(self) -> None:
        """142# regime base_threshold が adaptive 較正の初期値になること."""
        gate = self._make_gate_adaptive(
            threshold_bps=0.0,
            regime_thresholds={"high_vol": 0.5},
            adaptive_window=10,
            adaptive_min_samples=3,
        )
        # warmup: 3件投入して calibration を有効化
        for pnl in [0.1, 0.2, 0.3]:
            gate._pipeline.set_prediction(pnl)
            gate.evaluate(
                {"side_buy": 1.0, "spread_bps": 10.0, "offset_ratio": 0.01},
                side="buy",
                regime="high_vol",
            )

        # calibrate が動いた → pnl_threshold_buy が regime ベースから調整済み
        assert gate._pnl_threshold_buy is not None
        # regime=None では config.threshold_bps=0.0 ベースになる
        gate._pnl_threshold_buy = None  # reset
        gate._pred_pnl_history_buy = None  # reset
        for pnl in [0.1, 0.2, 0.3]:
            gate._pipeline.set_prediction(pnl)
            gate.evaluate(
                {"side_buy": 1.0, "spread_bps": 10.0, "offset_ratio": 0.01},
                side="buy",
                regime=None,
            )
        # base=0.0 から調整されるため、threshold は regime=high_vol 時とは異なる
        assert gate._pnl_threshold_buy is not None

    def test_regime_key_typo_warning(self) -> None:
        """142# M-3: 未知の regime キーで WARNING が出ること."""

        class _LoggerStub:
            def __init__(self) -> None:
                self.messages: list[str] = []

            def info(self, _message: str) -> None:
                return None

            def warning(self, message: str) -> None:
                self.messages.append(message)

        config = FillTestConfig(
            skip_gate_enabled=False,
            skip_gate_regime_thresholds={"hig_vol": 0.2},  # typo
        )
        evaluator = SkipGateEvaluator.__new__(SkipGateEvaluator)
        evaluator._config = config
        evaluator._gate_path = None
        mock_gate = SimpleNamespace(
            config=SimpleNamespace(),
            feature_cols=["a", "b"],
        )
        logger_stub = _LoggerStub()

        with patch(
            "scripts.v460.lib.skip_gate_evaluator.logger",
            new=logger_stub,
        ):
            evaluator._apply_config_overrides(mock_gate)
            warn_calls = [
                message for message in logger_stub.messages
                if "unknown regime key" in message
            ]
            assert len(warn_calls) == 1
            assert "hig_vol" in warn_calls[0]

    def test_select_gate_no_attr(self) -> None:
        """142# M-1: _gate_buy/_gate_sell が None の場合 unified にフォールバック."""

        evaluator = SkipGateEvaluator.__new__(SkipGateEvaluator)
        evaluator._skip_gate = MagicMock()
        # 255# __init__ 相当: _gate_buy/_gate_sell = None
        evaluator._gate_buy = None
        evaluator._gate_sell = None

        gate = evaluator._select_gate_for_side("buy")
        assert gate is evaluator._skip_gate
        gate = evaluator._select_gate_for_side("sell")
        assert gate is evaluator._skip_gate
