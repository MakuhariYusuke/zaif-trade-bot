"""124# SkipGate v3 テスト: GBM really_bad30 モデル + skip_unknown_sell ルール.

Tests:
  - 新モデル (skip_gate_rb30.pkl) のロード・評価
  - mode=as での P(really_bad) 判定
  - skip_sell_unknown_regime ルール
  - 設定更新の整合性
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.v460.ml.skip_gate import (
    SkipGate,
    SkipGateConfig,
    _BASE_FEATURE_COLS,
    build_features_from_market_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rb30_gate(
    *,
    predict_prob: float = 0.40,
    as_threshold: float = 0.50,
    buy_enabled: bool = True,
    sell_enabled: bool = True,
    adaptive_threshold: bool = False,
) -> SkipGate:
    """124# really_bad30 モデル互換の SkipGate を生成."""
    config = SkipGateConfig(
        mode="as",
        enabled=True,
        buy_enabled=buy_enabled,
        sell_enabled=sell_enabled,
        as_threshold=as_threshold,
        as_threshold_buy=as_threshold,
        as_threshold_sell=as_threshold,
        adaptive_threshold=adaptive_threshold,
    )
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array(
        [[1.0 - predict_prob, predict_prob]]
    )
    return SkipGate(
        model=MagicMock(),
        scaler=MagicMock(),
        feature_cols=list(_BASE_FEATURE_COLS),
        config=config,
        pipeline=mock_pipeline,
    )


def _make_full_features() -> dict[str, float]:
    """全 16 base 特徴量を含む辞書."""
    return {col: 0.0 for col in _BASE_FEATURE_COLS}


class _AdapterStub:
    async def get_recent_trades(self, symbol: str, *, limit: int = 200) -> list[object]:
        del symbol, limit
        return []

    async def get_orderbook(self, symbol: str, *, depth: int | None = None) -> None:
        del symbol, depth
        return None


def _make_bypassed_evaluator(config: "FillTestConfig") -> "SkipGateEvaluator":
    from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

    with patch(
        "scripts.v460.lib.skip_gate_evaluator.SkipGateEvaluator.__init__",
        lambda self, *a, **kw: None,
    ):
        evaluator = SkipGateEvaluator.__new__(SkipGateEvaluator)
        evaluator._config = config
        evaluator._skip_gate = MagicMock()
        evaluator._gate_buy = None
        evaluator._gate_sell = None
        evaluator._gate_path_buy = None
        evaluator._gate_path_sell = None
        evaluator._model_file_hash_buy = ""
        evaluator._model_file_hash_sell = ""
        evaluator._last_reload_check = 0.0
        evaluator._gate_path = None
        evaluator._model_file_hash = ""
        evaluator._project_root = Path("/tmp")
        return evaluator


# =====================================================================
# T1: really_bad30 classifier のスキップ判定
# =====================================================================


class TestReallyBad30Classifier:
    """124# GBM really_bad30 モデルのテスト."""

    def test_high_prob_skips(self) -> None:
        """P(really_bad) = 0.65 > threshold 0.50 → skip."""
        gate = _make_rb30_gate(predict_prob=0.65, as_threshold=0.50)
        decision = gate.evaluate(_make_full_features(), side="buy")
        assert decision.should_skip is True
        assert decision.as_probability == pytest.approx(0.65, abs=0.01)

    def test_low_prob_passes(self) -> None:
        """P(really_bad) = 0.30 < threshold 0.50 → pass."""
        gate = _make_rb30_gate(predict_prob=0.30, as_threshold=0.50)
        decision = gate.evaluate(_make_full_features(), side="buy")
        assert decision.should_skip is False
        assert decision.as_probability == pytest.approx(0.30, abs=0.01)

    def test_both_sides_enabled(self) -> None:
        """124# 逆選別なし: buy/sell 両方で判定有効."""
        gate = _make_rb30_gate(predict_prob=0.70, as_threshold=0.50)

        buy_dec = gate.evaluate(_make_full_features(), side="buy")
        assert buy_dec.should_skip is True

        sell_dec = gate.evaluate(_make_full_features(), side="sell")
        assert sell_dec.should_skip is True

    def test_sell_enabled_flag_works(self) -> None:
        """sell_enabled=True が反映される (118# A3 からの変更)."""
        gate = _make_rb30_gate(
            predict_prob=0.70, as_threshold=0.50,
            sell_enabled=True,
        )
        decision = gate.evaluate(_make_full_features(), side="sell")
        assert decision.should_skip is True
        assert decision.reason != "sell_gate_disabled"

    def test_boundary_threshold(self) -> None:
        """P(really_bad) == threshold → skip (>=)."""
        gate = _make_rb30_gate(predict_prob=0.50, as_threshold=0.50)
        decision = gate.evaluate(_make_full_features(), side="buy")
        assert decision.should_skip is True

    def test_features_used_count(self) -> None:
        """全 16 特徴量が使用される."""
        gate = _make_rb30_gate(predict_prob=0.30)
        decision = gate.evaluate(_make_full_features(), side="buy")
        assert decision.features_used == 16


# =====================================================================
# T2: モデル save/load round-trip
# =====================================================================


class TestRB30SaveLoad:
    """124# really_bad30 モデルの保存・読み込み."""

    def test_save_load_roundtrip(self) -> None:
        """save → load で同一の設定・特徴量が復元される."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        model = LogisticRegression(max_iter=100)
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])
        # dummy fit
        X_dummy = np.random.randn(50, 16)
        y_dummy = np.random.randint(0, 2, 50)
        pipeline.fit(X_dummy, y_dummy)

        config = SkipGateConfig(
            mode="as",
            buy_enabled=True,
            sell_enabled=True,
        )
        gate = SkipGate(
            model=pipeline.named_steps["model"],
            scaler=pipeline.named_steps["scaler"],
            feature_cols=list(_BASE_FEATURE_COLS),
            config=config,
            metadata={"version": "v3_really_bad30", "target": "really_bad30"},
            pipeline=pipeline,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_rb30.pkl"
            gate.save(path)
            loaded = SkipGate.load(path)

        assert loaded.config.mode == "as"
        assert loaded.config.buy_enabled is True
        assert loaded.config.sell_enabled is True
        assert loaded.feature_cols == list(_BASE_FEATURE_COLS)
        assert loaded.metadata["version"] == "v3_really_bad30"

    def test_actual_model_loads(self) -> None:
        """実際のデプロイモデルがロードできる (存在する場合)."""
        model_path = Path("models/v460/skip_gate_rb30.pkl")
        if not model_path.exists():
            pytest.skip("Deployed model not found")

        gate = SkipGate.load(model_path)
        assert gate.config.mode == "as"
        assert gate.config.buy_enabled is True
        assert gate.config.sell_enabled is True
        assert len(gate.feature_cols) == 16

        # 実際の特徴量で evaluate
        features = build_features_from_market_state(
            side="buy",
            spread_jpy=2000.0,
            offset_ratio=0.05,
            regime="unknown",
            market_timestamp=1700000000.0,
        )
        decision = gate.evaluate(features, side="buy")
        assert decision.as_probability is not None
        assert 0.0 <= decision.as_probability <= 1.0
        assert decision.features_used >= 10


# =====================================================================
# T3: skip_sell_unknown_regime ルール
# =====================================================================


class TestSkipSellUnknownRegime:
    """124# Rule: unknown regime での sell スキップ."""

    @pytest.fixture()
    def config(self) -> "FillTestConfig":
        """最小 FillTestConfig."""
        from scripts.v460.lib.fill_config import FillTestConfig

        return FillTestConfig(
            skip_gate_enabled=True,
            skip_gate_model_path="models/v460/skip_gate_rb30.pkl",
            skip_sell_unknown_regime=True,
        )

    def test_rule_config_default_false(self) -> None:
        """skip_sell_unknown_regime のデフォルトは False."""
        from scripts.v460.lib.fill_config import FillTestConfig

        cfg = FillTestConfig()
        assert cfg.skip_sell_unknown_regime is False

    def test_rule_config_yaml_mapping(self) -> None:
        """YAML から skip_sell_unknown_regime が読み込まれる."""
        from scripts.v460.lib.fill_config import FillTestConfig

        yaml_data = {
            "skip_gate": {
                "enabled": True,
                "skip_sell_unknown_regime": True,
            },
        }
        cfg = FillTestConfig.from_yaml(yaml_data)
        assert cfg.skip_sell_unknown_regime is True

    def test_evaluator_skips_sell_unknown(self, config: "FillTestConfig") -> None:
        """sell + unknown regime → skip."""
        evaluator = _make_bypassed_evaluator(config)

        result = asyncio.run(
            evaluator.evaluate(
                side="sell",
                cycle_id="test_001",
                order_price=15000000.0,
                spread_at_order=2000.0,
                effective_offset_ratio=0.05,
                adapter=_AdapterStub(),
                symbol="btc_jpy",
                current_lot=0.001,
                run_id="test_run",
                git_sha=None,
                regime_value="unknown",
                last_imbalance=None,
                last_bid_depth=None,
                last_ask_depth=None,
                imbalance_enabled=False,
            )
        )
        assert result.skipped is True
        assert result.reason == "rule_skip_unknown_sell"
        assert result.model_used == "rule"
        assert result.early_return_record is not None
        assert result.early_return_record.cancel_reason == "skip_gate_rule_unknown_sell"
        assert result.early_return_record.skip_gate_reason == "rule_skip_unknown_sell"

    def test_normalize_recent_trades_accepts_dict_and_object(self) -> None:
        """recent_trades は dict/object 混在でも共通形式に正規化される."""
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

        class TradeObj:
            def __init__(self) -> None:
                self.ts = 101.0
                self.price = 502.0
                self.quantity = 3.0
                self.side = "sell"

        normalized = SkipGateEvaluator._normalize_recent_trades(
            [
                {
                    "timestamp": 100.0,
                    "price": 500.0,
                    "amount": 2.0,
                    "side": "buy",
                },
                TradeObj(),
                None,
            ],
            fallback_timestamp=999.0,
        )

        assert normalized == [
            {"ts": 100.0, "price": 500.0, "amount": 2.0, "side": "buy"},
            {"ts": 101.0, "price": 502.0, "amount": 3.0, "side": "sell"},
        ]

    def test_evaluator_passes_sell_trending(self, config: "FillTestConfig") -> None:
        """sell + trending regime → ルールは発火しない (ML判定に委譲)."""
        evaluator = _make_bypassed_evaluator(config)

        # build_features_from_market_state → mock
        mock_decision = SimpleNamespace(
            should_skip=False,
            predicted_pnl_bps=0.5,
            reason="pass",
            model_used="primary",
            as_probability=0.30,
            threshold_used=0.50,
            features_used=16,
        )
        evaluator._skip_gate.evaluate.return_value = mock_decision
        evaluator._skip_gate.config = SkipGateConfig(use_ob_features=False)

        result = asyncio.run(
            evaluator.evaluate(
                side="sell",
                cycle_id="test_002",
                order_price=15000000.0,
                spread_at_order=2000.0,
                effective_offset_ratio=0.05,
                adapter=_AdapterStub(),
                symbol="btc_jpy",
                current_lot=0.001,
                run_id="test_run",
                git_sha=None,
                regime_value="trending",
                last_imbalance=None,
                last_bid_depth=None,
                last_ask_depth=None,
                imbalance_enabled=False,
            )
        )
        assert result.skipped is False

    def test_evaluator_passes_buy_unknown(self, config: "FillTestConfig") -> None:
        """buy + unknown regime → ルールは発火しない (sell のみ)."""
        evaluator = _make_bypassed_evaluator(config)

        mock_decision = SimpleNamespace(
            should_skip=False,
            predicted_pnl_bps=0.5,
            reason="pass",
            model_used="primary",
            as_probability=0.30,
            threshold_used=0.50,
            features_used=16,
        )
        evaluator._skip_gate.evaluate.return_value = mock_decision
        evaluator._skip_gate.config = SkipGateConfig(use_ob_features=False)

        result = asyncio.run(
            evaluator.evaluate(
                side="buy",
                cycle_id="test_003",
                order_price=15000000.0,
                spread_at_order=2000.0,
                effective_offset_ratio=0.05,
                adapter=_AdapterStub(),
                symbol="btc_jpy",
                current_lot=0.001,
                run_id="test_run",
                git_sha=None,
                regime_value="unknown",
                last_imbalance=None,
                last_bid_depth=None,
                last_ask_depth=None,
                imbalance_enabled=False,
            )
        )
        assert result.skipped is False

    def test_rule_none_regime_triggers(self, config: "FillTestConfig") -> None:
        """sell + regime_value=None → unknown 扱いでルール発火."""
        from scripts.v460.lib.skip_gate_evaluator import SkipGateEvaluator

        with patch(
            "scripts.v460.lib.skip_gate_evaluator.SkipGateEvaluator.__init__",
            lambda self, *a, **kw: None,
        ):
            evaluator = SkipGateEvaluator.__new__(SkipGateEvaluator)
            evaluator._config = config
            evaluator._skip_gate = MagicMock()

        import asyncio

        result = asyncio.run(
            evaluator.evaluate(
                side="sell",
                cycle_id="test_004",
                order_price=15000000.0,
                spread_at_order=2000.0,
                effective_offset_ratio=0.05,
                adapter=MagicMock(),
                symbol="btc_jpy",
                current_lot=0.001,
                run_id="test_run",
                git_sha=None,
                regime_value=None,
                last_imbalance=None,
                last_bid_depth=None,
                last_ask_depth=None,
                imbalance_enabled=False,
            )
        )
        assert result.skipped is True
        assert result.reason == "rule_skip_unknown_sell"
