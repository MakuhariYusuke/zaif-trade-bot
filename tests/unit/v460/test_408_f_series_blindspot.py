"""408# F-Series + Blind Spot fixes tests.

Tests for:
- F6: OOS best-checkpoint in _train_with_checkpoints
- F4: Default value alignment (balance_penalty, consistency_penalty)
- B1: _record_action single-call guarantee
- B2: BPC else-branch attribute completeness
- B3: continuous_action_value parameter not shadowed
- B4: avg_gross_per_trade semantic correctness
- B5: train_val_split empty DataFrame guard
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scripts.v460.lib.tasks.sac_train import _train_with_checkpoints
from scripts.v460.lib.sac_common import cleanup_training_resources
from ztb.trading.environment.components.calculators.reward_calculator import (
    RewardCalculator,
)

_TRAIN_WITH_CHECKPOINTS_SIG = inspect.signature(_train_with_checkpoints)
_REWARD_CALCULATOR_INIT_SOURCE = inspect.getsource(RewardCalculator.__init__)
_REWARD_CALCULATOR_CALCULATE_SOURCE = inspect.getsource(RewardCalculator.calculate_reward)
_REWARD_CALCULATE_SIMPLE_SOURCE = inspect.getsource(RewardCalculator.calculate_reward_simple)


# ======================================================================
# F6: OOS Best-Checkpoint
# ======================================================================


class TestF6OOSBestCheckpoint:
    """F6: _train_with_checkpoints に OOS eval + best-model save を検証."""

    def test_extract_best_checkpoint_returns_best(self):
        """_extract_best_checkpoint は最後の is_best=1 を返す."""
        from scripts.v460.lib.tasks.sac_train import _extract_best_checkpoint

        metrics = [
            {"timesteps": 5000, "roi": 0.001, "oos_roi": 0.002, "is_best": 1},
            {"timesteps": 10000, "roi": 0.003, "oos_roi": 0.005, "is_best": 1},
            {"timesteps": 15000, "roi": 0.002, "oos_roi": 0.003, "is_best": 0},
            {"timesteps": 20000, "roi": 0.004, "oos_roi": 0.004, "is_best": 0},
        ]
        best = _extract_best_checkpoint(metrics)
        assert best is not None
        assert best["timesteps"] == 10000
        assert best["oos_roi"] == 0.005

    def test_extract_best_checkpoint_no_best(self):
        """OOS 無効の場合、None を返す."""
        from scripts.v460.lib.tasks.sac_train import _extract_best_checkpoint

        metrics = [
            {"timesteps": 5000, "roi": 0.001},
            {"timesteps": 10000, "roi": 0.003},
        ]
        assert _extract_best_checkpoint(metrics) is None

    def test_extract_best_checkpoint_empty(self):
        """空リストの場合、None を返す."""
        from scripts.v460.lib.tasks.sac_train import _extract_best_checkpoint

        assert _extract_best_checkpoint([]) is None

    def test_train_with_checkpoints_signature_has_oos_params(self):
        """_train_with_checkpoints が oos_eval_env と best_model_path を受け付ける."""
        params = _TRAIN_WITH_CHECKPOINTS_SIG.parameters
        assert "oos_eval_env" in params, "oos_eval_env parameter missing"
        assert "best_model_path" in params, "best_model_path parameter missing"
        # Both should be keyword-only
        assert params["oos_eval_env"].kind == inspect.Parameter.KEYWORD_ONLY
        assert params["best_model_path"].kind == inspect.Parameter.KEYWORD_ONLY


class TestSACCleanupTrainingResources:
    def test_cleanup_detaches_model_buffers_and_runs_gc(self) -> None:
        model = MagicMock()
        model.replay_buffer = object()
        model.env = object()
        model._vec_normalize_env = object()
        env = MagicMock()

        with patch("gc.collect", return_value=7) as mock_gc:
            cleanup_training_resources(models=[model], envs=[env], dataframes=[pd.DataFrame({"x": [1]})])

        assert model.replay_buffer is None
        assert model.env is None
        assert model._vec_normalize_env is None
        env.close.assert_called_once()
        mock_gc.assert_called_once()

    def test_cleanup_ignores_missing_torch(self) -> None:
        model = MagicMock()

        with patch.dict("sys.modules", {"torch": None}), patch("gc.collect", return_value=3):
            cleanup_training_resources(models=[model], envs=[], dataframes=[])


# ======================================================================
# F4: Default Value Alignment
# ======================================================================


class TestF4DefaultAlignment:
    """F4: balance_penalty と consistency_penalty のデフォルト値が SSOT に統一されていること."""

    def test_reward_settings_balance_penalty_default(self):
        """RewardSettings.balance_penalty のデフォルトは 0.1."""
        from ztb.trading.environment.utils.config import RewardSettings

        rs = RewardSettings()
        assert rs.balance_penalty == 0.1

    def test_reward_settings_consistency_penalty_default(self):
        """RewardSettings.consistency_penalty のデフォルトは 0.0."""
        from ztb.trading.environment.utils.config import RewardSettings

        rs = RewardSettings()
        assert rs.consistency_penalty == 0.0

    def test_bpc_else_branch_balance_penalty_aligned(self):
        """BPC の else-branch の balance_penalty_value が RS (0.1) に一致."""
        from ztb.trading.environment.components.behavioral_penalty_calculator import (
            BehavioralPenaltyCalculator,
        )
        from ztb.trading.environment.utils.config import EnvironmentConfig

        # reward_settings=None で else-branch に入る
        config = EnvironmentConfig()
        config.reward_settings = None  # type: ignore[assignment]
        bpc = BehavioralPenaltyCalculator(config=config)
        assert bpc.balance_penalty_value == 0.1

    def test_bpc_else_branch_consistency_penalty_aligned(self):
        """BPC の else-branch の penalty_value が RS (0.0) に一致."""
        from ztb.trading.environment.components.behavioral_penalty_calculator import (
            BehavioralPenaltyCalculator,
        )
        from ztb.trading.environment.utils.config import EnvironmentConfig

        config = EnvironmentConfig()
        config.reward_settings = None  # type: ignore[assignment]
        bpc = BehavioralPenaltyCalculator(config=config)
        assert bpc.penalty_value == 0.0

    def test_rc_balance_penalty_fallback_aligned(self):
        """RC の balance_penalty fallback が 0.1 (RS aligned)."""
        # Should NOT contain the old '1.0' default for balance_penalty
        assert '"behavior_optimization.balance_penalty", 1.0' not in _REWARD_CALCULATOR_INIT_SOURCE
        # Should contain the aligned '0.1'
        assert '"behavior_optimization.balance_penalty", 0.1' in _REWARD_CALCULATOR_INIT_SOURCE


# ======================================================================
# B1: _record_action Single-Call Guarantee
# ======================================================================


class TestB1RecordActionSingleCall:
    """B1: _record_action が calculate_reward() で1回だけ呼ばれること."""

    def test_stage_methods_do_not_call_record_action(self):
        """各 stage メソッドのソースコードに self._record_action 呼び出しがないこと."""
        # 二重呼び出しが問題だった8つの stage メソッド
        stage_methods = [
            "_calculate_forced_balance_reward",
            "_calculate_action_discovery_reward",
            "_calculate_smart_incentive_reward",
            "_calculate_balanced_transition_reward",
            "_calculate_trading_focused_reward",
            "_calculate_profit_optimized_reward",
            "_calculate_risk_management_reward",
            "_calculate_opportunity_cost_reward",
        ]

        for method_name in stage_methods:
            method = getattr(RewardCalculator, method_name, None)
            if method is None:
                continue
            source = inspect.getsource(method)
            # _record_action への直接呼び出しがないことを確認
            # (コメント中の参照は除外)
            lines = source.split("\n")
            call_lines = [
                line
                for line in lines
                if "self._record_action" in line
                and not line.strip().startswith("#")
                and not line.strip().startswith("//")
            ]
            assert (
                len(call_lines) == 0
            ), f"{method_name} still calls _record_action: {call_lines}"

    def test_calculate_reward_calls_record_action(self):
        """calculate_reward() 自体は _record_action を呼ぶ."""
        lines = _REWARD_CALCULATOR_CALCULATE_SOURCE.split("\n")
        call_lines = [
            line
            for line in lines
            if "self._record_action" in line and not line.strip().startswith("#")
        ]
        assert len(call_lines) == 1, (
            f"calculate_reward should call _record_action exactly once, "
            f"found {len(call_lines)}"
        )


# ======================================================================
# B2: BPC Else-Branch Attribute Completeness
# ======================================================================


class TestB2BPCElseBranchCompleteness:
    """B2: reward_settings=None で BPC を初期化しても AttributeError が出ないこと."""

    def test_all_attributes_exist_when_no_reward_settings(self):
        """reward_settings=None で全属性がアクセス可能."""
        from ztb.trading.environment.components.behavioral_penalty_calculator import (
            BehavioralPenaltyCalculator,
        )
        from ztb.trading.environment.utils.config import EnvironmentConfig

        config = EnvironmentConfig()
        config.reward_settings = None  # type: ignore[assignment]
        bpc = BehavioralPenaltyCalculator(config=config)

        # 408# B2 で追加した属性群
        required_attrs = [
            "consistency_min_actions",
            "trend_adjustment_enabled",
            "trend_adjustment_strength",
            "balance_shaping_enabled",
            "balance_shaping_value",
            "action_entropy_shaping_enabled",
            "action_entropy_shaping_value",
            "action_entropy_lookback",
            "skewness_penalty_enabled",
            "skewness_penalty_value",
            "skewness_penalty_tolerance",
            "skewness_lookback",
            "emergency_intervention_threshold",
            "emergency_intervention_penalty",
        ]

        for attr in required_attrs:
            assert hasattr(bpc, attr), f"Missing attribute: {attr}"
            # アクセスして例外が出ないことを確認
            _ = getattr(bpc, attr)

    def test_calculate_methods_dont_crash_without_reward_settings(self):
        """reward_settings=None で各ペナルティ計算メソッドが動作する."""
        from ztb.trading.environment.components.behavioral_penalty_calculator import (
            BehavioralPenaltyCalculator,
        )
        from ztb.trading.environment.utils.config import EnvironmentConfig

        config = EnvironmentConfig()
        config.reward_settings = None  # type: ignore[assignment]
        bpc = BehavioralPenaltyCalculator(config=config)

        # Record some actions first
        for action in [0, 1, 2, 0, 1]:
            bpc.record_action(action)

        # These should not raise AttributeError
        _ = bpc.calculate_consistency_penalty()
        _ = bpc.calculate_skewness_penalty()


# ======================================================================
# B3: continuous_action_value Parameter Not Shadowed
# ======================================================================


class TestB3ContinuousActionValueNoShadow:
    """B3: calculate_reward_simple のパラメータがシャドーイングされないこと."""

    def test_no_local_reassignment_in_source(self):
        """calculate_reward_simple 内で continuous_action_value が再代入されないこと."""
        lines = _REWARD_CALCULATE_SIMPLE_SOURCE.split("\n")
        # メソッド本体内の型アノテーション再代入を検出 (シグネチャとコメント行は除外)
        shadow_lines = [
            line
            for line in lines
            if "continuous_action_value:" in line
            and "float | None" in line
            and not line.strip().startswith("def ")
            and not line.strip().startswith("#")
            # メソッドパラメータの行 (末尾が , または ) で終わる) を除外
            and not line.rstrip().endswith(",")
            and not line.rstrip().endswith(")")
            and not line.rstrip().endswith("):")
        ]
        assert len(shadow_lines) == 0, (
            f"continuous_action_value is still shadowed: {shadow_lines}"
        )


# ======================================================================
# B4: avg_gross_per_trade Semantic Correctness
# ======================================================================


class TestB4AvgGrossPerTrade:
    """B4: avg_gross_per_trade は abs() を使わない正しい計算."""

    def test_avg_gross_per_trade_without_abs(self):
        """損失取引がある場合、abs()なしで平均を計算する."""
        from scripts.v460.lib.sac_common import _compute_g3_metrics

        # 取引 PnL: +100, -50, +30, -20 → avg = 60/4 = 15.0
        pnl_steps = [0.0] * 96 + [100.0, -50.0, 30.0, -20.0]
        portfolio_values = list(range(1000000, 1000100))
        reward_steps = [0.1] * 100
        total_trades = 4

        metrics = _compute_g3_metrics(portfolio_values, pnl_steps, reward_steps, total_trades)
        # abs() なし: (100 - 50 + 30 - 20) / 4 = 15.0
        assert metrics["avg_gross_per_trade"] == pytest.approx(15.0, abs=0.01)

    def test_avg_gross_per_trade_all_positive(self):
        """全て正の場合は abs() の有無にかかわらず同じ."""
        from scripts.v460.lib.sac_common import _compute_g3_metrics

        pnl_steps = [0.0] * 97 + [100.0, 50.0, 50.0]
        portfolio_values = list(range(1000000, 1000100))
        reward_steps = [0.1] * 100
        total_trades = 3

        metrics = _compute_g3_metrics(portfolio_values, pnl_steps, reward_steps, total_trades)
        assert metrics["avg_gross_per_trade"] == pytest.approx(200.0 / 3, abs=0.01)


# ======================================================================
# B5: train_val_split Empty DataFrame Guard
# ======================================================================


class TestB5TrainValSplitGuard:
    """B5: 空 DataFrame が生成される場合にエラーを投げる."""

    def test_val_ratio_zero_raises(self):
        """val_ratio=0.0 で val_df が空 → ValueError."""
        from scripts.v460.lib.sac_common import train_val_split

        df = pd.DataFrame({"a": range(100)})
        with pytest.raises(ValueError, match="empty val_df"):
            train_val_split(df, val_ratio=0.0)

    def test_valid_split_works(self):
        """正常な split は例外なし."""
        from scripts.v460.lib.sac_common import train_val_split

        df = pd.DataFrame({"a": range(100)})
        train_df, val_df = train_val_split(df, val_ratio=0.2)
        assert len(train_df) == 80
        assert len(val_df) == 20

    def test_single_row_raises(self):
        """1行のデータで split → どちらかが空になる → ValueError."""
        from scripts.v460.lib.sac_common import train_val_split

        df = pd.DataFrame({"a": [1]})
        # val_ratio=0.5 → split_idx=0 → train_df empty
        with pytest.raises(ValueError):
            train_val_split(df, val_ratio=0.5)
