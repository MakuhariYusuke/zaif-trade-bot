from __future__ import annotations

from unittest.mock import MagicMock, patch

from tests.unit.v460._reward_calculator_test_helpers import make_reward_calculator
from ztb.trading.constants import ACTION_BUY
from ztb.trading.environment.components.calculators.reward_component_tracking import (
    build_reward_components,
    extend_reward_components,
    set_reward_telemetry,
)


def test_build_reward_components_filters_none_and_preserves_stage() -> None:
    payload = build_reward_components(
        "risk_management",
        base_reward=1.0,
        unrealized_loss_penalty=None,
        note="kept",
    )

    assert payload == {
        "stage": "risk_management",
        "base_reward": 1.0,
        "note": "kept",
    }


def test_extend_reward_components_updates_existing_stage_payload() -> None:
    payload = build_reward_components("default", pnl_reward=1.0)
    extend_reward_components(payload, action_bonus=0.5, skipped=None)

    assert payload["stage"] == "default"
    assert payload["pnl_reward"] == 1.0
    assert payload["action_bonus"] == 0.5
    assert "skipped" not in payload


def test_extend_reward_components_keeps_existing_stage_when_enriching_payload() -> None:
    payload = build_reward_components("forced_balance", base_reward=0.0)

    extend_reward_components(
        payload,
        scaled_reward=0.25,
        action_bonus=0.1,
        ignored=None,
    )

    assert payload == {
        "stage": "forced_balance",
        "base_reward": 0.0,
        "scaled_reward": 0.25,
        "action_bonus": 0.1,
    }


def test_build_reward_components_converts_boolean_flags_for_simple_reward_payload() -> None:
    payload = build_reward_components(
        "simple_reward",
        hold_penalty_applied=True,
        trade_bonus_applied=False,
    )

    assert payload["stage"] == "simple_reward"
    assert payload["hold_penalty_applied"] == 1.0
    assert payload["trade_bonus_applied"] == 0.0


def test_set_reward_telemetry_preserves_stage_and_accepts_nonscalar_payload() -> None:
    payload: dict[str, object] = build_reward_components("default", pnl_reward=1.0)

    set_reward_telemetry(payload, "mtf_weights", {"1m": 0.7, "5m": 0.3})

    assert payload["stage"] == "default"
    assert payload["mtf_weights"] == {"1m": 0.7, "5m": 0.3}


def test_risk_management_preserves_pre_and_post_trading_rewards() -> None:
    reward = make_reward_calculator()
    reward.unrealized_loss_penalty_calculator.calculate = MagicMock(return_value=-0.2)

    with (
        patch.object(reward, "_calculate_base_reward", return_value=1.0),
        patch.object(reward, "_calculate_base_trading_reward", return_value=1.5),
    ):
        total = reward._calculate_risk_management_reward(
            action=ACTION_BUY,
            pnl=0.0,
            position=0.2,
            atr_normalised=0.1,
            portfolio_return=0.0,
            effective_max_position=1.0,
            current_price=100.0,
            atr=1.0,
            observation=None,
        )

    assert total == 1.3
    assert reward._last_reward_components["stage"] == "risk_management"
    assert reward._last_reward_components["base_reward"] == 1.0
    assert reward._last_reward_components["base_trading_reward"] == 1.5
    assert reward._last_reward_components["unrealized_loss_penalty"] == -0.2
    assert reward._last_reward_components["total_reward"] == 1.3
