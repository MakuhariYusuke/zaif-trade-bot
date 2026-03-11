#!/usr/bin/env python3
"""Tests for PPO/SAC action recording fixes in callbacks."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from ztb.training.unified_trainer.base.callbacks import TrainingProgressCallback


class TestActionRecordingFixes:
    @pytest.fixture
    def mock_trainer(self) -> Mock:
        trainer = Mock()
        trainer.policy = Mock()
        trainer.policy.action_space = Mock()
        trainer.policy.optimizer = Mock(param_groups=[{"lr": 0.001}])
        trainer.logger = Mock()
        trainer.logger.name_to_value = {}
        return trainer

    @pytest.fixture
    def callback(self, mock_trainer: Mock) -> TrainingProgressCallback:
        callback = TrainingProgressCallback(
            check_freq=100,
            verbose=1,
            trainer_ref=mock_trainer,
        )
        callback.model = mock_trainer
        callback.trainer = mock_trainer
        callback._log_progress = Mock()  # type: ignore[method-assign]
        return callback

    def test_ppo_discrete_action_recording(
        self,
        callback: TrainingProgressCallback,
        mock_trainer: Mock,
    ) -> None:
        mock_trainer.policy.action_space.n = 3
        callback.locals = {"actions": np.array([1])}

        callback._on_step()

        assert list(callback.discrete_actions) == [1]
        assert list(callback.continuous_actions) == [1.0]

    def test_sac_continuous_action_recording(
        self,
        callback: TrainingProgressCallback,
        mock_trainer: Mock,
    ) -> None:
        mock_trainer.policy.action_space.n = None
        callback.locals = {"actions": np.array([0.5])}

        callback._on_step()

        assert list(callback.continuous_actions) == [0.5]
        assert list(callback.discrete_actions) == [1]

    @pytest.mark.parametrize(
        ("action", "expected_continuous"),
        [
            (0, 0.0),
            (1, 1.0),
            (2, -1.0),
        ],
    )
    def test_ppo_action_mapping(
        self,
        callback: TrainingProgressCallback,
        mock_trainer: Mock,
        action: int,
        expected_continuous: float,
    ) -> None:
        mock_trainer.policy.action_space.n = 3
        callback.locals = {"actions": np.array([action])}

        callback._on_step()

        assert callback.discrete_actions[-1] == action
        assert callback.continuous_actions[-1] == expected_continuous

    @pytest.mark.parametrize(
        ("action", "expected_discrete"),
        [
            (-1.0, -1),
            (-0.5, -1),
            (0.0, 0),
            (0.5, 1),
            (1.0, 1),
        ],
    )
    def test_sac_action_conversion_boundaries(
        self,
        callback: TrainingProgressCallback,
        mock_trainer: Mock,
        action: float,
        expected_discrete: int,
    ) -> None:
        mock_trainer.policy.action_space.n = None
        callback.locals = {"actions": np.array([action])}

        callback._on_step()

        assert callback.continuous_actions[-1] == action
        assert callback.discrete_actions[-1] == expected_discrete

    def test_action_recording_with_none_actions(
        self,
        callback: TrainingProgressCallback,
        mock_trainer: Mock,
    ) -> None:
        mock_trainer.policy.action_space.n = None
        callback.locals = {"actions": None}

        callback._on_step()

        assert len(callback.discrete_actions) == 0
        assert len(callback.continuous_actions) == 0

    def test_mixed_algorithm_detection(
        self,
        callback: TrainingProgressCallback,
    ) -> None:
        ppo_trainer = Mock()
        ppo_trainer.policy = Mock()
        ppo_trainer.policy.action_space = Mock(n=3)
        ppo_trainer.policy.optimizer = Mock(param_groups=[{"lr": 0.001}])
        ppo_trainer.logger = Mock(name_to_value={})
        callback.model = ppo_trainer
        callback.trainer = ppo_trainer
        callback.locals = {"actions": np.array([1])}

        callback._on_step()

        assert callback.discrete_actions[-1] == 1
        assert callback.continuous_actions[-1] == 1.0

        sac_trainer = Mock()
        sac_trainer.policy = Mock()
        sac_trainer.policy.action_space = Mock()
        sac_trainer.policy.action_space.n = None
        sac_trainer.policy.optimizer = Mock(param_groups=[{"lr": 0.001}])
        sac_trainer.logger = Mock(name_to_value={})
        callback.model = sac_trainer
        callback.trainer = sac_trainer
        callback.discrete_actions.clear()
        callback.continuous_actions.clear()
        callback.locals = {"actions": np.array([0.5])}

        callback._on_step()

        assert callback.continuous_actions[-1] == 0.5
        assert callback.discrete_actions[-1] == 1
