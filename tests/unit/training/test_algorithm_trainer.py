"""
AlgorithmTrainer統合テスト。

AlgorithmTrainerがAlgorithmFactoryを正しく使用し、
PPOおよびSACアルゴリズムを適切に処理できることを検証する。
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from ztb.training.algorithms import AlgorithmFactory
from ztb.training.core.algorithm_trainer import AlgorithmTrainer
from ztb.training.core.config_manager import ConfigManager

# ========================================
# Fixture
# ========================================


@pytest.fixture
def mock_config_manager():
    """ConfigManagerのモックを作成。"""
    mock_cm = Mock(spec=ConfigManager)
    mock_cm.config = {
        "algorithm": "ppo",
        "ppo_hyperparameters": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
        },
    }
    return mock_cm


@pytest.fixture
def algorithm_trainer(mock_config_manager):
    """AlgorithmTrainerインスタンスを作成。"""
    return AlgorithmTrainer(mock_config_manager, progress_bar_enabled=False)


# ========================================
# 初期化テスト
# ========================================


class TestAlgorithmTrainerInitialization:
    """AlgorithmTrainer初期化のテスト。"""

    def test_initialization(self, mock_config_manager):
        """基本的な初期化のテスト。"""
        trainer = AlgorithmTrainer(mock_config_manager, progress_bar_enabled=True)

        assert trainer.config_manager is mock_config_manager
        assert trainer.progress_bar_enabled is True

        # Legacy trainersが初期化されているか確認
        assert trainer.base_ml_trainer is not None
        assert trainer.iterative_trainer is not None
        assert trainer.ensemble_trainer is not None
        assert trainer.curriculum_trainer is not None

    def test_initialization_default_progress_bar(self, mock_config_manager):
        """デフォルトのprogress_bar設定。"""
        trainer = AlgorithmTrainer(mock_config_manager)

        assert trainer.progress_bar_enabled is False


# ========================================
# AlgorithmFactory統合テスト
# ========================================


class TestAlgorithmFactoryIntegration:
    """AlgorithmFactoryとの統合テスト。"""

    @pytest.mark.skip(
        reason="PPOAlgorithmTrainer class no longer exists; now using AlgorithmFactory pattern"
    )
    def test_ppo_uses_algorithm_factory(self, algorithm_trainer, mock_config_manager):
        """PPO訓練でAlgorithmFactoryが使用されることを確認。"""
        unified_config = {
            "algorithm": "ppo",
            "ppo_hyperparameters": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
            },
        }

        # PPOAlgorithmTrainerをモック化
        with patch(
            "ztb.training.core.algorithm_trainer.PPOAlgorithmTrainer"
        ) as mock_ppo_trainer:
            mock_trainer_instance = MagicMock()
            mock_trainer_instance.train.return_value = {"status": "success"}
            mock_ppo_trainer.return_value = mock_trainer_instance

            # 訓練実行
            result = algorithm_trainer.train("ppo", unified_config)

            # PPOAlgorithmTrainerが呼ばれたか確認
            mock_ppo_trainer.assert_called_once_with(
                mock_config_manager, False
            )  # progress_bar_enabled
            mock_trainer_instance.train.assert_called_once_with(unified_config)
            assert result == {"status": "success"}

    def test_algorithm_factory_creates_ppo(self, algorithm_trainer):
        """AlgorithmFactory.create("ppo")が正しく動作することを確認。"""
        # AlgorithmFactoryがPPOを作成できるか確認
        ppo = AlgorithmFactory.create("ppo")
        assert ppo is not None
        assert ppo.algorithm_name == "ppo"

    def test_algorithm_factory_creates_sac(self, algorithm_trainer):
        """AlgorithmFactory.create("sac")が正しく動作することを確認。"""
        # AlgorithmFactoryがSACを作成できるか確認
        sac = AlgorithmFactory.create("sac")
        assert sac is not None
        assert sac.algorithm_name == "sac"

    @pytest.mark.skip(
        reason="PPOAlgorithmTrainer class no longer exists; now using AlgorithmFactory pattern"
    )
    def test_case_insensitive_algorithm_name(
        self, algorithm_trainer, mock_config_manager
    ):
        """アルゴリズム名が大文字小文字を区別しないことを確認。"""
        unified_config = {"algorithm": "ppo"}

        with patch(
            "ztb.training.core.algorithm_trainer.PPOAlgorithmTrainer"
        ) as mock_ppo_trainer:
            mock_trainer_instance = MagicMock()
            mock_trainer_instance.train.return_value = {"status": "success"}
            mock_ppo_trainer.return_value = mock_trainer_instance

            # 大文字で呼び出し
            result = algorithm_trainer.train("PPO", unified_config)
            assert result == {"status": "success"}

            # 小文字で呼び出し
            result = algorithm_trainer.train("ppo", unified_config)
            assert result == {"status": "success"}


# ========================================
# Legacy Trainersテスト
# ========================================


class TestLegacyTrainersIntegration:
    """Legacy trainersとの統合テスト。"""

    def test_base_ml_trainer(self, algorithm_trainer):
        """base_mlアルゴリズムでlegacy trainerが使用されることを確認。"""
        unified_config = {"algorithm": "base_ml"}

        with patch.object(algorithm_trainer.base_ml_trainer, "train") as mock_train:
            mock_train.return_value = {"status": "base_ml_success"}

            result = algorithm_trainer.train("base_ml", unified_config)

            mock_train.assert_called_once_with(unified_config)
            assert result == {"status": "base_ml_success"}

    def test_iterative_trainer(self, algorithm_trainer):
        """iterativeアルゴリズムでlegacy trainerが使用されることを確認。"""
        unified_config = {"algorithm": "iterative"}

        with patch.object(algorithm_trainer.iterative_trainer, "train") as mock_train:
            mock_train.return_value = {"status": "iterative_success"}

            result = algorithm_trainer.train("iterative", unified_config)

            mock_train.assert_called_once_with(unified_config)
            assert result == {"status": "iterative_success"}

    def test_ensemble_trainer(self, algorithm_trainer):
        """ensembleアルゴリズムでlegacy trainerが使用されることを確認。"""
        unified_config = {"algorithm": "ensemble"}

        with patch.object(algorithm_trainer.ensemble_trainer, "train") as mock_train:
            mock_train.return_value = {"status": "ensemble_success"}

            result = algorithm_trainer.train("ensemble", unified_config)

            mock_train.assert_called_once_with(unified_config)
            assert result == {"status": "ensemble_success"}

    def test_curriculum_trainer(self, algorithm_trainer):
        """curriculumアルゴリズムでlegacy trainerが使用されることを確認。"""
        unified_config = {"algorithm": "curriculum"}

        with patch.object(algorithm_trainer.curriculum_trainer, "train") as mock_train:
            mock_train.return_value = {"status": "curriculum_success"}

            result = algorithm_trainer.train("curriculum", unified_config)

            mock_train.assert_called_once_with(unified_config)
            assert result == {"status": "curriculum_success"}


# ========================================
# エラーハンドリングテスト
# ========================================


class TestAlgorithmTrainerErrorHandling:
    """エラーハンドリングのテスト。"""

    def test_unknown_algorithm(self, algorithm_trainer):
        """未知のアルゴリズムでエラーが発生することを確認。"""
        unified_config = {"algorithm": "unknown_algo"}

        with pytest.raises(ValueError, match="Unknown algorithm: unknown_algo"):
            algorithm_trainer.train("unknown_algo", unified_config)

    def test_empty_algorithm_name(self, algorithm_trainer):
        """空のアルゴリズム名でエラーが発生することを確認。"""
        unified_config = {"algorithm": ""}

        with pytest.raises(ValueError, match="Unknown algorithm"):
            algorithm_trainer.train("", unified_config)


# ========================================
# 統合シナリオテスト
# ========================================


class TestAlgorithmTrainerIntegrationScenarios:
    """実際の使用シナリオの統合テスト。"""

    @pytest.mark.skip(
        reason="PPOAlgorithmTrainer class no longer exists; now using AlgorithmFactory pattern"
    )
    def test_full_ppo_training_flow(self, mock_config_manager):
        """完全なPPO訓練フローのテスト。"""
        # ConfigManagerのモック設定
        mock_config_manager.config = {
            "algorithm": "ppo",
            "ppo_hyperparameters": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
            },
            "total_timesteps": 100000,
        }

        # AlgorithmTrainer作成
        trainer = AlgorithmTrainer(mock_config_manager, progress_bar_enabled=True)

        # Unified config
        unified_config = {
            "algorithm": "ppo",
            "ppo_hyperparameters": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
            },
            "total_timesteps": 100000,
        }

        # PPOAlgorithmTrainerをモック化
        with patch(
            "ztb.training.core.algorithm_trainer.PPOAlgorithmTrainer"
        ) as mock_ppo_trainer:
            mock_trainer_instance = MagicMock()
            mock_trainer_instance.train.return_value = {
                "status": "success",
                "model_path": "models/ppo_v394d.zip",
                "final_metrics": {"mean_reward": 100.5},
            }
            mock_ppo_trainer.return_value = mock_trainer_instance

            # 訓練実行
            result = trainer.train("ppo", unified_config)

            # 結果検証
            assert result["status"] == "success"
            assert "model_path" in result
            assert "final_metrics" in result
            assert mock_trainer_instance.train.called

    def test_algorithm_switching(self, algorithm_trainer):
        """複数のアルゴリズムを順番に実行できることを確認。"""
        configs = [
            {"algorithm": "base_ml"},
            {"algorithm": "iterative"},
            {"algorithm": "ensemble"},
        ]

        results = []

        with patch.object(
            algorithm_trainer.base_ml_trainer,
            "train",
            return_value={"status": "base_ml"},
        ), patch.object(
            algorithm_trainer.iterative_trainer,
            "train",
            return_value={"status": "iterative"},
        ), patch.object(
            algorithm_trainer.ensemble_trainer,
            "train",
            return_value={"status": "ensemble"},
        ):
            for config in configs:
                result = algorithm_trainer.train(config["algorithm"], config)
                results.append(result)

        # 各アルゴリズムが正しく実行されたか確認
        assert results[0]["status"] == "base_ml"
        assert results[1]["status"] == "iterative"
        assert results[2]["status"] == "ensemble"
