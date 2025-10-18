""""""""""""

Unit tests for Online Learning SAC Trainer

"""Unit tests for Online Learning SAC Trainer



import unittest"""Unit tests for Online Learning SAC TrainerUnit tests for Online Learning SAC Trainer

from ztb.adaptation.online_learning.trainer import OnlineLearningSACTrainer

from ztb.adaptation.online_learning.config import OnlineLearningConfig



import unittestオンライン学習機能を統合したSACトレーナーのテストオンライン学習機能を統合したSACトレーナーのテスト

class TestOnlineLearningSACTrainer(unittest.TestCase):

from ztb.adaptation.online_learning.trainer import OnlineLearningSACTrainer

    def setUp(self):

        self.online_config = OnlineLearningConfig()from ztb.adaptation.online_learning.config import OnlineLearningConfig""""""

        self.sac_config = {'learning_rate': 0.001}

        self.env_config = {'observation_space': {'shape': (10,)}, 'action_space': {'n': 3}}

        self.trainer = OnlineLearningSACTrainer(

            online_config=self.online_config,

            sac_config=self.sac_config,

            env_config=self.env_configclass TestOnlineLearningSACTrainer(unittest.TestCase):

        )

import unittestimport unittest

    def test_initialization(self):

        self.assertIsInstance(self.trainer.online_config, OnlineLearningConfig)    def setUp(self):

        self.assertEqual(self.trainer.sac_config, self.sac_config)

        self.assertEqual(self.trainer.env_config, self.env_config)        self.online_config = OnlineLearningConfig()import torchimport torch

        self.assertFalse(self.trainer.is_online_learning_active)
        self.sac_config = {'learning_rate': 0.001}

        self.env_config = {'observation_space': {'shape': (10,)}, 'action_space': {'n': 3}}from unittest.mock import patch, MagicMockimport torch.nn as nn

        self.trainer = OnlineLearningSACTrainer(

            online_config=self.online_config,from datetime import datetimefrom unittest.mock import patch, MagicMock, Mock

            sac_config=self.sac_config,

            env_config=self.env_configfrom typing import Iterator, Dict, Any

        )

from ztb.adaptation.online_learning.trainer import OnlineLearningSACTrainerfrom datetime import datetime

    def test_initialization(self):

        """初期化テスト"""from ztb.adaptation.online_learning.config import OnlineLearningConfigimport threading

        self.assertIsInstance(self.trainer.online_config, OnlineLearningConfig)

        self.assertEqual(self.trainer.sac_config, self.sac_config)from ztb.adaptation.online_learning.types import DataBatchimport time

        self.assertEqual(self.trainer.env_config, self.env_config)

        self.assertFalse(self.trainer.is_online_learning_active)

from ztb.adaptation.online_learning.trainer import OnlineLearningSACTrainer

class MockDataStream:from ztb.adaptation.online_learning.config import OnlineLearningConfig

    """テスト用データストリーム"""from ztb.adaptation.online_learning.types import DataBatch



    def __init__(self, num_batches: int = 5):

        self.num_batches = num_batchesclass MockDataStream:

        self.current_batch = 0    """テスト用データストリーム"""



    def __iter__(self):    def __init__(self, num_batches: int = 5):

        return self        self.num_batches = num_batches

        self.current_batch = 0

    def __next__(self):

        if self.current_batch >= self.num_batches:    def __iter__(self):

            raise StopIteration        return self



        batch = DataBatch(    def __next__(self):

            features=torch.randn(16, 10),        if self.current_batch >= self.num_batches:

            targets=torch.randn(16, 1),            raise StopIteration

            weights=None,

            timestamps=[datetime.now()] * 16,        batch = DataBatch(

            batch_id=f"test_batch_{self.current_batch}"            features=torch.randn(16, 10),

        )            targets=torch.randn(16, 1),

        self.current_batch += 1            weights=None,

        return batch            timestamps=[datetime.now()] * 16,

            batch_id=f"test_batch_{self.current_batch}"

        )

class TestOnlineLearningSACTrainer(unittest.TestCase):        self.current_batch += 1

    """OnlineLearningSACTrainerのテスト"""        return batch



    def setUp(self):

        self.online_config = OnlineLearningConfig()class TestOnlineLearningSACTrainer(unittest.TestCase):

        self.online_config.batch_size = 16    """OnlineLearningSACTrainerのテスト"""

        self.online_config.max_memory_samples = 100

    def setUp(self):

        self.sac_config = {        self.online_config = OnlineLearningConfig()

            'learning_rate': 0.001,        self.online_config.batch_size = 16

            'batch_size': 64,        self.online_config.max_memory_samples = 100

            'gamma': 0.99,

            'tau': 0.005,        self.sac_config = {

            'alpha': 0.2            'learning_rate': 0.001,

        }            'batch_size': 64,

            'gamma': 0.99,

        self.env_config = {            'tau': 0.005,

            'observation_space': {'shape': (10,)},            'alpha': 0.2

            'action_space': {'n': 3}        }

        }

        self.env_config = {

        self.trainer = OnlineLearningSACTrainer(            'observation_space': {'shape': (10,)},

            online_config=self.online_config,            'action_space': {'n': 3}

            sac_config=self.sac_config,        }

            env_config=self.env_config

        )        self.trainer = OnlineLearningSACTrainer(

            online_config=self.online_config,

    def tearDown(self):            sac_config=self.sac_config,

        # オンライン学習がアクティブな場合は停止            env_config=self.env_config

        if self.trainer.is_online_learning_active:        )

            self.trainer.stop_online_learning()

    def tearDown(self):

    def test_initialization(self):        # オンライン学習がアクティブな場合は停止

        """初期化テスト"""        if self.trainer.is_online_learning_active:

        self.assertIsInstance(self.trainer.online_config, OnlineLearningConfig)            self.trainer.stop_online_learning()

        self.assertEqual(self.trainer.sac_config, self.sac_config)

        self.assertEqual(self.trainer.env_config, self.env_config)    def test_initialization(self):

        self.assertFalse(self.trainer.is_online_learning_active)        """初期化テスト"""

        self.assertIsNone(self.trainer.online_thread)        self.assertIsInstance(self.trainer.online_config, OnlineLearningConfig)

        self.assertIsNone(self.trainer.data_stream)        self.assertEqual(self.trainer.sac_config, self.sac_config)

        self.assertEqual(self.trainer.env_config, self.env_config)

    def test_start_online_learning(self):        self.assertFalse(self.trainer.is_online_learning_active)

        """オンライン学習開始テスト"""        self.assertIsNone(self.trainer.online_thread)

        data_stream = MockDataStream(num_batches=3)        self.assertIsNone(self.trainer.data_stream)



        # オンライン学習開始    def test_start_online_learning(self):

        self.trainer.start_online_learning(data_stream)        """オンライン学習開始テスト"""

        data_stream = MockDataStream(num_batches=3)

        # 状態確認

        self.assertTrue(self.trainer.is_online_learning_active)        # オンライン学習開始

        self.assertIsNotNone(self.trainer.online_thread)        self.trainer.start_online_learning(data_stream)



        # 学習停止        # 状態確認

        self.trainer.stop_online_learning()        self.assertTrue(self.trainer.is_online_learning_active)

        self.assertFalse(self.trainer.is_online_learning_active)        self.assertIsNotNone(self.trainer.online_thread)
        self.assertIsInstance(self.trainer.online_thread, threading.Thread)
        self.assertTrue(self.trainer.online_thread.is_alive())

        # スレッドがデーモンであることを確認
        self.assertTrue(self.trainer.online_thread.daemon)

        # 学習停止
        self.trainer.stop_online_learning()
        self.assertFalse(self.trainer.is_online_learning_active)

    def test_start_online_learning_already_active(self):
        """既にアクティブな場合の開始テスト"""
        data_stream = MockDataStream(num_batches=3)

        # 最初に開始
        self.trainer.start_online_learning(data_stream)
        self.assertTrue(self.trainer.is_online_learning_active)

        # 再度開始しようとする
        with patch('ztb.adaptation.online_learning.trainer.logger') as mock_logger:
            self.trainer.start_online_learning(data_stream)
            mock_logger.warning.assert_called_with("Online learning already active")

        # 学習停止
        self.trainer.stop_online_learning()

    def test_stop_online_learning(self):
        """オンライン学習停止テスト"""
        data_stream = MockDataStream(num_batches=3)

        # 開始してから停止
        self.trainer.start_online_learning(data_stream)
        self.assertTrue(self.trainer.is_online_learning_active)

        self.trainer.stop_online_learning()
        self.assertFalse(self.trainer.is_online_learning_active)

        # スレッドが終了するまで待機
        if self.trainer.online_thread:
            self.trainer.online_thread.join(timeout=5.0)

    def test_online_learning_worker_processing(self):
        """オンライン学習ワーカーのデータ処理テスト"""
        data_stream = MockDataStream(num_batches=2)

        # パイプラインのモック
        with patch.object(self.trainer.online_pipeline, 'start_streaming') as mock_start:
            with patch.object(self.trainer.online_pipeline, 'learning_state', new_callable=lambda: Mock()):
                # 学習開始
                self.trainer.start_online_learning(data_stream)

                # 少し待機して処理を開始
                time.sleep(0.1)

                # start_streamingが呼ばれたことを確認
                mock_start.assert_called_once_with(data_stream)

                # 学習停止
                self.trainer.stop_online_learning()

    def test_online_learning_worker_error_handling(self):
        """エラーハンドリングテスト"""
        # エラーを発生させるデータストリーム
        class ErrorDataStream:
            def __iter__(self):
                return self
            def __next__(self):
                raise RuntimeError("Test error")

        data_stream = ErrorDataStream()

        # ロガーをモックしてエラーがログに記録されることを確認
        with patch('ztb.adaptation.online_learning.trainer.logger') as mock_logger:
            self.trainer.start_online_learning(data_stream)
            time.sleep(0.1)  # エラーが発生するまで待機
            self.trainer.stop_online_learning()

            # エラーログが記録されたことを確認
            mock_logger.error.assert_called()

    def test_get_online_learning_status(self):
        """学習状態取得テスト"""
        # 初期状態
        status = self.trainer._get_online_learning_status()
        self.assertIn('is_active', status)
        self.assertIn('total_samples_processed', status)
        self.assertIn('current_loss', status)
        self.assertFalse(status['is_active'])

        # 学習開始後の状態
        data_stream = MockDataStream(num_batches=1)
        self.trainer.start_online_learning(data_stream)
        time.sleep(0.1)

        status = self.trainer._get_online_learning_status()
        self.assertTrue(status['is_active'])

        self.trainer.stop_online_learning()</content>
<parameter name="filePath">c:\Users\Admin\dev\zaif-trade-bot\ztb\tests\unit\training\test_online_learning_trainer.py