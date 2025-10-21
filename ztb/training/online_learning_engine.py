#!/usr/bin/env python3
"""
V433 Online Learning Engine
リアルタイム適応学習システム
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Tuple, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ztb.utils.logging_utils import get_logger

# Dummy MarketDataStream for testing
class MarketDataStream:
    async def start_stream(self):
        pass

    async def stop_stream(self):
        pass

    async def get_latest_data(self):
        # Return dummy market data
        return None
from ztb.training.adaptive_sac_core import AdaptiveSACCore, AdaptiveSACConfig
from ztb.optimization.unified_optimizer import UnifiedOptimizer, OptimizationConfig

logger = get_logger(__name__)

@dataclass
class OnlineLearningConfig:
    """オンライン学習設定"""
    # データストリーム設定
    stream_buffer_size: int = 10000
    data_update_interval: float = 1.0  # seconds
    max_data_age: timedelta = timedelta(minutes=5)

    # 学習設定
    learning_batch_size: int = 64
    learning_interval: int = 100  # steps
    experience_buffer_size: int = 50000
    mini_batch_updates: int = 10

    # 適応設定
    adaptation_threshold: float = 0.1
    drift_detection_window: int = 1000
    concept_drift_threshold: float = 0.05

    # パフォーマンス監視
    performance_check_interval: int = 300  # seconds
    retraining_trigger_threshold: float = 0.7
    emergency_retraining_threshold: float = 0.5

    # リソース管理
    max_concurrent_updates: int = 4
    memory_limit_mb: float = 1024.0
    cpu_limit_percent: float = 80.0

    # バックアップ設定
    auto_backup_interval: int = 3600  # seconds
    backup_retention_days: int = 7


@dataclass
class ExperienceTuple:
    """経験タプル"""
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningMetrics:
    """学習指標"""
    total_experiences: int = 0
    learning_steps: int = 0
    average_reward: float = 0.0
    learning_efficiency: float = 0.0
    adaptation_events: int = 0
    concept_drift_events: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class ConceptDriftDetector:
    """概念ドリフト検知器"""

    def __init__(self, window_size: int = 1000, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.reward_history = deque(maxlen=window_size)
        self.baseline_reward = 0.0
        self.drift_detected = False

    def update(self, reward: float) -> bool:
        """報酬を更新し、ドリフトを検知"""
        self.reward_history.append(reward)

        if len(self.reward_history) >= self.window_size // 2:
            current_avg = np.mean(list(self.reward_history)[-self.window_size//2:])
            baseline_avg = np.mean(list(self.reward_history)[:self.window_size//2])

            if baseline_avg > 0:
                drift_ratio = abs(current_avg - baseline_avg) / abs(baseline_avg)

                if drift_ratio > self.threshold and not self.drift_detected:
                    self.drift_detected = True
                    logger.warning(f"Concept drift detected: {drift_ratio:.4f} > {self.threshold}")
                    return True
                elif drift_ratio < self.threshold * 0.5:
                    self.drift_detected = False

        return False

    def reset_baseline(self):
        """ベースラインをリセット"""
        if len(self.reward_history) >= self.window_size // 2:
            self.baseline_reward = np.mean(list(self.reward_history)[:self.window_size//2])
        self.drift_detected = False


class OnlineExperienceBuffer:
    """オンライン経験バッファ"""

    def __init__(self, max_size: int = 50000, prioritized: bool = True):
        self.max_size = max_size
        self.prioritized = prioritized
        self.buffer: Deque[ExperienceTuple] = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size) if prioritized else None

        # インデックス管理
        self.next_idx = 0

    def add_experience(self, experience: ExperienceTuple, priority: float = 1.0):
        """経験を追加"""
        if len(self.buffer) >= self.max_size:
            self.buffer.popleft()
            if self.priorities:
                self.priorities.popleft()

        self.buffer.append(experience)
        if self.priorities:
            self.priorities.append(priority)

        self.next_idx = (self.next_idx + 1) % self.max_size

    def sample_batch(self, batch_size: int) -> List[ExperienceTuple]:
        """バッチをサンプリング"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        if self.prioritized and self.priorities:
            # 優先度に基づくサンプリング
            priorities = np.array(self.priorities)
            probabilities = priorities / np.sum(priorities)

            indices = np.random.choice(
                len(self.buffer),
                size=min(batch_size, len(self.buffer)),
                p=probabilities,
                replace=False
            )
        else:
            # ランダムサンプリング
            indices = np.random.choice(
                len(self.buffer),
                size=min(batch_size, len(self.buffer)),
                replace=False
            )

        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)

    def get_statistics(self) -> Dict[str, Any]:
        """バッファ統計を取得"""
        if not self.buffer:
            return {"size": 0}

        rewards = [exp.reward for exp in self.buffer]
        ages = [(datetime.now() - exp.timestamp).total_seconds() for exp in self.buffer]

        return {
            "size": len(self.buffer),
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "avg_age_seconds": np.mean(ages),
            "max_age_seconds": np.max(ages)
        }


class OnlineLearningEngine:
    """
    V433オンライン学習エンジン
    リアルタイムデータストリームからの継続学習
    """

    def __init__(self, config: OnlineLearningConfig, adaptive_sac: AdaptiveSACCore):
        self.config = config
        self.adaptive_sac = adaptive_sac
        self.logger = get_logger(__name__)

        # コンポーネントの初期化
        self.market_data_stream = MarketDataStream()
        self.experience_buffer = OnlineExperienceBuffer(
            max_size=config.experience_buffer_size,
            prioritized=True
        )
        self.concept_drift_detector = ConceptDriftDetector(
            window_size=config.drift_detection_window,
            threshold=config.concept_drift_threshold
        )
        self.unified_optimizer = UnifiedOptimizer(OptimizationConfig())

        # 学習状態
        self.is_learning = False
        self.learning_thread = None
        self.monitoring_thread = None
        self.backup_thread = None

        # パフォーマンス追跡
        self.learning_metrics = LearningMetrics()
        self.performance_history = deque(maxlen=100)

        # スレッドプール
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_updates)

        # 同期プリミティブ
        self.stop_event = threading.Event()
        self.data_lock = threading.Lock()
        self.learning_lock = threading.Lock()

    async def start_online_learning(self):
        """オンライン学習を開始"""
        self.logger.info("Starting online learning engine")
        self.is_learning = True
        self.stop_event.clear()

        # データストリームを開始
        await self.market_data_stream.start_stream()

        # 学習スレッドを開始
        self.learning_thread = threading.Thread(target=self._learning_loop)
        self.learning_thread.daemon = True
        self.learning_thread.start()

        # モニタリングスレッドを開始
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        # バックアップスレッドを開始
        self.backup_thread = threading.Thread(target=self._backup_loop)
        self.backup_thread.daemon = True
        self.backup_thread.start()

        # データ処理ループを開始
        await self._data_processing_loop()

    def stop_online_learning(self):
        """オンライン学習を停止"""
        self.logger.info("Stopping online learning engine")
        self.is_learning = False
        self.stop_event.set()

        # ストリームを停止
        asyncio.create_task(self.market_data_stream.stop_stream())

        # スレッドの終了を待機
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=10)

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)

        if self.backup_thread and self.backup_thread.is_alive():
            self.backup_thread.join(timeout=10)

        # エグゼキューターをシャットダウン
        self.executor.shutdown(wait=True)

    async def _data_processing_loop(self):
        """データ処理ループ"""
        self.logger.info("Starting data processing loop")

        while not self.stop_event.is_set():
            try:
                # 新しい市場データを取得
                market_data = await self.market_data_stream.get_latest_data()

                if market_data is not None:
                    # 経験を生成
                    experiences = await self._generate_experiences_from_data(market_data)

                    # 経験をバッファに追加
                    with self.data_lock:
                        for exp in experiences:
                            self.experience_buffer.add_experience(exp)
                            self.learning_metrics.total_experiences += 1

                            # 概念ドリフト検知
                            if self.concept_drift_detector.update(exp.reward):
                                await self._handle_concept_drift()

                # 処理間隔
                await asyncio.sleep(self.config.data_update_interval)

            except Exception as e:
                self.logger.error(f"Data processing error: {e}")
                await asyncio.sleep(5)

    async def _generate_experiences_from_data(self, market_data: pd.DataFrame) -> List[ExperienceTuple]:
        """市場データから経験を生成"""
        experiences = []

        try:
            # データの前処理
            processed_data = self._preprocess_market_data(market_data)

            # 特徴量エンジニアリング
            features = self._extract_features(processed_data)

            # モデルの予測を取得
            if self.adaptive_sac.sac_model:
                # 現在の観測
                current_obs = features.iloc[-1:].values

                # 行動の予測
                action, _ = self.adaptive_sac.sac_model.predict(current_obs, deterministic=False)

                # 報酬の計算（実際の取引結果に基づく）
                reward = self._calculate_reward_from_market_data(processed_data, action)

                # 次の観測
                if len(features) > 1:
                    next_obs = features.iloc[-2:-1].values
                else:
                    next_obs = current_obs

                # 完了フラグ
                done = len(processed_data) >= 100  # エピソード長に基づく

                # 経験タプルの作成
                experience = ExperienceTuple(
                    observation=current_obs.flatten(),
                    action=action.flatten() if hasattr(action, 'flatten') else action,
                    reward=float(reward),
                    next_observation=next_obs.flatten(),
                    done=bool(done),
                    timestamp=datetime.now(),
                    metadata={
                        "market_data_shape": processed_data.shape,
                        "features_shape": features.shape,
                        "data_timestamp": processed_data.index[-1] if len(processed_data) > 0 else None
                    }
                )

                experiences.append(experience)

        except Exception as e:
            self.logger.error(f"Experience generation failed: {e}")

        return experiences

    def _preprocess_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """市場データの前処理"""
        # 基本的な前処理
        processed = data.copy()

        # 欠損値処理
        processed = processed.fillna(method='forward').fillna(0)

        # 異常値処理
        numeric_columns = processed.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # IQR法による異常値除去
            Q1 = processed[col].quantile(0.25)
            Q3 = processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            processed[col] = processed[col].clip(lower_bound, upper_bound)

        return processed

    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """特徴量抽出"""
        features = pd.DataFrame(index=data.index)

        # 価格ベースの特徴量
        if 'close' in data.columns:
            # 移動平均
            features['sma_5'] = data['close'].rolling(5).mean()
            features['sma_20'] = data['close'].rolling(20).mean()

            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            features['macd'] = ema_12 - ema_26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()

        # 出来高ベースの特徴量
        if 'volume' in data.columns:
            features['volume_sma_5'] = data['volume'].rolling(5).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma_5']

        # 欠損値を埋める
        features = features.fillna(0)

        return features

    def _calculate_reward_from_market_data(self, data: pd.DataFrame, action: np.ndarray) -> float:
        """市場データと行動から報酬を計算"""
        try:
            if len(data) < 2:
                return 0.0

            # 価格変化に基づく報酬
            current_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-2]
            price_change = (current_price - prev_price) / prev_price

            # 行動の解釈（買い/売り/ホールド）
            if isinstance(action, np.ndarray) and len(action) > 0:
                action_value = action[0] if len(action) == 1 else action[0]
            else:
                action_value = 0.0

            # 報酬計算: 行動が価格変化の方向と一致すれば正の報酬
            reward = price_change * action_value * 100  # スケーリング

            # 取引コストのペナルティ
            transaction_cost = abs(action_value) * 0.001  # 0.1%の手数料
            reward -= transaction_cost

            return float(reward)

        except Exception as e:
            self.logger.error(f"Reward calculation failed: {e}")
            return 0.0

    def _learning_loop(self):
        """学習ループ"""
        self.logger.info("Starting learning loop")

        while not self.stop_event.is_set():
            try:
                # 学習タイミングのチェック
                if (len(self.experience_buffer) >= self.config.learning_batch_size and
                    self.learning_metrics.total_experiences % self.config.learning_interval == 0):

                    with self.learning_lock:
                        self._perform_learning_update()

                time.sleep(1)  # 1秒ごとにチェック

            except Exception as e:
                self.logger.error(f"Learning loop error: {e}")
                time.sleep(5)

    def _perform_learning_update(self):
        """学習更新を実行"""
        try:
            # バッチをサンプリング
            batch = self.experience_buffer.sample_batch(self.config.learning_batch_size)

            if len(batch) < self.config.learning_batch_size // 2:
                return  # 十分なデータがない

            # ミニバッチ学習
            for _ in range(self.config.mini_batch_updates):
                mini_batch = np.random.choice(batch, size=min(32, len(batch)), replace=False)

                # SACのオンライン学習
                for exp in mini_batch:
                    self.adaptive_sac.online_learn(
                        exp.observation,
                        exp.action,
                        exp.reward,
                        exp.next_observation,
                        exp.done
                    )

            self.learning_metrics.learning_steps += 1

            # 指標の更新
            rewards = [exp.reward for exp in batch]
            self.learning_metrics.average_reward = np.mean(rewards)

            self.logger.debug(f"Learning update completed: step {self.learning_metrics.learning_steps}, "
                            f"avg_reward={self.learning_metrics.average_reward:.4f}")

        except Exception as e:
            self.logger.error(f"Learning update failed: {e}")

    async def _handle_concept_drift(self):
        """概念ドリフトの処理"""
        self.logger.warning("Handling concept drift")

        self.learning_metrics.concept_drift_events += 1

        # 適応最適化の実行
        try:
            adaptation_result = self.unified_optimizer.adaptive_optimize(
                current_performance={"score": 0.5},
                market_regime="volatile"  # ドリフト時はvolatileと仮定
            )

            # ドリフト検知器のリセット
            self.concept_drift_detector.reset_baseline()

            # 緊急学習のトリガー
            await self._emergency_learning_update()

            self.logger.info("Concept drift adaptation completed")

        except Exception as e:
            self.logger.error(f"Concept drift handling failed: {e}")

    async def _emergency_learning_update(self):
        """緊急学習更新"""
        self.logger.info("Performing emergency learning update")

        try:
            # 大きなバッチで学習
            emergency_batch_size = min(len(self.experience_buffer), self.config.learning_batch_size * 2)
            batch = self.experience_buffer.sample_batch(emergency_batch_size)

            # 並列学習
            tasks = []
            for i in range(0, len(batch), 32):
                mini_batch = batch[i:i+32]
                task = self.executor.submit(self._process_emergency_batch, mini_batch)
                tasks.append(task)

            # タスクの完了を待機
            for task in tasks:
                task.result(timeout=30)

            self.logger.info(f"Emergency learning completed with {len(batch)} experiences")

        except Exception as e:
            self.logger.error(f"Emergency learning failed: {e}")

    def _process_emergency_batch(self, batch: List[ExperienceTuple]):
        """緊急バッチ処理"""
        for exp in batch:
            self.adaptive_sac.online_learn(
                exp.observation,
                exp.action,
                exp.reward,
                exp.next_observation,
                exp.done
            )

    def _monitoring_loop(self):
        """モニタリングループ"""
        while not self.stop_event.is_set():
            try:
                # パフォーマンスチェック
                if self.learning_metrics.total_experiences > 0:
                    self._check_performance()

                # リソース使用量チェック
                self._check_resources()

                time.sleep(self.config.performance_check_interval)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(30)

    def _check_performance(self):
        """パフォーマンスチェック"""
        try:
            # 学習効率の計算
            if self.learning_metrics.learning_steps > 0:
                efficiency = (self.learning_metrics.average_reward /
                            max(1, self.learning_metrics.learning_steps))
                self.learning_metrics.learning_efficiency = efficiency

            # パフォーマンス履歴に追加
            self.performance_history.append({
                "timestamp": datetime.now(),
                "metrics": self.learning_metrics.__dict__.copy()
            })

            # 再訓練トリガーのチェック
            if self.learning_metrics.learning_efficiency < self.config.emergency_retraining_threshold:
                self.logger.warning("Emergency retraining triggered due to low efficiency")
                # ここで再訓練をトリガー

        except Exception as e:
            self.logger.error(f"Performance check failed: {e}")

    def _check_resources(self):
        """リソース使用量チェック"""
        try:
            import psutil
            process = psutil.Process()

            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent(interval=1)

            if memory_mb > self.config.memory_limit_mb:
                self.logger.warning(f"Memory usage high: {memory_mb:.1f}MB > {self.config.memory_limit_mb}MB")

            if cpu_percent > self.config.cpu_limit_percent:
                self.logger.warning(f"CPU usage high: {cpu_percent:.1f}% > {self.config.cpu_limit_percent}%")

        except ImportError:
            pass  # psutilが利用できない場合
        except Exception as e:
            self.logger.error(f"Resource check failed: {e}")

    def _backup_loop(self):
        """バックアップループ"""
        while not self.stop_event.is_set():
            try:
                time.sleep(self.config.auto_backup_interval)

                # バックアップの実行
                self._perform_backup_sync()

            except Exception as e:
                self.logger.error(f"Backup loop error: {e}")

    async def _perform_backup(self):
        """バックアップ実行"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path("backups") / "online_learning"
            backup_dir.mkdir(parents=True, exist_ok=True)

            # 経験バッファのバックアップ
            buffer_backup = backup_dir / f"experience_buffer_{timestamp}.pkl"
            torch.save(list(self.experience_buffer.buffer), buffer_backup)

            # 学習状態のバックアップ
            state_backup = backup_dir / f"learning_state_{timestamp}.pkl"
            torch.save({
                "metrics": self.learning_metrics.__dict__,
                "performance_history": list(self.performance_history),
                "config": self.config.__dict__
            }, state_backup)

            # 古いバックアップの削除
            self._cleanup_old_backups(backup_dir)

            self.logger.info(f"Backup completed: {buffer_backup}, {state_backup}")

        except Exception as e:
            self.logger.error(f"Backup failed: {e}")

    def _perform_backup_sync(self):
        """同期バックアップ実行"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._perform_backup())
        finally:
            loop.close()

    def _cleanup_old_backups(self, backup_dir: Path):
        """古いバックアップの削除"""
        try:
            retention_period = timedelta(days=self.config.backup_retention_days)
            cutoff_time = datetime.now() - retention_period

            for file_path in backup_dir.glob("*.pkl"):
                if file_path.stat().st_mtime < cutoff_time.timestamp():
                    file_path.unlink()
                    self.logger.debug(f"Removed old backup: {file_path}")

        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {e}")

    def get_learning_status(self) -> Dict[str, Any]:
        """学習状態を取得"""
        return {
            "is_learning": self.is_learning,
            "metrics": self.learning_metrics.__dict__,
            "buffer_stats": self.experience_buffer.get_statistics(),
            "performance_history_size": len(self.performance_history),
            "concept_drift_events": self.learning_metrics.concept_drift_events,
            "adaptation_events": self.learning_metrics.adaptation_events
        }


def create_online_learning_engine(config: OnlineLearningConfig = None,
                                adaptive_sac: AdaptiveSACCore = None) -> OnlineLearningEngine:
    """OnlineLearningEngineのファクトリ関数"""
    if config is None:
        config = OnlineLearningConfig()

    if adaptive_sac is None:
        # デフォルトのAdaptiveSACCoreを作成
        sac_config = AdaptiveSACConfig()
        adaptive_sac = AdaptiveSACCore(sac_config, observation_dim=10, action_dim=3)

    return OnlineLearningEngine(config, adaptive_sac)


# 使用例
async def example_usage():
    """使用例"""
    # 設定の作成
    config = OnlineLearningConfig(
        stream_buffer_size=5000,
        learning_batch_size=32,
        experience_buffer_size=10000,
        adaptation_threshold=0.15
    )

    # AdaptiveSACの作成
    sac_config = AdaptiveSACConfig(enable_online_learning=True)
    adaptive_sac = AdaptiveSACCore(sac_config, observation_dim=15, action_dim=3)

    # オンライン学習エンジンの作成
    engine = create_online_learning_engine(config, adaptive_sac)

    try:
        # オンライン学習を開始
        await engine.start_online_learning()

        # 実行中のステータスを表示
        while True:
            status = engine.get_learning_status()
            print(f"Learning Status: {status}")
            await asyncio.sleep(10)

    except KeyboardInterrupt:
        print("Stopping online learning...")
    finally:
        engine.stop_online_learning()


if __name__ == "__main__":
    asyncio.run(example_usage())