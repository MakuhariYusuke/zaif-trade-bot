"""
SAC (Soft Actor-Critic) Algorithm Trainer.

SACアルゴリズム専用のトレーナー。
AlgorithmFactoryから生成されたSACAlgorithmを使用して訓練を実行する。
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import csv
from datetime import datetime

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback

from ztb.training.algorithms import AlgorithmFactory
from ztb.training.algorithms.sac import SACAlgorithm
from ztb.training.core.config_manager import ConfigManager
from ztb.trading.environment.environment import HeavyTradingEnv  # 🔧 Fixed import
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.training.callbacks.advanced_callbacks import EarlyStoppingCallback, BestModelCallback
from ztb.training.utils.training_logger import create_training_logger
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SACMetricsCallback(BaseCallback):
    """
    SAC訓練中のメトリクスを出力・記録するコールバック。
    
    訓練中のCritic Loss, Actor Loss, Entropy Coefficientを:
    - 標準出力とロガーに出力
    - CSVファイルに記録（確実な永続化）
    - TensorBoardに直接書き込み（可視化用）
    
    これにより、TensorBoardのデフォルト記録頻度に依存せず、
    100ステップごとに確実にメトリクスを記録できる。
    """
    
    def __init__(
        self, 
        log_interval: int = 100, 
        training_logger: Optional[Any] = None, 
        csv_path: Optional[Path] = None,
        verbose: int = 0
    ) -> None:
        """
        Args:
            log_interval: ログ出力間隔（ステップ数）
            training_logger: TrainingLoggerインスタンス（オプション）
            csv_path: CSVファイルの保存パス（オプション）
            verbose: 詳細度（0: エラーのみ、1: 情報、2: デバッグ）
        """
        super().__init__(verbose)
        self.log_interval = log_interval
        self.training_logger = training_logger
        self.csv_path = csv_path
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.csv_writer: Optional[Any] = None
        self.csv_file: Optional[Any] = None
        
        # CSVファイルを初期化
        if self.csv_path:
            self._init_csv_file()
    
    def _init_csv_file(self) -> None:
        """CSVファイルを初期化してヘッダーを書き込む"""
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            # ヘッダー行を書き込み（拡張メトリクス追加）
            self.csv_writer.writerow([
                'timestamp', 'step', 
                # 基本メトリクス
                'critic_loss', 'actor_loss', 'ent_coef', 'ent_coef_loss', 'learning_rate', 'fps',
                # 拡張メトリクス
                'episode_reward_mean', 'episode_reward_std', 'episode_length_mean',
                'q_value_mean', 'q_value_std', 'q_value_min', 'q_value_max',
                'total_episodes'
            ])
            self.csv_file.flush()
            logger.info(f"📝 Metrics CSV initialized: {self.csv_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize CSV file: {e}")
            self.csv_writer = None
            self.csv_file = None
    
    def _write_to_csv(self, step: int, metrics: Dict[str, float]) -> None:
        """CSVファイルにメトリクスを書き込む"""
        if self.csv_writer is None:
            return
        
        try:
            timestamp = datetime.now().isoformat()
            row = [
                timestamp,
                step,
                # 基本メトリクス
                metrics.get('critic_loss', ''),
                metrics.get('actor_loss', ''),
                metrics.get('ent_coef', ''),
                metrics.get('ent_coef_loss', ''),
                metrics.get('learning_rate', ''),
                metrics.get('fps', ''),
                # 拡張メトリクス
                metrics.get('episode_reward_mean', ''),
                metrics.get('episode_reward_std', ''),
                metrics.get('episode_length_mean', ''),
                metrics.get('q_value_mean', ''),
                metrics.get('q_value_std', ''),
                metrics.get('q_value_min', ''),
                metrics.get('q_value_max', ''),
                metrics.get('total_episodes', '')
            ]
            self.csv_writer.writerow(row)
            self.csv_file.flush()  # 即座にディスクに書き込み
        except Exception as e:
            logger.warning(f"Failed to write to CSV: {e}")
    
    def _write_to_tensorboard(self, step: int, metrics: Dict[str, float]) -> None:
        """TensorBoardに直接書き込む"""
        if not hasattr(self.model, 'logger') or self.model.logger is None:
            return
        
        try:
            # TensorBoard writerを取得
            tb_writer = self.model.logger.output_formats[0] if self.model.logger.output_formats else None
            
            if tb_writer and hasattr(tb_writer, 'writer'):
                # TensorBoardのSummaryWriterに直接書き込み
                for key, value in metrics.items():
                    tag = f"train/{key}" if not key.startswith('train/') else key
                    tb_writer.writer.add_scalar(tag, value, step)
                tb_writer.writer.flush()
        except Exception as e:
            # TensorBoard書き込みエラーは警告のみ（訓練は継続）
            if self.verbose > 1:
                logger.debug(f"TensorBoard write failed: {e}")
    
    def _on_step(self) -> bool:
        """
        各ステップで呼ばれるコールバック。
        
        log_intervalごとにメトリクスを:
        1. モデルから抽出
        2. 標準出力に表示
        3. CSVファイルに記録
        4. TensorBoardに直接書き込み
        
        Returns:
            True: 訓練を継続
        """
        # log_intervalごとにメトリクスを出力
        if self.n_calls % self.log_interval == 0:
            # メトリクスを収集
            metrics = {}
            total_steps = self.model._total_timesteps if hasattr(self.model, '_total_timesteps') else 0
            
            # SACモデルの内部メトリクスを確認
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                try:
                    name_to_value = self.model.logger.name_to_value
                    
                    # 基本メトリクスを抽出
                    for key in name_to_value:
                        if 'train/critic_loss' in key:
                            metrics['critic_loss'] = name_to_value[key]
                        elif 'train/actor_loss' in key:
                            metrics['actor_loss'] = name_to_value[key]
                        elif 'train/ent_coef' in key:
                            metrics['ent_coef'] = name_to_value[key]
                        elif 'train/ent_coef_loss' in key:
                            metrics['ent_coef_loss'] = name_to_value[key]
                        elif 'train/learning_rate' in key:
                            metrics['learning_rate'] = name_to_value[key]
                        elif 'time/fps' in key:
                            metrics['fps'] = name_to_value[key]
                        # 拡張メトリクス（エピソード報酬・長さ）
                        elif 'rollout/ep_rew_mean' in key:
                            metrics['episode_reward_mean'] = name_to_value[key]
                        elif 'rollout/ep_len_mean' in key:
                            metrics['episode_length_mean'] = name_to_value[key]
                except (AttributeError, KeyError):
                    pass
            
            # エピソード報酬の統計を計算（直近のエピソード）
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                try:
                    ep_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer if 'r' in ep_info]
                    ep_lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer if 'l' in ep_info]
                    
                    if ep_rewards:
                        metrics['episode_reward_mean'] = np.mean(ep_rewards)
                        metrics['episode_reward_std'] = np.std(ep_rewards)
                        metrics['total_episodes'] = len(ep_rewards)
                    
                    if ep_lengths:
                        metrics['episode_length_mean'] = np.mean(ep_lengths)
                except Exception as e:
                    if self.verbose > 1:
                        logger.debug(f"Failed to extract episode info: {e}")
            
            # Q値統計を取得（可能であれば）
            try:
                if hasattr(self.model, 'replay_buffer') and self.model.replay_buffer.size() > 128:
                    import torch
                    
                    # リプレイバッファから最近のサンプルを取得してQ値を推定
                    sample_size = min(128, self.model.replay_buffer.size())
                    replay_data = self.model.replay_buffer.sample(sample_size)
                    
                    if hasattr(self.model, 'critic'):
                        with torch.no_grad():  # 勾配計算不要
                            # Observationsとactionsをtensorに変換
                            obs_tensor = torch.as_tensor(replay_data.observations, device=self.model.device)
                            act_tensor = torch.as_tensor(replay_data.actions, device=self.model.device)
                            
                            # Q値を計算（両方のCriticの平均）
                            q_values_1 = self.model.critic.q1_forward(obs_tensor, act_tensor)
                            q_values_2 = self.model.critic.q2_forward(obs_tensor, act_tensor)
                            
                            q_values = (q_values_1 + q_values_2) / 2.0
                            q_values_np = q_values.cpu().numpy().flatten()
                            
                            metrics['q_value_mean'] = float(np.mean(q_values_np))
                            metrics['q_value_std'] = float(np.std(q_values_np))
                            metrics['q_value_min'] = float(np.min(q_values_np))
                            metrics['q_value_max'] = float(np.max(q_values_np))
            except Exception as e:
                # Q値取得は失敗しても継続
                if self.verbose > 1:
                    logger.debug(f"Failed to extract Q-values: {e}")
            
            # メトリクスが取得できなかった場合、モデルの内部状態から直接取得を試みる
            if not metrics and hasattr(self.model, '_last_obs'):
                try:
                    # SAC特有の属性から取得
                    if hasattr(self.model, 'ent_coef'):
                        metrics['ent_coef'] = float(self.model.ent_coef)
                    if hasattr(self.model, 'actor') and hasattr(self.model.actor, 'optimizer'):
                        # Learning rateを取得
                        for param_group in self.model.actor.optimizer.param_groups:
                            metrics['learning_rate'] = param_group['lr']
                            break
                except Exception:
                    pass
            
            # TrainingLoggerを使用して標準出力
            if self.training_logger:
                self.training_logger.print_training_progress(
                    current_step=self.n_calls,
                    total_steps=total_steps,
                    metrics=metrics
                )
            else:
                # フォールバック：基本的な出力
                output_parts = [f"Step {self.n_calls}/{total_steps}"]
                for key, value in metrics.items():
                    output_parts.append(f"{key}={value:.6f}")
                message = " | ".join(output_parts)
                print(message)
                
                if self.verbose > 0:
                    logger.info(message)
            
            # CSVファイルに書き込み
            self._write_to_csv(self.n_calls, metrics)
            
            # TensorBoardに直接書き込み
            self._write_to_tensorboard(self.n_calls, metrics)
        
        return True
    
    def _on_training_end(self) -> None:
        """
        訓練終了時のコールバック。
        
        最終的なメトリクスを出力し、CSVファイルをクローズする。
        """
        # 最終メトリクスを取得
        critic_loss = None
        actor_loss = None
        ent_coef = None
        
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            try:
                name_to_value = self.model.logger.name_to_value
                
                # メトリクスを抽出
                for key in name_to_value:
                    if 'train/critic_loss' in key:
                        critic_loss = name_to_value[key]
                    elif 'train/actor_loss' in key:
                        actor_loss = name_to_value[key]
                    elif 'train/ent_coef' in key:
                        ent_coef = name_to_value[key]
            except (AttributeError, KeyError):
                pass
        
        # 最終メトリクスを出力
        print("\n" + "="*80)
        print("SAC Training Completed - Final Metrics:")
        print("="*80)
        
        if critic_loss is not None:
            print(f"  Final Critic Loss: {critic_loss:.6f}")
        else:
            print(f"  Final Critic Loss: Not available")
        
        if actor_loss is not None:
            print(f"  Final Actor Loss: {actor_loss:.6f}")
        else:
            print(f"  Final Actor Loss: Not available")
        
        if ent_coef is not None:
            print(f"  Final Entropy Coef: {ent_coef:.6f}")
        else:
            print(f"  Final Entropy Coef: Not available")
        
        print("="*80 + "\n")
        
        # ロガーにも出力（メトリクスがある場合のみ）
        if critic_loss is not None or actor_loss is not None:
            logger.info(f"Training completed: critic_loss={critic_loss}, actor_loss={actor_loss}, ent_coef={ent_coef}")
        
        # CSVファイルをクローズ
        if self.csv_file:
            try:
                self.csv_file.close()
                logger.info(f"✅ Metrics CSV saved: {self.csv_path}")
            except Exception as e:
                logger.warning(f"Failed to close CSV file: {e}")


class SACAlgorithmTrainer:
    """
    SAC (Soft Actor-Critic) アルゴリズムのトレーナー。
    
    AlgorithmFactoryから生成されたSACAlgorithmを使用して、
    環境の作成、モデルの初期化、訓練、保存を実行する。
    """
    
    def __init__(self, config_manager: ConfigManager, progress_bar_enabled: bool = False):
        """
        SACAlgorithmTrainerを初期化。
        
        Args:
            config_manager: ConfigManager instance
            progress_bar_enabled: Whether progress bar is enabled
        """
        self.config_manager = config_manager
        self.progress_bar_enabled = progress_bar_enabled
        self.logger = get_logger(__name__)
    
    def train(self, unified_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        SAC訓練を実行。
        
        Args:
            unified_config: 統合設定辞書
            
        Returns:
            訓練結果（モデルパス、ログパス等）
        """
        # TrainingLoggerを作成
        model_name = unified_config.get("model_name", "sac_model")
        training_logger = create_training_logger("sac", model_name, verbose=True)
        
        self.logger.info("🚀 Starting SAC training")
        
        # 1. SAC設定を取得
        sac_config = (
            unified_config.get("sac_hyperparameters")
            or unified_config.get("sac_params")
            or {}
        )
        if sac_config:
            # Merge with defaults to ensure required keys are present
            sac_config = {**SACAlgorithm.get_default_config(), **dict(sac_config)}
        else:
            self.logger.warning("No SAC hyperparameters found in config, using defaults")
            sac_config = SACAlgorithm.get_default_config()
        
        # Expose resolved config for logging/debugging
        unified_config["sac_hyperparameters"] = sac_config
        if "sac_params" in unified_config:
            unified_config["sac_params"] = sac_config
        
        # 2. 環境を作成
        raw_env_config = unified_config.get("environment", {})
        if isinstance(raw_env_config, EnvironmentConfig):
            env_config = raw_env_config.as_dict()
        elif isinstance(raw_env_config, dict):
            env_config = dict(raw_env_config)
        else:
            env_config = {}
        
        # SAC requires continuous action space
        env_config["use_continuous_actions"] = True
        env_config["enable_action_masking"] = False  # SAC doesn't support action masking
        
        self.logger.info("🔧 Configured environment for SAC: continuous actions enabled")
        
        # ConfigManagerからデータを取得
        from ztb.utils.data_utils import load_csv_data_optimized
        data_path = unified_config.get("data_path", "btc_jpy_real_dataset.csv")
        
        df = load_csv_data_optimized(data_path)
        
        # HeavyTradingEnvを直接作成（env_configをconfigオブジェクトに変換）
        config_obj = EnvironmentConfig.from_dict(env_config)
        env = HeavyTradingEnv(df=df, config=config_obj)
        vec_env = DummyVecEnv([lambda: env])
        self.logger.info(f"✅ Environment created with {len(df)} timesteps (continuous action space)")
        self.logger.info(f"   Action space: {env.action_space}")
        
        # データ情報を収集
        total_timesteps = unified_config.get("total_timesteps", 100000)
        data_info = {
            "Data Rows": len(df),
            "Data Source": data_path,
            "Action Space": str(env.action_space),
            "Observation Dim": env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 'N/A'
        }
        
        # 訓練開始バナーを表示
        training_logger.print_training_start_banner(
            config=unified_config,
            total_timesteps=total_timesteps,
            data_info=data_info
        )
        
        # 3. SACAlgorithmを作成
        sac_algo = AlgorithmFactory.create("sac")
        self.logger.info(f"✅ Algorithm created: {sac_algo}")
        
        # 4. モデルを作成
        model_name = unified_config.get("model_name", "sac_model")
        session_id = unified_config.get("session_id", "sac_session")
        log_dir = Path("checkpoints") / session_id
        log_dir.mkdir(parents=True, exist_ok=True)
        
        model = sac_algo.create_model(
            env=vec_env,
            config=sac_config,
            tensorboard_log=str(log_dir)
        )
        self.logger.info(f"✅ Model created: {model}")
        
        # 5. コールバックを作成
        callbacks = []
        
        # CSVメトリクスファイルのパスを準備
        csv_metrics_path = log_dir / f"{model_name}_training_metrics.csv"
        
        # メトリクス出力コールバック（拡張メトリクス対応）
        metrics_log_interval = unified_config.get("metrics_log_interval", 100)
        metrics_callback = SACMetricsCallback(
            log_interval=metrics_log_interval,
            training_logger=training_logger,  # TrainingLoggerを渡す
            csv_path=csv_metrics_path,  # CSVパスを渡す
            verbose=1
        )
        callbacks.append(metrics_callback)
        
        self.logger.info(f"📊 Metrics will be logged to: {csv_metrics_path}")
        
        # Early Stopping機能（オプション）
        if unified_config.get("enable_early_stopping", False):
            early_stopping_config = unified_config.get("early_stopping", {})
            early_stopping_callback = EarlyStoppingCallback(
                metric_name=early_stopping_config.get("metric", "critic_loss"),
                min_delta=early_stopping_config.get("min_delta", 0.0001),
                patience=early_stopping_config.get("patience", 5000),
                check_interval=early_stopping_config.get("check_interval", 1000),
                window_size=early_stopping_config.get("window_size", 1000),
                cv_threshold=early_stopping_config.get("cv_threshold", 0.05),
                verbose=1
            )
            callbacks.append(early_stopping_callback)
            self.logger.info("🛑 Early Stopping enabled")
        
        # Best Model保存機能（オプション）
        if unified_config.get("enable_best_model_saving", False):
            best_model_config = unified_config.get("best_model", {})
            best_model_callback = BestModelCallback(
                save_path=log_dir / "best_models",
                model_name=model_name,
                metric_name=best_model_config.get("metric", "critic_loss"),
                mode=best_model_config.get("mode", "min"),
                check_interval=best_model_config.get("check_interval", 1000),
                verbose=1
            )
            callbacks.append(best_model_callback)
            self.logger.info("🏆 Best Model saving enabled")
        
        # チェックポイントコールバック
        checkpoint_interval = unified_config.get("checkpoint_interval", 10000)
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_interval,
            save_path=str(log_dir / "checkpoints"),
            name_prefix=model_name
        )
        callbacks.append(checkpoint_callback)
        
        callback_list = CallbackList(callbacks)
        
        # 6. 訓練実行
        total_timesteps = unified_config.get("total_timesteps", 100000)
        self.logger.info(f"🏃 Training for {total_timesteps} timesteps...")
        
        trained_model = sac_algo.train(
            model=model,
            total_timesteps=total_timesteps,
            callback=callback_list
        )
        
        # 7. モデル保存
        model_path = log_dir / f"{model_name}_final.zip"
        sac_algo.save(trained_model, str(model_path))
        self.logger.info(f"💾 Model saved to: {model_path}")
        
        # 8. 最終メトリクスを取得（TensorBoardから）
        final_metrics = {}
        try:
            # SACメトリクスコールバックから最終値を取得
            if hasattr(trained_model, 'logger') and trained_model.logger is not None:
                name_to_value = trained_model.logger.name_to_value
                for key in name_to_value:
                    if 'train/critic_loss' in key:
                        final_metrics['critic_loss'] = name_to_value[key]
                    elif 'train/actor_loss' in key:
                        final_metrics['actor_loss'] = name_to_value[key]
                    elif 'train/ent_coef' in key:
                        final_metrics['ent_coef'] = name_to_value[key]
        except Exception as e:
            self.logger.warning(f"Could not extract final metrics: {e}")
        
        # 9. 結果を返す
        result = {
            "model_path": str(model_path),
            "log_path": str(log_dir),
            "total_timesteps": total_timesteps,
            "algorithm": "sac",
            "success": True
        }
        
        # 訓練完了バナーを表示
        training_logger.print_training_complete_banner(result, final_metrics)
        
        # メトリクスをJSONで保存（最適化スクリプト用）
        if final_metrics:
            metrics_path = log_dir / f"{model_name}_final_metrics.json"
            training_logger.save_metrics_json(final_metrics, metrics_path)
        
        self.logger.info("🎉 SAC training completed successfully!")
        return result
