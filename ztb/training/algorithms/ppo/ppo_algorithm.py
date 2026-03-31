"""
PPOアルゴリズムのラッパー実装。

既存のPPOTrainerをBaseRLAlgorithmインターフェースに適合させる。
既存コードを最大限再利用し、段階的な移行を可能にする。

Example:
    >>> from ztb.training.algorithms import AlgorithmFactory
    >>> ppo = AlgorithmFactory.create("ppo")
    >>> model = ppo.create_model(env, config)
    >>> ppo.train(model, total_timesteps=100000)
"""

from typing import Any, Callable, cast

from sb3_contrib import MaskablePPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from ztb.training.algorithms.base_algorithm import BaseRLAlgorithm
from ztb.training.core.ppo_trainer import PPOTrainerAutoHalt, PPOTrainingConfig
from ztb.training.custom_ppo import CustomPPO
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class PPOAlgorithm(BaseRLAlgorithm):
    """
    PPO (Proximal Policy Optimization) アルゴリズムの実装。

    既存のPPOTrainerをラップし、BaseRLAlgorithmインターフェースを提供する。
    これにより、他のアルゴリズム（SAC, TD3等）と統一的に扱える。

    Attributes:
        _trainer: 既存のPPOTrainerインスタンス
        _model: 作成されたPPOモデル
    """

    def __init__(self, use_auto_halt: bool = False):
        """
        初期化。

        Args:
            use_auto_halt: PPOTrainerAutoHaltを使用するか（デフォルト: False）
        """
        self._use_auto_halt = use_auto_halt
        self._trainer: PPOTrainerAutoHalt | None = None
        self._model: BaseAlgorithm | None = None
        self._config: dict[str, Any] | None = None

    @property
    def algorithm_name(self) -> str:
        """アルゴリズム名: "ppo" """
        return "ppo"

    def create_model(
        self,
        env: VecEnv,
        config: dict[str, Any],
        tensorboard_log: str | None = None,
    ) -> BaseAlgorithm:
        """
        PPOモデルを作成。

        既存のPPOTrainerまたはPPOTrainerAutoHaltを使用してモデルを作成する。

        Args:
            env: 訓練環境（VecEnv）
            config: PPO設定（ppo_hyperparameters等を含む）
            tensorboard_log: TensorBoardログディレクトリ

        Returns:
            作成されたPPOモデル

        Example:
            >>> ppo = PPOAlgorithm()
            >>> model = ppo.create_model(
            ...     env=vec_env,
            ...     config={
            ...         "ppo_hyperparameters": {
            ...             "learning_rate": 0.0003,
            ...             "n_steps": 2048,
            ...             "batch_size": 64
            ...         }
            ...     }
            ... )
        """
        logger.info("Creating PPO model (use_auto_halt=%s)", self._use_auto_halt)
        self._config = dict(config)

        training_config = PPOTrainingConfig.from_dict(config)
        model_cls = CustomPPO if training_config.use_custom_ppo else MaskablePPO
        self._model = model_cls(
            "MlpPolicy",
            env,
            learning_rate=training_config.learning_rate,
            n_steps=training_config.n_steps,
            batch_size=training_config.batch_size,
            n_epochs=training_config.n_epochs,
            gamma=training_config.gamma,
            gae_lambda=training_config.gae_lambda,
            clip_range=training_config.clip_range,
            clip_range_vf=training_config.clip_range_vf,
            normalize_advantage=training_config.normalize_advantage,
            ent_coef=training_config.ent_coef,
            vf_coef=training_config.vf_coef,
            max_grad_norm=training_config.max_grad_norm,
            use_sde=training_config.use_sde,
            sde_sample_freq=training_config.sde_sample_freq,
            target_kl=training_config.target_kl,
            verbose=training_config.verbose,
            tensorboard_log=tensorboard_log,
        )
        model_name = getattr(model_cls, "__name__", model_cls.__class__.__name__)
        logger.info("✅ PPO model created successfully (%s)", model_name)
        return self._model

    def train(
        self,
        model: BaseAlgorithm,
        total_timesteps: int,
        callback: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> BaseAlgorithm:
        """
        PPOモデルを訓練。

        Args:
            model: 訓練するPPOモデル
            total_timesteps: 総ステップ数
            callback: コールバック関数
            **kwargs: その他のパラメータ

        Returns:
            訓練済みモデル

        Example:
            >>> ppo = PPOAlgorithm()
            >>> model = ppo.create_model(env, config)
            >>> trained_model = ppo.train(
            ...     model=model,
            ...     total_timesteps=100000,
            ...     callback=checkpoint_callback
            ... )
        """
        logger.info("Training PPO model for %s timesteps", total_timesteps)

        learnable_model = cast(Any, model)
        if not hasattr(learnable_model, "learn"):
            raise TypeError("PPO model must expose learn(...)")

        learn_kwargs: dict[str, Any] = {"total_timesteps": total_timesteps}
        if callback is not None:
            learn_kwargs["callback"] = callback
        learn_kwargs.update(kwargs)
        trained_model = cast(BaseAlgorithm, learnable_model.learn(**learn_kwargs))
        self._model = trained_model
        logger.info("✅ PPO training completed")
        return trained_model

    def get_default_config(self) -> dict[str, Any]:
        """
        PPOのデフォルト設定を取得。

        Returns:
            デフォルト設定の辞書

        Note:
            SACAlgorithmと統一したインターフェース。
            ハイパーパラメータのみを返す（ppo_hyperparametersキーは含まない）。
        """
        return {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "normalize_advantage": True,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "target_kl": None,
            "verbose": 1,
        }

    def validate_config(self, config: dict[str, Any]) -> bool:
        """
        PPO設定の妥当性を検証。

        Args:
            config: 検証する設定（PPOハイパーパラメータのみ、またはppo_hyperparametersを含む完全な設定）

        Returns:
            設定が妥当ならTrue

        Raises:
            ValueError: 必須パラメータが不足している場合

        Note:
            SACAlgorithmと同様のインターフェースに統一。
        """
        # ppo_hyperparametersが含まれている場合は取り出す
        if "ppo_hyperparameters" in config:
            ppo_params = config["ppo_hyperparameters"]
        else:
            # 直接PPOパラメータが渡された場合
            ppo_params = config

        # 必須パラメータの確認
        required_params = ["learning_rate", "n_steps", "batch_size"]

        for param in required_params:
            if param not in ppo_params:
                raise ValueError(f"Missing required PPO parameter: {param}")

        # 値の範囲チェック
        if ppo_params["learning_rate"] <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {ppo_params['learning_rate']}"
            )

        if ppo_params["n_steps"] <= 0:
            raise ValueError(f"n_steps must be positive, got {ppo_params['n_steps']}")

        if ppo_params["batch_size"] <= 0:
            raise ValueError(
                f"batch_size must be positive, got {ppo_params['batch_size']}"
            )

        logger.debug(f"✅ PPO config validation passed: {ppo_params}")
        return True

    def __repr__(self) -> str:
        """PPOアルゴリズムの文字列表現"""
        trainer_type = "AutoHalt" if self._use_auto_halt else "Standard"
        return f"PPOAlgorithm(trainer={trainer_type}, model={'loaded' if self._model else 'not loaded'})"
