"""SAC (Soft Actor-Critic) algorithm implementation.

This module provides a SAC wrapper and utilities for training SAC-based
agents. It includes configuration defaults and helpers for model creation,
transfer learning, and optional model compression.

References:
    - Haarnoja et al., 2018: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
    - Haarnoja et al., 2019: "Soft Actor-Critic Algorithms and Applications"
"""

import logging
from collections.abc import Sequence
from typing import Callable, Optional, cast

import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from ztb.adaptation.explainability.analyzer import ExplainabilityAnalyzer
from ztb.adaptation.explainability.config import ExplainabilityConfig

from ztb.training.algorithms.base_algorithm import BaseRLAlgorithm
from ztb.training.models.advanced_networks import LSTMPolicy, TransformerPolicy
from ztb.training.model_compression import create_compression_pipeline
from ztb.types.common import SACLikeModelProtocol
from ztb.utils.safety import ensure_dict, safe_to_float, safe_to_int
from ztb.utils.types import ConfigMap

logger = logging.getLogger(__name__)

# Default SAC configuration used by the tests
DEFAULT_SAC_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 50000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    # Additional SB3 defaults used by create_model
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_entropy": "auto",
    "target_update_interval": 1,
    "use_sde": False,
    "sde_sample_freq": -1,
    "use_sde_at_warmup": False,
    "verbose": 0,
    "device": "auto",
    # Advanced network defaults
    "network_type": "mlp",
    "sequence_length": 10,
    "lstm_hidden_size": 128,
    "lstm_layers": 2,
    "transformer_d_model": 128,
    "transformer_n_heads": 8,
    "transformer_n_layers": 4,
    "transformer_d_ff": 512,
    "network_dropout": 0.1,
    # Model compression settings
    "compression_enabled": False,
    "compression_techniques": [],
    "quantization_type": "dynamic",
    "pruning_type": "l1_unstructured",
    "pruning_amount": 0.3,
    "distillation_temperature": 2.0,
    "distillation_alpha": 0.5,
    "compressed_model_path": None,

    # Explainability settings
    "explainability_enabled": False,
    "shap_enabled": True,
    "shap_max_evals": 1000,
    "shap_batch_size": 50,
    "plot_format": "png",
    "plot_dpi": 150,
    "explanation_cache_enabled": True,
    "cache_ttl_seconds": 3600,
    "natural_language_enabled": True,
    "market_context_analysis": True,
    "risk_warnings": True,
    "report_generation": True,
    "report_format": "html",
}

class SACAlgorithm(BaseRLAlgorithm):
    """
    SAC (Soft Actor-Critic) algorithm wrapper.

    Implements a wrapper around Stable-Baselines3 SAC providing project
    specific defaults, model creation helpers, and optional compression
    integration for testing and production.

    Example:
        >>> from ztb.training.algorithms import AlgorithmFactory
        >>> AlgorithmFactory.register("sac", SACAlgorithm)
        >>> sac = AlgorithmFactory.create("sac")
        >>> model = sac.create_model(env, config)
        >>> sac.train(model, total_timesteps=100000)
    """

    def __init__(self) -> None:
        """Initialize SACAlgorithm."""
        # Use conservative protocol to reduce raw Any in downstream code
        self._model: SACLikeModelProtocol | None = None
        self.compression_manager = None
        self.explainability_analyzer: ExplainabilityAnalyzer | None = None
        logger.info("SACAlgorithm initialized")

    @property
    def algorithm_name(self) -> str:
        """
        Get the algorithm name.

        Returns:
            "sac"
        """
        return "sac"

    def __repr__(self) -> str:
        """Return a string representation."""
        return (
            f"SACAlgorithm(model={'initialized' if self._model else 'not_initialized'})"
        )

    @staticmethod
    def _resolve_policy_kwargs(
        raw_kwargs: ConfigMap | None,
        network_type: str = "mlp",  # "mlp", "lstm", "transformer", "efficient"
        sequence_length: int = 10,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        transformer_d_model: int = 128,
        transformer_n_heads: int = 8,
        transformer_n_layers: int = 4,
        transformer_d_ff: int = 512,
        network_dropout: float = 0.1,
    ) -> ConfigMap | None:
        """
        Normalize policy kwargs and configure advanced network architectures.
        """
        if raw_kwargs is None:
            raw_kwargs = {}

        policy_kwargs: ConfigMap = dict(raw_kwargs)

        # Configure network architecture based on type
        if network_type == "lstm":
            policy_kwargs.update(
                {
                    "features_extractor_class": LSTMPolicy,
                    "features_extractor_kwargs": {
                        "lstm_hidden_size": lstm_hidden_size,
                        "lstm_layers": lstm_layers,
                        "dropout": network_dropout,
                        "sequence_length": sequence_length,
                    },
                }
            )
        elif network_type == "transformer":
            policy_kwargs.update(
                {
                    "features_extractor_class": TransformerPolicy,
                    "features_extractor_kwargs": {
                        "d_model": transformer_d_model,
                        "n_heads": transformer_n_heads,
                        "n_layers": transformer_n_layers,
                        "d_ff": transformer_d_ff,
                        "dropout": network_dropout,
                        "sequence_length": sequence_length,
                    },
                }
            )
        elif network_type == "efficient":
            # Import efficient network classes
            from ztb.training.models.advanced_networks import EfficientSACPolicy

            policy_kwargs.update(
                {
                    "features_extractor_class": EfficientSACPolicy,
                    "features_extractor_kwargs": {
                        "use_depthwise_conv": raw_kwargs.get(
                            "use_depthwise_conv", True
                        ),
                        "use_efficient_attention": raw_kwargs.get(
                            "use_efficient_attention", True
                        ),
                        "use_dynamic_network": raw_kwargs.get(
                            "use_dynamic_network", True
                        ),
                        "attention_method": raw_kwargs.get(
                            "attention_method", "linformer"
                        ),
                        "sequence_length": sequence_length,
                    },
                }
            )
        # For MLP, use default policy_kwargs

        # Handle activation function
        activation = policy_kwargs.get("activation_fn")
        if isinstance(activation, str):
            # 365#: Lazy import to handle torch DLL load failures gracefully
            try:
                activation_map: dict[str, Callable[[], nn.Module]] = {
                    "relu": nn.ReLU,
                    "leaky_relu": nn.LeakyReLU,
                    "elu": nn.ELU,
                    "selu": nn.SELU,
                    "gelu": nn.GELU,
                    "tanh": nn.Tanh,
                    "sigmoid": nn.Sigmoid,
                    "softplus": nn.Softplus,
                }
            except AttributeError:
                # torch.nn stub (DLL not loaded) — fall back to string mapping
                logger.warning(
                    "torch.nn activation classes unavailable; "
                    "activation_fn left as string"
                )
                return policy_kwargs
            key = activation.strip().lower()
            if key not in activation_map:
                raise ValueError(
                    f"Unsupported activation function '{activation}' in policy_kwargs. "
                    f"Supported values: {sorted(activation_map.keys())}"
                )
            policy_kwargs["activation_fn"] = activation_map[key]

        return policy_kwargs

    @classmethod
    def get_default_config(cls) -> ConfigMap:
        """
        Get the default SAC configuration.

        Returns:
            A dictionary containing default SAC configuration values.

        Example:
            >>> config = SACAlgorithm.get_default_config()
            >>> config["learning_rate"]
            0.0003
        """
        return DEFAULT_SAC_CONFIG.copy()

    @classmethod
    def validate_config(cls, config: ConfigMap) -> bool:
        """
        Validate a SAC configuration dictionary.

        Args:
            config: Configuration dictionary to validate.

        Returns:
            True if the configuration is valid.

        Raises:
            ValueError: If required parameters are missing or invalid.

        Example:
            >>> config = {"learning_rate": 3e-4, "buffer_size": 50000}
            >>> SACAlgorithm.validate_config(config)
            True
        """
        # 必須パラメータ
        required_params = ["learning_rate", "buffer_size", "batch_size"]

        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required SAC parameter: {param}")

        learning_rate = safe_to_float(config.get("learning_rate", -1.0), -1.0)
        buffer_size = safe_to_int(config.get("buffer_size", -1), -1)
        batch_size = safe_to_int(config.get("batch_size", -1), -1)

        # 値の範囲チェック
        if learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {config.get('learning_rate')}"
            )

        if buffer_size <= 0:
            raise ValueError(
                f"buffer_size must be positive, got {config.get('buffer_size')}"
            )

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {config.get('batch_size')}")

        # buffer_size >= batch_size
        if buffer_size < batch_size:
            raise ValueError(
                f"buffer_size ({buffer_size}) must be >= batch_size ({batch_size})"
            )

        # ネットワークタイプの検証
        network_type = str(config.get("network_type", "mlp"))
        if network_type not in ["mlp", "lstm", "transformer", "efficient"]:
            raise ValueError(
                f"Unsupported network_type: {network_type}. Must be one of: mlp, lstm, transformer, efficient"
            )

        # LSTM/Transformer固有のパラメータ検証
        if network_type == "lstm":
            if safe_to_int(config.get("lstm_hidden_size", 128), 128) <= 0:
                raise ValueError("lstm_hidden_size must be positive")
            if safe_to_int(config.get("lstm_layers", 2), 2) <= 0:
                raise ValueError("lstm_layers must be positive")
        elif network_type == "transformer":
            d_model = safe_to_int(config.get("transformer_d_model", 128), 128)
            n_heads = safe_to_int(config.get("transformer_n_heads", 8), 8)
            if d_model <= 0:
                raise ValueError("transformer_d_model must be positive")
            if n_heads <= 0:
                raise ValueError("transformer_n_heads must be positive")
            if d_model % n_heads != 0:
                raise ValueError(
                    "transformer_d_model must be divisible by transformer_n_heads"
                )

        # sequence_lengthの検証（LSTM/Transformer共通）
        if network_type in ["lstm", "transformer"]:
            if safe_to_int(config.get("sequence_length", 10), 10) <= 0:
                raise ValueError("sequence_length must be positive")

        # 転移学習設定の検証
        if config.get("transfer_learning_enabled", False):
            pretrained_path = config.get("pretrained_model_path")
            if not pretrained_path:
                raise ValueError(
                    "transfer_learning_enabled is True but pretrained_model_path is not specified"
                )

            freeze_layers = safe_to_float(config.get("freeze_layers", 0), 0.0)
            if freeze_layers < 0:
                raise ValueError("freeze_layers must be non-negative")

            if network_type == "mlp" and freeze_layers > 10:  # MLPの最大層数
                raise ValueError(
                    f"freeze_layers ({freeze_layers}) too large for MLP network"
                )
            elif network_type in ["lstm", "transformer"] and freeze_layers > 1.0:
                raise ValueError(
                    f"freeze_layers ({freeze_layers}) must be <= 1.0 for {network_type} network"
                )

            fine_tune_lr = config.get("fine_tune_learning_rate")
            if fine_tune_lr is not None and safe_to_float(fine_tune_lr, 0.0) <= 0:
                raise ValueError(
                    "fine_tune_learning_rate must be positive if specified"
                )

        # モデル圧縮設定の検証
        if config.get("compression_enabled", False):
            compression_techniques = config.get("compression_techniques", [])
            if not compression_techniques:
                logger.warning(
                    "compression_enabled is True but compression_techniques is empty"
                )
                # 空のtechniquesは警告のみで許可する

            valid_techniques = ["quantization", "pruning", "distillation"]
            for technique in compression_techniques:
                if technique not in valid_techniques:
                    raise ValueError(
                        f"Unsupported compression technique: {technique}. Must be one of: {valid_techniques}"
                    )

            # 量子化設定の検証
            if "quantization" in compression_techniques:
                quant_type = config.get("quantization_type", "dynamic")
                if quant_type not in ["dynamic", "static", "mixed_precision"]:
                    raise ValueError(f"Unsupported quantization_type: {quant_type}")

            # プルーニング設定の検証
            if "pruning" in compression_techniques:
                pruning_type = config.get("pruning_type", "l1_unstructured")
                if pruning_type not in [
                    "l1_unstructured",
                    "l2_unstructured",
                    "structured",
                ]:
                    raise ValueError(f"Unsupported pruning_type: {pruning_type}")

                pruning_amount = config.get("pruning_amount", 0.3)
                pruning_amount_f = safe_to_float(pruning_amount, 0.3)
                if not (0.0 < pruning_amount_f < 1.0):
                    raise ValueError(
                        f"pruning_amount must be between 0.0 and 1.0, got {pruning_amount_f}"
                    )

            # 蒸留設定の検証
            if "distillation" in compression_techniques:
                teacher_path = config.get("teacher_model_path")
                if not teacher_path:
                    raise ValueError(
                        "distillation requested but teacher_model_path not specified"
                    )

                distillation_temp = config.get("distillation_temperature", 2.0)
                if safe_to_float(distillation_temp, 0.0) <= 0:
                    raise ValueError("distillation_temperature must be positive")

                distillation_alpha = config.get("distillation_alpha", 0.5)
                distillation_alpha_f = safe_to_float(distillation_alpha, -1.0)
                if not (0.0 <= distillation_alpha_f <= 1.0):
                    raise ValueError("distillation_alpha must be between 0.0 and 1.0")

        logger.debug(f"SAC config validation passed: {config}")
        return True

    def create_model(
        self,
        env: VecEnv,
        config: ConfigMap,
        tensorboard_log: str | None = None,
    ) -> BaseAlgorithm:
        """Create a SAC model instance using the provided environment and config.

        Args:
            env: Training environment (VecEnv).
            config: SAC configuration dictionary.
            tensorboard_log: Optional TensorBoard log directory.

        Returns:
            The created SAC model instance.

        Raises:
            ValueError: If the configuration is invalid.

        Example:
            >>> sac = SACAlgorithm()
            >>> config = sac.get_default_config()
            >>> model = sac.create_model(vec_env, config, "./logs")
        """
        # デフォルト設定とマージして検証
        config = {**self.get_default_config(), **config}
        self.validate_config(config)

        logger.info(f"Creating SAC model with config: {config}")
        logger.info(
            "compression_enabled: %s, compression_techniques: %s",
            config.get("compression_enabled"),
            config.get("compression_techniques"),
        )

        # Stable-Baselines3 SAC用パラメータ抽出
        sac_params = {
            "policy": config.get("policy", "MlpPolicy"),
            "env": env,
            "learning_rate": safe_to_float(config["learning_rate"], 3e-4),
            "buffer_size": safe_to_int(config["buffer_size"], 50000),
            "learning_starts": safe_to_int(
                config.get("learning_starts", DEFAULT_SAC_CONFIG["learning_starts"]),
                DEFAULT_SAC_CONFIG["learning_starts"],
            ),
            "batch_size": safe_to_int(config["batch_size"], 256),
            "tau": safe_to_float(config.get("tau", DEFAULT_SAC_CONFIG["tau"]), DEFAULT_SAC_CONFIG["tau"]),
            "gamma": safe_to_float(config.get("gamma", DEFAULT_SAC_CONFIG["gamma"]), DEFAULT_SAC_CONFIG["gamma"]),
            "train_freq": safe_to_int(
                config.get("train_freq", DEFAULT_SAC_CONFIG["train_freq"]),
                DEFAULT_SAC_CONFIG["train_freq"],
            ),
            "gradient_steps": safe_to_int(
                config.get("gradient_steps", DEFAULT_SAC_CONFIG["gradient_steps"]),
                DEFAULT_SAC_CONFIG["gradient_steps"],
            ),
            "ent_coef": config.get("ent_coef", DEFAULT_SAC_CONFIG["ent_coef"]),
            "target_entropy": config.get(
                "target_entropy", DEFAULT_SAC_CONFIG["target_entropy"]
            ),
            "target_update_interval": safe_to_int(
                config.get(
                    "target_update_interval",
                    DEFAULT_SAC_CONFIG["target_update_interval"],
                ),
                DEFAULT_SAC_CONFIG["target_update_interval"],
            ),
            "use_sde": config.get("use_sde", DEFAULT_SAC_CONFIG["use_sde"]),
            "sde_sample_freq": safe_to_int(
                config.get("sde_sample_freq", DEFAULT_SAC_CONFIG["sde_sample_freq"]),
                DEFAULT_SAC_CONFIG["sde_sample_freq"],
            ),
            "use_sde_at_warmup": config.get(
                "use_sde_at_warmup", DEFAULT_SAC_CONFIG["use_sde_at_warmup"]
            ),
            "verbose": safe_to_int(
                config.get("verbose", DEFAULT_SAC_CONFIG["verbose"]),
                DEFAULT_SAC_CONFIG["verbose"],
            ),
            "tensorboard_log": tensorboard_log,
            "device": config.get("device", DEFAULT_SAC_CONFIG["device"]),
        }

        # policy_kwargsがあれば追加（文字列定義の活性化関数を解決）
        network_type = str(config.get("network_type", "mlp"))
        raw_policy_kwargs = config.get("policy_kwargs")
        policy_kwargs_source = (
            ensure_dict(raw_policy_kwargs)
            if isinstance(raw_policy_kwargs, dict)
            else None
        )
        policy_kwargs = self._resolve_policy_kwargs(
            policy_kwargs_source,
            network_type=network_type,
            sequence_length=safe_to_int(
                config.get("sequence_length", DEFAULT_SAC_CONFIG["sequence_length"]),
                DEFAULT_SAC_CONFIG["sequence_length"],
            ),
            lstm_hidden_size=safe_to_int(
                config.get("lstm_hidden_size", DEFAULT_SAC_CONFIG["lstm_hidden_size"]),
                DEFAULT_SAC_CONFIG["lstm_hidden_size"],
            ),
            lstm_layers=safe_to_int(
                config.get("lstm_layers", DEFAULT_SAC_CONFIG["lstm_layers"]),
                DEFAULT_SAC_CONFIG["lstm_layers"],
            ),
            transformer_d_model=safe_to_int(
                config.get(
                    "transformer_d_model", DEFAULT_SAC_CONFIG["transformer_d_model"]
                ),
                DEFAULT_SAC_CONFIG["transformer_d_model"],
            ),
            transformer_n_heads=safe_to_int(
                config.get(
                    "transformer_n_heads", DEFAULT_SAC_CONFIG["transformer_n_heads"]
                ),
                DEFAULT_SAC_CONFIG["transformer_n_heads"],
            ),
            transformer_n_layers=safe_to_int(
                config.get(
                    "transformer_n_layers", DEFAULT_SAC_CONFIG["transformer_n_layers"]
                ),
                DEFAULT_SAC_CONFIG["transformer_n_layers"],
            ),
            transformer_d_ff=safe_to_int(
                config.get("transformer_d_ff", DEFAULT_SAC_CONFIG["transformer_d_ff"]),
                DEFAULT_SAC_CONFIG["transformer_d_ff"],
            ),
            network_dropout=safe_to_float(
                config.get("network_dropout", DEFAULT_SAC_CONFIG["network_dropout"]),
                DEFAULT_SAC_CONFIG["network_dropout"],
            ),
        )
        if policy_kwargs is not None:
            sac_params["policy_kwargs"] = policy_kwargs

        # set policy based on network type
        if network_type == "lstm":
            sac_params["policy"] = LSTMPolicy
        elif network_type == "transformer":
            sac_params["policy"] = TransformerPolicy
        else:
            sac_params["policy"] = config.get("policy", "MlpPolicy")

        # SACモデル作成
        self._model = SAC(**sac_params)

        # 転移学習の適用
        if config.get("transfer_learning_enabled", False):
            self._apply_transfer_learning(self._model, config)

        # モデル圧縮の適用
        if config.get("compression_enabled", False):
            # For diagnostics in tests, emit a visible message when compression
            # would be applied.
            print("APPLY_MODEL_COMPRESSION_TRIGGERED")
            self._apply_model_compression(self._model, config)

        # 特徴量情報をログに出力
        feature_set = config.get("feature_set", "curated")
        expected_features = safe_to_int(config.get("expected_features", 88), 88)
        learning_rate = safe_to_float(config.get("learning_rate", 3e-4), 3e-4)
        buffer_size = safe_to_int(config.get("buffer_size", 50000), 50000)
        batch_size = safe_to_int(config.get("batch_size", 256), 256)
        try:
            # 環境の観測空間から特徴量数を取得
            if hasattr(env, "observation_space"):
                obs_shape = env.observation_space.shape
                actual_features = obs_shape[0] if len(obs_shape) > 0 else 0
                logger.info(
                    f"SAC model created with feature integration: "
                    f"feature_set={feature_set}, expected_features={expected_features}, "
                    f"actual_features={actual_features}, lr={learning_rate}, "
                    f"buffer_size={buffer_size}, batch_size={batch_size}, "
                    f"ent_coef={config.get('ent_coef', 'auto')}"
                )
                if abs(actual_features - expected_features) > 5:  # 許容誤差
                    logger.warning(
                        f"Feature count mismatch: expected {expected_features}, got {actual_features}. "
                        f"Check environment configuration for feature_set='{feature_set}'"
                    )
            else:
                logger.info(
                    f"SAC model created: feature_set={feature_set}, expected_features={expected_features}, "
                    f"lr={learning_rate}, buffer_size={buffer_size}, "
                    f"batch_size={batch_size}, ent_coef={config.get('ent_coef', 'auto')}"
                )
        except Exception as e:
            logger.warning(
                f"Could not retrieve feature information from environment: {e}"
            )
            logger.info(
                f"SAC model created: feature_set={feature_set}, expected_features={expected_features}, "
                f"lr={learning_rate}, buffer_size={buffer_size}, "
                f"batch_size={batch_size}, ent_coef={config.get('ent_coef', 'auto')}"
            )

        # 説明可能性アナライザーの初期化
        if config.get("explainability_enabled", False):
            self._initialize_explainability_analyzer(config)

        return self._model

    def _apply_transfer_learning(
        self, model: BaseAlgorithm, config: ConfigMap
    ) -> None:
        """Apply transfer learning to a model.

        Args:
            model: The SAC model to apply transfer learning to.
            config: Configuration dictionary for transfer learning.
        """
        pretrained_path = config.get("pretrained_model_path")
        if not pretrained_path:
            logger.warning(
                "Transfer learning enabled but no pretrained_model_path specified"
            )
            return

        try:
            # 事前学習済みモデルの読み込み
            logger.info(f"Loading pretrained model from: {pretrained_path}")
            pretrained_model = SAC.load(pretrained_path, device=model.device)

            # モデルの検証
            self._validate_pretrained_model(model, pretrained_model, config)

            # 層の凍結
            freeze_layers = safe_to_float(config.get("freeze_layers", 0.0), 0.0)
            if freeze_layers > 0:
                self._freeze_layers(model, freeze_layers, config)

            # ファインチューニング学習率の設定
            fine_tune_lr = config.get("fine_tune_learning_rate")
            if fine_tune_lr is not None:
                self._set_fine_tune_learning_rate(
                    model, safe_to_float(fine_tune_lr, 3e-4)
                )

            logger.info("Transfer learning applied successfully")

        except Exception as e:
            logger.error(f"Failed to apply transfer learning: {e}")
            raise

    def _apply_model_compression(
        self, model: BaseAlgorithm, config: ConfigMap
    ) -> None:
        """Apply model compression techniques to the SAC model.

        Args:
            model: The SAC model to compress.
            config: Compression configuration dictionary.
        """
        raw_techniques = config.get("compression_techniques", [])
        if isinstance(raw_techniques, list):
            compression_techniques = [str(item) for item in raw_techniques]
        elif raw_techniques is None:
            compression_techniques = []
        else:
            compression_techniques = [str(raw_techniques)]
        if not compression_techniques:
            logger.warning(
                "Model compression enabled but no compression_techniques specified"
            )
            return

        try:
            logger.info(
                f"Applying model compression techniques: {compression_techniques}"
            )
            print("COMPRESSION_TECHNIQUES:", compression_techniques)
            logger.debug("compression_techniques value: %s", compression_techniques)

            # 圧縮パイプラインの設定
            techniques_config = {}

            if "quantization" in compression_techniques:
                techniques_config["quantization"] = {
                    "type": "quantization",
                    "quantization_type": config.get("quantization_type", "dynamic"),
                }

            if "pruning" in compression_techniques:
                techniques_config["pruning"] = {
                    "type": "pruning",
                    "pruning_type": config.get("pruning_type", "l1_unstructured"),
                    "amount": safe_to_float(config.get("pruning_amount", 0.3), 0.3),
                }

            if "distillation" in compression_techniques:
                # 教師モデルが必要
                teacher_model_path = config.get("teacher_model_path")
                if teacher_model_path:
                    SAC.load(teacher_model_path, device=model.device)
                    techniques_config["distillation"] = {
                        "type": "distillation",
                        "temperature": safe_to_float(
                            config.get("distillation_temperature", 2.0), 2.0
                        ),
                        "alpha": safe_to_float(
                            config.get("distillation_alpha", 0.5), 0.5
                        ),
                    }
                else:
                    logger.warning(
                        "Knowledge distillation requested but teacher_model_path not specified"
                    )

                # distillation handling only populates techniques_config; actual
                # pipeline creation and compression are handled below.

            # At this point, if any techniques were configured, create the
            # compression pipeline and attempt to apply it.
            if not techniques_config:
                logger.warning("No valid compression techniques configured")
                return

            logger.debug("techniques_config populated: %s", techniques_config)

            try:
                # Create compression pipeline (this is what tests patch)
                self.compression_manager = create_compression_pipeline(techniques_config)

                # Attempt to compress the model's policy if possible
                policy = getattr(model, "policy", None)
                if policy is not None and hasattr(self.compression_manager, "compress_model"):
                    compressed_policy = self.compression_manager.compress_model(
                        policy, list(techniques_config.keys())
                    )
                    if compressed_policy is not None:
                        model.policy = compressed_policy

                # Save compressed model if a path is provided
                compressed_path = config.get("compressed_model_path")
                if compressed_path and hasattr(self.compression_manager, "save_compressed_model"):
                    try:
                        self.compression_manager.save_compressed_model(model, compressed_path)
                        logger.info(f"Compressed model saved to {compressed_path}")
                    except Exception:
                        logger.warning("Failed to save compressed model")

            except Exception as e:
                logger.error(f"Model compression pipeline failed: {e}")
                # Keep compression manager set for inspection even if compress failed

        except Exception as e:
            logger.error(f"Failed to apply model compression: {e}")
            raise

    def _validate_pretrained_model(
        self,
        model: BaseAlgorithm,
        pretrained_model: BaseAlgorithm,
        config: ConfigMap,
    ) -> None:
        """Validate a pretrained model.

        Args:
            model: The current model instance
            pretrained_model: The pretrained model to validate
            config: Configuration dictionary
        """
        # ネットワークタイプの一致を確認
        current_network_type = str(config.get("network_type", "mlp"))

        # ポリシータイプの確認
        if current_network_type == "lstm":
            expected_policy = LSTMPolicy
        elif current_network_type == "transformer":
            expected_policy = TransformerPolicy
        else:
            expected_policy = type(model.policy).__name__

        pretrained_policy_type = type(pretrained_model.policy).__name__

        if current_network_type in ["lstm", "transformer"]:
            if not isinstance(pretrained_model.policy, expected_policy):
                raise ValueError(
                    f"Network type mismatch: expected {expected_policy}, "
                    f"got {pretrained_policy_type} in pretrained model"
                )

        logger.info(f"Pretrained model validation passed: {pretrained_policy_type}")

    def _freeze_layers(
        self,
        model: BaseAlgorithm,
        freeze_layers: int | float,
        config: ConfigMap,
    ) -> None:
        """Freeze specified layers of the model.

        Args:
            model: The target model
            freeze_layers: Number of layers or fraction to freeze
            config: Configuration dictionary
        """
        network_type = str(config.get("network_type", "mlp"))

        if network_type == "mlp":
            # MLPの場合：層数を指定
            self._freeze_mlp_layers(model, freeze_layers)
        elif network_type in ["lstm", "transformer"]:
            # LSTM/Transformerの場合：割合を指定
            self._freeze_advanced_layers(model, freeze_layers, network_type)

        logger.info(f"Froze {freeze_layers} layers for {network_type} network")

    def _freeze_mlp_layers(
        self, model: BaseAlgorithm, freeze_layers: int | float
    ) -> None:
        """Freeze layers in an MLP network.

        Args:
            model: SAC model
            freeze_layers: Number of layers to freeze
        """
        policy = model.policy

        # Freeze actor network layers
        if hasattr(policy, "actor"):
            self._freeze_network_layers(policy.actor, freeze_layers)

        # Freeze critic network layers
        if hasattr(policy, "critic"):
            self._freeze_network_layers(policy.critic, freeze_layers)

    def _freeze_network_layers(
        self, network: nn.Module, freeze_layers: int | float
    ) -> None:
        """Freeze layers in a network.

        Args:
            network: Target network
            freeze_layers: Number of layers to freeze
        """
        layer_count = 0
        # intに変換
        freeze_count = (
            int(freeze_layers)
            if isinstance(freeze_layers, (int, float))
            else freeze_layers
        )
        # Support both real nn.Module (modules()) and mocks that provide children()
        try:
            modules_iter = network.modules()
            # Ensure it's iterable
            iter(modules_iter)
        except Exception:
            modules_iter = getattr(network, "children", lambda: [])()

        for module in modules_iter:
            # Check for Linear layers without relying on isinstance (handles mocks)
            is_linear = (
                getattr(getattr(module, "__class__", None), "__name__", "") == "Linear"
            )
            if is_linear and layer_count < freeze_count:
                params_func = getattr(module, "parameters", None)
                if callable(params_func):
                    try:
                        for param in params_func():
                            try:
                                param.requires_grad_(False)
                            except Exception:
                                # If param is a Mock or doesn't support requires_grad_, skip
                                pass
                    except TypeError:
                        # parameters() returned non-iterable; skip
                        pass
                logger.debug(f"Froze layer {layer_count}")
                layer_count += 1

    def _freeze_advanced_layers(
        self, model: BaseAlgorithm, freeze_ratio: float, network_type: str
    ) -> None:
        """Freeze layers in LSTM/Transformer networks by ratio.

        Args:
            model: SAC model
            freeze_ratio: Fraction of layers to freeze (0.0-1.0)
            network_type: Type of network ('lstm' or 'transformer')
        """
        policy = model.policy

        if not hasattr(policy, "features_extractor"):
            return

        extractor = policy.features_extractor

        if network_type == "lstm":
            # Freeze LSTM layers
            if hasattr(extractor, "lstm"):
                lstm_layers = (
                    list(extractor.lstm.children())
                    if hasattr(extractor.lstm, "children")
                    else [extractor.lstm]
                )
                freeze_count = int(len(lstm_layers) * freeze_ratio)
                for i, layer in enumerate(lstm_layers):
                    if i < freeze_count:
                        for param in layer.parameters():
                            param.requires_grad_(False)
                        logger.debug(f"Froze LSTM layer {i}")

        elif network_type == "transformer":
            # Freeze Transformer layers
            if hasattr(extractor, "transformer_layers"):
                transformer_layers = (
                    list(extractor.transformer_layers.children())
                    if hasattr(extractor.transformer_layers, "children")
                    else [extractor.transformer_layers]
                )
                freeze_count = int(len(transformer_layers) * freeze_ratio)
                for i, layer in enumerate(transformer_layers):
                    if i < freeze_count:
                        for param in layer.parameters():
                            param.requires_grad_(False)
                        logger.debug(f"Froze Transformer layer {i}")

    def _set_fine_tune_learning_rate(
        self, model: BaseAlgorithm, learning_rate: float
    ) -> None:
        """set learning rate for fine-tuning.

        Args:
            model: SAC model
            learning_rate: Fine-tuning learning rate
        """
        # Update optimizer learning rates
        from collections.abc import Iterable

        if hasattr(model, "policy_optimizer") and hasattr(
            model.policy_optimizer, "param_groups"
        ):
            param_groups_obj = getattr(model.policy_optimizer, "param_groups", [])
            if isinstance(param_groups_obj, Iterable):
                for param_group in param_groups_obj:
                    try:
                        param_group["lr"] = learning_rate
                    except Exception:
                        # If the param_group isn't a mapping (e.g., Mock), skip
                        continue

        logger.info(f"set fine-tuning learning rate to {learning_rate}")

    def train(
        self,
        model: BaseAlgorithm,
        total_timesteps: int,
        callback: Callable[..., object] | None = None,
        **kwargs: object,
    ) -> BaseAlgorithm:
        """Train the SAC model.

        Args:
            model: The SAC model to train
            total_timesteps: Number of training timesteps
            callback: Optional training callback
            **kwargs: Additional arguments passed to model.learn()

        Returns:
            The trained model

        Raises:
            ValueError: If the model is not initialized

        Example:
            >>> sac = SACAlgorithm()
            >>> model = sac.create_model(env, config)
            >>> trained = sac.train(model, total_timesteps=100000)
        """
        if model is None:
            raise ValueError(
                "Model must be initialized before training. Call create_model() first."
            )

        logger.info(f"Starting SAC training for {total_timesteps} timesteps")

        # 訓練実行
        model.learn(total_timesteps=total_timesteps, callback=callback, **kwargs)

        logger.info(f"SAC training completed: {total_timesteps} timesteps")

        return model

    def save(self, model: BaseAlgorithm, save_path: str) -> None:
        """Save the model to disk.

        Args:
            model: Model to save
            save_path: Path where the model will be saved

        Raises:
            OSError: If the save operation fails

        Example:
            >>> sac.save(model, "models/sac_v395a.zip")
        """
        from pathlib import Path as _Path

        save_dir = _Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        logger.info(f"SAC model saved to {save_path}")

    @staticmethod
    def load(load_path: str, env: VecEnv | None = None) -> BaseAlgorithm:
        """Load a model from disk.

        Args:
            load_path: Path to the model file
            env: Optional environment for the model (useful at inference time)

        Returns:
            The loaded model

        Raises:
            FileNotFoundError: If the model file does not exist

        Example:
            >>> model = SACAlgorithm.load("models/sac_v395a.zip", env=vec_env)
        """
        from pathlib import Path as _Path

        # SB3 adds .zip automatically, check both
        p = _Path(load_path)
        if not p.exists() and not p.with_suffix(".zip").exists():
            raise FileNotFoundError(f"SAC model not found: {load_path}")
        model = SAC.load(load_path, env=env)
        logger.info(f"SAC model loaded from {load_path}")
        return model

    @staticmethod
    def save_replay_buffer(
        model: BaseAlgorithm, buffer_path: str
    ) -> None:
        """Save replay buffer to disk for warm-start training.

        365# P1: Replay buffer persistence enables incremental training
        without catastrophic forgetting.

        Args:
            model: Trained SAC model with populated replay buffer
            buffer_path: Path where the buffer will be saved (pickle format)

        Raises:
            RuntimeError: If the model has no replay buffer
        """
        from pathlib import Path as _Path

        if not hasattr(model, "save_replay_buffer"):
            raise RuntimeError(
                "Model does not support replay buffer persistence. "
                "Requires SB3 OffPolicyAlgorithm (SAC/TD3/DQN)."
            )
        buf_dir = _Path(buffer_path).parent
        buf_dir.mkdir(parents=True, exist_ok=True)
        model.save_replay_buffer(buffer_path)
        logger.info(f"Replay buffer saved to {buffer_path}")

    @staticmethod
    def load_replay_buffer(
        model: BaseAlgorithm, buffer_path: str
    ) -> None:
        """Load a previously saved replay buffer into the model.

        365# P1: Enables warm-start by restoring past experiences.

        Args:
            model: SAC model to load the buffer into
            buffer_path: Path to the saved buffer file

        Raises:
            FileNotFoundError: If the buffer file does not exist
            RuntimeError: If the model does not support replay buffers
        """
        from pathlib import Path as _Path

        if not _Path(buffer_path).exists():
            raise FileNotFoundError(f"Replay buffer not found: {buffer_path}")
        if not hasattr(model, "load_replay_buffer"):
            raise RuntimeError(
                "Model does not support replay buffer persistence."
            )
        model.load_replay_buffer(buffer_path)
        logger.info(f"Replay buffer loaded from {buffer_path}")

    def _initialize_explainability_analyzer(self, config: ConfigMap) -> None:
        """Initialize the explainability analyzer using provided config.

        Args:
            config: Configuration dictionary containing explainability settings
        """
        try:
            # 説明可能性設定の作成
            explainability_config = ExplainabilityConfig(
                enabled=bool(config.get("explainability_enabled", True)),
                explanation_method=str(config.get("explanation_method", "shap")),
                shap_max_evals=safe_to_int(config.get("shap_max_evals", 1000), 1000),
                shap_batch_size=safe_to_int(config.get("shap_batch_size", 50), 50),
                generate_natural_language=bool(
                    config.get("natural_language_enabled", True)
                ),
                enable_visualization=bool(config.get("enable_visualization", True)),
                plot_format=str(config.get("plot_format", "png")),
                cache_explanations=bool(config.get("explanation_cache_enabled", True)),
                cache_ttl_seconds=safe_to_int(
                    config.get("cache_ttl_seconds", 3600), 3600
                ),
                generate_reports=bool(config.get("report_generation", True)),
            )

            # 説明可能性アナライザーの作成
            self.explainability_analyzer = ExplainabilityAnalyzer(explainability_config)

            logger.info("Explainability analyzer initialized for SAC algorithm")

        except Exception as e:
            logger.error(f"Failed to initialize explainability analyzer: {e}")
            self.explainability_analyzer = None

    def explain_decision(
        self,
        observation: object,
        action: object | None = None,
        context: ConfigMap | None = None,
    ) -> ConfigMap | None:
        """Explain a model decision.

        Args:
            observation: Observation input data
            action: Executed action (Optional)
            context: Additional contextual information (Optional)

        Returns:
            A dictionary with explanation results, or None if explainability is disabled
        """
        if self.explainability_analyzer is None or self._model is None:
            logger.warning("Explainability analyzer or model not initialized")
            return None

        try:
            # 説明結果の生成
            explanation_result = self.explainability_analyzer.explain_prediction(
                model=self._model, input_data=observation, prediction=action
            )

            # 辞書形式に変換して返す
            return cast(ConfigMap, explanation_result.to_dict())

        except Exception as e:
            logger.error(f"Failed to explain decision: {e}")
            return None

    def generate_explanation_report(
        self,
        observations: object | Sequence[object],
        actions: Sequence[object] | None = None,
        output_path: str | None = None,
    ) -> str | None:
        """Generate an explanation report for a batch of observations.

        Args:
            observations: Batch of observation inputs
            actions: Corresponding batch of actions (Optional)
            output_path: Optional path to write the report

        Returns:
            Path to the generated report file or None on failure
        """
        if self.explainability_analyzer is None or self._model is None:
            logger.warning("Explainability analyzer or model not initialized")
            return None

        try:
            # Generate explanations for multiple observations
            explanations = []

            # Determine if observations is a single item or a batch
            if isinstance(observations, (list, tuple)) and len(observations) > 0:
                obs_list = observations
            else:
                obs_list = [observations]

            # Generate an explanation for each observation
            for i, obs in enumerate(obs_list):
                action = (
                    actions[i] if actions is not None and i < len(actions) else None
                )
                explanation = self.explainability_analyzer.explain_prediction(
                    model=self._model, input_data=obs, prediction=action
                )
                explanations.append(explanation)

            # Generate the report. The return type of
            # explainability_analyzer.generate_explanation_report may vary by
            # implementation, so cast to str | None.
            return cast(
                str | None,
                self.explainability_analyzer.generate_explanation_report(
                    explanations=explanations, output_path=output_path
                ),
            )

        except Exception as e:
            logger.error(f"Failed to generate explanation report: {e}")
            return None
