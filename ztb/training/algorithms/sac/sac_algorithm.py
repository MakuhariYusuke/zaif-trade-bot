"""
SAC（Soft Actor-Critic）アルゴリズム実装。

SACは、エントロピー正則化を用いたoff-policyアクター・クリティック手法。
金融取引において、PPOよりも探索と活用のバランスが優れている可能性がある。

主な特徴：
- Off-policy: 過去の経験（Replay Buffer）を効率的に再利用
- エントロピー正則化: 自動的な探索・活用バランス調整
- alpha自動調整: target_entropyに基づく適応的なエントロピー係数
- ソフトアップデート: targetネットワークの安定的な更新

References:
    - Haarnoja et al., 2018: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
    - Haarnoja et al., 2019: "Soft Actor-Critic Algorithms and Applications"
"""

import logging
from typing import Any, Callable, Dict, Optional, Union
# Note: keep heavier algorithm internals untyped for now to avoid broad mypy noise

import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from ztb.adaptation.explainability.analyzer import ExplainabilityAnalyzer
from ztb.adaptation.explainability.config import ExplainabilityConfig
from ztb.optimization.model_compression import (
    ModelCompressionManager,
    create_compression_pipeline,
)
from ztb.training.algorithms.base_algorithm import BaseRLAlgorithm
from ztb.training.models.advanced_networks import LSTMPolicy, TransformerPolicy

logger = logging.getLogger(__name__)


# SACデフォルト設定（Stable-Baselines3準拠 + 金融取引向け調整）
DEFAULT_SAC_CONFIG = {
    # 学習率
    "learning_rate": 3e-4,
    # Replay Buffer
    "buffer_size": 50000,  # 金融データの季節性を考慮したサイズ
    # 訓練パラメータ
    "learning_starts": 1000,  # バッファが十分溜まってから学習開始
    "batch_size": 256,
    "tau": 0.005,  # ソフトアップデート係数（小さいほど安定）
    "gamma": 0.99,  # 割引率
    "train_freq": 1,  # 毎ステップ訓練
    "gradient_steps": 1,  # 各訓練で1回勾配更新
    # エントロピー正則化
    "ent_coef": "auto",  # 自動調整（推奨）
    "target_entropy": "auto",  # 自動設定（-dim(A)）
    # 特徴量設定（拡張されたIchimoku特徴量を含む）
    "feature_set": "curated",  # 使用する特徴量セット
    "expected_features": 80,  # 予想される特徴量数（CURATED_FEATURESの80個）
    # ネットワーク構造
    "policy_kwargs": None,  # デフォルトのMLP
    "network_type": "mlp",  # "mlp", "lstm", "transformer"
    # LSTM/Transformer設定
    "sequence_length": 10,  # シーケンス長
    "lstm_hidden_size": 128,  # LSTM隠れ層サイズ
    "lstm_layers": 2,  # LSTM層数
    "transformer_d_model": 128,  # Transformerモデル次元
    "transformer_n_heads": 8,  # Transformer注意ヘッド数
    "transformer_n_layers": 4,  # Transformer層数
    "transformer_d_ff": 512,  # Transformerフィードフォワード次元
    "network_dropout": 0.1,  # ネットワークドロップアウト
    # 効率的ネットワーク設定
    "use_efficient_network": False,  # 効率的ネットワークを使用するか
    "use_depthwise_conv": True,  # 深度分離畳み込みを使用するか
    "use_efficient_attention": True,  # 効率的アテンションを使用するか
    "use_dynamic_network": True,  # 動的ネットワークを使用するか
    "attention_method": "linformer",  # アテンション方法 ("linformer", "performer")
    # 転移学習設定
    "transfer_learning_enabled": False,  # 転移学習を使用するか
    "pretrained_model_path": None,  # 事前学習済みモデルのパス
    "freeze_layers": 0,  # 凍結する層の数（MLPの場合）または割合（LSTM/Transformerの場合）
    "fine_tune_learning_rate": None,  # ファインチューニング時の学習率（Noneの場合は通常のlearning_rateを使用）
    "fine_tune_layers_only": False,  # 最後の層のみファインチューニングするか
    # その他
    "verbose": 1,
    "tensorboard_log": None,
    "device": "auto",
    "target_update_interval": 1,
    "use_sde": False,
    "sde_sample_freq": -1,
    "use_sde_at_warmup": False,
    # モデル圧縮設定
    "compression_enabled": False,  # モデル圧縮を使用するか
    "compression_techniques": [],  # 使用する圧縮手法のリスト
    "quantization_type": "dynamic",  # 量子化タイプ ("dynamic", "static", "mixed_precision")
    "pruning_type": "l1_unstructured",  # プルーニングタイプ ("l1_unstructured", "l2_unstructured", "structured")
    "pruning_amount": 0.3,  # プルーニング量 (0.0-1.0)
    "distillation_temperature": 2.0,  # 蒸留温度
    "distillation_alpha": 0.5,  # 蒸留損失の重み
    "compressed_model_path": None,  # 圧縮モデルの保存パス
    # 説明可能性設定
    "explainability_enabled": False,  # 説明可能性機能を有効にするか
    "shap_enabled": True,  # SHAP分析を使用するか
    "shap_max_evals": 1000,  # SHAPの最大評価数
    "shap_batch_size": 50,  # SHAPのバッチサイズ
    "plot_format": "png",  # プロットのフォーマット ("png", "svg", "pdf")
    "plot_dpi": 150,  # プロットのDPI
    "explanation_cache_enabled": True,  # 説明結果のキャッシュを有効にするか
    "cache_ttl_seconds": 3600,  # キャッシュのTTL（秒）
    "natural_language_enabled": True,  # 自然言語説明を有効にするか
    "market_context_analysis": True,  # 市場文脈分析を有効にするか
    "risk_warnings": True,  # リスク警告を有効にするか
    "report_generation": True,  # レポート生成を有効にするか
    "report_format": "html",  # レポートフォーマット ("html", "json")
}


class SACAlgorithm(BaseRLAlgorithm):
    """
    SAC（Soft Actor-Critic）アルゴリズムのラッパークラス。

    BaseRLAlgorithmインターフェースに準拠し、
    AlgorithmFactoryから使用可能。

    金融取引への適用において、PPOと比較して以下の利点がある：
    - 過去の経験を効率的に再利用（sample efficiency）
    - エントロピー係数の自動調整による適応的な探索
    - 連続行動・離散行動両対応

    Example:
        >>> from ztb.training.algorithms import AlgorithmFactory
        >>> AlgorithmFactory.register("sac", SACAlgorithm)
        >>> sac = AlgorithmFactory.create("sac")
        >>> model = sac.create_model(env, config)
        >>> sac.train(model, total_timesteps=100000)
    """

    def __init__(self) -> None:
        """SACAlgorithmを初期化。"""
        # Use conservative protocol to reduce raw Any in downstream code
        self._model: Optional[SACLikeModelProtocol] = None
        self.compression_manager: Optional[ModelCompressionManager] = None
        self.explainability_analyzer: Optional[ExplainabilityAnalyzer] = None
        logger.info("SACAlgorithm initialized")

    @property
    def algorithm_name(self) -> str:
        """
        アルゴリズム名を取得。

        Returns:
            "sac"
        """
        return "sac"

    def __repr__(self) -> str:
        """文字列表現。"""
        return (
            f"SACAlgorithm(model={'initialized' if self._model else 'not_initialized'})"
        )

    @staticmethod
    def _resolve_policy_kwargs(
        raw_kwargs: Optional[Dict[str, Any]],
        network_type: str = "mlp",
        sequence_length: int = 10,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        transformer_d_model: int = 128,
        transformer_n_heads: int = 8,
        transformer_n_layers: int = 4,
        transformer_d_ff: int = 512,
        network_dropout: float = 0.1,
    ) -> Optional[Dict[str, Any]]:
        """
        Normalize policy kwargs and configure advanced network architectures.
        """
        if raw_kwargs is None:
            raw_kwargs = {}

        policy_kwargs: Dict[str, Any] = dict(raw_kwargs)

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
            activation_map: Dict[str, Callable[[], nn.Module]] = {
                "relu": nn.ReLU,
                "leaky_relu": nn.LeakyReLU,
                "elu": nn.ELU,
                "selu": nn.SELU,
                "gelu": nn.GELU,
                "tanh": nn.Tanh,
                "sigmoid": nn.Sigmoid,
                "softplus": nn.Softplus,
            }
            key = activation.strip().lower()
            if key not in activation_map:
                raise ValueError(
                    f"Unsupported activation function '{activation}' in policy_kwargs. "
                    f"Supported values: {sorted(activation_map.keys())}"
                )
            policy_kwargs["activation_fn"] = activation_map[key]

        return policy_kwargs

    def get_default_config(self) -> Dict[str, Any]:
        """
        SACのデフォルト設定を取得。

        Returns:
            デフォルト設定の辞書

        Example:
            >>> config = SACAlgorithm.get_default_config()
            >>> config["learning_rate"]
            0.0003
        """
        return DEFAULT_SAC_CONFIG.copy()

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        SAC設定の妥当性を検証。

        Args:
            config: 検証する設定辞書

        Returns:
            設定が妥当ならTrue

        Raises:
            ValueError: 必須パラメータが不足している場合

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

        # 値の範囲チェック
        if config["learning_rate"] <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {config['learning_rate']}"
            )

        if config["buffer_size"] <= 0:
            raise ValueError(
                f"buffer_size must be positive, got {config['buffer_size']}"
            )

        if config["batch_size"] <= 0:
            raise ValueError(f"batch_size must be positive, got {config['batch_size']}")

        # buffer_size >= batch_size
        if config["buffer_size"] < config["batch_size"]:
            raise ValueError(
                f"buffer_size ({config['buffer_size']}) must be >= batch_size ({config['batch_size']})"
            )

        # ネットワークタイプの検証
        network_type = config.get("network_type", "mlp")
        if network_type not in ["mlp", "lstm", "transformer", "efficient"]:
            raise ValueError(
                f"Unsupported network_type: {network_type}. Must be one of: mlp, lstm, transformer, efficient"
            )

        # LSTM/Transformer固有のパラメータ検証
        if network_type == "lstm":
            if config.get("lstm_hidden_size", 128) <= 0:
                raise ValueError("lstm_hidden_size must be positive")
            if config.get("lstm_layers", 2) <= 0:
                raise ValueError("lstm_layers must be positive")
        elif network_type == "transformer":
            if config.get("transformer_d_model", 128) <= 0:
                raise ValueError("transformer_d_model must be positive")
            if config.get("transformer_n_heads", 8) <= 0:
                raise ValueError("transformer_n_heads must be positive")
            if (
                config.get("transformer_d_model", 128)
                % config.get("transformer_n_heads", 8)
                != 0
            ):
                raise ValueError(
                    "transformer_d_model must be divisible by transformer_n_heads"
                )

        # sequence_lengthの検証（LSTM/Transformer共通）
        if network_type in ["lstm", "transformer"]:
            if config.get("sequence_length", 10) <= 0:
                raise ValueError("sequence_length must be positive")

        # 転移学習設定の検証
        if config.get("transfer_learning_enabled", False):
            pretrained_path = config.get("pretrained_model_path")
            if not pretrained_path:
                raise ValueError(
                    "transfer_learning_enabled is True but pretrained_model_path is not specified"
                )

            freeze_layers = config.get("freeze_layers", 0)
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
            if fine_tune_lr is not None and fine_tune_lr <= 0:
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
                if not (0.0 < pruning_amount < 1.0):
                    raise ValueError(
                        f"pruning_amount must be between 0.0 and 1.0, got {pruning_amount}"
                    )

            # 蒸留設定の検証
            if "distillation" in compression_techniques:
                teacher_path = config.get("teacher_model_path")
                if not teacher_path:
                    raise ValueError(
                        "distillation requested but teacher_model_path not specified"
                    )

                distillation_temp = config.get("distillation_temperature", 2.0)
                if distillation_temp <= 0:
                    raise ValueError("distillation_temperature must be positive")

                distillation_alpha = config.get("distillation_alpha", 0.5)
                if not (0.0 <= distillation_alpha <= 1.0):
                    raise ValueError("distillation_alpha must be between 0.0 and 1.0")

        logger.debug(f"SAC config validation passed: {config}")
        return True

    def create_model(
        self,
        env: VecEnv,
        config: Dict[str, Any],
        tensorboard_log: Optional[str] = None,
    ) -> BaseAlgorithm:
        """
        SACモデルを作成。

        Args:
            env: 訓練環境（VecEnv）
            config: SAC設定（ハイパーパラメータ等）
            tensorboard_log: TensorBoardログディレクトリ（Optional）

        Returns:
            作成されたSACモデル

        Raises:
            ValueError: 設定が不正な場合

        Example:
            >>> sac = SACAlgorithm()
            >>> config = sac.get_default_config()
            >>> model = sac.create_model(vec_env, config, "./logs")
        """
        # デフォルト設定とマージして検証
        config = {**self.get_default_config(), **config}
        self.validate_config(config)

        logger.info(f"Creating SAC model with config: {config}")

        # Stable-Baselines3 SAC用パラメータ抽出
        sac_params = {
            "policy": config.get("policy", "MlpPolicy"),
            "env": env,
            "learning_rate": config["learning_rate"],
            "buffer_size": config["buffer_size"],
            "learning_starts": config.get(
                "learning_starts", DEFAULT_SAC_CONFIG["learning_starts"]
            ),
            "batch_size": config["batch_size"],
            "tau": config.get("tau", DEFAULT_SAC_CONFIG["tau"]),
            "gamma": config.get("gamma", DEFAULT_SAC_CONFIG["gamma"]),
            "train_freq": config.get("train_freq", DEFAULT_SAC_CONFIG["train_freq"]),
            "gradient_steps": config.get(
                "gradient_steps", DEFAULT_SAC_CONFIG["gradient_steps"]
            ),
            "ent_coef": config.get("ent_coef", DEFAULT_SAC_CONFIG["ent_coef"]),
            "target_entropy": config.get(
                "target_entropy", DEFAULT_SAC_CONFIG["target_entropy"]
            ),
            "target_update_interval": config.get(
                "target_update_interval", DEFAULT_SAC_CONFIG["target_update_interval"]
            ),
            "use_sde": config.get("use_sde", DEFAULT_SAC_CONFIG["use_sde"]),
            "sde_sample_freq": config.get(
                "sde_sample_freq", DEFAULT_SAC_CONFIG["sde_sample_freq"]
            ),
            "use_sde_at_warmup": config.get(
                "use_sde_at_warmup", DEFAULT_SAC_CONFIG["use_sde_at_warmup"]
            ),
            "verbose": config.get("verbose", DEFAULT_SAC_CONFIG["verbose"]),
            "tensorboard_log": tensorboard_log,
            "device": config.get("device", DEFAULT_SAC_CONFIG["device"]),
        }

        # policy_kwargsがあれば追加（文字列定義の活性化関数を解決）
        network_type = config.get("network_type", "mlp")
        policy_kwargs = self._resolve_policy_kwargs(
            config.get("policy_kwargs"),
            network_type=network_type,
            sequence_length=config.get(
                "sequence_length", DEFAULT_SAC_CONFIG["sequence_length"]
            ),
            lstm_hidden_size=config.get(
                "lstm_hidden_size", DEFAULT_SAC_CONFIG["lstm_hidden_size"]
            ),
            lstm_layers=config.get("lstm_layers", DEFAULT_SAC_CONFIG["lstm_layers"]),
            transformer_d_model=config.get(
                "transformer_d_model", DEFAULT_SAC_CONFIG["transformer_d_model"]
            ),
            transformer_n_heads=config.get(
                "transformer_n_heads", DEFAULT_SAC_CONFIG["transformer_n_heads"]
            ),
            transformer_n_layers=config.get(
                "transformer_n_layers", DEFAULT_SAC_CONFIG["transformer_n_layers"]
            ),
            transformer_d_ff=config.get(
                "transformer_d_ff", DEFAULT_SAC_CONFIG["transformer_d_ff"]
            ),
            network_dropout=config.get(
                "network_dropout", DEFAULT_SAC_CONFIG["network_dropout"]
            ),
        )
        if policy_kwargs is not None:
            sac_params["policy_kwargs"] = policy_kwargs

        # Set policy based on network type
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
            self._apply_model_compression(self._model, config)

        # 特徴量情報をログに出力
        feature_set = config.get("feature_set", "curated")
        expected_features = config.get("expected_features", 88)
        try:
            # 環境の観測空間から特徴量数を取得
            if hasattr(env, "observation_space"):
                obs_shape = env.observation_space.shape
                actual_features = obs_shape[0] if len(obs_shape) > 0 else 0
                logger.info(
                    f"SAC model created with feature integration: "
                    f"feature_set={feature_set}, expected_features={expected_features}, "
                    f"actual_features={actual_features}, lr={config['learning_rate']}, "
                    f"buffer_size={config['buffer_size']}, batch_size={config['batch_size']}, "
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
                    f"lr={config['learning_rate']}, buffer_size={config['buffer_size']}, "
                    f"batch_size={config['batch_size']}, ent_coef={config.get('ent_coef', 'auto')}"
                )
        except Exception as e:
            logger.warning(
                f"Could not retrieve feature information from environment: {e}"
            )
            logger.info(
                f"SAC model created: feature_set={feature_set}, expected_features={expected_features}, "
                f"lr={config['learning_rate']}, buffer_size={config['buffer_size']}, "
                f"batch_size={config['batch_size']}, ent_coef={config.get('ent_coef', 'auto')}"
            )

        # 説明可能性アナライザーの初期化
        if config.get("explainability_enabled", False):
            self._initialize_explainability_analyzer(config)

        return self._model

    def _apply_transfer_learning(
        self, model: BaseAlgorithm, config: Dict[str, Any]
    ) -> None:
        """
        転移学習をモデルに適用。

        Args:
            model: 適用対象のSACモデル
            config: 転移学習設定を含む設定辞書
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
            freeze_layers = config.get("freeze_layers", 0)
            if freeze_layers > 0:
                self._freeze_layers(model, freeze_layers, config)

            # ファインチューニング学習率の設定
            fine_tune_lr = config.get("fine_tune_learning_rate")
            if fine_tune_lr is not None:
                self._set_fine_tune_learning_rate(model, fine_tune_lr)

            logger.info("Transfer learning applied successfully")

        except Exception as e:
            logger.error(f"Failed to apply transfer learning: {e}")
            raise

    def _apply_model_compression(
        self, model: BaseAlgorithm, config: Dict[str, Any]
    ) -> None:
        """
        モデル圧縮をモデルに適用。

        Args:
            model: 適用対象のSACモデル
            config: モデル圧縮設定を含む設定辞書
        """
        compression_techniques = config.get("compression_techniques", [])
        if not compression_techniques:
            logger.warning(
                "Model compression enabled but no compression_techniques specified"
            )
            return

        try:
            logger.info(
                f"Applying model compression techniques: {compression_techniques}"
            )

            # 圧縮パイプラインの設定
            techniques_config = {}
            teacher_model = None  # 蒸留用の教師モデル

            if "quantization" in compression_techniques:
                techniques_config["quantization"] = {
                    "type": "quantization",
                    "quantization_type": config.get("quantization_type", "dynamic"),
                }

            if "pruning" in compression_techniques:
                techniques_config["pruning"] = {
                    "type": "pruning",
                    "pruning_type": config.get("pruning_type", "l1_unstructured"),
                    "amount": config.get("pruning_amount", 0.3),
                }

            if "distillation" in compression_techniques:
                # 教師モデルが必要
                teacher_model_path = config.get("teacher_model_path")
                if teacher_model_path:
                    teacher_model = SAC.load(teacher_model_path, device=model.device)
                    techniques_config["distillation"] = {
                        "type": "distillation",
                        "temperature": config.get("distillation_temperature", 2.0),
                        "alpha": config.get("distillation_alpha", 0.5),
                    }
                else:
                    logger.warning(
                        "Knowledge distillation requested but teacher_model_path not specified"
                    )

            if not techniques_config:
                logger.warning("No valid compression techniques configured")
                return

            # 圧縮マネージャーの作成と適用
            self.compression_manager = create_compression_pipeline(techniques_config)

            # モデルのポリシーを取得して圧縮
            policy = model.policy

            # 蒸留の場合は教師モデルを渡す
            compress_kwargs = {}
            if teacher_model is not None:
                compress_kwargs["teacher_model"] = teacher_model

            compressed_policy = self.compression_manager.compress_model(
                policy, list(techniques_config.keys()), **compress_kwargs
            )

            # 圧縮されたポリシーをモデルに設定
            model.policy = compressed_policy

            # 圧縮モデルの保存
            compressed_path = config.get("compressed_model_path")
            if compressed_path:
                self.compression_manager.save_compressed_model(model, compressed_path)
                logger.info(f"Compressed model saved to {compressed_path}")

            # 圧縮レポートの取得
            compression_report = self.compression_manager.get_compression_report()
            logger.info(f"Model compression completed: {compression_report}")

        except Exception as e:
            logger.error(f"Failed to apply model compression: {e}")
            raise

    def _validate_pretrained_model(
        self,
        model: BaseAlgorithm,
        pretrained_model: BaseAlgorithm,
        config: Dict[str, Any],
    ) -> None:
        """
        事前学習済みモデルの妥当性を検証。

        Args:
            model: 現在のモデル
            pretrained_model: 事前学習済みモデル
            config: 設定辞書
        """
        # ネットワークタイプの一致を確認
        current_network_type = config.get("network_type", "mlp")

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
                    f"Network type mismatch: expected {expected_policy.__name__}, "
                    f"got {pretrained_policy_type} in pretrained model"
                )

        logger.info(f"Pretrained model validation passed: {pretrained_policy_type}")

    def _freeze_layers(
        self,
        model: BaseAlgorithm,
        freeze_layers: Union[int, float],
        config: Dict[str, Any],
    ) -> None:
        """
        指定された層を凍結。

        Args:
            model: 対象モデル
            freeze_layers: 凍結する層の数または割合
            config: 設定辞書
        """
        network_type = config.get("network_type", "mlp")

        if network_type == "mlp":
            # MLPの場合：層数を指定
            self._freeze_mlp_layers(model, freeze_layers)
        elif network_type in ["lstm", "transformer"]:
            # LSTM/Transformerの場合：割合を指定
            self._freeze_advanced_layers(model, freeze_layers, network_type)

        logger.info(f"Froze {freeze_layers} layers for {network_type} network")

    def _freeze_mlp_layers(
        self, model: BaseAlgorithm, freeze_layers: Union[int, float]
    ) -> None:
        """
        MLPネットワークの層を凍結。

        Args:
            model: SACモデル
            freeze_layers: 凍結する層数
        """
        policy = model.policy

        # Actorネットワークの層を凍結
        if hasattr(policy, "actor"):
            self._freeze_network_layers(policy.actor, freeze_layers)

        # Criticネットワークの層を凍結
        if hasattr(policy, "critic"):
            self._freeze_network_layers(policy.critic, freeze_layers)

    def _freeze_network_layers(
        self, network: nn.Module, freeze_layers: Union[int, float]
    ) -> None:
        """
        ネットワークの層を凍結。

        Args:
            network: 対象ネットワーク
            freeze_layers: 凍結する層数
        """
        layer_count = 0
        # intに変換
        freeze_count = (
            int(freeze_layers)
            if isinstance(freeze_layers, (int, float))
            else freeze_layers
        )
        for module in network.modules():
            if isinstance(module, nn.Linear) and layer_count < freeze_count:
                for param in module.parameters():
                    param.requires_grad_(False)
                logger.debug(f"Froze layer {layer_count}")
                layer_count += 1

    def _freeze_advanced_layers(
        self, model: BaseAlgorithm, freeze_ratio: float, network_type: str
    ) -> None:
        """
        LSTM/Transformerネットワークの層を割合で凍結。

        Args:
            model: SACモデル
            freeze_ratio: 凍結する割合（0.0-1.0）
            network_type: ネットワークタイプ
        """
        policy = model.policy

        if not hasattr(policy, "features_extractor"):
            return

        extractor = policy.features_extractor

        if network_type == "lstm":
            # LSTM層の凍結
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
            # Transformer層の凍結
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
        """
        ファインチューニング用の学習率を設定。

        Args:
            model: SACモデル
            learning_rate: ファインチューニング学習率
        """
        # オプティマイザの学習率を更新
        if hasattr(model, "policy_optimizer"):
            for param_group in model.policy_optimizer.param_groups:
                param_group["lr"] = learning_rate

        logger.info(f"Set fine-tuning learning rate to {learning_rate}")

    def train(
        self,
        model: BaseAlgorithm,
        total_timesteps: int,
        callback: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ) -> BaseAlgorithm:
        """
        SACモデルを訓練。

        Args:
            model: 訓練するSACモデル
            total_timesteps: 訓練ステップ数
            callback: 訓練中のコールバック（Optional）
            **kwargs: その他のlearn()引数

        Returns:
            訓練済みモデル

        Raises:
            ValueError: モデルが初期化されていない場合

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
        """
        モデルを保存。

        Args:
            model: 保存するモデル
            save_path: 保存先パス

        Example:
            >>> sac.save(model, "models/sac_v395a.zip")
        """
        model.save(save_path)
        logger.info(f"SAC model saved to {save_path}")

    @staticmethod
    def load(load_path: str, env: Optional[VecEnv] = None) -> BaseAlgorithm:
        """
        モデルを読み込み。

        Args:
            load_path: モデルファイルパス
            env: 環境（Optional、推論時に必要）

        Returns:
            読み込まれたモデル

        Example:
            >>> model = SACAlgorithm.load("models/sac_v395a.zip", env=vec_env)
        """
        model = SAC.load(load_path, env=env)
        logger.info(f"SAC model loaded from {load_path}")
        return model

    def _initialize_explainability_analyzer(self, config: Dict[str, Any]) -> None:
        """
        説明可能性アナライザーを初期化。

        Args:
            config: 説明可能性設定を含む設定辞書
        """
        try:
            # 説明可能性設定の作成
            explainability_config = ExplainabilityConfig(
                enabled=config.get("explainability_enabled", True),
                explanation_method=config.get("explanation_method", "shap"),
                shap_max_evals=config.get("shap_max_evals", 1000),
                shap_batch_size=config.get("shap_batch_size", 50),
                generate_natural_language=config.get("natural_language_enabled", True),
                enable_visualization=config.get("enable_visualization", True),
                plot_format=config.get("plot_format", "png"),
                cache_explanations=config.get("explanation_cache_enabled", True),
                cache_ttl_seconds=config.get("cache_ttl_seconds", 3600),
                generate_reports=config.get("report_generation", True),
            )

            # 説明可能性アナライザーの作成
            self.explainability_analyzer = ExplainabilityAnalyzer(explainability_config)

            logger.info("Explainability analyzer initialized for SAC algorithm")

        except Exception as e:
            logger.error(f"Failed to initialize explainability analyzer: {e}")
            self.explainability_analyzer = None

    def explain_decision(
        self,
        observation: Any,
        action: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        モデルの決定を説明。

        Args:
            observation: 観測データ
            action: 実行された行動（Optional）
            context: 追加の文脈情報（Optional）

        Returns:
            説明結果の辞書、またはNone（説明可能性が無効の場合）
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
            return explanation_result.to_dict()

        except Exception as e:
            logger.error(f"Failed to explain decision: {e}")
            return None

    def generate_explanation_report(
        self,
        observations: Any,
        actions: Optional[Any] = None,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        説明レポートを生成。

        Args:
            observations: 観測データのバッチ
            actions: 対応する行動のバッチ（Optional）
            output_path: レポート出力パス（Optional）

        Returns:
            レポートファイルのパス、またはNone
        """
        if self.explainability_analyzer is None or self._model is None:
            logger.warning("Explainability analyzer or model not initialized")
            return None

        try:
            # 複数の観測データに対して説明を生成
            explanations = []

            # observationsが単一の観測かバッチかを判定
            if isinstance(observations, (list, tuple)) and len(observations) > 0:
                obs_list = observations
            else:
                obs_list = [observations]

            # 各観測に対して説明を生成
            for i, obs in enumerate(obs_list):
                action = (
                    actions[i] if actions is not None and i < len(actions) else None
                )
                explanation = self.explainability_analyzer.explain_prediction(
                    model=self._model, input_data=obs, prediction=action
                )
                explanations.append(explanation)

            # レポートを生成
            return self.explainability_analyzer.generate_explanation_report(
                explanations=explanations, output_path=output_path
            )

        except Exception as e:
            logger.error(f"Failed to generate explanation report: {e}")
            return None
