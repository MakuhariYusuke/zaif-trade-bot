"""設定管理モジュール

マルチモーダル学習モジュールの設定を管理するクラスを提供。
YAMLファイルからの設定読み込みと検証を行う。
"""

__version__ = "1.0.0"

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml  # type: ignore


@dataclass
class DataConfig:
    """データ収集設定"""

    symbols: list = field(default_factory=lambda: ["USDJPY", "EURUSD", "GBPUSD"])
    timeframe: str = "1m"
    lookback_days: int = 30
    sources: list = field(default_factory=lambda: ["newsapi", "alphavantage"])
    keywords: list = field(default_factory=lambda: ["forex", "currency", "economic"])
    sentiment_threshold: float = 0.1
    indicators: list = field(default_factory=lambda: ["GDP", "CPI", "UNEMPLOYMENT"])
    countries: list = field(default_factory=lambda: ["US", "JP", "EU"])


@dataclass
class FeaturesConfig:
    """特徴量エンジニアリング設定"""

    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    max_length: int = 512
    embedding_dim: int = 768
    normalization: str = "standard"
    outlier_threshold: float = 3.0
    attention_heads: int = 8
    fusion_layers: int = 2


@dataclass
class ModelConfig:
    """モデルアーキテクチャ設定"""

    price_encoder_hidden_dims: list = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1
    text_encoder_model_name: str = "bert-base-uncased"
    fine_tune: bool = True
    economic_encoder_hidden_dims: list = field(default_factory=lambda: [64, 32])
    attention_dim: int = 256
    num_heads: int = 8
    actor_hidden_dims: list = field(default_factory=lambda: [256, 128])
    critic_hidden_dims: list = field(default_factory=lambda: [256, 128])
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2


@dataclass
class TrainingConfig:
    """トレーニング設定"""

    batch_size: int = 64
    learning_rate: float = 3e-4
    epochs: int = 100
    optimizer_type: str = "adam"
    weight_decay: float = 1e-4
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.1
    cross_modal_weight: float = 1.0
    patience: int = 10
    min_delta: float = 1e-4


@dataclass
class EvaluationConfig:
    """評価設定"""

    metrics: list = field(
        default_factory=lambda: [
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
        ]
    )
    test_split: float = 0.2
    cross_validation_folds: int = 5
    attention_maps: bool = True
    feature_importance: bool = True


@dataclass
class HardwareConfig:
    """ハードウェア設定"""

    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class APIConfig:
    """API設定"""

    newsapi_key: Optional[str] = None
    alphavantage_key: Optional[str] = None
    fred_key: Optional[str] = None


@dataclass
class MultimodalConfig:
    """マルチモーダル学習モジュールのメイン設定クラス"""

    # バージョン情報
    version: str = "1.0.0"

    # サブ設定
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    api: APIConfig = field(default_factory=APIConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MultimodalConfig":
        """
        YAMLファイルから設定を読み込む

        Args:
            yaml_path: YAMLファイルのパス

        Returns:
            MultimodalConfigインスタンス
        """
        yaml_file = Path(yaml_path)

        if not yaml_file.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {yaml_path}")

        with open(yaml_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # 環境変数の展開
        config_dict = cls._expand_env_vars(config_dict)

        return cls._from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MultimodalConfig":
        """辞書から設定を作成"""
        config_dict = cls._expand_env_vars(config_dict)
        return cls._from_dict(config_dict)

    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "MultimodalConfig":
        """辞書から設定オブジェクトを作成"""
        # 各サブ設定を作成
        data_config = DataConfig(**config_dict.get("data", {}))
        features_config = FeaturesConfig(**config_dict.get("features", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        evaluation_config = EvaluationConfig(**config_dict.get("evaluation", {}))
        hardware_config = HardwareConfig(**config_dict.get("hardware", {}))
        api_config = APIConfig(**config_dict.get("api", {}))

        return cls(
            version=config_dict.get("version", "1.0.0"),
            data=data_config,
            features=features_config,
            model=model_config,
            training=training_config,
            evaluation=evaluation_config,
            hardware=hardware_config,
            api=api_config,
        )

    @staticmethod
    def _expand_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """設定内の環境変数を展開"""

        def expand_value(value: Any) -> Any:
            if (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                env_var = value[2:-1]
                return os.getenv(env_var, value)
            elif isinstance(value, dict):
                return {k: expand_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [expand_value(item) for item in value]
            else:
                return value

        return expand_value(config_dict)  # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書に変換"""
        return {
            "version": self.version,
            "data": {
                "symbols": self.data.symbols,
                "timeframe": self.data.timeframe,
                "lookback_days": self.data.lookback_days,
                "sources": self.data.sources,
                "keywords": self.data.keywords,
                "sentiment_threshold": self.data.sentiment_threshold,
                "indicators": self.data.indicators,
                "countries": self.data.countries,
            },
            "features": {
                "model_name": self.features.model_name,
                "max_length": self.features.max_length,
                "embedding_dim": self.features.embedding_dim,
                "normalization": self.features.normalization,
                "outlier_threshold": self.features.outlier_threshold,
                "attention_heads": self.features.attention_heads,
                "fusion_layers": self.features.fusion_layers,
            },
            "model": {
                "price_encoder_hidden_dims": self.model.price_encoder_hidden_dims,
                "dropout": self.model.dropout,
                "text_encoder_model_name": self.model.text_encoder_model_name,
                "fine_tune": self.model.fine_tune,
                "economic_encoder_hidden_dims": self.model.economic_encoder_hidden_dims,
                "attention_dim": self.model.attention_dim,
                "num_heads": self.model.num_heads,
                "actor_hidden_dims": self.model.actor_hidden_dims,
                "critic_hidden_dims": self.model.critic_hidden_dims,
                "gamma": self.model.gamma,
                "tau": self.model.tau,
                "alpha": self.model.alpha,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "epochs": self.training.epochs,
                "optimizer_type": self.training.optimizer_type,
                "weight_decay": self.training.weight_decay,
                "reconstruction_weight": self.training.reconstruction_weight,
                "kl_weight": self.training.kl_weight,
                "cross_modal_weight": self.training.cross_modal_weight,
                "patience": self.training.patience,
                "min_delta": self.training.min_delta,
            },
            "evaluation": {
                "metrics": self.evaluation.metrics,
                "test_split": self.evaluation.test_split,
                "cross_validation_folds": self.evaluation.cross_validation_folds,
                "attention_maps": self.evaluation.attention_maps,
                "feature_importance": self.evaluation.feature_importance,
            },
            "hardware": {
                "device": self.hardware.device,
                "num_workers": self.hardware.num_workers,
                "pin_memory": self.hardware.pin_memory,
            },
            "api": {
                "newsapi_key": self.api.newsapi_key,
                "alphavantage_key": self.api.alphavantage_key,
                "fred_key": self.api.fred_key,
            },
        }

    def save_yaml(self, yaml_path: str) -> None:
        """設定をYAMLファイルに保存"""
        config_dict = self.to_dict()
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
