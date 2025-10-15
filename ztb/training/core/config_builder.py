"""
統一設定構築モジュール。

UnifiedTrainerの肥大化を防ぐため、設定構築ロジックを分離。
各アルゴリズムで共通して使用する設定構築機能を提供。

Example:
    >>> builder = ConfigBuilder(config)
    >>> memory_config = builder.get_memory_optimization_config()
    >>> env_config = builder.get_environment_config()
    >>> unified = builder.build_unified_config()
"""

from typing import Any, Dict, List, Optional
from ztb.utils.logging_utils import get_logger
from ztb.training.core.config_manager import ConfigManager
from ztb.trading.constants import SAC_CONTINUOUS_THRESHOLD

logger = get_logger(__name__)

# Type aliases
MemoryOptimizationConfig = Dict[str, Optional[int]]
EnvironmentConfig = Dict[str, Any]
PPOCoreConfig = Dict[str, Any]
UnifiedConfig = Dict[str, Any]


class ConfigBuilder:
    """
    設定構築の統一インターフェース。
    
    UnifiedTrainerから設定構築ロジックを抽出し、
    再利用可能なコンポーネントとして提供する。
    
    Attributes:
        config: 元の設定辞書
        config_manager: ConfigManagerインスタンス（遅延初期化）
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化。
        
        Args:
            config: 設定辞書（JSON設定ファイルから読み込んだもの）
        """
        self.config = config
        self._config_manager: Optional[ConfigManager] = None
    
    @property
    def config_manager(self) -> ConfigManager:
        """ConfigManagerの遅延初期化"""
        if self._config_manager is None:
            from ztb.training.core.config_manager import ConfigManager
            self._config_manager = ConfigManager(self.config)
        return self._config_manager
    
    def get_config_value(
        self, 
        key: str, 
        sections: Optional[List[str]] = None, 
        default: Any = None
    ) -> Any:
        """
        設定値を優先順位付きで取得。
        
        優先順位: top-level > sections（順序通り） > default
        
        Args:
            key: 設定キー
            sections: 検索するセクションのリスト（優先順）
            default: デフォルト値（基本型のみ）
            
        Returns:
            見つかった設定値、またはdefault
            
        Example:
            >>> builder = ConfigBuilder(config)
            >>> lr = builder.get_config_value(
            ...     "learning_rate",
            ...     sections=["ppo_hyperparameters", "ppo"],
            ...     default=0.0003
            ... )
        """
        # トップレベルを最初にチェック
        if key in self.config:
            return self.config[key]
        
        # 指定されたセクションを順番にチェック
        if sections:
            for section in sections:
                section_data = self.config.get(section, {})
                if key in section_data:
                    return section_data[key]
        
        return default
    
    def get_memory_optimization_config(self) -> MemoryOptimizationConfig:
        """
        メモリ最適化設定を抽出。
        
        Returns:
            メモリ最適化設定の辞書
            - data_rows_limit: 読み込む最大データ行数
            - max_features: 使用する最大特徴量数
            
        Note:
            Bug #52対応として追加。メモリクラッシュ防止。
        """
        return {
            "data_rows_limit": self.get_config_value("data_rows_limit"),
            "max_features": self.get_config_value("max_features"),
        }
    
    def get_environment_config(self) -> EnvironmentConfig:
        """
        環境設定を抽出。
        
        Returns:
            環境設定の辞書
            - max_position_size: 最大ポジションサイズ
            - initial_balance: 初期残高
            - transaction_cost: 取引コスト
            - reward_scaling: 報酬スケーリング
            - continuous_to_discrete_threshold: SAC用アクション変換閾値
        """
        from ztb.training.config.ppo_config import DEFAULT_PPO_CONFIG
        
        return {
            "max_position_size": self.get_config_value(
                "max_position_size",
                sections=["environment"],
                default=DEFAULT_PPO_CONFIG.get("max_position_size", 1.0)
            ),
            "initial_balance": self.get_config_value(
                "initial_balance",
                sections=["environment"],
                default=DEFAULT_PPO_CONFIG.get("initial_balance", 1000000)
            ),
            "transaction_cost": self.get_config_value(
                "transaction_cost",
                sections=["environment"],
                default=DEFAULT_PPO_CONFIG.get("transaction_cost", 0.001)
            ),
            "reward_scaling": self.get_config_value(
                "reward_scaling",
                sections=["environment"],
                default=DEFAULT_PPO_CONFIG.get("reward_scaling", 1.0)
            ),
            "continuous_to_discrete_threshold": self.get_config_value(
                "continuous_to_discrete_threshold",
                sections=["environment"],
                default=SAC_CONTINUOUS_THRESHOLD
            ),
        }
    
    def get_ppo_core_config(self) -> PPOCoreConfig:
        """
        PPOアルゴリズム固有の設定を抽出。
        
        Returns:
            PPOハイパーパラメータの辞書
            - learning_rate, n_steps, batch_size, n_epochs, gamma, etc.
        """
        from ztb.training.config.ppo_config import DEFAULT_PPO_CONFIG
        
        sections = ["ppo_hyperparameters", "ppo"]
        
        return {
            "learning_rate": self.get_config_value(
                "learning_rate", sections, DEFAULT_PPO_CONFIG.get("learning_rate", 3e-4)
            ),
            "n_steps": self.get_config_value(
                "n_steps", sections, DEFAULT_PPO_CONFIG.get("n_steps", 1024)
            ),
            "batch_size": self.get_config_value(
                "batch_size", sections, DEFAULT_PPO_CONFIG.get("batch_size", 32)
            ),
            "n_epochs": self.get_config_value(
                "n_epochs", sections, DEFAULT_PPO_CONFIG.get("n_epochs", 10)
            ),
            "gamma": self.get_config_value(
                "gamma", sections, DEFAULT_PPO_CONFIG.get("gamma", 0.99)
            ),
            "gae_lambda": self.get_config_value(
                "gae_lambda", sections, DEFAULT_PPO_CONFIG.get("gae_lambda", 0.95)
            ),
            "clip_range": self.get_config_value(
                "clip_range", sections, DEFAULT_PPO_CONFIG.get("clip_range", 0.2)
            ),
            "clip_range_vf": self.get_config_value(
                "clip_range_vf", sections, None
            ),
            "normalize_advantage": self.get_config_value(
                "normalize_advantage", sections, DEFAULT_PPO_CONFIG.get("normalize_advantage", True)
            ),
            "ent_coef": self.get_config_value(
                "ent_coef", sections, DEFAULT_PPO_CONFIG.get("ent_coef", 0.0)
            ),
            "vf_coef": self.get_config_value(
                "vf_coef", sections, DEFAULT_PPO_CONFIG.get("vf_coef", 0.5)
            ),
            "max_grad_norm": self.get_config_value(
                "max_grad_norm", sections, DEFAULT_PPO_CONFIG.get("max_grad_norm", 0.5)
            ),
            "use_sde": self.get_config_value(
                "use_sde", sections, DEFAULT_PPO_CONFIG.get("use_sde", False)
            ),
            "sde_sample_freq": self.get_config_value(
                "sde_sample_freq", sections, DEFAULT_PPO_CONFIG.get("sde_sample_freq", -1)
            ),
            "target_kl": self.get_config_value(
                "target_kl", sections, None
            ),
            "verbose": self.get_config_value(
                "verbose", sections, DEFAULT_PPO_CONFIG.get("verbose", 1)
            ),
        }
    
    def get_feature_config(self) -> Dict[str, Any]:
        """
        特徴量関連の設定を抽出。
        
        Returns:
            特徴量設定の辞書
            - feature_set: 使用する特徴量セット
            - custom_features: カスタム特徴量リスト
            - feature_config_path: 特徴量設定ファイルパス
            - max_features: 最大特徴量数
        """
        return {
            "feature_set": self.config.get("feature_set", "curated"),
            "custom_features": self.config.get("custom_features", None),
            "feature_config_path": self.config.get("feature_config_path", None),
            "max_features": self.config.get("max_features", None),
        }
    
    def build_unified_config(
        self,
        enable_streaming: bool = False,
        stream_batch_size: int = 256,
        total_timesteps_override: Optional[int] = None
    ) -> UnifiedConfig:
        """
        統合設定を構築。
        
        ConfigManagerを使用して全ての設定を統合する。
        
        Args:
            enable_streaming: ストリーミングデータパイプライン有効化
            stream_batch_size: ストリーミングバッチサイズ
            total_timesteps_override: total_timestepsの上書き値
            
        Returns:
            統合された設定辞書
            
        Example:
            >>> builder = ConfigBuilder(config)
            >>> unified = builder.build_unified_config(
            ...     enable_streaming=False,
            ...     total_timesteps_override=10000
            ... )
        """
        return self.config_manager.build_unified_config(
            enable_streaming=enable_streaming,
            stream_batch_size=stream_batch_size,
            total_timesteps_override=total_timesteps_override
        )
    
    def get_sac_core_config(self) -> Dict[str, Any]:
        """
        SAC（Soft Actor-Critic）アルゴリズム固有の設定を抽出。
        
        将来のSAC実装のために追加。
        
        Returns:
            SACハイパーパラメータの辞書
        """
        sections = ["sac_hyperparameters", "sac_params", "sac"]
        
        # SACデフォルト値（Stable-Baselines3準拠）
        return {
            "learning_rate": self.get_config_value(
                "learning_rate", sections, 3e-4
            ),
            "buffer_size": self.get_config_value(
                "buffer_size", sections, 50000
            ),
            "learning_starts": self.get_config_value(
                "learning_starts", sections, 1000
            ),
            "batch_size": self.get_config_value(
                "batch_size", sections, 256
            ),
            "tau": self.get_config_value(
                "tau", sections, 0.005
            ),
            "gamma": self.get_config_value(
                "gamma", sections, 0.99
            ),
            "train_freq": self.get_config_value(
                "train_freq", sections, 1
            ),
            "gradient_steps": self.get_config_value(
                "gradient_steps", sections, 1
            ),
            "target_update_interval": self.get_config_value(
                "target_update_interval", sections, 1
            ),
            "ent_coef": self.get_config_value(
                "ent_coef", sections, "auto"
            ),
            "target_entropy": self.get_config_value(
                "target_entropy", sections, "auto"
            ),
            "use_sde": self.get_config_value(
                "use_sde", sections, False
            ),
            "sde_sample_freq": self.get_config_value(
                "sde_sample_freq", sections, -1
            ),
            "use_sde_at_warmup": self.get_config_value(
                "use_sde_at_warmup", sections, False
            ),
            "policy_kwargs": self.get_config_value(
                "policy_kwargs", sections, None
            ),
            "device": self.get_config_value(
                "device", sections, "auto"
            ),
            "verbose": self.get_config_value(
                "verbose", sections, 1
            ),
        }

    def __repr__(self) -> str:
        """ConfigBuilderの文字列表現"""
        algorithm = self.config.get("algorithm", "unknown")
        model_name = self.config.get("model_name", "unnamed")
        return f"ConfigBuilder(algorithm='{algorithm}', model='{model_name}')"
