"""
Reward Configuration Schema and Validation

Phase 3 Day 3: Reward Config作成の型安全性確保
YAMLファイルの検証とRewardSettingsへの変換を提供
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ztb.trading.environment.utils.config import RewardSettings
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RewardConfigValidationError(Exception):
    """Reward Config検証エラー"""
    field: str
    message: str
    value: Any = None


class RewardConfigSchema:
    """Reward Configuration Schema定義とバリデーション"""

    # 必須フィールド
    REQUIRED_FIELDS = {
        "name": str,
        "description": str,
        "curriculum_stage": str,
        "reward_scale": (int, float),
    }

    # オプショナルフィールド（型チェック用）
    OPTIONAL_FIELDS = {
        # 基本設定
        "use_simple_reward": bool,
        "trading_bonus": (int, float),
        "balance_penalty": (int, float),
        "balance_penalty_tolerance": (int, float),
        
        # 重み設定
        "profit_weight": (int, float),
        "risk_weight": (int, float),
        "consistency_weight": (int, float),
        
        # ポジション管理
        "position_soft_cap": (int, float),
        "position_penalty_scale": (int, float),
        "position_penalty_exponent": (int, float),
        
        # 取引管理
        "trade_frequency_penalty": (int, float),
        "trade_frequency_halflife": (int, float),
        "trade_cooldown_steps": int,
        "trade_cooldown_penalty": (int, float),
        "max_consecutive_trades": int,
        "consecutive_trade_penalty": (int, float),
        
        # 在庫管理
        "inventory_window": int,
        "inventory_penalty_scale": (int, float),
        
        # ボラティリティ
        "volatility_window": int,
        "volatility_penalty_scale": (int, float),
        
        # リスク調整リターン
        "sharpe_bonus_scale": (int, float),
        "sortino_bonus_scale": (int, float),
        "calmar_bonus_scale": (int, float),
        
        # クリッピング
        "reward_clip_value": (int, float),
        "reward_clip_min": (int, float),
        "reward_clip_max": (int, float),
        
        # その他
        "entropy_bonus": (int, float),
        "entropy_regularization": (int, float),
        "action_smoothing": (int, float),
        "consistency_penalty": (int, float),
        "enable_forced_diversity": bool,
        
        # Ultra Profit
        "ultra_profit_multiplier": (int, float),
        "ultra_risk_multiplier": (int, float),
        
        # 未実現損失
        "unrealized_loss_penalty_enabled": bool,
        "unrealized_loss_penalty_base": (int, float),
        "unrealized_loss_penalty_max_steps": int,
        
        # リスト/辞書フィールド
        "profit_bonus_multipliers": list,
        "asymmetric_reward_scaling": dict,
        "dynamic_reward_shaping": dict,
        "behavior_optimization": dict,
        "regime_detection_config": dict,
        "forced_balance": dict,
        "curriculum_learning": dict,
        "metadata": dict,
    }

    # 値範囲制約
    VALUE_CONSTRAINTS = {
        "reward_scale": (0.0, None),  # 正の値
        "balance_penalty": (0.0, None),
        "profit_weight": (0.0, None),
        "risk_weight": (0.0, None),
        "consistency_weight": (0.0, None),
        "position_soft_cap": (0.0, 1.0),
        "position_penalty_exponent": (1.0, 5.0),
        "trade_cooldown_steps": (0, 100),
        "max_consecutive_trades": (1, 20),
        "inventory_window": (1, 200),
        "volatility_window": (5, 200),
        "ultra_profit_multiplier": (0.5, 5.0),
        "ultra_risk_multiplier": (0.1, 2.0),
    }

    # 許可されたcurriculum_stage値
    VALID_CURRICULUM_STAGES = {
        "simple",
        "trading_focused",
        "profit_optimized",
        "ultra_profit",
        "forced_balance",
        "confidence_penalty",
    }

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> List[str]:
        """Config検証

        Args:
            config: YAMLから読み込んだ設定辞書

        Returns:
            検証エラーのリスト（空なら検証成功）
        """
        errors: List[str] = []

        # 必須フィールドチェック
        for field, expected_type in cls.REQUIRED_FIELDS.items():
            if field not in config:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(config[field], expected_type):
                errors.append(
                    f"Field '{field}' must be {expected_type}, got {type(config[field])}"
                )

        # curriculum_stage検証
        if "curriculum_stage" in config:
            stage = config["curriculum_stage"]
            if stage not in cls.VALID_CURRICULUM_STAGES:
                errors.append(
                    f"Invalid curriculum_stage '{stage}'. "
                    f"Must be one of: {cls.VALID_CURRICULUM_STAGES}"
                )

        # オプショナルフィールドの型チェック
        for field, expected_type in cls.OPTIONAL_FIELDS.items():
            if field in config:
                value = config[field]
                if not isinstance(value, expected_type):
                    errors.append(
                        f"Field '{field}' must be {expected_type}, got {type(value)}"
                    )

        # 値範囲制約チェック
        for field, (min_val, max_val) in cls.VALUE_CONSTRAINTS.items():
            if field in config:
                value = config[field]
                if isinstance(value, (int, float)):
                    if min_val is not None and value < min_val:
                        errors.append(
                            f"Field '{field}' value {value} is below minimum {min_val}"
                        )
                    if max_val is not None and value > max_val:
                        errors.append(
                            f"Field '{field}' value {value} is above maximum {max_val}"
                        )

        # profit_bonus_multipliersの検証
        if "profit_bonus_multipliers" in config:
            multipliers = config["profit_bonus_multipliers"]
            if not all(isinstance(x, (int, float)) and x >= 1.0 for x in multipliers):
                errors.append(
                    "profit_bonus_multipliers must be a list of numbers >= 1.0"
                )

        # asymmetric_reward_scalingの検証
        if "asymmetric_reward_scaling" in config:
            ars = config["asymmetric_reward_scaling"]
            required_keys = {
                "long_position_reward_multiplier",
                "short_position_reward_multiplier",
                "long_position_penalty_multiplier",
                "short_position_penalty_multiplier",
            }
            for key in required_keys:
                if key not in ars:
                    errors.append(
                        f"asymmetric_reward_scaling missing required key: {key}"
                    )

        # dynamic_reward_shapingの検証
        if "dynamic_reward_shaping" in config:
            drs = config["dynamic_reward_shaping"]
            if "enabled" not in drs:
                errors.append("dynamic_reward_shaping must have 'enabled' field")

        return errors

    @classmethod
    def load_and_validate(cls, config_path: Union[str, Path]) -> Dict[str, Any]:
        """YAMLファイルを読み込んで検証

        Args:
            config_path: YAMLファイルパス

        Returns:
            検証済み設定辞書

        Raises:
            FileNotFoundError: ファイルが見つからない
            ValueError: 検証エラー
            yaml.YAMLError: YAML解析エラー
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML: {e}") from e

        # 検証実行
        errors = cls.validate(config)
        if errors:
            error_msg = "\n".join(f"  - {err}" for err in errors)
            raise ValueError(f"Config validation failed:\n{error_msg}")

        logger.info(f"✓ Validated config: {config.get('name', 'unknown')}")
        return config


def load_reward_config(config_path: Union[str, Path]) -> RewardSettings:
    """YAML ConfigをRewardSettingsオブジェクトに変換

    Args:
        config_path: YAMLファイルパス

    Returns:
        RewardSettings: 型安全な設定オブジェクト

    Raises:
        FileNotFoundError: ファイルが見つからない
        ValueError: 検証エラー
    """
    # 検証付き読み込み
    config = RewardConfigSchema.load_and_validate(config_path)

    # RewardSettingsへの変換（YAMLキーとRewardSettings属性のマッピング）
    settings_dict: Dict[str, Any] = {}

    # 基本設定
    for key in [
        "use_simple_reward",
        "reward_scale",
        "trading_bonus",
        "balance_penalty",
        "balance_penalty_tolerance",
        "profit_weight",
        "risk_weight",
        "consistency_weight",
        "position_soft_cap",
        "position_penalty_scale",
        "position_penalty_exponent",
        "inventory_window",
        "inventory_penalty_scale",
        "trade_frequency_penalty",
        "trade_frequency_halflife",
        "trade_cooldown_steps",
        "trade_cooldown_penalty",
        "max_consecutive_trades",
        "consecutive_trade_penalty",
        "volatility_window",
        "volatility_penalty_scale",
        "sharpe_bonus_scale",
        "sortino_bonus_scale",
        "calmar_bonus_scale",
        "reward_clip_value",
        "reward_clip_min",
        "reward_clip_max",
        "entropy_bonus",
        "entropy_regularization",
        "action_smoothing",
        "consistency_penalty",
        "enable_forced_diversity",
        "ultra_profit_multiplier",
        "ultra_risk_multiplier",
        "unrealized_loss_penalty_enabled",
        "unrealized_loss_penalty_base",
        "unrealized_loss_penalty_max_steps",
        "profit_bonus_multipliers",
        "asymmetric_reward_scaling",
        "dynamic_reward_shaping",
    ]:
        if key in config:
            settings_dict[key] = config[key]

    # curriculum_stageもRewardSettingsに追加
    settings_dict["curriculum_stage"] = config.get("curriculum_stage", "simple")

    # custom_reward_paramsに追加情報を格納
    custom_params = {}
    
    # behavior_optimization設定
    if "behavior_optimization" in config:
        custom_params.update(config["behavior_optimization"])
    
    # forced_balance設定
    if "forced_balance" in config:
        fb = config["forced_balance"]
        if fb.get("enabled", False):
            for key, value in fb.items():
                custom_params[f"forced_balance_{key}"] = value
    
    # curriculum_learning設定
    if "curriculum_learning" in config:
        custom_params["curriculum_learning"] = config["curriculum_learning"]
    
    # regime_detection_config設定
    if "regime_detection_config" in config:
        custom_params["regime_detection_config"] = config["regime_detection_config"]
    
    # メタデータ
    if "metadata" in config:
        custom_params["metadata"] = config["metadata"]

    settings_dict["custom_reward_params"] = custom_params

    # RewardSettingsオブジェクト作成
    try:
        reward_settings = RewardSettings(**settings_dict)
        logger.info(
            f"✓ Loaded reward config: {config.get('name', 'unknown')} "
            f"(stage={reward_settings.curriculum_stage})"
        )
        return reward_settings
    except Exception as e:
        raise ValueError(f"Failed to create RewardSettings: {e}") from e


def list_available_configs(config_dir: Union[str, Path] = "configs/rewards") -> List[Path]:
    """利用可能なReward Configファイルをリスト

    Args:
        config_dir: Configディレクトリパス

    Returns:
        YAMLファイルパスのリスト
    """
    config_dir = Path(config_dir)
    if not config_dir.exists():
        logger.warning(f"Config directory not found: {config_dir}")
        return []

    yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
    return sorted(yaml_files)


def compare_configs(
    config_paths: List[Union[str, Path]],
) -> Dict[str, Dict[str, Any]]:
    """複数のConfigを比較

    Args:
        config_paths: 比較するConfigファイルパスのリスト

    Returns:
        Config名をキーとした設定辞書
    """
    comparison: Dict[str, Dict[str, Any]] = {}

    for path in config_paths:
        config = RewardConfigSchema.load_and_validate(path)
        name = config.get("name", Path(path).stem)
        comparison[name] = {
            "curriculum_stage": config.get("curriculum_stage"),
            "profit_weight": config.get("profit_weight"),
            "risk_weight": config.get("risk_weight"),
            "consistency_weight": config.get("consistency_weight"),
            "balance_penalty": config.get("balance_penalty"),
            "ultra_profit_multiplier": config.get("ultra_profit_multiplier", 1.0),
            "dynamic_shaping_enabled": config.get("dynamic_reward_shaping", {}).get(
                "enabled", False
            ),
        }

    return comparison


__all__ = [
    "RewardConfigSchema",
    "RewardConfigValidationError",
    "load_reward_config",
    "list_available_configs",
    "compare_configs",
]
