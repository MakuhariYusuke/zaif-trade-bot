from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from ztb.training.core.feature_schema_manager import FeatureSchemaManager
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

def create_env_from_schema(
    model_name: str,
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    models_dir: Optional[Path] = None
) -> HeavyTradingEnv:
    if models_dir is None:
        models_dir = Path("models")
    
    manager = FeatureSchemaManager(model_name, models_dir)
    metadata = manager.load_schema()
    scaler = manager.load_scaler()
    
    logger.info(f"Creating environment from schema: {model_name}")
    
    missing_features = set(metadata.feature_names) - set(df.columns)
    if missing_features:
        raise ValueError(f"Dataset is missing required features: {missing_features}")
    
    # 設定を構築（ユーザー設定を尊重）
    env_config = config.copy() if config else {}
    logger.info(f"Schema factory config: {env_config}")
    
    # 🔧 訓練時の環境設定を適用（CRITICAL FIX for BUG #HOLD_BIAS）
    # metadata.training_config["environment"] には訓練時の重要な設定が含まれる:
    # - initial_balance: 資金
    # - max_position_size: 最大ポジションサイズ
    # - transaction_cost: 取引手数料
    # - curriculum_stage: カリキュラム段階
    # - random_start: ランダム開始位置（バックテスト用）
    # - enable_action_masking: アクションマスキング
    # これらを適用しないと、ActionValidatorが取引を非合法と判定してHOLD 100%になる
    # または全エピソードで同じ開始位置になり、決定論的な同一結果になる
    training_env_config_raw = metadata.training_config.get("environment", {})
    if training_env_config_raw:
        # 辞書をコピーして変更（元のmetadataを保護）
        training_env_config = training_env_config_raw.copy()
        logger.info(f"Applying training environment config: {training_env_config}")
        
        # initial_balance → initial_portfolio_value に変換
        if "initial_balance" in training_env_config:
            training_env_config["initial_portfolio_value"] = training_env_config.pop("initial_balance")
        
        # 訓練時設定を適用（ユーザー設定で上書きされていないもののみ）
        for key, value in training_env_config.items():
            if key not in env_config:
                env_config[key] = value
                logger.debug(f"Applied training config: {key}={value}")
    else:
        logger.warning("No training environment config found in metadata")
    
    # スキーマ情報を設定に追加（特徴量情報は最優先で上書き）
    env_config.update({
        "feature_names": metadata.feature_names,
        "num_features": metadata.num_features,
        "schema_hash": metadata.schema_hash,
        "model_name": model_name,
    })
    
    # スキーマベース環境では相関削減を無効化（デフォルト）
    # ただし、ユーザーが明示的に設定した場合は尊重
    if "enable_correlation_reduction" not in (config or {}):
        env_config["enable_correlation_reduction"] = False
    
    logger.info(f"Final env_config: enable_correlation_reduction={env_config.get('enable_correlation_reduction')}")
    
    if scaler:
        env_config.update({
            "scaler_mean": scaler["mean"],
            "scaler_std": scaler["std"],
        })
    
    # 🔧 CRITICAL FIX: random_startは位置引数なので明示的に渡す
    # HeavyTradingEnv.__init__(df, config, random_start=False, ...)
    # config辞書に含めるだけでは適用されない！
    random_start = env_config.pop("random_start", False)
    logger.info(f"Creating HeavyTradingEnv with random_start={random_start}")
    
    env = HeavyTradingEnv(df=df, config=env_config, random_start=random_start)
    return env

def create_env_from_model_path(
    model_path: str,
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> HeavyTradingEnv:
    model_name = Path(model_path).stem
    model_path_obj = Path(model_path)
    if model_path_obj.parent.name == "models":
        models_dir = model_path_obj.parent
    else:
        models_dir = Path("models")
    
    return create_env_from_schema(model_name, df, config, models_dir)
