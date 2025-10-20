"""
強化学習アルゴリズムモジュール。

このモジュールは複数の強化学習アルゴリズムを統一的に扱うための
インターフェースと実装を提供する。

Architecture:
    - BaseRLAlgorithm: 全アルゴリズムの基底クラス
    - AlgorithmFactory: アルゴリズムを動的に生成するファクトリー
    - PPOAlgorithm: PPO実装
    - SACAlgorithm: SAC実装
    - (将来) TD3Algorithm: TD3実装

Usage:
    >>> from ztb.training.algorithms import AlgorithmFactory
    >>>
    >>> # 利用可能なアルゴリズムを確認
    >>> print(AlgorithmFactory.list_algorithms())
    ['ppo', 'sac']
    >>>
    >>> # SACアルゴリズムを作成
    >>> sac = AlgorithmFactory.create("sac")
    >>> model = sac.create_model(env, config)
    >>> sac.train(model, total_timesteps=100000)
    >>>
    >>> # 設定ファイルでアルゴリズムを指定
    >>> config = {
    ...     "algorithm": "sac",  # ここで切り替え可能
    ...     "sac_hyperparameters": {...}
    ... }
"""

from .algorithm_factory import AlgorithmFactory
from .base_algorithm import BaseRLAlgorithm
from .ppo import PPOAlgorithm
from .sac import SACAlgorithm

# ========================================
# アルゴリズム登録
# ========================================

# PPOを登録
AlgorithmFactory.register("ppo", PPOAlgorithm)

# SACを登録
AlgorithmFactory.register("sac", SACAlgorithm)
#
# from .td3 import TD3Algorithm
# AlgorithmFactory.register("td3", TD3Algorithm)

# ========================================
# 公開API
# ========================================

__all__ = [
    # Core classes
    "BaseRLAlgorithm",
    "AlgorithmFactory",
    # Implementations
    "PPOAlgorithm",
    "SACAlgorithm",
    # Future implementations (commented out)
    # "TD3Algorithm",
]

# ========================================
# モジュール情報
# ========================================

__version__ = "1.0.0"
__author__ = "Zaif Trade Bot Team"

# 初期化時にアルゴリズム情報を表示
import logging

logger = logging.getLogger(__name__)

info = AlgorithmFactory.get_info()
logger.debug(f"Algorithms module initialized: {info['count']} algorithms registered")
logger.debug(f"Available algorithms: {info['algorithms']}")
