"""
マルチモーダル学習モジュール

SAC v421取引AIへのマルチモーダル統合を提供する包括的なモジュール。
価格データに加え、ニュース感情分析、経済指標、ソーシャルメディアデータを統合。

主な機能:
- 複数モダリティのデータ収集と前処理
- クロスモーダル特徴量エンジニアリング
- マルチモーダルSACエージェント
- 統合トレーニングパイプライン
- 評価と説明可能性

作成日: 2025-10-17
バージョン: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Multi-modal learning module for SAC v421 trading AI"

# 設定管理
from .config import MultimodalConfig

# 主要コンポーネントのインポート
from .core import *
from .data import *
from .features import *
from .models import *
from .training import *

# NOTE: evaluation submodule does not exist yet — removed to fix ImportError
# from .evaluation import *  # TODO: 365# add when evaluation module is created

# デフォルト設定の読み込み
def get_default_config() -> MultimodalConfig:
    """デフォルト設定を取得"""
    return MultimodalConfig.from_yaml("ztb/multimodal/config/default.yaml")

# クイックスタート関数
def create_multimodal_agent(
    price_dim: int = 156,
    text_dim: int = 768,
    economic_dim: int = 20,
    action_dim: int = 3,
):
    """
    マルチモーダルSACエージェントを簡単に作成

    Args:
        price_dim: 価格特徴量の次元数
        text_dim: テキスト埋め込みの次元数
        economic_dim: 経済指標の次元数
        action_dim: 行動の次元数

    Returns:
        設定済みのマルチモーダルエージェント
    """
    from .models.architectures.multimodal_architecture import MultiModalTradingAgent

    agent = MultiModalTradingAgent(
        price_feature_dim=price_dim,
        text_embedding_dim=text_dim,
        economic_feature_dim=economic_dim,
        action_dim=action_dim,
    )

    return agent

def create_data_pipeline():
    """
    マルチモーダルデータ収集パイプラインを作成

    Returns:
        設定済みのデータパイプライン
    """
    from .data.collectors.data_sources import FreeDataSourceManager
    from .data.preprocessors import MultiModalDataPreprocessor

    data_manager = FreeDataSourceManager()
    preprocessor = MultiModalDataPreprocessor()

    return {"data_manager": data_manager, "preprocessor": preprocessor}

# バージョン情報
def get_version_info():
    """バージョン情報を取得"""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "components": [
            "core.encoders",
            "core.attention",
            "core.fusion",
            "data.collectors",
            "data.preprocessors",
            "features.text",
            "features.numeric",
            "models.agents",
            "training.trainers",
            "evaluation.metrics",
        ],
    }

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

# 初期化完了メッセージ
logger.info("🤖 MultiModal Learning Module v%s initialized", __version__)
logger.info("📊 Ready for multi-modal trading AI development")

def __getattr__(name: str):
    """Lazily import submodules like `pretraining` when accessed as attributes.

    This avoids importing heavy submodules during top-level package import while
    allowing attribute-based access (e.g., `ztb.multimodal.pretraining`) used by
    tests and patchers.
    """
    if name == "pretraining":
        import importlib

        mod = importlib.import_module(f"{__name__}.pretraining")
        globals()[name] = mod
        return mod
    raise AttributeError(name)
