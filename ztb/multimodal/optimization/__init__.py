"""最適化モジュール

マルチモーダル学習モデルのパフォーマンス最適化を提供。
モデル圧縮、量子化、推論最適化を含む。
"""

__version__ = "1.0.0"

from .compression import *
from .quantization import *
from .inference import *