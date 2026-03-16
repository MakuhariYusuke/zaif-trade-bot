"""Walk-Forward Splitter: 時系列安全な複数分割生成"""

import logging
from typing import List, NamedTuple

import pandas as pd

logger = logging.getLogger(__name__)


class TimeSeriesWindow(NamedTuple):
    """個別ウィンドウ定義"""
    window_id: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


class WalkForwardSplitter:
    """時系列安全な複数分割生成"""

    def __init__(
        self,
        initial_train_pct: float = 0.50,
        val_pct: float = 0.15,
        test_pct: float = 0.15,
        step_pct: float = 0.15,
        embargo_days: int = 7,
    ):
        """
        Args:
            initial_train_pct: 初期訓練比率
            val_pct: 検証セット比率
            test_pct: テストセット比率
            step_pct: ウィンドウシフト比率
            embargo_days: Embargo期間
        """
        self.initial_train_pct = initial_train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.step_pct = step_pct
        self.embargo_days = embargo_days

    def split(self, df: pd.DataFrame) -> List[TimeSeriesWindow]:
        """複数ウィンドウを生成"""
        n = len(df)
        
        # サイズ計算
        train_size = int(n * self.initial_train_pct)
        val_size = int(n * self.val_pct)
        test_size = int(n * self.test_pct)
        step_size = int(n * self.step_pct)
        
        windows: List[TimeSeriesWindow] = []
        window_id = 0
        
        # ウィンドウ生成
        train_end = train_size
        
        while train_end + val_size + test_size <= n:
            val_start = train_end
            val_end = val_start + val_size
            test_start = val_end
            test_end = test_start + test_size
            
            if test_end > n:
                break
            
            window = TimeSeriesWindow(
                window_id=window_id,
                train_start=0,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
            
            windows.append(window)
            logger.info(
                f"Window {window_id}: "
                f"Train [{window.train_start}:{window.train_end}] "
                f"Val [{window.val_start}:{window.val_end}] "
                f"Test [{window.test_start}:{window.test_end}]"
            )
            
            # ウィンドウシフト
            train_end += step_size
            window_id += 1
        
        logger.info(f"\n✓ Created {len(windows)} walk-forward windows\n")
        return windows
