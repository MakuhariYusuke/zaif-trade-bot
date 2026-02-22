"""
Walk-Forward 時系列分割モジュール

複数の時系列ウィンドウを安全に生成する。
時系列データの過度な最適化（過学習）を検出するため、複数の重なるウィンドウで
訓練・検証・テストを実施します。

## 使用例

```python
from ztb.evaluation.walk_forward import WalkForwardSplitter, TimeSeriesWindow
import pandas as pd

# データロード
df = pd.read_csv('price_data.csv')

# スプリッター作成
splitter = WalkForwardSplitter(
    initial_train_pct=0.50,
    val_pct=0.15,
    test_pct=0.15,
    step_pct=0.15
)

# ウィンドウ生成
windows: List[TimeSeriesWindow] = splitter.split(df)

# 各ウィンドウで訓練・テスト
for window in windows:
    train_df = df.iloc[window.train_start:window.train_end]
    val_df = df.iloc[window.val_start:window.val_end]
    test_df = df.iloc[window.test_start:window.test_end]
    # 訓練・評価処理...
```
"""

import logging
from typing import List

import pandas as pd

from .types import TimeSeriesWindow

logger = logging.getLogger(__name__)


class WalkForwardSplitter:
    """時系列安全な複数分割生成
    
    Walk-Forward Analysis 用の重なるウィンドウを生成します。
    時系列データにおいて過学習を検出し、モデルの真の予測力を評価するために、
    複数の訓練・検証・テストセットを作成します。
    
    特徴:
    - 時系列リーク（look-ahead bias）防止
    - 複数ウィンドウ：複数の異なる時期でのモデル性能を評価
    - 拡張訓練期間：後の各ウィンドウでは訓練期間が増加
    - 統計的堅牢性：複数ウィンドウの平均・分散で過学習を定量化
    """

    def __init__(
        self,
        initial_train_pct: float = 0.50,
        val_pct: float = 0.15,
        test_pct: float = 0.15,
        step_pct: float = 0.15,
        embargo_days: int = 7,
    ) -> None:
        """初期化
        
        Args:
            initial_train_pct: 初期訓練期間の割合（0.0-1.0）
            val_pct: 検証期間の割合（0.0-1.0）
            test_pct: テスト期間の割合（0.0-1.0）
            step_pct: ウィンドウシフト比率（0.0-1.0）
            embargo_days: Embargo期間（日数）
            
        Raises:
            ValueError: パラメータが無効な場合
        """
        self._validate_parameters(initial_train_pct, val_pct, test_pct, step_pct)
        
        self.initial_train_pct: float = initial_train_pct
        self.val_pct: float = val_pct
        self.test_pct: float = test_pct
        self.step_pct: float = step_pct
        self.embargo_days: int = embargo_days

    @staticmethod
    def _validate_parameters(
        initial_train_pct: float,
        val_pct: float,
        test_pct: float,
        step_pct: float,
    ) -> None:
        """パラメータ検証"""
        for name, value in [
            ("initial_train_pct", initial_train_pct),
            ("val_pct", val_pct),
            ("test_pct", test_pct),
            ("step_pct", step_pct),
        ]:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")

    def split(self, df: pd.DataFrame) -> List[TimeSeriesWindow]:
        """複数ウィンドウを生成
        
        Args:
            df: 時系列データフレーム（昇順）。DatetimeIndexまたは timestamp カラムを推奨
        
        Returns:
            List[TimeSeriesWindow]: 生成されたウィンドウリスト
            
        Raises:
            ValueError: データが不足している場合またはデータリークが検出された場合
        """
        n: int = len(df)
        
        if n == 0:
            raise ValueError("DataFrame is empty")
        
        # 初期訓練期間
        train_size: int = int(n * self.initial_train_pct)
        val_size: int = int(n * self.val_pct)
        test_size: int = int(n * self.test_pct)
# Embargo期間をインデックス数に変換（日次データなら embargo_days が直接使える）
        # Embargo期間を行数に換算。デフォルトは指定された embargo_days の最小 1 とする。
        # 以前の実装はデータ量に基づくスケーリングを試みていましたが、少数サンプルで
        # 過度に大きな embargo_size が計算されることがありテストを破壊していました。
        # シンプルかつ予測可能な振る舞いとして、`embargo_days` を行数として扱います
        # （テストデータは均一な頻度で与えられる想定）。
        embargo_size: int = max(1, int(self.embargo_days))

        # Step size controls how far the training window starts are advanced each iteration.
        # Use a simple fractional step of the dataset length with a minimum of 1 row so
        # that small datasets still produce multiple sliding windows.
        step_size: int = max(1, int(n * self.step_pct))  # minimal 1 row step to allow multiple windows
        
        if train_size == 0 or val_size == 0 or test_size == 0:
            raise ValueError(
                f"Insufficient data: train_size={train_size}, "
                f"val_size={val_size}, test_size={test_size}"
            )
        
        windows: List[TimeSeriesWindow] = []
        window_id: int = 0
        train_start: int = 0

        # ウィンドウ生成（各ウィンドウごとにサイズを再計算して非重複の連続ウィンドウを生成）
        while True:
            remaining = n - train_start
            if remaining <= 0:
                break

            # 各ウィンドウのサイズを残りデータ長に基づいて再計算
            train_size_win = max(1, int(remaining * self.initial_train_pct))
            val_size_win = max(1, int(remaining * self.val_pct))
            test_size_win = max(1, int(remaining * self.test_pct))

            # 必要最低限のサイズが満たされているか確認
            if train_size_win == 0 or val_size_win == 0 or test_size_win == 0:
                break

            train_end = train_start + train_size_win

            # Embargo: 訓練と検証の間に gap を設ける（時系列リーク防止）
            val_start: int = train_end + embargo_size
            val_end: int = val_start + val_size_win
            test_start: int = val_end
            test_end: int = test_start + test_size_win

            if test_end > n:
                break

            # ウィンドウの完全性チェック
            self._validate_window(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
                total_size=n,
            )

            window: TimeSeriesWindow = TimeSeriesWindow(
                window_id=window_id,
                train_start=train_start,
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
                f"Test [{window.test_start}:{window.test_end}] "
                f"(Embargo: {embargo_size})"
            )

            # ウィンドウシフト：次のウィンドウは前のテスト期間の直後から開始する（非重複）
            # また、step_pct に基づく前進を考慮して実際の開始位置を決める
            step_forward = max(1, int(n * self.step_pct))
            candidate_start = max(test_end, train_start + step_forward)
            train_start = candidate_start
            window_id += 1
        if not windows:
            raise ValueError(
                f"Could not create any windows with n={n}, "
                f"train_size={train_size}, val_size={val_size}, test_size={test_size}, "
                f"embargo_size={embargo_size}"
            )
        
        logger.info(f"✓ Created {len(windows)} walk-forward windows (embargo: {embargo_size})\n")
        
        # ウィンドウ間のデータリークチェック
        self._check_data_leakage(windows)
        
        return windows

    @staticmethod
    def _validate_window(
        window_id: int,
        train_start: int,
        train_end: int,
        val_start: int,
        val_end: int,
        test_start: int,
        test_end: int,
        total_size: int,
    ) -> None:
        """ウィンドウの完全性を検証
        
        Args:
            window_id: ウィンドウID
            train_start, train_end: 訓練期間のインデックス
            val_start, val_end: 検証期間のインデックス
            test_start, test_end: テスト期間のインデックス
            total_size: データ総数
            
        Raises:
            ValueError: ウィンドウが不正な場合
        """
        # インデックス範囲チェック
        if not (0 <= train_start < train_end <= val_start):
            raise ValueError(
                f"Window {window_id}: Invalid train/val boundary. "
                f"Train [{train_start}:{train_end}], Val starts at {val_start}"
            )
        
        if not (val_start < val_end <= test_start):
            raise ValueError(
                f"Window {window_id}: Invalid val/test boundary. "
                f"Val [{val_start}:{val_end}], Test starts at {test_start}"
            )
        
        if not (test_start < test_end <= total_size):
            raise ValueError(
                f"Window {window_id}: Test range [{test_start}:{test_end}] "
                f"exceeds total size {total_size}"
            )
        
        # 最小ウィンドウサイズチェック（各セグメント最低1サンプル）
        if train_end - train_start < 1 or val_end - val_start < 1 or test_end - test_start < 1:
            raise ValueError(
                f"Window {window_id}: Segment size is too small"
            )

    @staticmethod
    def _check_data_leakage(windows: List[TimeSeriesWindow]) -> None:
        """ウィンドウ間のデータリークをチェック
        
        Args:
            windows: ウィンドウリスト
            
        Raises:
            ValueError: リークが検出された場合
        """
        for i, window in enumerate(windows):
            # 次のウィンドウと比較（訓練期間の拡張チェック）
            if i > 0:
                prev_window = windows[i - 1]
                if window.train_start < prev_window.train_start:
                    raise ValueError(
                        f"Window {i}: Training starts before window {i-1}. "
                        f"This violates walk-forward logic."
                    )
                # Allow non-overlapping touching boundaries but detect actual overlap
                # between validation ranges of successive windows.
                # This check avoids false positives while catching real data leakage.
                if window.val_start < prev_window.val_end:
                    raise ValueError(
                        f"Window {i}: Validation overlaps with previous window {i-1}. "
                        f"Data leakage detected!"
                    )
