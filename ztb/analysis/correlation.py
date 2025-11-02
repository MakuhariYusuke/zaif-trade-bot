"""
Correlation Analysis Module

相関係数分析機能を提供するモジュール
"""

from typing import Dict, Optional, Literal
import pandas as pd
import numpy as np
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_correlations(
    frames: Dict[str, pd.DataFrame],
    nan_strategy: Literal["drop", "fill", "none"] = "none",
    fill_value: float = 0.0
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    複数のデータフレーム間の相関係数を計算

    Args:
        frames: データフレームの辞書 {"name": DataFrame}
        nan_strategy: NaN処理戦略 ("drop", "fill", "none")
        fill_value: NaNを埋める値 (nan_strategy="fill"の場合)

    Returns:
        相関係数結果の辞書 {"pearson": DataFrame, "spearman": DataFrame}
        空の場合は {"pearson": None, "spearman": None}
    """
    if not frames:
        return {"pearson": None, "spearman": None}

    try:
        # すべてのデータフレームを結合（プレフィックス付き）
        combined_df = pd.concat(frames.values(), axis=1, keys=frames.keys())

        # MultiIndexをプレフィックス付きカラム名に変換
        combined_df.columns = [f"{level0}_{level1}" for level0, level1 in combined_df.columns]

        # NaNが多い列を除外（80%以上がNaNの場合）
        nan_threshold = len(combined_df) * 0.8
        valid_columns = combined_df.columns[combined_df.isna().sum() < nan_threshold]
        filtered_df = combined_df[valid_columns]

        if filtered_df.empty or len(filtered_df.columns) < 2:
            logger.warning("Not enough valid data for correlation analysis")
            return {"pearson": None, "spearman": None}

        # 数値列のみを選択
        numeric_df = filtered_df.select_dtypes(include=[np.number])

        if numeric_df.empty or len(numeric_df.columns) < 2:
            logger.warning("Not enough numeric columns for correlation analysis")
            return {"pearson": None, "spearman": None}

        # 定数カラムを除外（すべての値が同じ場合）
        non_constant_columns = []
        for col in numeric_df.columns:
            if numeric_df[col].nunique() > 1:  # ユニークな値が1つ以上
                non_constant_columns.append(col)

        if len(non_constant_columns) < 2:
            logger.warning("Not enough non-constant columns for correlation analysis")
            return {"pearson": None, "spearman": None}

        numeric_df = numeric_df[non_constant_columns]

        # NaN処理
        if nan_strategy == "drop":
            numeric_df = numeric_df.dropna()
        elif nan_strategy == "fill":
            numeric_df = numeric_df.fillna(fill_value)

        # 相関係数を計算
        pearson_corr = numeric_df.corr(method='pearson')
        spearman_corr = numeric_df.corr(method='spearman')

        logger.info(f"Computed correlations for {len(numeric_df.columns)} features")

        return {
            "pearson": pearson_corr,
            "spearman": spearman_corr
        }

    except Exception as e:
        logger.error(f"Error computing correlations: {e}")
        return {"pearson": None, "spearman": None}