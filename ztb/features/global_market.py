"""
Global Market Feature Engineer

外部市場データ（Binance等）を統合し、Lead-Lag効果や市場全体のトレンドを
特徴量として生成するモジュール。

Features:
- 外部市場データのマージ (Timestamp alignment)
- Lead-Lag特徴量の計算 (Returns correlation, Price delta)
- グローバル市場トレンド指標 (BTC Dominance, Global Volatility)
"""

import pandas as pd

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class GlobalMarketFeatureEngineer:
    """
    グローバル市場データ統合・特徴量生成クラス
    """

    def __init__(self) -> None:
        pass

    def merge_external_data(
        self,
        main_df: pd.DataFrame,
        external_df: pd.DataFrame,
        suffix: str = "_global",
        fill_method: str = "ffill",
    ) -> pd.DataFrame:
        """
        メインのDataFrameに外部市場データをマージする。
        タイムスタンプをキーにして結合し、欠損値は前方埋めする。

        Args:
            main_df: メイン取引所のデータ (indexはDatetimeIndexまたはtimestampカラムを持つ)
            external_df: 外部取引所のデータ (indexはDatetimeIndexまたはtimestampカラムを持つ)
            suffix: カラム名のサフィックス (例: '_binance')
            fill_method: 欠損値の埋め方 ('ffill' or None)

        Returns:
            Merged DataFrame
        """
        # コピーを作成
        df = main_df.copy()
        ext = external_df.copy()

        # インデックスの正規化
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        if "timestamp" in ext.columns:
            ext = ext.set_index("timestamp")

        # 必要なカラムのみ抽出 (OHLCV)
        cols_to_merge = ["open", "high", "low", "close", "volume"]
        ext = ext[[c for c in cols_to_merge if c in ext.columns]]

        # カラム名のリネーム
        ext = ext.rename(columns={c: f"{c}{suffix}" for c in ext.columns})

        # マージ (左外部結合)
        # メインデータのタイムスタンプに合わせて結合
        merged = df.join(ext, how="left")

        # 欠損値処理 (外部データが遅れている場合や欠落している場合)
        if fill_method == "ffill":
            merged = merged.ffill()

        # インデックスをカラムに戻す (元の形式による)
        if "timestamp" in main_df.columns:
            merged = merged.reset_index()

        return merged

    def generate_lead_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        global_col: str = "close_global",
        windows: list[int] = [1, 5, 15],
    ) -> pd.DataFrame:
        """
        Lead-Lag特徴量を生成する。

        Args:
            df: マージ済みのDataFrame
            target_col: メインデータの価格カラム
            global_col: 外部データの価格カラム
            windows: 計算期間のリスト

        Returns:
            Features DataFrame
        """
        features = pd.DataFrame(index=df.index)

        # 1. 価格乖離 (Price Delta)
        # 正規化された価格差: (Main - Global) / Global
        # ※ 絶対値の違いを吸収するため、リターンまたは対数差分を使うのが一般的だが、
        #    ここでは単純な乖離率を見る
        features[f"price_divergence_{global_col}"] = (
            df[target_col] - df[global_col]
        ) / df[global_col]

        # 2. リターン差分 (Return Spread)
        # Mainのリターン - Globalのリターン
        # Globalが上がっているのにMainが上がっていない -> 遅れて上がる可能性
        main_ret = df[target_col].pct_change()
        global_ret = df[global_col].pct_change()
        features[f"return_spread_{global_col}"] = main_ret - global_ret

        # 3. 相関 (Correlation)
        for w in windows:
            features[f"corr_{global_col}_{w}"] = (
                main_ret.rolling(window=w).corr(global_ret).fillna(0)
            )

        # 4. Global先行指標 (Global Momentum)
        for w in windows:
            features[f"global_mom_{w}"] = df[global_col].pct_change(w).fillna(0)

        return features
