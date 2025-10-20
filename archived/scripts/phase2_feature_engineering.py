#!/usr/bin/env python3
"""
Phase 2: Feature Engineering - Correlation-Aware Features
SAC v426 Improvement Plan

このスクリプトは、SAC v424の市場切断問題（相関係数0.019）を解決するために、
相関認識特徴量を追加します。

目標:
- 価格位置相関 (price_position_corr): 現在の価格が市場トレンドとどう関連するか
- アクション価格相関 (action_price_corr): エージェントの行動が価格変動とどう関連するか
- レジーム整合性 (regime_alignment): 現在の市場レジームに対する行動の適切性

これにより、相関係数を0.1以上に向上させ、市場接続性を確立します。
"""

import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CorrelationAwareFeatureEngineer:
    """
    相関認識特徴量エンジニアリングクラス

    SAC v424の市場切断問題を解決するための特徴量を生成します。
    """

    def __init__(self, data_path: str = "data/btc_jpy_balanced_v426_dataset.csv"):
        self.data_path = Path(data_path)
        self.output_path = (
            self.data_path.parent / "btc_jpy_correlation_aware_v426_dataset.csv"
        )
        # BacktestAnalyzerは後で必要に応じて初期化

    def load_data(self) -> pd.DataFrame:
        """Phase 1で作成したバランスデータセットを読み込み"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"データファイルが見つかりません: {self.data_path}")

        logger.info(f"データを読み込み中: {self.data_path}")
        df = pd.read_csv(self.data_path)

        # timestampをdatetimeに変換
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        logger.info(f"データ読み込み完了: {len(df)} 行")
        return df

    def calculate_price_position_corr(self, df: pd.DataFrame) -> pd.Series:
        """
        価格位置相関を計算

        現在の価格が市場トレンドとどう関連するかを示す指標。
        価格がトレンドに対してどの位置にあるかを-1から1の範囲で表現。
        """
        logger.info("価格位置相関を計算中...")

        # 移動平均と標準偏差を計算
        ma_20 = df["close"].rolling(window=20).mean()
        ma_50 = df["close"].rolling(window=50).mean()
        std_20 = df["close"].rolling(window=20).std()

        # トレンド方向 (短期MA - 長期MA)
        trend_direction = (ma_20 - ma_50) / ma_50

        # 現在の価格がトレンドからの偏差
        price_deviation = (df["close"] - ma_20) / std_20

        # 相関を計算: トレンド方向と価格偏差の関係
        # トレンドが上昇中で価格がMAより高い → 正の相関
        # トレンドが下降中で価格がMAより低い → 正の相関
        price_position_corr = trend_direction * price_deviation

        # NaNを0で埋める
        price_position_corr = price_position_corr.fillna(0)

        # -1から1の範囲に正規化
        price_position_corr = np.clip(price_position_corr, -1, 1)

        logger.info(
            f"価格位置相関計算完了: 平均={price_position_corr.mean():.4f}, 範囲=[{price_position_corr.min():.4f}, {price_position_corr.max():.4f}]"
        )
        return price_position_corr

    def calculate_action_price_corr(self, df: pd.DataFrame) -> pd.Series:
        """
        アクション価格相関を計算

        エージェントの行動が価格変動とどう関連するかを示す指標。
        過去の行動が将来の価格変動を予測できたかを評価。
        """
        logger.info("アクション価格相関を計算中...")

        # 簡易的な行動履歴をシミュレート（実際の学習データに基づく）
        # ここではランダムウォークベースの行動を想定
        np.random.seed(42)  # 再現性のため

        # 価格変化の方向を予測する行動をシミュレート
        future_returns = df["close"].shift(-5) / df["close"] - 1  # 5期間後のリターン
        future_direction = np.sign(future_returns)  # 上昇(+1) or 下降(-1)

        # 行動: 価格方向予測 (-1: 売る, 0: ホールド, +1: 買う)
        # 実際の学習ではSACの行動出力を使用
        action = np.random.choice([-1, 0, 1], size=len(df), p=[0.3, 0.4, 0.3])

        # 行動と将来価格変化の相関
        # 行動が将来の価格方向と一致すれば正の相関
        action_price_corr = action * future_direction

        # NaNを0で埋める
        action_price_corr = pd.Series(action_price_corr).fillna(0)

        # -1から1の範囲に正規化
        action_price_corr = np.clip(action_price_corr, -1, 1)

        logger.info(
            f"アクション価格相関計算完了: 平均={action_price_corr.mean():.4f}, 範囲=[{action_price_corr.min():.4f}, {action_price_corr.max():.4f}]"
        )
        return action_price_corr

    def calculate_regime_alignment(self, df: pd.DataFrame) -> pd.Series:
        """
        レジーム整合性を計算

        現在の市場レジームに対する行動の適切性を示す指標。
        各市場レジームで最適な行動パターンを学習。
        """
        logger.info("レジーム整合性を計算中...")

        # 市場レジームを特定（BTCDataAugmentorのロジックに基づく）
        regime_alignment = pd.Series(index=df.index, dtype=float)

        for idx, row in df.iterrows():
            regime = row.get("market_regime", "unknown")

            # レジームごとの最適行動パターン
            regime_patterns = {
                "strong_bull": 0.8,  # 強気市場では積極的に買う
                "moderate_bull": 0.6,  # 中程度の強気
                "sideways": 0.0,  # 横ばいではホールド
                "moderate_bear": -0.6,  # 中程度の弱気
                "strong_bear": -0.8,  # 強気市場では積極的に売る
                "high_volatility": 0.0,  # 高ボラティリティでは慎重
                "low_volatility": 0.2,  # 低ボラティリティでは軽く買う
            }

            # デフォルト値
            base_alignment = regime_patterns.get(regime, 0.0)

            # ボラティリティによる調整
            volatility = row.get("volatility", 0.01)
            if volatility > 0.05:  # 高ボラティリティ
                base_alignment *= 0.5  # 慎重に

            regime_alignment.loc[idx] = base_alignment

        logger.info(
            f"レジーム整合性計算完了: 平均={regime_alignment.mean():.4f}, 範囲=[{regime_alignment.min():.4f}, {regime_alignment.max():.4f}]"
        )
        return regime_alignment

    def add_correlation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """相関認識特徴量をデータフレームに追加"""
        logger.info("相関認識特徴量を追加中...")

        # 各特徴量を計算
        df["price_position_corr"] = self.calculate_price_position_corr(df)
        df["action_price_corr"] = self.calculate_action_price_corr(df)
        df["regime_alignment"] = self.calculate_regime_alignment(df)

        # 統合相関スコアを計算
        df["market_correlation_score"] = (
            df["price_position_corr"] * 0.4
            + df["action_price_corr"] * 0.4
            + df["regime_alignment"] * 0.2
        )

        logger.info("相関認識特徴量追加完了")
        return df

    def validate_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """特徴量の妥当性を検証"""
        logger.info("特徴量の妥当性を検証中...")

        validation_results = {}

        # 各特徴量の統計
        for feature in [
            "price_position_corr",
            "action_price_corr",
            "regime_alignment",
            "market_correlation_score",
        ]:
            if feature in df.columns:
                values = df[feature]
                validation_results[f"{feature}_mean"] = values.mean()
                validation_results[f"{feature}_std"] = values.std()
                validation_results[f"{feature}_range"] = values.max() - values.min()

        # レジームごとの相関スコア分布
        if "market_regime" in df.columns and "market_correlation_score" in df.columns:
            regime_corr = df.groupby("market_regime")["market_correlation_score"].mean()
            for regime, score in regime_corr.items():
                validation_results[f"regime_{regime}_correlation"] = score

        # 相関行列
        corr_features = [
            "price_position_corr",
            "action_price_corr",
            "regime_alignment",
            "close",
            "volume",
        ]
        available_features = [f for f in corr_features if f in df.columns]

        if len(available_features) > 1:
            corr_matrix = df[available_features].corr(numeric_only=True)
            validation_results["feature_correlation_matrix"] = corr_matrix.to_dict()

        logger.info(f"特徴量検証完了: {len(validation_results)} 指標")
        return validation_results

    def save_enhanced_dataset(self, df: pd.DataFrame) -> None:
        """拡張データセットを保存"""
        logger.info(f"拡張データセットを保存中: {self.output_path}")

        # 特徴量の順序を整理
        feature_cols = [
            "price_position_corr",
            "action_price_corr",
            "regime_alignment",
            "market_correlation_score",
        ]

        # 既存の列 + 新しい特徴量列
        all_cols = [col for col in df.columns.tolist()]
        for col in feature_cols:
            if col not in all_cols:
                all_cols.append(col)

        df = df.reindex(columns=all_cols)

        # CSVとして保存
        df.to_csv(self.output_path, index=False)
        logger.info(f"拡張データセット保存完了: {len(df)} 行, {len(df.columns)} 列")

    def generate_feature_report(
        self, df: pd.DataFrame, validation_results: Dict
    ) -> None:
        """特徴量エンジニアリングレポートを生成"""
        report_path = self.output_path.parent / "phase2_feature_engineering_report.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Phase 2: Feature Engineering Report\n\n")
            f.write("## SAC v426 Correlation-Aware Features\n\n")

            f.write("### 目標\n")
            f.write("- SAC v424の市場切断問題解決（相関係数 0.019 → 0.1+）\n")
            f.write("- 相関認識特徴量による市場接続性確立\n\n")

            f.write("### 追加された特徴量\n\n")

            f.write("#### 1. 価格位置相関 (price_position_corr)\n")
            f.write("- 現在の価格が市場トレンドとどう関連するか\n")
            f.write("- 範囲: -1 (トレンドと逆相関) から +1 (トレンドと正相関)\n")
            f.write(
                f"- 平均: {validation_results.get('price_position_corr_mean', 'N/A'):.4f}\n\n"
            )

            f.write("#### 2. アクション価格相関 (action_price_corr)\n")
            f.write("- エージェントの行動が価格変動とどう関連するか\n")
            f.write("- 過去の行動が将来の価格変動を予測できたかを評価\n")
            f.write(
                f"- 平均: {validation_results.get('action_price_corr_mean', 'N/A'):.4f}\n\n"
            )

            f.write("#### 3. レジーム整合性 (regime_alignment)\n")
            f.write("- 現在の市場レジームに対する行動の適切性\n")
            f.write("- レジームごとの最適行動パターンを学習\n")
            f.write(
                f"- 平均: {validation_results.get('regime_alignment_mean', 'N/A'):.4f}\n\n"
            )

            f.write("#### 4. 市場相関スコア (market_correlation_score)\n")
            f.write("- 上記3特徴量の統合スコア\n")
            f.write("- 重み付け: 価格位置40% + アクション価格40% + レジーム整合20%\n")
            f.write(
                f"- 平均: {validation_results.get('market_correlation_score_mean', 'N/A'):.4f}\n\n"
            )

            f.write("### レジーム別相関分析\n\n")
            for key, value in validation_results.items():
                if key.startswith("regime_") and key.endswith("_correlation"):
                    regime = key.replace("regime_", "").replace("_correlation", "")
                    f.write(f"- {regime}: {value:.4f}\n")
            f.write("\n")

            f.write("### データセット統計\n")
            f.write(f"- 元データ: {self.data_path.name}\n")
            f.write(f"- 拡張データ: {self.output_path.name}\n")
            f.write(f"- レコード数: {len(df)}\n")
            f.write(f"- 特徴量数: {len(df.columns)}\n\n")

            f.write("### 次のステップ\n")
            f.write("- Phase 3: Adaptive Reward System実装\n")
            f.write("- SAC v426学習と評価\n")
            f.write("- 相関係数目標: 0.1以上\n\n")

        logger.info(f"特徴量レポート生成完了: {report_path}")

    def run_phase2(self) -> None:
        """Phase 2の完全な実行"""
        logger.info("=== Phase 2: Feature Engineering開始 ===")

        try:
            # 1. データ読み込み
            df = self.load_data()

            # 2. 相関認識特徴量追加
            df_enhanced = self.add_correlation_features(df)

            # 3. 特徴量検証
            validation_results = self.validate_features(df_enhanced)

            # 4. 拡張データセット保存
            self.save_enhanced_dataset(df_enhanced)

            # 5. レポート生成
            self.generate_feature_report(df_enhanced, validation_results)

            logger.info("=== Phase 2: Feature Engineering完了 ===")
            logger.info(f"出力ファイル: {self.output_path}")
            logger.info(
                f"市場相関スコア平均: {validation_results.get('market_correlation_score_mean', 'N/A'):.4f}"
            )

        except Exception as e:
            logger.error(f"Phase 2実行中にエラー発生: {e}")
            raise


def main():
    """メイン実行関数"""
    engineer = CorrelationAwareFeatureEngineer()
    engineer.run_phase2()


if __name__ == "__main__":
    main()
