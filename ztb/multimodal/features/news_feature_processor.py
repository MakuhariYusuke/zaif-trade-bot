"""
ニュース特徴量統合システム
TextSentimentEncoderを活用したニュース情報特徴量の実装
"""

import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ztb.multimodal.features.text.nlp_processor import (
    FinancialTextProcessor,
    MultiModalFeatureIntegrator,
    TextSentimentEncoder,
)

logger = logging.getLogger(__name__)

class NewsFeatureProcessor:
    """
    ニュース情報を特徴量として統合するクラス
    """

    def __init__(
        self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    ):
        """
        初期化

        Args:
            model_name: 使用する感情分析モデルの名前
        """
        self.sentiment_encoder = TextSentimentEncoder(model_name=model_name)
        self.financial_processor = FinancialTextProcessor()
        self.multimodal_integrator = MultiModalFeatureIntegrator()

        # ニュースデータのキャッシュ
        self.news_cache = {}
        self.feature_cache = {}

    def load_news_data(self, news_file_path: str) -> pd.DataFrame:
        """
        ニュースデータを読み込み

        Args:
            news_file_path: ニュースデータファイルのパス

        Returns:
            ニュースデータのDataFrame
        """
        try:
            if not Path(news_file_path).exists():
                logger.warning(f"ニュースファイルが見つかりません: {news_file_path}")
                return pd.DataFrame()

            # CSVまたはJSONから読み込み
            if news_file_path.endswith(".csv"):
                news_df = pd.read_csv(news_file_path)
            elif news_file_path.endswith(".json"):
                news_df = pd.read_json(news_file_path)
            else:
                logger.error(f"未対応のファイル形式: {news_file_path}")
                return pd.DataFrame()

            # 日時列の処理
            if "timestamp" in news_df.columns:
                news_df["timestamp"] = pd.to_datetime(news_df["timestamp"])
                news_df = news_df.sort_values("timestamp")

            # テキスト列の確認
            if "text" not in news_df.columns and "content" in news_df.columns:
                news_df["text"] = news_df["content"]

            logger.info(f"ニュースデータを読み込みました: {len(news_df)}件")
            return news_df

        except Exception as e:
            logger.error(f"ニュースデータの読み込みに失敗: {e}")
            return pd.DataFrame()

    def extract_sentiment_features(
        self, news_texts: list[str]
    ) -> dict[str, np.ndarray]:
        """
        ニューステキストから感情特徴量を抽出

        Args:
            news_texts: ニューステキストのリスト

        Returns:
            感情特徴量辞書
        """
        try:
            if not news_texts:
                return {}

            # TextSentimentEncoderを使用
            sentiment_results = self.sentiment_encoder(news_texts)

            # 金融特化の感情分析も追加
            financial_sentiments = []
            for text in news_texts:
                fin_sentiment = self.financial_processor.extract_financial_sentiment(
                    text
                )
                financial_sentiments.append(fin_sentiment["sentiment_score"])

            # 特徴量統合
            features = {
                "sentiment_scores": sentiment_results["sentiment_scores"]
                .detach()
                .numpy(),
                "sentiment_intensity": sentiment_results["intensity"].detach().numpy(),
                "bert_embeddings": sentiment_results["embeddings"].detach().numpy(),
                "financial_sentiment": np.array(financial_sentiments),
            }

            return features

        except Exception as e:
            logger.error(f"感情特徴量抽出に失敗: {e}")
            return {}

    def aggregate_news_features_by_time(
        self, news_df: pd.DataFrame, time_window_hours: int = 24
    ) -> pd.DataFrame:
        """
        時間帯ごとにニュース特徴量を集約

        Args:
            news_df: ニュースデータのDataFrame
            time_window_hours: 集約時間窓（時間）

        Returns:
            時間集約された特徴量DataFrame
        """
        try:
            if news_df.empty:
                return pd.DataFrame()

            # 時間窓でグループ化
            news_df["time_window"] = news_df["timestamp"].dt.floor(
                f"{time_window_hours}H"
            )

            # 各時間窓のニュースを集約
            aggregated_features = []

            for window, group in news_df.groupby("time_window"):
                texts = group["text"].tolist()

                # 感情特徴量抽出
                sentiment_features = self.extract_sentiment_features(texts)

                if sentiment_features:
                    # 時間窓ごとの集約特徴量
                    window_features = {
                        "timestamp": window,
                        "news_count": len(texts),
                        "avg_sentiment_score": np.mean(
                            sentiment_features["sentiment_scores"], axis=0
                        ),
                        "avg_sentiment_intensity": np.mean(
                            sentiment_features["sentiment_intensity"]
                        ),
                        "avg_financial_sentiment": np.mean(
                            sentiment_features["financial_sentiment"]
                        ),
                        "sentiment_volatility": np.std(
                            sentiment_features["sentiment_scores"], axis=0
                        ),
                        "max_sentiment_score": np.max(
                            sentiment_features["sentiment_scores"], axis=0
                        ),
                        "min_sentiment_score": np.min(
                            sentiment_features["sentiment_scores"], axis=0
                        ),
                    }

                    # BERT埋め込みの平均
                    if "bert_embeddings" in sentiment_features:
                        window_features["avg_bert_embedding"] = np.mean(
                            sentiment_features["bert_embeddings"], axis=0
                        )

                    aggregated_features.append(window_features)

            result_df = pd.DataFrame(aggregated_features)
            logger.info(f"ニュース特徴量を集約しました: {len(result_df)}時間窓")
            return result_df

        except Exception as e:
            logger.error(f"ニュース特徴量集約に失敗: {e}")
            return pd.DataFrame()

    def integrate_with_price_features(
        self, news_features_df: pd.DataFrame, price_features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        ニュース特徴量と価格特徴量を統合

        Args:
            news_features_df: ニュース特徴量DataFrame
            price_features_df: 価格特徴量DataFrame

        Returns:
            統合された特徴量DataFrame
        """
        try:
            if news_features_df.empty or price_features_df.empty:
                return pd.DataFrame()

            # タイムスタンプでマージ
            merged_df = pd.merge_asof(
                price_features_df.sort_values("timestamp"),
                news_features_df.sort_values("timestamp"),
                on="timestamp",
                direction="backward",  # 直前のニュースを使用
            )

            # 欠損値補完（直近のニュース特徴量を使用）
            merged_df = merged_df.fillna(method="ffill")

            logger.info(f"ニュースと価格特徴量を統合しました: {len(merged_df)}行")
            return merged_df

        except Exception as e:
            logger.error(f"特徴量統合に失敗: {e}")
            return pd.DataFrame()

    def create_news_impact_features(
        self,
        news_features_df: pd.DataFrame,
        price_data_df: pd.DataFrame,
        impact_window_minutes: int = 60,
    ) -> pd.DataFrame:
        """
        ニュースの価格影響特徴量を作成

        Args:
            news_features_df: ニュース特徴量DataFrame
            price_data_df: 価格データDataFrame
            impact_window_minutes: 影響分析時間窓（分）

        Returns:
            ニュース影響特徴量DataFrame
        """
        try:
            impact_features = []

            for _, news_row in news_features_df.iterrows():
                news_time = news_row["timestamp"]

                # ニュース後の価格変化を分析
                end_time = news_time + timedelta(minutes=impact_window_minutes)
                price_window = price_data_df[
                    (price_data_df["timestamp"] >= news_time)
                    & (price_data_df["timestamp"] <= end_time)
                ]

                if len(price_window) >= 2:
                    # 価格変化率
                    price_change = (
                        price_window["price"].iloc[-1] - price_window["price"].iloc[0]
                    ) / price_window["price"].iloc[0]

                    # ボラティリティ
                    returns = price_window["price"].pct_change().dropna()
                    volatility = returns.std() if len(returns) > 0 else 0

                    # 取引量変化
                    volume_change = 0
                    if "volume" in price_window.columns and len(price_window) > 1:
                        volume_change = (
                            price_window["volume"].iloc[-1]
                            - price_window["volume"].iloc[0]
                        ) / price_window["volume"].iloc[0]

                    impact_feature = {
                        "timestamp": news_time,
                        "news_sentiment": news_row.get("avg_sentiment_score", 0),
                        "price_impact": price_change,
                        "volatility_impact": volatility,
                        "volume_impact": volume_change,
                        "impact_window_minutes": impact_window_minutes,
                    }

                    impact_features.append(impact_feature)

            result_df = pd.DataFrame(impact_features)
            logger.info(f"ニュース影響特徴量を作成しました: {len(result_df)}件")
            return result_df

        except Exception as e:
            logger.error(f"ニュース影響特徴量作成に失敗: {e}")
            return pd.DataFrame()

class TradingFeatureIntegrator:
    """
    取引特徴量統合クラス
    ニュース、価格、テクニカル指標を統合
    """

    def __init__(self):
        self.news_processor = NewsFeatureProcessor()

    def create_comprehensive_features(
        self, price_data_path: str, news_data_path: str | None = None
    ) -> pd.DataFrame:
        """
        包括的な特徴量セットを作成

        Args:
            price_data_path: 価格データファイルパス
            news_data_path: ニュースデータファイルパス（オプション）

        Returns:
            統合された特徴量DataFrame
        """
        try:
            # 価格データを読み込み
            price_df = pd.read_csv(price_data_path)
            price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])

            # ニュースデータがある場合
            if news_data_path and Path(news_data_path).exists():
                news_df = self.news_processor.load_news_data(news_data_path)
                if not news_df.empty:
                    # ニュース特徴量集約
                    news_features_df = (
                        self.news_processor.aggregate_news_features_by_time(news_df)
                    )

                    # ニュース影響特徴量
                    news_impact_df = self.news_processor.create_news_impact_features(
                        news_features_df, price_df
                    )

                    # 特徴量統合
                    integrated_df = self.news_processor.integrate_with_price_features(
                        news_features_df, price_df
                    )

                    logger.info("ニュース特徴量を統合しました")
                    return integrated_df

            # ニュースデータがない場合は価格データのみ
            logger.info("ニュースデータなしで価格特徴量のみを使用します")
            return price_df

        except Exception as e:
            logger.error(f"包括的特徴量作成に失敗: {e}")
            return pd.DataFrame()

def create_sample_news_features():
    """
    サンプルニュース特徴量生成
    """
    processor = NewsFeatureProcessor()

    # サンプルニュースデータ
    sample_news = [
        "市場が上昇トレンドを継続、投資家心理が改善",
        "経済指標が予想を上回り、強気相場が続く",
        "企業業績が好調で株価が最高値を更新",
        "市場が下落トレンドにあり、投資家心理悪化",
        "経済指標が失望を誘い、弱気相場が継続",
    ]

    # 感情特徴量抽出
    features = processor.extract_sentiment_features(sample_news)

    print("ニュース感情特徴量抽出結果:")
    print(f"感情スコア形状: {features['sentiment_scores'].shape}")
    print(f"感情強度形状: {features['sentiment_intensity'].shape}")
    print(f"金融感情スコア: {features['financial_sentiment']}")

    # 平均感情スコア
    avg_sentiment = np.mean(features["sentiment_scores"], axis=0)
    print(f"平均感情スコア [ネガティブ, ニュートラル, ポジティブ]: {avg_sentiment}")

    print("ニュース特徴量処理が完了しました")

if __name__ == "__main__":
    create_sample_news_features()
