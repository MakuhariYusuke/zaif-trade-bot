"""
マルチモーダル学習 - データソース管理
無料/公開データセットの調査と統合
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class FreeDataSourceManager:
    """
    無料データソース統合マネージャー
    経済指標、ニュース、ソーシャルメディアデータを統合
    """

    def __init__(self):
        self.cache = {}
        self.last_request_time = {}

        # APIレート制限管理
        self.rate_limits = {
            "fred": {"requests_per_minute": 120, "last_request": None},
            "newsapi": {"requests_per_minute": 100, "last_request": None},
            "alphavantage": {"requests_per_minute": 5, "last_request": None},
        }

    def _check_rate_limit(self, source: str) -> bool:
        """レート制限チェック"""
        if source not in self.rate_limits:
            return True

        limit_info = self.rate_limits[source]
        if limit_info["last_request"] is None:
            return True

        elapsed = (datetime.now() - limit_info["last_request"]).seconds
        min_interval = 60 / limit_info["requests_per_minute"]

        return elapsed >= min_interval

    def _update_rate_limit(self, source: str):
        """レート制限更新"""
        if source in self.rate_limits:
            self.rate_limits[source]["last_request"] = datetime.now()

    def fetch_fred_economic_data(
        self,
        series_ids: List[str],
        start_date: str = "2020-01-01",
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        FRED (Federal Reserve Economic Data) から経済指標を取得
        完全に無料で利用可能

        Args:
            series_ids: FREDシリーズIDリスト
            start_date: 開始日
            end_date: 終了日（Noneの場合は今日）

        Returns:
            経済指標データフレーム
        """

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if not self._check_rate_limit("fred"):
            logger.warning("FRED API rate limit exceeded, using cached data")
            return self.cache.get("fred", pd.DataFrame())

        base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"

        all_data = {}
        for series_id in series_ids:
            params = {"id": series_id, "cosd": start_date, "coed": end_date}

            try:
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()

                # CSVデータをDataFrameに変換
                df = pd.read_csv(pd.io.common.StringIO(response.text))
                df["DATE"] = pd.to_datetime(df["DATE"])
                df.set_index("DATE", inplace=True)

                all_data[series_id] = df[series_id]

                self._update_rate_limit("fred")
                time.sleep(0.1)  # 礼儀的な遅延

            except Exception as e:
                logger.error(f"Failed to fetch FRED data for {series_id}: {e}")
                continue

        if all_data:
            result_df = pd.DataFrame(all_data)
            self.cache["fred"] = result_df
            return result_df

        return pd.DataFrame()

    def fetch_newsapi_data(
        self,
        query: str = "cryptocurrency OR bitcoin OR ethereum",
        language: str = "en",
        days_back: int = 7,
    ) -> List[Dict]:
        """
        NewsAPIからニュースデータを取得
        無料枠：1日100リクエストまで

        Args:
            query: 検索クエリ
            language: 言語
            days_back: 何日前まで取得するか

        Returns:
            ニュース記事リスト
        """

        # NewsAPIキーが必要（無料で取得可能）
        api_key = self._get_newsapi_key()
        if not api_key:
            logger.warning("NewsAPI key not found, returning empty data")
            return []

        if not self._check_rate_limit("newsapi"):
            logger.warning("NewsAPI rate limit exceeded")
            return self.cache.get("newsapi", [])

        base_url = "https://newsapi.org/v2/everything"

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        params = {
            "q": query,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "language": language,
            "sortBy": "relevancy",
            "apiKey": api_key,
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            articles = data.get("articles", [])

            # 必要な情報のみ抽出
            processed_articles = []
            for article in articles:
                processed_articles.append(
                    {
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "content": article.get("content", ""),
                        "publishedAt": article.get("publishedAt", ""),
                        "source": article.get("source", {}).get("name", ""),
                        "url": article.get("url", ""),
                    }
                )

            self.cache["newsapi"] = processed_articles
            self._update_rate_limit("newsapi")

            return processed_articles

        except Exception as e:
            logger.error(f"Failed to fetch NewsAPI data: {e}")
            return []

    def fetch_alpha_vantage_crypto(
        self, symbol: str = "BTC", market: str = "USD", outputsize: str = "compact"
    ) -> pd.DataFrame:
        """
        Alpha Vantageから暗号通貨データを取得
        無料枠：1日5リクエストまで

        Args:
            symbol: 暗号通貨シンボル
            market: 市場
            outputsize: データサイズ（compact/full）

        Returns:
            価格データフレーム
        """

        api_key = self._get_alpha_vantage_key()
        if not api_key:
            logger.warning("Alpha Vantage key not found")
            return pd.DataFrame()

        if not self._check_rate_limit("alphavantage"):
            logger.warning("Alpha Vantage rate limit exceeded")
            return self.cache.get("alphavantage", pd.DataFrame())

        base_url = "https://www.alphavantage.co/query"

        params = {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": symbol,
            "market": market,
            "apikey": api_key,
            "outputsize": outputsize,
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if "Time Series (Digital Currency Daily)" not in data:
                logger.error(f"Invalid response from Alpha Vantage: {data}")
                return pd.DataFrame()

            time_series = data["Time Series (Digital Currency Daily)"]

            # データ整形
            records = []
            for date, values in time_series.items():
                record = {
                    "date": date,
                    "open": float(values["1a. open (USD)"]),
                    "high": float(values["2a. high (USD)"]),
                    "low": float(values["3a. low (USD)"]),
                    "close": float(values["4a. close (USD)"]),
                    "volume": float(values["5. volume"]),
                }
                records.append(record)

            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            self.cache["alphavantage"] = df
            self._update_rate_limit("alphavantage")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch Alpha Vantage data: {e}")
            return pd.DataFrame()

    def _get_newsapi_key(self) -> Optional[str]:
        """NewsAPIキーを取得（環境変数または設定ファイルから）"""
        # 実際の実装では環境変数や設定ファイルから取得
        # ここではサンプルとしてNoneを返す
        return None  # TODO: 実際のAPIキーを設定

    def _get_alpha_vantage_key(self) -> Optional[str]:
        """Alpha Vantageキーを取得"""
        # 実際の実装では環境変数や設定ファイルから取得
        return None  # TODO: 実際のAPIキーを設定

    def get_available_economic_indicators(self) -> Dict[str, str]:
        """
        利用可能な経済指標のリストを返す

        Returns:
            指標IDと説明の辞書
        """
        return {
            # 主要経済指標
            "GDP": "Gross Domestic Product",
            "UNRATE": "Unemployment Rate",
            "FEDFUNDS": "Federal Funds Rate",
            "CPIAUCSL": "Consumer Price Index",
            "DEXUSEU": "USD/EUR Exchange Rate",
            "DEXJPUS": "USD/JPY Exchange Rate",
            # 株式市場指標
            "SP500": "S&P 500 Index",
            "NASDAQCOM": "NASDAQ Composite",
            "DJIA": "Dow Jones Industrial Average",
            # 債券市場
            "DGS10": "10-Year Treasury Rate",
            "DGS2": "2-Year Treasury Rate",
            "T10Y2Y": "10-Year Minus 2-Year Treasury Spread",
            # 商品価格
            "DCOILWTICO": "WTI Crude Oil Price",
            "GOLDAMGBD228NLBM": "Gold Fixing Price",
            "DHHNGSP": "Henry Hub Natural Gas Spot Price",
        }


class PublicDatasetLoader:
    """
    公開データセットローダー
    Kaggle, UCI, 政府機関などの公開データを活用
    """

    def __init__(self):
        self.data_dir = "data/public_datasets"

    def load_financial_news_dataset(self) -> pd.DataFrame:
        """
        公開の金融ニュースデータセットを読み込み
        （例: Financial PhraseBank, 感情分析済みニュース）

        Returns:
            ニュースデータフレーム
        """
        # サンプルデータ生成（実際には公開データセットを使用）
        sample_data = {
            "text": [
                "Company reports strong quarterly earnings",
                "Market shows signs of recovery",
                "Economic indicators point to slowdown",
                "Stock prices surge on positive news",
                "Investors remain cautious amid uncertainty",
            ],
            "sentiment": ["positive", "positive", "negative", "positive", "neutral"],
            "date": pd.date_range("2025-01-01", periods=5, freq="D"),
        }

        df = pd.DataFrame(sample_data)
        return df

    def load_economic_indicators_dataset(self) -> pd.DataFrame:
        """
        公開の経済指標データセットを読み込み

        Returns:
            経済指標データフレーム
        """
        # FREDなどの公開データを模擬
        dates = pd.date_range("2020-01-01", "2025-01-01", freq="M")

        np.random.seed(42)
        data = {
            "GDP_growth": np.random.normal(2.0, 1.0, len(dates)),
            "unemployment_rate": np.random.normal(4.0, 0.5, len(dates)),
            "inflation_rate": np.random.normal(2.5, 0.8, len(dates)),
            "interest_rate": np.random.normal(2.0, 0.3, len(dates)),
        }

        df = pd.DataFrame(data, index=dates)
        return df


# 使用例
if __name__ == "__main__":
    # データソースマネージャーの初期化
    data_manager = FreeDataSourceManager()

    # 利用可能な経済指標を表示
    indicators = data_manager.get_available_economic_indicators()
    print("利用可能な経済指標:")
    for indicator_id, description in list(indicators.items())[:5]:
        print(f"  {indicator_id}: {description}")

    # FREDデータ取得の例（APIキーが必要）
    print("\nFREDデータ取得例:")
    print("注意: 実際の使用にはインターネット接続とAPIレート制限の考慮が必要です")

    # 公開データセットローダー
    dataset_loader = PublicDatasetLoader()

    news_data = dataset_loader.load_financial_news_dataset()
    print(f"\nサンプルニュースデータ: {len(news_data)}件")
    print(news_data.head())

    economic_data = dataset_loader.load_economic_indicators_dataset()
    print(f"\nサンプル経済指標データ: {economic_data.shape}")
    print(economic_data.head())

    print(
        """
    📋 無料データソース利用ガイド:

    1. FRED (Federal Reserve Economic Data)
       - URL: https://fred.stlouisfed.org/
       - 利用: 完全に無料、レート制限なし
       - データ: 経済指標、為替、商品価格など

    2. NewsAPI
       - URL: https://newsapi.org/
       - 利用: 無料枠（1日100リクエスト）
       - データ: ニュース記事、リアルタイム

    3. Alpha Vantage
       - URL: https://www.alphavantage.co/
       - 利用: 無料枠（1日5リクエスト）
       - データ: 株価、暗号通貨、経済指標

    4. 公開データセット
       - Kaggle: 金融ニュースデータセット
       - UCI Machine Learning Repository
       - 政府機関（日本の場合: 総務省統計局、東証など）

    注意事項:
    - APIキーは無料で取得可能
    - レート制限を遵守すること
    - 商用利用の場合は有料プランを検討
    """
    )
