#!/usr/bin/env python3
"""
V433 Data Pipeline: Yahoo Financeから現実市場データを取得
現実データ中心主義に基づく堅牢なデータ取得システム
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class YahooFinanceDataPipeline:
    """
    Yahoo Financeから高品質な金融データを取得するパイプライン
    v433設計原則: 現実データ中心主義
    """

    def __init__(self, data_dir: str = "data", cache_dir: str = "data/cache"):
        """
        Args:
            data_dir: データ保存ディレクトリ
            cache_dir: キャッシュディレクトリ
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

        # BTC/JPYのティッカーシンボル
        self.symbol = "BTC-JPY"
        self.cryptocompare_api_key = os.getenv(
            "CRYPTOCOMPARE_API_KEY"
        )  # 環境変数から取得

        # データ取得設定
        self.interval = "1d"  # 日足データ (1分足は制限があるため)
        self.max_retries = 3
        self.retry_delay = 5  # 秒

    def fetch_historical_data(
        self, start_date: str, end_date: str, symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        指定期間の過去データを取得

        Args:
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)
            symbol: ティッカーシンボル (デフォルト: BTC-JPY)

        Returns:
            価格データ (timestamp, open, high, low, close, volume)
        """
        symbol = symbol or self.symbol

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Fetching {symbol} data from {start_date} to {end_date} (attempt {attempt + 1})"
                )

                # Yahoo Financeからデータを取得
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date, end=end_date, interval=self.interval, prepost=True
                )  # 取引前後のデータも含む

                if df.empty:
                    logger.warning(f"No data retrieved for {symbol}")
                    return pd.DataFrame()

                # データのクリーニングと整形
                df = self._clean_data(df)
                df = self._add_technical_indicators(df)

                logger.info(f"Successfully retrieved {len(df)} records for {symbol}")
                return df

            except Exception as e:
                logger.error(f"Error fetching data (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise e

        return pd.DataFrame()

    def fetch_from_cryptocompare(
        self, start_date: str, end_date: str, symbol: str = "BTC", market: str = "JPY"
    ) -> pd.DataFrame:
        """
        CryptoCompare APIからデータを取得

        Args:
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)
            symbol: 暗号通貨シンボル
            market: 市場シンボル

        Returns:
            価格データ
        """
        base_url = "https://min-api.cryptocompare.com/data/v2/histominute"

        # 日付をUnixタイムスタンプに変換
        start_ts = int(pd.Timestamp(start_date).timestamp())
        end_ts = int(pd.Timestamp(end_date).timestamp())

        params = {
            "fsym": symbol,
            "tsym": market,
            "limit": 2000,  # 最大2000レコード
            "toTs": end_ts,
            "api_key": self.cryptocompare_api_key,
        }

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Fetching {symbol}/{market} data from CryptoCompare (attempt {attempt + 1})"
                )

                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                if data["Response"] != "Success":
                    logger.error(
                        f"CryptoCompare API error: {data.get('Message', 'Unknown error')}"
                    )
                    return pd.DataFrame()

                # データをDataFrameに変換
                df = pd.DataFrame(data["Data"]["Data"])
                if df.empty:
                    logger.warning("No data retrieved from CryptoCompare")
                    return pd.DataFrame()

                # カラム名を統一
                df = df.rename(
                    columns={
                        "time": "timestamp",
                        "open": "open",
                        "high": "high",
                        "low": "low",
                        "close": "close",
                        "volumefrom": "volume_from",
                        "volumeto": "volume_to",
                    }
                )

                # タイムスタンプをdatetimeに変換
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
                df.set_index("timestamp", inplace=True)

                # 日本時間に変換
                df.index = df.index.tz_convert("Asia/Tokyo")

                # 出来高を統合 (from + toの平均を使用)
                df["volume"] = (df["volume_from"] + df["volume_to"]) / 2
                df = df.drop(["volume_from", "volume_to"], axis=1)

                # データのクリーニングと整形
                df = self._clean_data(df)
                df = self._add_technical_indicators(df)

                logger.info(
                    f"Successfully retrieved {len(df)} records from CryptoCompare"
                )
                return df

            except Exception as e:
                logger.error(
                    f"Error fetching from CryptoCompare (attempt {attempt + 1}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise e

        return pd.DataFrame()

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データのクリーニングと品質チェック
        """
        # 欠損値の処理
        df = df.dropna()

        # 異常値の検知と除去 (価格が0以下、または極端な値)
        df = df[df["Close"] > 0]
        df = df[df["Volume"] >= 0]

        # 価格の論理的一貫性チェック
        # High >= Close >= Low, High >= Open >= Low
        valid_prices = (
            (df["High"] >= df["Close"])
            & (df["Close"] >= df["Low"])
            & (df["High"] >= df["Open"])
            & (df["Open"] >= df["Low"])
        )
        df = df[valid_prices]

        # タイムスタンプを日本時間に変換
        df.index = df.index.tz_convert("Asia/Tokyo")

        # カラム名を小文字に統一
        df.columns = df.columns.str.lower()

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基本的なテクニカル指標を追加
        """
        # 価格変化率
        df["returns"] = df["close"].pct_change()

        # 対数リターン
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # ボラティリティ (20期間)
        df["volatility"] = df["returns"].rolling(window=20).std()

        # 移動平均
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        # RSI (14期間)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        return df

    def save_data(self, df: pd.DataFrame, filename: str, format: str = "csv") -> str:
        """
        データを保存

        Args:
            df: 保存するデータ
            filename: ファイル名
            format: 保存形式 (csv, pkl, json)

        Returns:
            保存されたファイルのパス
        """
        if format == "csv":
            filepath = self.data_dir / f"{filename}.csv"
            df.to_csv(filepath)
        elif format == "pkl":
            filepath = self.data_dir / f"{filename}.pkl"
            df.to_pickle(filepath)
        elif format == "json":
            filepath = self.data_dir / f"{filename}.json"
            # JSON形式で保存（インデックスをリセット）
            df.reset_index().to_json(filepath, orient="records", date_format="iso")
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Data saved to {filepath}")
        return str(filepath)

    def load_data(self, filename: str, format: str = "csv") -> pd.DataFrame:
        """
        データを読み込み

        Args:
            filename: ファイル名
            format: ファイル形式

        Returns:
            読み込まれたデータ
        """
        if format == "csv":
            filepath = self.data_dir / f"{filename}.csv"
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif format == "pkl":
            filepath = self.data_dir / f"{filename}.pkl"
            df = pd.read_pickle(filepath)
        elif format == "json":
            filepath = self.data_dir / f"{filename}.json"
            df = pd.read_json(filepath, orient="records")
            df.set_index("Datetime", inplace=True)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Data loaded from {filepath}")
        return df

    def update_real_time_data(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        リアルタイムデータを取得して既存データに追加

        Args:
            symbol: ティッカーシンボル

        Returns:
            更新されたデータ
        """
        symbol = symbol or self.symbol

        # 最新のデータを取得 (過去2日分)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)

        new_data = self.fetch_historical_data(
            start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), symbol
        )

        if new_data.empty:
            logger.warning("No new data available")
            return pd.DataFrame()

        return new_data

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        データ品質の検証

        Args:
            df: 検証するデータ

        Returns:
            品質メトリクス
        """
        quality_report = {
            "total_records": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_timestamps": df.index.duplicated().sum(),
            "price_anomalies": self._detect_price_anomalies(df),
            "volume_anomalies": self._detect_volume_anomalies(df),
            "data_completeness": self._check_data_completeness(df),
            "date_range": {
                "start": df.index.min().isoformat() if not df.empty else None,
                "end": df.index.max().isoformat() if not df.empty else None,
            },
        }

        return quality_report

    def _detect_price_anomalies(self, df: pd.DataFrame, threshold: float = 3.0) -> int:
        """
        価格異常値の検知 (Z-scoreベース)
        """
        if "close" not in df.columns:
            return 0

        z_scores = np.abs((df["close"] - df["close"].mean()) / df["close"].std())
        return (z_scores > threshold).sum()

    def _detect_volume_anomalies(self, df: pd.DataFrame, threshold: float = 5.0) -> int:
        """
        出来高異常値の検知
        """
        if "volume" not in df.columns:
            return 0

        z_scores = np.abs((df["volume"] - df["volume"].mean()) / df["volume"].std())
        return (z_scores > threshold).sum()

    def _check_data_completeness(self, df: pd.DataFrame) -> float:
        """
        データ完全性のチェック (期待されるレコード数に対する割合)
        """
        if df.empty:
            return 0.0

        # 1分足データを想定して、営業時間内の期待レコード数を計算
        total_period = (df.index.max() - df.index.min()).total_seconds() / 60  # 分単位
        expected_records = total_period * 0.8  # 80%を期待 (市場休場考慮)

        return min(len(df) / expected_records, 1.0) if expected_records > 0 else 0.0

    def _make_serializable(self, obj):
        """
        numpy型をPython型に変換してJSONシリアライズ可能にする
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj


def main():
    """
    メイン実行関数
    """
    # データパイプラインの初期化
    pipeline = YahooFinanceDataPipeline()

    # データ取得期間の設定 (過去1年 - 日足データ)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    try:
        # 過去データの取得 (Yahoo Financeを優先)
        logger.info("Starting data collection...")
        btc_data = pipeline.fetch_historical_data(
            start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )

        # Yahoo Financeが失敗したらCryptoCompareを試す
        if btc_data.empty:
            logger.info("Yahoo Finance failed, trying CryptoCompare...")
            btc_data = pipeline.fetch_from_cryptocompare(
                start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )

        if not btc_data.empty:
            # データ品質検証
            quality_report = pipeline.validate_data_quality(btc_data)
            logger.info(f"Data quality report: {quality_report}")

            # データを保存
            filename = f"btc_jpy_yahoo_real_{datetime.now().strftime('%Y%m%d')}"
            saved_path = pipeline.save_data(btc_data, filename, format="csv")
            saved_path_pkl = pipeline.save_data(btc_data, filename, format="pkl")

            logger.info("Data collection completed successfully!")
            logger.info(f"Records: {len(btc_data)}")
            logger.info(f"Saved to: {saved_path}")

            # 品質レポートを保存 (numpy型をPython型に変換)
            quality_report_serializable = self._make_serializable(quality_report)
            report_path = pipeline.data_dir / f"{filename}_quality_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(quality_report_serializable, f, indent=2, ensure_ascii=False)

        else:
            logger.error("No data was collected")

    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise


if __name__ == "__main__":
    main()
