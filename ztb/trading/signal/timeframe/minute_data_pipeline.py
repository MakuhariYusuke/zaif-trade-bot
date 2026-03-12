"""
Minute-level Data Pipeline for Phase 4

分足データ取得・処理・管理パイプライン
"""

from typing import Any
import asyncio
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

class MinuteDataPipeline:
    """
    分足データパイプライン

    高頻度取引のための分足データ取得、処理、管理を担当
    Phase 4のデータ基盤コンポーネント
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize minute data pipeline

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()

        # データソース設定
        self.data_sources = self.config.get('data_sources', {
            'primary': 'zaif_api',
            'supplemental': ['binance', 'bitflyer'],
            'fallback': 'bybit'
        })

        # タイムフレーム設定
        self.supported_timeframes = ['1m', '5m', '15m', '1h']
        self.max_data_age = self.config.get('max_data_age', 300)  # 5分以内のデータ

        # パフォーマンス設定
        self.max_workers = self.config.get('max_workers', 4)
        self.request_timeout = self.config.get('request_timeout', 10)
        self.retry_attempts = self.config.get('retry_attempts', 3)

        # データキャッシュ
        self.data_cache = {}
        self.cache_expiry = {}

        # 非同期実行設定
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        logger.info("MinuteDataPipeline initialized")

    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'data_sources': {
                'primary': 'zaif_api',
                'supplemental': ['binance', 'bitflyer'],
                'fallback': 'bybit'
            },
            'max_data_age': 300,
            'max_workers': 4,
            'request_timeout': 10,
            'retry_attempts': 3,
            'cache_size': 1000,
            'rate_limits': {
                'zaif_api': {'requests_per_minute': 60},
                'binance': {'requests_per_minute': 1200},
                'bitflyer': {'requests_per_minute': 200},
                'bybit': {'requests_per_minute': 50}
            }
        }

    async def get_minute_data(self, symbol: str, timeframe: str = '5m',
                            limit: int = 100) -> pd.DataFrame | None:
        """
        指定シンボルの分足データを取得

        Args:
            symbol: 取引シンボル（例: 'btc_jpy'）
            timeframe: タイムフレーム（'1m', '5m', '15m', '1h'）
            limit: 取得データ数

        Returns:
            pd.DataFrame | None: 分足データ
        """
        if timeframe not in self.supported_timeframes:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return None

        # キャッシュチェック
        cache_key = f"{symbol}_{timeframe}"
        if self._is_cache_valid(cache_key):
            logger.debug(f"Returning cached data for {cache_key}")
            return self.data_cache[cache_key]

        try:
            # データ取得（プライマリソースから）
            data = await self._fetch_from_primary_source(symbol, timeframe, limit)

            if data is None or len(data) == 0:
                # プライマリが失敗したら補完ソースを使用
                logger.warning(f"Primary source failed for {symbol}, trying supplemental sources")
                data = await self._fetch_from_supplemental_sources(symbol, timeframe, limit)

            if data is not None and len(data) > 0:
                # データ検証とクリーニング
                data = self._validate_and_clean_data(data)

                # キャッシュ保存
                self._cache_data(cache_key, data)

                logger.info(f"Successfully retrieved {len(data)} {timeframe} bars for {symbol}")
                return data
            else:
                logger.error(f"Failed to retrieve data for {symbol} {timeframe}")
                return None

        except Exception as e:
            logger.error(f"Error getting minute data for {symbol}: {e}")
            return None

    async def get_multi_timeframe_data(self, symbol: str,
                                     timeframes: list[str] = None) -> dict[str, pd.DataFrame]:
        """
        複数タイムフレームのデータを並行取得

        Args:
            symbol: 取引シンボル
            timeframes: 取得対象タイムフレームリスト

        Returns:
            dict[str, pd.DataFrame]: タイムフレーム別データ
        """
        if timeframes is None:
            timeframes = ['1m', '5m', '15m']

        # 無効なタイムフレームを除去
        valid_timeframes = [tf for tf in timeframes if tf in self.supported_timeframes]

        if not valid_timeframes:
            logger.error("No valid timeframes specified")
            return {}

        try:
            # 並行データ取得
            tasks = []
            for tf in valid_timeframes:
                task = self.get_minute_data(symbol, tf, limit=100)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 結果整理
            multi_tf_data = {}
            for tf, result in zip(valid_timeframes, results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to get {tf} data: {result}")
                elif result is not None:
                    multi_tf_data[tf] = result

            logger.info(f"Retrieved multi-timeframe data for {symbol}: {list(multi_tf_data.keys())}")
            return multi_tf_data

        except Exception as e:
            logger.error(f"Error in multi-timeframe data retrieval: {e}")
            return {}

    async def _fetch_from_primary_source(self, symbol: str, timeframe: str,
                                       limit: int) -> pd.DataFrame | None:
        """
        プライマリデータソースからデータを取得

        Args:
            symbol: 取引シンボル
            timeframe: タイムフレーム
            limit: データ数

        Returns:
            pd.DataFrame | None: 取得データ
        """
        primary_source = self.data_sources['primary']

        for attempt in range(self.retry_attempts):
            try:
                if primary_source == 'zaif_api':
                    return await self._fetch_zaif_data(symbol, timeframe, limit)
                else:
                    logger.warning(f"Unknown primary source: {primary_source}")
                    return None

            except Exception as e:
                logger.warning(f"Primary source attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(1)  # リトライ待機

        return None

    async def _fetch_from_supplemental_sources(self, symbol: str, timeframe: str,
                                             limit: int) -> pd.DataFrame | None:
        """
        補完データソースからデータを取得

        Args:
            symbol: 取引シンボル
            timeframe: タイムフレーム
            limit: データ数

        Returns:
            pd.DataFrame | None: 取得データ
        """
        supplemental_sources = self.data_sources.get('supplemental', [])

        for source in supplemental_sources:
            try:
                if source == 'binance':
                    data = await self._fetch_binance_data(symbol, timeframe, limit)
                elif source == 'bitflyer':
                    data = await self._fetch_bitflyer_data(symbol, timeframe, limit)
                else:
                    continue

                if data is not None and len(data) > 0:
                    logger.info(f"Retrieved data from supplemental source: {source}")
                    return data

            except Exception as e:
                logger.warning(f"Supplemental source {source} failed: {e}")
                continue

        # 最終フォールバック
        fallback_source = self.data_sources.get('fallback')
        if fallback_source:
            try:
                return await self._fetch_bybit_data(symbol, timeframe, limit)
            except Exception as e:
                logger.error(f"Fallback source {fallback_source} also failed: {e}")

        return None

    async def _fetch_zaif_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
        """Zaif APIからデータを取得（実際の実装ではAPIコール）"""
        # 実際のAPI実装はここに
        # 現在はモックデータ生成
        try:
            # モックデータ生成（実際のAPI実装に置き換え）
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=limit * self._timeframe_to_minutes(timeframe))

            timestamps = pd.date_range(start=start_time, end=end_time,
                                     freq=f'{self._timeframe_to_minutes(timeframe)}min')[:limit]

            np.random.seed(42)  # 再現性のために固定シード
            base_price = 5000000  # BTC/JPYの基準価格

            # リアルな価格変動をシミュレート
            price_changes = np.random.normal(0, 0.005, len(timestamps))  # 0.5%ボラティリティ
            prices = base_price * (1 + np.cumsum(price_changes))

            # OHLCVデータ生成
            high_mult = 1 + np.abs(np.random.normal(0, 0.002, len(timestamps)))
            low_mult = 1 - np.abs(np.random.normal(0, 0.002, len(timestamps)))
            volume = np.random.uniform(0.1, 5.0, len(timestamps))

            data = pd.DataFrame({
                'timestamp': timestamps,
                'open': prices * (1 + np.random.normal(0, 0.001, len(timestamps))),
                'high': prices * high_mult,
                'low': prices * low_mult,
                'close': prices,
                'volume': volume
            })

            data.set_index('timestamp', inplace=True)
            return data

        except Exception as e:
            logger.error(f"Error fetching Zaif data: {e}")
            return None

    async def _fetch_binance_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
        """Binance APIからデータを取得"""
        # Binance API実装（実際のAPIキーが必要）
        logger.info("Binance data fetching not implemented yet")
        return None

    async def _fetch_bitflyer_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
        """BitFlyer APIからデータを取得"""
        # BitFlyer API実装
        logger.info("BitFlyer data fetching not implemented yet")
        return None

    async def _fetch_bybit_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
        """Bybit APIからデータを取得"""
        # Bybit API実装
        logger.info("Bybit data fetching not implemented yet")
        return None

    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        データの検証とクリーニング

        Args:
            data: 生データ

        Returns:
            pd.DataFrame: クリーニング済みデータ
        """
        try:
            # 必須カラムの存在確認
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError("Missing required OHLCV columns")

            # データ型の確認と変換
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            # NaN値の除去
            data = data.dropna()

            # 価格の妥当性チェック（負の値や極端な値の除去）
            data = data[
                (data['open'] > 0) & (data['high'] > 0) & (data['low'] > 0) & (data['close'] > 0) &
                (data['high'] >= data['open']) & (data['high'] >= data['close']) &
                (data['low'] <= data['open']) & (data['low'] <= data['close'])
            ]

            # タイムスタンプのソート
            if 'timestamp' in data.index.names or 'timestamp' in data.columns:
                data = data.sort_index() if data.index.name == 'timestamp' else data.sort_values('timestamp')

            return data

        except Exception as e:
            logger.warning(f"Error cleaning data: {e}")
            return data

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """タイムフレームを分単位に変換"""
        mapping = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '1d': 1440
        }
        return mapping.get(timeframe, 5)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """キャッシュの有効性を確認"""
        if cache_key not in self.data_cache:
            return False

        if cache_key not in self.cache_expiry:
            return False

        return time.time() < self.cache_expiry[cache_key]

    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """データをキャッシュ"""
        self.data_cache[cache_key] = data.copy()
        self.cache_expiry[cache_key] = time.time() + self.max_data_age

        # キャッシュサイズ制限
        if len(self.data_cache) > self.config.get('cache_size', 1000):
            self._cleanup_cache()

    def _cleanup_cache(self):
        """古いキャッシュをクリーンアップ"""
        current_time = time.time()
        expired_keys = [k for k, expiry in self.cache_expiry.items() if current_time > expiry]

        for key in expired_keys:
            del self.data_cache[key]
            del self.cache_expiry[key]

        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def get_data_quality_metrics(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        データ品質メトリクスを取得

        Args:
            data: 評価対象データ

        Returns:
            dict: 品質メトリクス
        """
        if data is None or len(data) == 0:
            return {'quality_score': 0, 'issues': ['empty_data']}

        try:
            metrics = {
                'total_bars': len(data),
                'missing_data_ratio': data.isnull().sum().sum() / (len(data) * len(data.columns)),
                'price_anomalies': self._detect_price_anomalies(data),
                'volume_anomalies': self._detect_volume_anomalies(data),
                'time_gaps': self._detect_time_gaps(data)
            }

            # 総合品質スコア計算
            quality_score = self._calculate_quality_score(metrics)

            metrics['quality_score'] = quality_score

            return metrics

        except Exception as e:
            logger.warning(f"Error calculating quality metrics: {e}")
            return {'quality_score': 0, 'error': str(e)}

    def _detect_price_anomalies(self, data: pd.DataFrame) -> int:
        """価格異常を検出"""
        try:
            # 極端な価格変動の検出
            returns = data['close'].pct_change().abs()
            anomalies = (returns > 0.1).sum()  # 10%以上変動
            return int(anomalies)
        except Exception:
            return 0

    def _detect_volume_anomalies(self, data: pd.DataFrame) -> int:
        """出来高異常を検出"""
        try:
            # 極端な出来高の検出
            volume_zscore = (data['volume'] - data['volume'].mean()) / data['volume'].std()
            anomalies = (volume_zscore.abs() > 3).sum()  # 3σ以上
            return int(anomalies)
        except Exception:
            return 0

    def _detect_time_gaps(self, data: pd.DataFrame) -> int:
        """時間ギャップを検出"""
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                return 0

            # 期待される間隔の計算
            expected_diff = data.index.to_series().diff().mode().iloc[0]
            gaps = (data.index.to_series().diff() > expected_diff * 2).sum()
            return int(gaps)
        except Exception:
            return 0

    def _calculate_quality_score(self, metrics: dict[str, Any]) -> float:
        """品質スコアを計算"""
        try:
            score = 100.0

            # 欠損データペナルティ
            score -= metrics['missing_data_ratio'] * 50

            # 異常値ペナルティ
            total_anomalies = metrics['price_anomalies'] + metrics['volume_anomalies']
            anomaly_penalty = min(total_anomalies * 5, 30)
            score -= anomaly_penalty

            # 時間ギャップペナルティ
            gap_penalty = min(metrics['time_gaps'] * 2, 20)
            score -= gap_penalty

            return max(0, score)

        except Exception as e:
            logger.warning(f"Error calculating quality score: {e}")
            return 0.0
