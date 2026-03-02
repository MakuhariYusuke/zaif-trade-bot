#!/usr/bin/env python3
"""
v456 環境初期化ファクトリー（型安全）

FastIntradayEnvV456 の複雑な初期化を簡潔にするファクトリー実装。
特徴量計算パイプラインを統合し、型安全性を確保。

Usage:
    factory = EnvironmentFactory(data_df)
    env = factory.create_training_env()
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
from ztb.utils.error_utils import safe_operation

logger = logging.getLogger(__name__)

class FeaturePipeline:
    """特徴量計算パイプライン（型安全）"""
    
    def __init__(self, base_features_count: int = 30, mtf_features_count: int = 27, regime_features_count: int = 13):
        self.base_features_count = base_features_count
        self.mtf_features_count = mtf_features_count
        self.regime_features_count = regime_features_count
    
    def validate_base_features(self, df: pd.DataFrame) -> list[str]:
        """Base 特徴量（30次元）を検証・構築"""
        base_cols = [
            "open", "high", "low", "close", "volume",
            "sma_5", "sma_20", "sma_50",
            "ema_5", "ema_20", "ema_50",
            "rsi_14", "rsi_20",
            "atr_14", "atr_20",
            "bb_upper_20", "bb_lower_20", "bb_pct_b_20",
            "macd_line", "macd_signal",
            "adx_14", "plus_di_14", "minus_di_14",
            "obv", "vpt",
            "sma_5_close_ratio", "atr_pct_close",
            "hl_ratio", "hml_ratio", "trend_direction"
        ]
        
        # 存在する列のみを使用
        available_cols = [col for col in base_cols if col in df.columns]
        
        if len(available_cols) < self.base_features_count:
            logger.warning(
                f"Expected {self.base_features_count} base features, "
                f"found {len(available_cols)}. Using available features."
            )
        
        return available_cols
    
    def calculate_mtf_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """MTF 特徴量（27次元）を計算 (Resampling + Forward Fill)"""
        df_copy = df.copy()
        mtf_cols: list[str] = []
        
        if "close" not in df_copy.columns:
            logger.warning("'close' column not found, skipping MTF features")
            return df_copy, mtf_cols
            
        # Ensure DatetimeIndex for resampling
        temp_df = df_copy.copy()
        if "timestamp" in temp_df.columns:
            temp_df["timestamp"] = pd.to_datetime(temp_df["timestamp"])
            temp_df.set_index("timestamp", inplace=True)
        elif not isinstance(temp_df.index, pd.DatetimeIndex):
            # Fallback: Create dummy index
            temp_df.index = pd.date_range("2024-01-01", periods=len(temp_df), freq="1min")
            
        features_per_timeframe = [
            ("rsi", lambda c: self._calculate_rsi(c, period=14)),
            ("rsi_long", lambda c: self._calculate_rsi(c, period=21)),
            ("macd", lambda c: self._calculate_macd(c)),
            ("bb_width", lambda c: self._calculate_bb_width(c)),
            ("bb_pct", lambda c: self._calculate_bb_pct(c)),
            ("volatility", lambda c: self._calculate_volatility(c, period=20)),
            ("volatility_long", lambda c: self._calculate_volatility(c, period=30)),
            ("trend_strength", lambda c: self._calculate_trend_strength(c)),
            ("momentum", lambda c: self._calculate_momentum(c)),
        ]
        
        # Mapping timeframe strings to pandas offsets
        tf_map = {"5m": "5min", "15m": "15min", "1h": "1h"}
        
        for timeframe, offset in tf_map.items():
            try:
                # Resample: OHLC logic
                # Fix Finding 1: Prevent Lookahead Leakage
                # default resample (label='left', closed='left') for minutes means 12:00 row contains 12:00-12:05 data.
                # This data is only available at 12:05.
                # We shift the logic to ensure causal availability.
                resampled = temp_df.resample(offset, label='left', closed='left').agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                }).dropna()
                
                # Calculate features on resampled data
                resampled_feats = pd.DataFrame(index=resampled.index)
                for feat_name, feat_func in features_per_timeframe:
                    col_name = f"mtf_{feat_name}_{timeframe}"
                    # Calculate on resampled series
                    resampled_vals = feat_func(resampled["close"].values)
                    resampled_feats[col_name] = resampled_vals
                    mtf_cols.append(col_name)
                
                # Shift the index forward by one offset to enforce causality
                # e.g. Data computed from 12:00-12:05 (labeled 12:00) becomes available at 12:05.
                # So we move 12:00 -> 12:05.
                # When reindexing at 12:01, ffill will find 11:55 (shifted to 12:00), which is correct.
                # 12:01 should NOT see 12:00-12:05 data (which is at 12:05 now).
                resampled_feats.index = resampled_feats.index + pd.to_timedelta(offset)

                # Project back to original timeframe (Forward Fill)
                projected_feats = resampled_feats.reindex(temp_df.index, method='ffill').fillna(0)
                
                # Assign to df_copy
                for col in projected_feats.columns:
                    df_copy[col] = projected_feats[col].values
                    
            except Exception as e:
                logger.warning(f"Failed to calculate MTF {timeframe}: {e}")
                # Fallback: fill with zeros if resampling fails
                for feat_name, _ in features_per_timeframe:
                    col_name = f"mtf_{feat_name}_{timeframe}"
                    df_copy[col_name] = 0.0
                    mtf_cols.append(col_name)
        
        # Scale and clip MTF features to prevent overflow
        for col in mtf_cols:
            if col in df_copy.columns:
                # Robust scaling: clip to reasonable range and normalize
                values = df_copy[col].values
                # Clip extreme values
                values = np.clip(values, -1e6, 1e6)
                # Normalize by median absolute deviation for robustness
                if len(values) > 0:
                    median = np.median(np.abs(values))
                    if median > 0:
                        values = values / median
                    # Final clip to [-10, 10] range
                    values = np.clip(values, -10.0, 10.0)
                df_copy[col] = values
        
        logger.info(f"✓ Calculated {len(mtf_cols)} MTF features (Resampled & Scaled)")
        return df_copy, mtf_cols
    
    @staticmethod
    def _calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI 計算"""
        if len(close) < period + 1:
            return np.zeros(len(close))
        
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(len(close))
        avg_loss = np.zeros(len(close))
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
            rsi = 100 - (100 / (1 + rs))
        
        return np.nan_to_num(rsi, nan=50.0)
    
    @staticmethod
    def _calculate_macd(close: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
        """MACD 計算"""
        if len(close) < slow:
            return np.zeros(len(close))
        
        ema_fast = FeaturePipeline._calculate_ema(close, fast)
        ema_slow = FeaturePipeline._calculate_ema(close, slow)
        macd = ema_fast - ema_slow
        
        return np.nan_to_num(macd, nan=0.0)
    
    @staticmethod
    def _calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
        """EMA 計算"""
        if len(data) < period:
            return np.zeros(len(data))
        
        ema = np.zeros(len(data))
        multiplier = 2.0 / (period + 1)
        ema[period - 1] = np.mean(data[:period])
        
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]
        
        return ema
    
    @staticmethod
    def _calculate_bb_width(close: np.ndarray, period: int = 20) -> np.ndarray:
        """Bollinger Bands 幅計算 (causal)"""
        if len(close) < period:
            return np.zeros(len(close))
        
        # Causal rolling mean (no look-ahead)
        sma = np.full_like(close, fill_value=np.nan)
        for i in range(period - 1, len(close)):
            sma[i] = np.mean(close[max(0, i - period + 1):i + 1])
        sma[0:period-1] = sma[period-1]  # Forward fill initial
        
        std = np.zeros(len(close))
        
        for i in range(period - 1, len(close)):
            std[i] = np.std(close[max(0, i - period + 1):i + 1])
        
        width = 2 * std
        return np.nan_to_num(width, nan=0.0)
    
    @staticmethod
    def _calculate_volatility(close: np.ndarray, period: int = 20) -> np.ndarray:
        """ボラティリティ計算"""
        if len(close) < period:
            return np.zeros(len(close))
        
        returns = np.diff(close) / close[:-1]
        volatility = np.zeros(len(close))
        
        for i in range(period, len(close)):
            volatility[i] = np.std(returns[max(0, i - period):i])
        
        return np.nan_to_num(volatility, nan=0.0)
    
    @staticmethod
    def _calculate_bb_pct(close: np.ndarray, period: int = 20) -> np.ndarray:
        """Bollinger Bands %B 計算 (causal)"""
        if len(close) < period:
            return np.ones(len(close)) * 0.5
        
        # Causal rolling mean (no look-ahead)
        sma = np.full_like(close, fill_value=np.nan)
        for i in range(period - 1, len(close)):
            sma[i] = np.mean(close[max(0, i - period + 1):i + 1])
        sma[0:period-1] = sma[period-1]  # Forward fill initial
        
        std = np.zeros(len(close))
        
        for i in range(period - 1, len(close)):
            std[i] = np.std(close[max(0, i - period + 1):i + 1])
        
        upper = sma + 2 * std
        lower = sma - 2 * std
        
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_b = (close - lower) / (upper - lower)
        
        return np.nan_to_num(pct_b, nan=0.5)
    
    @staticmethod
    def _calculate_trend_strength(close: np.ndarray, period: int = 14) -> np.ndarray:
        """Trend strength 計算"""
        if len(close) < period:
            return np.zeros(len(close))
        
        high_low_diff = np.zeros(len(close))
        close_close_diff = np.zeros(len(close))
        
        for i in range(period, len(close)):
            close_high = np.max(close[i - period:i])
            close_low = np.min(close[i - period:i])
            high_low_diff[i] = close_high - close_low
            close_close_diff[i] = np.abs(close[i] - close[i - period])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            trend_strength = close_close_diff / (high_low_diff + 1e-6)
        
        return np.nan_to_num(trend_strength, nan=0.0)
    
    @staticmethod
    def _calculate_momentum(close: np.ndarray, period: int = 12) -> np.ndarray:
        """Momentum 計算"""
        if len(close) < period:
            return np.zeros(len(close))
        
        momentum = np.zeros(len(close))
        for i in range(period, len(close)):
            momentum[i] = close[i] - close[i - period]
        
        return momentum
    
    def calculate_regime_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Regime 特徴量（13次元）を計算"""
        df_copy = df.copy()
        regime_cols: list[str] = []
        
        # Trend regime (3)
        trend_cols = self._calculate_trend_regime(df_copy)
        regime_cols.extend(trend_cols)
        
        # Volatility regime (3)
        vol_cols = self._calculate_volatility_regime(df_copy)
        regime_cols.extend(vol_cols)
        
        # Volume regime (3)
        vol_regime_cols = self._calculate_volume_regime(df_copy)
        regime_cols.extend(vol_regime_cols)
        
        # Price regime (4)
        price_cols = self._calculate_price_regime(df_copy)
        regime_cols.extend(price_cols)
        
        logger.info(f"✓ Calculated {len(regime_cols)} regime features")
        return df_copy, regime_cols
    
    @staticmethod
    def _calculate_trend_regime(df: pd.DataFrame) -> list[str]:
        """Trend Regime (uptrend, neutral, downtrend)"""
        cols: list[str] = []
        
        if "close" in df.columns:
            close = df["close"].values
            sma_fast = pd.Series(close).rolling(window=10).mean().values
            sma_slow = pd.Series(close).rolling(window=30).mean().values
            
            uptrend = (sma_fast > sma_slow).astype(float)
            downtrend = (sma_fast < sma_slow).astype(float)
            neutral = 1 - uptrend - downtrend
            
            df["regime_trend_up"] = uptrend
            df["regime_trend_down"] = downtrend
            df["regime_trend_neutral"] = neutral
            
            cols = ["regime_trend_up", "regime_trend_down", "regime_trend_neutral"]
        
        return cols
    
    @staticmethod
    def _calculate_volatility_regime(df: pd.DataFrame) -> list[str]:
        """Volatility Regime (low, medium, high)"""
        cols: list[str] = []
        
        if "close" not in df.columns:
            return cols
        
        close = df["close"].values
        returns = np.diff(close) / close[:-1]
        
        # length mismatch を避けるため、返品の最初に 0 を追加
        returns_padded = np.concatenate([[0], returns])
        
        volatility = pd.Series(returns_padded).rolling(window=20).std().values
        volatility = np.nan_to_num(volatility, nan=0.0)
        
        vol_25 = np.percentile(volatility[volatility > 0], 33) if np.any(volatility > 0) else 0.001
        vol_75 = np.percentile(volatility[volatility > 0], 67) if np.any(volatility > 0) else 0.01
        
        low_vol = (volatility <= vol_25).astype(float)
        med_vol = ((volatility > vol_25) & (volatility <= vol_75)).astype(float)
        high_vol = (volatility > vol_75).astype(float)
        
        df["regime_vol_low"] = low_vol
        df["regime_vol_med"] = med_vol
        df["regime_vol_high"] = high_vol
        
        cols = ["regime_vol_low", "regime_vol_med", "regime_vol_high"]
        return cols
    
    @staticmethod
    def _calculate_volume_regime(df: pd.DataFrame) -> list[str]:
        """Volume Regime (low, medium, high)"""
        cols: list[str] = []
        
        if "volume" in df.columns:
            volume = df["volume"].values
            vol_25 = np.percentile(volume, 33)
            vol_75 = np.percentile(volume, 67)
            
            low_vol_regime = (volume <= vol_25).astype(float)
            med_vol_regime = ((volume > vol_25) & (volume <= vol_75)).astype(float)
            high_vol_regime = (volume > vol_75).astype(float)
            
            df["regime_vol_low"] = low_vol_regime
            df["regime_vol_med"] = med_vol_regime
            df["regime_vol_high"] = high_vol_regime
            
            cols = ["regime_vol_low", "regime_vol_med", "regime_vol_high"]
        
        return cols
    
    @staticmethod
    def _calculate_price_regime(df: pd.DataFrame) -> list[str]:
        """Price Regime (oversold, neutral, overbought, extreme)"""
        cols: list[str] = []
        
        if "close" in df.columns:
            close = df["close"].values
            sma = pd.Series(close).rolling(window=20).mean().values
            std = pd.Series(close).rolling(window=20).std().values
            
            z_score = (close - sma) / (std + 1e-6)
            
            oversold = (z_score < -2).astype(float)
            neutral = (np.abs(z_score) <= 2).astype(float)
            overbought = (z_score > 2).astype(float)
            extreme = (np.abs(z_score) > 3).astype(float)
            
            df["regime_price_oversold"] = oversold
            df["regime_price_neutral"] = neutral
            df["regime_price_overbought"] = overbought
            df["regime_price_extreme"] = extreme
            
            cols = ["regime_price_oversold", "regime_price_neutral", 
                    "regime_price_overbought", "regime_price_extreme"]
        
        return cols

class EnvironmentFactory:
    """FastIntradayEnvV456 の型安全ファクトリー"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 1_000_000.0,
        max_position: float = 1.0,
        commission_rate: float = 0.001,
        config: dict[str, Any] | None = None,
    ):
        """
        Args:
            df: OHLCV データフレーム (close, volume 必須)
            initial_balance: 初期残高
            max_position: 最大ポジション
            commission_rate: 手数料
            config: 環境設定辞書
        """
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.commission_rate = commission_rate
        self.config = config or {}
        self.pipeline = FeaturePipeline()
    
    def prepare_features(self) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """全特徴量を準備"""
        df: pd.DataFrame = self.df.copy()
        feature_cols: dict[str, list[str]] = {}
        
        # Base 特徴量
        def prepare_base() -> list[str]:
            nonlocal df
            cols = self.pipeline.validate_base_features(df)
            if len(cols) < 30:
                # 足りない分は警告ログして、deterministic ダミー特徴量を追加
                logger.warning(f"Missing {30 - len(cols)} base features. Adding deterministic fillers.")
                # np.random.seed(42)  # Deterministic seed for reproducibility - removed for config-driven seed
                for i in range(len(cols), 30):
                    col_name = f"base_dummy_{i}"
                    df[col_name] = np.random.randn(len(df))
                    cols.append(col_name)
            return cols[:30]
        
        base_cols = safe_operation(
            prepare_base,
            default_result=[],
            operation_name="prepare_base_features"
        )
        feature_cols["base"] = base_cols
        
        # MTF 特徴量
        def prepare_mtf() -> list[str]:
            nonlocal df
            df_with_mtf, mtf_cols = self.pipeline.calculate_mtf_features(df)
            df = df_with_mtf
            return mtf_cols[:27]
        
        mtf_cols = safe_operation(
            prepare_mtf,
            default_result=[],
            operation_name="prepare_mtf_features"
        )
        feature_cols["mtf"] = mtf_cols
        
        # Regime 特徴量
        def prepare_regime() -> list[str]:
            nonlocal df
            df_with_regime, regime_cols = self.pipeline.calculate_regime_features(df)
            df = df_with_regime
            return regime_cols[:13]
        
        regime_cols = safe_operation(
            prepare_regime,
            default_result=[],
            operation_name="prepare_regime_features"
        )
        feature_cols["regime"] = regime_cols
        
        return df, feature_cols
    
    def create_training_env(self, env_kwargs: dict[str, Any] | None = None) -> FastIntradayEnvV456 | None:
        """訓練環境を作成（型安全）"""
        try:
            # 特徴量準備
            df_prepared, feature_cols = self.prepare_features()
            
            logger.info(
                f"Feature Summary:\n"
                f"  Base: {len(feature_cols['base'])} columns\n"
                f"  MTF: {len(feature_cols['mtf'])} columns\n"
                f"  Regime: {len(feature_cols['regime'])} columns\n"
                f"  Total: {len(feature_cols['base']) + len(feature_cols['mtf']) + len(feature_cols['regime'])} columns"
            )
            
            # 環境パラメータの準備
            env_params = {
                "df": df_prepared,
                "base_feature_columns": feature_cols["base"],
                "mtf_feature_columns": feature_cols["mtf"],
                "regime_feature_columns": feature_cols["regime"],
                "initial_balance": self.initial_balance,
                "max_position": self.max_position,
                "commission_rate": self.commission_rate,
                "env_config": self.config,  # env_configを追加
            }
            
            # env_kwargs マージ
            if env_kwargs:
                # 既存のキーと重複する場合は env_kwargs を優先か、警告を出すか
                # ここでは安全に update する
                for k, v in env_kwargs.items():
                    if k in env_params and k not in ["df", "base_feature_columns", "mtf_feature_columns", "regime_feature_columns"]:
                        # 重要なキーは上書き許可
                        env_params[k] = v
                    elif k not in env_params:
                        # 新規キーは更に追加 (min_delta など)
                        env_params[k] = v
                        
            # 環境作成
            env = FastIntradayEnvV456(**env_params)
            
            logger.info(f"✓ Environment created: obs_shape={env.observation_space.shape}")
            return env
        
        except Exception as e:
            logger.error(f"Failed to create environment: {e}", exc_info=True)
            return None
