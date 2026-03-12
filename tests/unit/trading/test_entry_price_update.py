"""
Phase 2 P1-2: Entry Price更新のテスト

ポジション反転時にentry_priceが正しく更新されることを検証
"""

import numpy as np
import pandas as pd
import pytest
import pytz

from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456


@pytest.fixture
def utc_tz():
    return pytz.UTC


@pytest.fixture
def sample_data(utc_tz) -> tuple:
    """テスト用データ生成"""
    n_steps = 300
    dates = pd.date_range("2023-01-01", periods=n_steps, freq="5min", tz=utc_tz)
    
    # シンプルなトレンドデータ
    base_price = 1000.0
    np.random.seed(42)
    prices = base_price + np.linspace(0, 200, n_steps) + np.random.randn(n_steps) * 5
    
    # 30個のBase特徴量
    base_cols = {f'base_{i}': np.random.randn(n_steps) for i in range(30)}
    
    # 27個のMTF特徴量
    mtf_cols = {f'mtf_{i}': np.random.randn(n_steps) for i in range(27)}
    
    # 13個のRegime特徴量
    regime_cols = {f'regime_{i}': np.random.rand(n_steps) for i in range(13)}
    
    df = pd.DataFrame({
        'close': prices,
        'atr': np.abs(np.random.randn(n_steps)) + 5,
        'impact_proxy': np.random.rand(n_steps) * 0.01,
        **base_cols,
        **mtf_cols,
        **regime_cols,
    }, index=dates)
    
    return df, list(base_cols.keys()), list(mtf_cols.keys()), list(regime_cols.keys())


@pytest.fixture
def env(sample_data: tuple) -> FastIntradayEnvV456:
    """テスト用環境構築"""
    df, base_cols, mtf_cols, regime_cols = sample_data
    
    env = FastIntradayEnvV456(
        df=df,
        base_feature_columns=base_cols,
        mtf_feature_columns=mtf_cols,
        regime_feature_columns=regime_cols,
        initial_balance=100000.0,
        max_position=1.0,
        max_steps=200,
        prewarm_steps=50,
        max_delta_per_step=1.0,  # ★ 反転テストのため制約緩和
        env_config={"entry_gate": {"enabled": False}},  # Entry Gate無効化
    )
    return env


class TestEntryPriceUpdateOnReversal:
    """P1-2: Entry Price更新テスト"""
    
    def test_long_to_short_reversal_updates_entry_price(self, env: FastIntradayEnvV456):
        """Long→Short反転時、entry_priceがShort約定価格に更新される"""
        env.reset()
        
        # Step 1: Long エントリー（2D action: [position, ttl]）
        action = np.array([0.8, 1.0])  # Long position 80%, full TTL
        obs, reward, done, truncated, info = env.step(action)
        
        assert env.position > 0, "Long position should be established"
        long_entry_price = env.entry_price
        assert long_entry_price > 0, "Entry price should be set"
        
        # 数ステップ進める（position維持）
        for _ in range(5):
            obs, reward, done, truncated, info = env.step(action)
        
        # Step 2: Short反転（2D action）
        action = np.array([-0.8, 1.0])  # Short position 80%, full TTL
        obs, reward, done, truncated, info = env.step(action)
        
        assert env.position < 0, f"Short position should be established, got {env.position}"
        short_entry_price = env.entry_price
        
        # ★ P1-2検証: entry_priceが更新されている
        assert short_entry_price != long_entry_price, \
            "Entry price should be updated on reversal"
        assert abs(short_entry_price - env.last_execution_price) < 1.0, \
            "Entry price should match Short execution price"
    
    def test_short_to_long_reversal_updates_entry_price(self, env: FastIntradayEnvV456):
        """Short→Long反転時、entry_priceがLong約定価格に更新される"""
        env.reset()
        
        # Step 1: Short エントリー（2D action）
        action = np.array([-0.8, 1.0])  # Short position 80%, full TTL
        obs, reward, done, truncated, info = env.step(action)
        
        assert env.position < 0, f"Short position should be established, got {env.position}"
        short_entry_price = env.entry_price
        assert short_entry_price > 0, "Entry price should be set"
        
        # 数ステップ進める
        for _ in range(5):
            obs, reward, done, truncated, info = env.step(action)
        
        # Step 2: Long反転（2D action）
        action = np.array([0.8, 1.0])  # Long position 80%, full TTL
        obs, reward, done, truncated, info = env.step(action)
        
        assert env.position > 0, f"Long position should be established, got {env.position}"
        long_entry_price = env.entry_price
        
        # ★ P1-2検証: entry_priceが更新されている
        assert long_entry_price != short_entry_price, \
            "Entry price should be updated on reversal"
        assert abs(long_entry_price - env.last_execution_price) < 1.0, \
            "Entry price should match Long execution price"
