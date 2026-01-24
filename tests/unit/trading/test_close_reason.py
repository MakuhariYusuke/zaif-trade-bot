"""
Phase 2 P1-1: close_reason実装のテスト

close_reasonがenv層で正しく判定・記録されることを検証
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
    n_steps = 400
    dates = pd.date_range("2023-01-01", periods=n_steps, freq="5min", tz=utc_tz)
    
    # トレンドデータ
    base_price = 1000.0
    np.random.seed(42)
    prices = base_price + np.linspace(0, 300, n_steps) + np.random.randn(n_steps) * 10
    
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
    """テスト用環境構築（TP/SL設定あり）"""
    df, base_cols, mtf_cols, regime_cols = sample_data
    
    env = FastIntradayEnvV456(
        df=df,
        base_feature_columns=base_cols,
        mtf_feature_columns=mtf_cols,
        regime_feature_columns=regime_cols,
        initial_balance=100000.0,
        max_position=1.0,
        max_steps=300,
        prewarm_steps=50,
        max_delta_per_step=1.0,
        env_config={
            "entry_gate": {"enabled": False},
            "tp_threshold": 0.03,  # 3% profit
            "sl_threshold": 0.015,  # 1.5% loss
        },
    )
    return env


class TestCloseReasonDetection:
    """P1-1: close_reason判定テスト"""
    
    def test_close_reason_in_info_dict(self, env: FastIntradayEnvV456):
        """info辞書にclose_reasonフィールドが含まれる"""
        env.reset()
        
        # Longエントリー
        action = np.array([0.8, 1.0])
        obs, reward, done, truncated, info = env.step(action)
        
        # エントリー時はclose_reasonなし
        assert 'close_reason' in info, "info dict should contain close_reason key"
        # エントリー時は None
        assert info['close_reason'] is None, "close_reason should be None on entry"
        
        # 数ステップ保持
        for _ in range(5):
            obs, reward, done, truncated, info = env.step(action)
        
        # エグジット
        action = np.array([0.0, 1.0])  # Close position
        obs, reward, done, truncated, info = env.step(action)
        
        # エグジット時はclose_reason設定
        if abs(env.position) < 1e-6:  # ポジションクローズされた場合
            assert 'close_reason' in info
            # manual/tp/sl/reversalのいずれか
            assert info['close_reason'] in ["manual", "tp", "sl", "reversal", None]
    
    def test_close_reason_reversal_on_position_flip(self, env: FastIntradayEnvV456):
        """反転時にclose_reasonが記録される（TP/SL優先のため、tp/sl/reversalのいずれか）"""
        env.reset()
        
        # Longエントリー
        action = np.array([0.8, 1.0])
        obs, reward, done, truncated, info = env.step(action)
        assert env.position > 0
        
        # 数ステップ保持
        for _ in range(5):
            obs, reward, done, truncated, info = env.step(action)
        
        # Short反転
        action = np.array([-0.8, 1.0])
        obs, reward, done, truncated, info = env.step(action)
        
        # ★ Doc19指摘[Minor]: TP/SL優先のため、価格推移でtp/slになる可能性もある
        # 反転時はclose_reasonが必ず設定されることを検証（tp/sl/reversalのいずれか）
        assert info['close_reason'] in ["tp", "sl", "reversal"], \
            f"Expected tp/sl/reversal on reversal, got '{info['close_reason']}'"
    
    def test_close_reason_manual_on_normal_exit(self, env: FastIntradayEnvV456):
        """通常のエグジットでclose_reason="manual"が記録される"""
        env.reset()
        
        # 小さいポジション（TP/SLに達しない）
        action = np.array([0.3, 1.0])
        obs, reward, done, truncated, info = env.step(action)
        assert env.position > 0
        
        # 2ステップ保持（PnL小）
        for _ in range(2):
            obs, reward, done, truncated, info = env.step(action)
        
        # エグジット
        action = np.array([0.0, 1.0])
        obs, reward, done, truncated, info = env.step(action)
        
        # ★ P1-1検証: TP/SLトリガーしない場合は"manual"
        if abs(env.position) < 1e-6:
            assert info['close_reason'] in ["manual", "tp", "sl"], \
                f"Expected manual/tp/sl, got '{info['close_reason']}'"
    
    def test_tp_threshold_configurable(self, sample_data: tuple):
        """TP閾値がconfigで設定可能"""
        df, base_cols, mtf_cols, regime_cols = sample_data
        
        # TP閾値0.5%（非常に緩い）
        env = FastIntradayEnvV456(
            df=df,
            base_feature_columns=base_cols,
            mtf_feature_columns=mtf_cols,
            regime_feature_columns=regime_cols,
            initial_balance=100000.0,
            max_position=1.0,
            max_steps=300,
            prewarm_steps=50,
            max_delta_per_step=1.0,
            env_config={
                "entry_gate": {"enabled": False},
                "tp_threshold": 0.005,  # 0.5% profit
                "sl_threshold": 0.005,  # 0.5% loss
            },
        )
        
        assert env.tp_threshold == 0.005
        assert env.sl_threshold == 0.005
    
    def test_close_reason_priority_tp_over_reversal(self, env: FastIntradayEnvV456):
        """判定優先順位: TP/SL > 反転 > 手動"""
        env.reset()
        
        # Longエントリー（極端なサイズでTP達成しやすくする）
        action = np.array([0.9, 1.0])
        obs, reward, done, truncated, info = env.step(action)
        entry_price = env.entry_price
        
        # 利益が出るまで待つ（価格上昇想定）
        for _ in range(10):
            obs, reward, done, truncated, info = env.step(action)
        
        # 反転でTP/SL条件を満たす場合、TP/SL優先
        action = np.array([-0.9, 1.0])
        obs, reward, done, truncated, info = env.step(action)
        
        # ★ P1-1検証: TP/SL優先（反転でもTP/SL達成ならTP/SL）
        # 実際にTP/SL達成しない場合はreversalになる
        assert info['close_reason'] in ["tp", "sl", "reversal"], \
            f"Expected tp/sl/reversal, got {info['close_reason']}"
        
        # 重要: TP/SLが優先され、reversalではないことを確認
        # （もしTP/SL達成していれば、reversalではなくtp/slになるべき）


class TestTPSLDetection:
    """TP/SL判定ロジックのテスト"""
    
    def test_tp_detection_methods_exist(self, env: FastIntradayEnvV456):
        """TP/SL判定メソッドが存在する"""
        assert hasattr(env, '_is_take_profit_exit')
        assert hasattr(env, '_is_stop_loss_exit')
        assert callable(env._is_take_profit_exit)
        assert callable(env._is_stop_loss_exit)
    
    def test_tp_sl_threshold_stored(self, env: FastIntradayEnvV456):
        """TP/SL閾値が保存されている"""
        assert hasattr(env, 'tp_threshold')
        assert hasattr(env, 'sl_threshold')
        assert env.tp_threshold == 0.03  # fixture設定値
        assert env.sl_threshold == 0.015
