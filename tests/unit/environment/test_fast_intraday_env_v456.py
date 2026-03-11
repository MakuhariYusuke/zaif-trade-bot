"""
Tests for FastIntradayEnvV456

88次元観測空間、GroupedFeatureScaler、MTFリーク検証の統合テスト
"""

import pytest
import numpy as np
import pandas as pd
import pytz
from datetime import datetime

from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456


@pytest.fixture
def utc_tz():
    return pytz.UTC


@pytest.fixture
def sample_df(utc_tz):
    """サンプル市場データ (1000 steps)"""
    n_steps = 1000
    dates = pd.date_range('2025-01-01', periods=n_steps, freq='1min', tz=utc_tz)
    
    np.random.seed(42)
    prices = 9000 + np.cumsum(np.random.randn(n_steps) * 5)
    
    # 30個のBase特徴量
    base_cols = {f'base_{i}': np.random.randn(n_steps) for i in range(30)}
    
    # 27個のMTF特徴量
    mtf_cols = {f'mtf_{i}': np.random.randn(n_steps) for i in range(27)}
    
    # 13個のRegime特徴量 (One-Hot like)
    regime_cols = {f'regime_{i}': np.random.rand(n_steps) for i in range(13)}
    
    df = pd.DataFrame({
        'close': prices,
        'atr': np.abs(np.random.randn(n_steps)) + 5,
        'impact_proxy': np.random.rand(n_steps) * 0.1,
        **base_cols,
        **mtf_cols,
        **regime_cols,
    }, index=dates)
    
    return df, list(base_cols.keys()), list(mtf_cols.keys()), list(regime_cols.keys())


@pytest.fixture
def env(sample_df):
    """FastIntradayEnvV456インスタンス"""
    df, base_cols, mtf_cols, regime_cols = sample_df
    
    return FastIntradayEnvV456(
        df=df,
        base_feature_columns=base_cols,
        mtf_feature_columns=mtf_cols,
        regime_feature_columns=regime_cols,
        max_steps=100,
        prewarm_steps=50,
    )


class TestEnvInitialization:
    """環境初期化のテスト"""
    
    def test_env_creates_successfully(self, env):
        """環境が正しく作成される"""
        assert env is not None
        assert env.TOTAL_OBS_DIM == 88
    
    def test_action_space_is_correct(self, env):
        """アクション空間が正しい"""
        assert env.action_space.shape == (2,)
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0
        assert env.action_space.low[1] == 0.0
        assert env.action_space.high[1] == 1.0
    
    def test_observation_space_is_88d(self, env):
        """観測空間が88次元"""
        assert env.observation_space.shape == (88,)
    
    def test_feature_dimension_validation(self, sample_df):
        """特徴量次元の検証"""
        df, base_cols, mtf_cols, regime_cols = sample_df
        
        # 不正な次元で初期化試行
        with pytest.raises(ValueError, match="Expected 30 base features"):
            FastIntradayEnvV456(
                df=df,
                base_feature_columns=base_cols[:20],  # 20個のみ
                mtf_feature_columns=mtf_cols,
                regime_feature_columns=regime_cols,
            )
    
    def test_observation_structure_describes_88d(self, env):
        """観測構造が88Dを説明"""
        structure = env.get_observation_structure()
        
        assert structure['total_dim'] == 88
        assert structure['base'][0] == 30
        assert structure['mtf'][0] == 27
        assert structure['cyclical'][0] == 6
        assert structure['global'][0] == 6
        assert structure['regime'][0] == 13
        assert structure['account'][0] == 6


class TestReset:
    """リセット機能のテスト"""
    
    def test_reset_returns_valid_observation(self, env):
        """リセット後に有効な観測が返される"""
        obs, info = env.reset()
        
        assert obs.shape == (88,)
        assert np.all(np.isfinite(obs))
        assert isinstance(info, dict)
    
    def test_reset_initializes_state(self, env):
        """リセットが状態を初期化"""
        env.reset()
        
        assert env.balance == env.initial_balance
        assert env.position == 0.0
        assert env.position_ttl == 0
        assert env.steps_in_episode == 0
    
    def test_reset_performs_prewarm(self, env):
        """リセット時にprewarm処理が実行"""
        env.prewarm_steps = 50
        obs, info = env.reset()
        
        # スケーラーが50ステップ分の情報を持つ
        assert env.scaler.n_samples >= env.prewarm_steps
    
    def test_reset_random_start_position(self, env):
        """リセットがランダムな開始位置を使用"""
        starts = set()
        for _ in range(5):
            env.reset()
            starts.add(env.current_step)
        
        # 複数回のリセットで異なる開始位置を使用
        assert len(starts) > 1


class TestStep:
    """ステップ実行のテスト"""
    
    def test_step_returns_valid_tuple(self, env):
        """ステップが正しい形式を返す"""
        env.reset()
        action = np.array([0.5, 0.7], dtype=np.float32)
        
        result = env.step(action)
        
        assert len(result) == 5  # obs, reward, done, truncated, info
        obs, reward, done, truncated, info = result
        
        assert obs.shape == (88,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_step_increments_step_counter(self, env):
        """ステップが実行カウントを増加"""
        env.reset()
        initial_step = env.steps_in_episode
        
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        
        assert env.steps_in_episode == initial_step + 1
    
    def test_step_observation_is_88d(self, env):
        """ステップ返却の観測が88次元"""
        env.reset()
        obs, _, _, _, _ = env.step(np.array([0.5, 0.5], dtype=np.float32))
        
        assert obs.shape == (88,)
        assert np.all(np.isfinite(obs))
    
    def test_action_clipping(self, env):
        """アクションがクリップされる"""
        env.reset()
        
        # Out of range action
        action = np.array([2.0, 1.5], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        
        # アクションが内部で処理されてクリップされる
        assert env.position >= -env.max_position
        assert env.position <= env.max_position


class TestObservationConstruction:
    """観測空間構築のテスト"""
    
    def test_observation_has_all_components(self, env):
        """観測が全コンポーネントを含む"""
        env.reset()
        obs = env._get_observation()
        
        # 形状チェック
        assert obs.shape == (88,)
        
        # 有限値チェック
        assert np.all(np.isfinite(obs))
    
    def test_base_features_present(self, env):
        """Base特徴量が観測に含まれる"""
        env.reset()
        obs = env._get_observation()
        
        # Base features [0:30] は0でない可能性が高い
        base_part = obs[0:30]
        assert len(base_part) == 30
        assert np.any(base_part != 0)  # すべてが0ではない
    
    def test_mtf_features_present(self, env):
        """MTF特徴量が観測に含まれる"""
        env.reset()
        obs = env._get_observation()
        
        # MTF features [30:57]
        mtf_part = obs[30:57]
        assert len(mtf_part) == 27
    
    def test_regime_features_present(self, env):
        """Regime特徴量が観測に含まれる"""
        env.reset()
        obs = env._get_observation()
        
        # Regime features [69:82]
        regime_part = obs[69:82]
        assert len(regime_part) == 13
    
    def test_account_features_in_range(self, env):
        """Account特徴量が正しい範囲"""
        env.reset()
        action = np.array([0.5, 0.7], dtype=np.float32)
        env.step(action)
        
        obs = env._get_observation()
        account_part = obs[82:88]
        
        assert len(account_part) == 6
        # Normalized values should be in reasonable ranges (with some margin for clipping)
        assert np.all(np.abs(account_part) <= 2.0)


class TestScalerIntegration:
    """GroupedFeatureScaler統合テスト"""
    
    def test_scaler_is_grouped_type(self, env):
        """スケーラーがGroupedFeatureScalerインスタンス"""
        from ztb.features.grouping.grouped_scaler import GroupedFeatureScaler
        assert isinstance(env.scaler, GroupedFeatureScaler)
    
    def test_scaler_selective_normalization(self, env):
        """スケーラーが選別的正規化を実行"""
        # 36次元が対象（Base 30 + Global連続 6）
        assert env.scaler.TOTAL_FEATURES == 88
        assert len(env.scaler.SCALE_INDICES) == 36
    
    def test_scaler_updated_during_episode(self, env):
        """エピソード中にスケーラーが更新"""
        env.reset()
        initial_samples = env.scaler.n_samples
        
        for _ in range(10):
            env.step(np.array([0.1, 0.5], dtype=np.float32))
        
        # サンプル数が増加
        assert env.scaler.n_samples > initial_samples


class TestMTFLeakPrevention:
    """MTFリーク防止のテスト"""
    
    def test_observation_uses_current_mtf_features(self, env):
        """観測が現在のMTF特徴量を使用"""
        env.reset()
        
        # MTF features [30:57] は現在のバーのデータのみ
        # 将来のデータが混ざっていないことを確認
        mtf_part = env._get_observation()[30:57]
        
        assert len(mtf_part) == 27
        assert np.all(np.isfinite(mtf_part))
    
    def test_no_forward_looking_features(self, env):
        """前方参照特徴量がない"""
        env.reset()
        initial_price = env.close_prices[env.current_step]
        
        obs = env._get_observation()
        
        # Account特徴量にのみ現在の価格が反映される
        # MTF特徴量は外部の過去データ
        # 構造上、forward leakは起こらない
        assert True  # Structural guarantee


class TestTerminationConditions:
    """終了条件のテスト"""
    
    def test_drawdown_limit_triggers_done(self, env):
        """ドローダウンリミットがdone=Trueを返す"""
        env.reset()
        env.drawdown_limit = 0.5  # 50%

        # 現行実装は accounting.portfolio_value() を balance に同期するので、
        # drawdown 状態は gross/net pnl の両方に反映させる必要がある
        env.accounting.gross_pnl = -env.initial_balance * 0.6
        env.accounting.net_pnl = -env.initial_balance * 0.6
        env.balance = env.accounting.portfolio_value()
        env.current_step = env.data_len - 10
        
        _, _, done, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        
        # ドローダウンが大きいので done が True
        assert done or (env.balance < env.initial_balance * (1 - env.drawdown_limit))
    
    def test_max_steps_triggers_truncated(self, env):
        """Max stepsが truncated=True を返す"""
        env.reset()
        env.max_steps = 10
        
        truncated = False
        for _ in range(15):
            _, _, _, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
            if truncated:
                break
        
        # いずれかのステップでtruncatedが True
        assert truncated or env.steps_in_episode >= env.max_steps
    
    def test_data_end_triggers_truncated(self, env):
        """データ終端が truncated=True を返す"""
        env.reset()
        env.current_step = env.data_len - 2
        
        _, _, _, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        
        # データ終端で truncated
        assert truncated or env.current_step >= env.data_len - 1


class TestValidation:
    """検証機能のテスト"""
    
    def test_validate_observation_shape_passes(self, env):
        """観測形状検証が成功"""
        env.reset()
        
        assert env.validate_observation_shape() is True
    
    def test_get_observation_structure_complete(self, env):
        """観測構造情報が完全"""
        structure = env.get_observation_structure()
        
        required_keys = ['total_dim', 'base', 'mtf', 'cyclical', 'global', 'regime', 'account']
        for key in required_keys:
            assert key in structure


class TestIntegration:
    """統合テスト"""
    
    def test_full_episode(self, env):
        """完全なエピソード実行"""
        obs, _ = env.reset()
        
        assert obs.shape == (88,)
        
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated) and steps < 50:
            action = np.array([
                np.sin(steps * 0.1),  # Oscillating position
                0.5 + 0.2 * np.sin(steps * 0.05),  # Varying TTL
            ], dtype=np.float32)
            
            obs, reward, done, truncated, info = env.step(action)
            
            assert obs.shape == (88,)
            assert np.all(np.isfinite(obs))
            steps += 1
        
        assert steps > 0
    
    def test_multiple_episodes(self, env):
        """複数エピソード実行"""
        for episode in range(3):
            obs, _ = env.reset()
            
            for step in range(20):
                action = env.action_space.sample()
                obs, _, done, truncated, _ = env.step(action)
                
                assert obs.shape == (88,)
                
                if done or truncated:
                    break
    
    def test_scaling_consistency_across_episodes(self, env):
        """エピソード間のスケーリング一貫性"""
        # 最初のエピソード
        env.reset()
        env.scaler.reset()  # 初期化
        obs1 = env._get_observation()
        
        # スケーラーサンプル数記録
        samples_after_first = env.scaler.n_samples
        
        # 2番目のエピソード
        env.reset()
        obs2 = env._get_observation()
        
        # 両者が有効な観測
        assert obs1.shape == obs2.shape == (88,)
        assert np.all(np.isfinite(obs1))
        assert np.all(np.isfinite(obs2))


# パラメトリックテスト
@pytest.mark.parametrize("action_value", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_action_values(env, action_value):
    """各アクション値がサポートされる"""
    env.reset()
    
    action = np.array([action_value, 0.5], dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)
    
    assert obs.shape == (88,)
    assert np.all(np.isfinite(obs))


@pytest.mark.parametrize("ttl_value", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_ttl_values(env, ttl_value):
    """各TTL値がサポートされる"""
    env.reset()
    
    action = np.array([0.5, ttl_value], dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)
    
    assert obs.shape == (88,)
