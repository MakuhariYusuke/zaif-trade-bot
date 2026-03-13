"""
v459 Phase 0.2b: Entry Gate安全性の単体テスト
Doc04仕様準拠の検証: exit/close常時許可
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, Mock

# Direct method testing without full environment setup


class TestEntryGateLogic:
    """Entry Gate安全性ロジックのテスト（メソッド直接）"""
    
    def test_is_entry_action_long_entry(self):
        """ロングエントリーはゲートチェック対象"""
        from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
        
        # Create mock instance
        env = Mock(spec=FastIntradayEnvV456)
        env._is_entry_action = FastIntradayEnvV456._is_entry_action.__get__(env, FastIntradayEnvV456)
        
        # 0 → 1.0 (エントリー)
        assert env._is_entry_action(1.0, 0.0) is True
        
        # 0.5 → 1.0 (拡大)
        assert env._is_entry_action(1.0, 0.5) is True
    
    def test_is_entry_action_short_entry(self):
        """ショートエントリーはゲートチェック対象"""
        from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
        
        env = Mock(spec=FastIntradayEnvV456)
        env._is_entry_action = FastIntradayEnvV456._is_entry_action.__get__(env, FastIntradayEnvV456)
        
        # 0 → -1.0 (エントリー)
        assert env._is_entry_action(-1.0, 0.0) is True
        
        # -0.5 → -1.0 (拡大)
        assert env._is_entry_action(-1.0, -0.5) is True
    
    def test_is_entry_action_exit_allowed(self):
        """Exit/closeは常時許可（ゲートチェック不要）"""
        from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
        
        env = Mock(spec=FastIntradayEnvV456)
        env._is_entry_action = FastIntradayEnvV456._is_entry_action.__get__(env, FastIntradayEnvV456)
        
        # 1.0 → 0.0 (ロングクローズ)
        assert env._is_entry_action(0.0, 1.0) is False
        
        # -1.0 → 0.0 (ショートクローズ)
        assert env._is_entry_action(0.0, -1.0) is False
    
    def test_is_entry_action_reduce_allowed(self):
        """Reduceは常時許可"""
        from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
        
        env = Mock(spec=FastIntradayEnvV456)
        env._is_entry_action = FastIntradayEnvV456._is_entry_action.__get__(env, FastIntradayEnvV456)
        
        # 1.0 → 0.5 (ロング縮小)
        assert env._is_entry_action(0.5, 1.0) is False
        
        # -1.0 → -0.5 (ショート縮小)
        assert env._is_entry_action(-0.5, -1.0) is False
    
    def test_is_entry_action_hold(self):
        """Holdは許可（ゲートチェック不要）"""
        from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
        
        env = Mock(spec=FastIntradayEnvV456)
        env._is_entry_action = FastIntradayEnvV456._is_entry_action.__get__(env, FastIntradayEnvV456)
        
        assert env._is_entry_action(0.0, 0.0) is False
        assert env._is_entry_action(1.0, 1.0) is False
        assert env._is_entry_action(-1.0, -1.0) is False
    
    def test_convert_to_hold_action_1d(self):
        """1次元action spaceでのHOLD変換"""
        from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
        
        env = Mock(spec=FastIntradayEnvV456)
        env.action_space_type = "1d_position"
        env._convert_to_hold_action = FastIntradayEnvV456._convert_to_hold_action.__get__(env, FastIntradayEnvV456)
        
        hold_action = env._convert_to_hold_action()
        
        assert hold_action.shape == (1,)
        assert hold_action[0] == 0.0
    
    def test_convert_to_hold_action_2d(self):
        """2次元action space (position+ttl)でのHOLD変換"""
        from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
        
        env = Mock(spec=FastIntradayEnvV456)
        env.action_space_type = "2d_position_ttl"
        env._convert_to_hold_action = FastIntradayEnvV456._convert_to_hold_action.__get__(env, FastIntradayEnvV456)
        
        hold_action = env._convert_to_hold_action()
        
        assert hold_action.shape == (2,)
        assert hold_action[0] == 0.0  # position = hold
        assert hold_action[1] == 0.5  # ttl = default
    
    def test_entry_blocked_converted_to_zero_position(self):
        """エントリーブロック時はposition=0に変換される"""
        from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
        
        env = Mock(spec=FastIntradayEnvV456)
        env.action_space_type = "2d_position_ttl"
        env._convert_to_hold_action = FastIntradayEnvV456._convert_to_hold_action.__get__(env, FastIntradayEnvV456)
        
        # エントリーアクション
        original_action = np.array([0.8, 0.7])
        
        # ゲートブロック → HOLD変換
        hold_action = env._convert_to_hold_action()
        
        # position成分が0になっているか
        assert hold_action[0] == 0.0
    
    def test_exit_not_affected_by_entry_logic(self):
        """Exitは_is_entry_action=Falseなのでゲートチェックされない"""
        from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
        
        env = Mock(spec=FastIntradayEnvV456)
        env._is_entry_action = FastIntradayEnvV456._is_entry_action.__get__(env, FastIntradayEnvV456)
        
        # Long position → Close
        current_position = 1.0
        target_position = 0.0
        
        # これはentry扱いではない
        is_entry = env._is_entry_action(target_position, current_position)
        assert is_entry is False
        
        # 実際のstep()では、is_entry=Falseならゲートチェックをスキップ


class TestEntryGateSafetySpec:
    """Doc04仕様準拠の検証"""
    
    def test_doc04_spec_entry_expand_gated(self):
        """Doc04仕様: 新規エントリー/拡大のみゲートチェック"""
        from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
        
        env = Mock(spec=FastIntradayEnvV456)
        env._is_entry_action = FastIntradayEnvV456._is_entry_action.__get__(env, FastIntradayEnvV456)
        
        # 新規エントリー
        assert env._is_entry_action(1.0, 0.0) is True
        assert env._is_entry_action(-1.0, 0.0) is True
        
        # 拡大
        assert env._is_entry_action(1.0, 0.5) is True
        assert env._is_entry_action(-1.0, -0.5) is True
    
    def test_doc04_spec_exit_reduce_always_allowed(self):
        """Doc04仕様: exit/close/reduceは常時許可"""
        from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
        
        env = Mock(spec=FastIntradayEnvV456)
        env._is_entry_action = FastIntradayEnvV456._is_entry_action.__get__(env, FastIntradayEnvV456)
        
        # exit/close
        assert env._is_entry_action(0.0, 1.0) is False
        assert env._is_entry_action(0.0, -1.0) is False
        
        # reduce
        assert env._is_entry_action(0.5, 1.0) is False
        assert env._is_entry_action(-0.5, -1.0) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

